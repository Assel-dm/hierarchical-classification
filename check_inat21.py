"""Check prepared iNat21 images, loaders and MBM losses without starting a campaign"""
import argparse
from pathlib import Path

import torch

import config as cfg
from common.data import build_loaders
from common.models import SpeciesClassifier, resolve_model_settings
from datasets.inat21 import load_inat21, show_summary
from methods.mbm import build_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=cfg.DATASET_ROOT)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--model", action="store_true",
                        help="Also check a backbone forward/backward pass; no pretrained download")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    device = torch.device(args.device)
    torch.manual_seed(42)
    data = load_inat21(args.root, ranks=getattr(cfg, "INAT21_RANKS", None))
    show_summary(data)
    settings = resolve_model_settings(cfg.MODEL_KEY, cfg.IMAGE_SIZE,
                                      cfg.HEAD_DROPOUT, cfg.FREEZE_BATCH_NORM)
    loaders = build_loaders(data, settings, args.batch_size, 0, 42,
                            cfg.AUGMENTATION, device)
    batch = None
    for split, loader in loaders.items():
        images, labels = next(iter(loader))
        assert images.shape[1:] == (3, *settings["image_size"])
        assert torch.isfinite(images).all()
        assert labels.dtype == torch.long
        assert ((labels >= 0) & (labels < len(data["species"]))).all()
        print(f"{split}: first batch decoded and transformed, shape={tuple(images.shape)}")
        if split == "train":
            batch = (images.to(device), labels.to(device))

    images, labels = batch
    experiments = getattr(cfg, "EXPERIMENTS_MBM", [("ce", None), ("hxe", 0.2), ("soft", 10)])
    normalize = getattr(cfg, "HXE_NORMALIZE_WEIGHTS", False)
    for variant, parameter in experiments:
        logits = torch.randn(len(labels), len(data["species"]), device=device, requires_grad=True)
        criterion = build_loss(variant, parameter, data, normalize).to(device)
        loss = criterion(logits, labels)
        assert torch.isfinite(loss), f"Nonfinite loss: {variant}"
        loss.backward()
        assert logits.grad is not None and torch.isfinite(logits.grad).all()
        print(f"{variant}({parameter}): loss={loss.item():.6f}, finite gradients")

    # Independent mathematical check: unnormalised HXE(alpha=0) telescopes to CE.
    logits = torch.randn(len(labels), len(data["species"]), device=device)
    ce = build_loss("ce", None, data).to(device)(logits, labels)
    hxe = build_loss("hxe", 0.0, data, False).to(device)(logits, labels)
    torch.testing.assert_close(hxe, ce)
    raw = build_loss("hxe", 0.2, data, False).to(device)
    normalized = build_loss("hxe", 0.2, data, True).to(device)
    assert torch.all(normalized.weights < raw.weights), "HXE normalization flag was ignored"
    print("HXE(alpha=0) = CE; normalization flag active")

    if args.model:
        model = SpeciesClassifier(settings, len(data["species"]), pretrained=False).to(device)
        model.set_stage("full")
        logits = model(images)
        assert logits.shape == (len(labels), len(data["species"]))
        loss = build_loss("ce", None, data).to(device)(logits, labels)
        assert torch.isfinite(loss)
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert gradients and all(torch.isfinite(grad).all() for grad in gradients)
        print("Backbone + species head: forward/backward OK (random weights)")
    print("Checks passed. No training campaign or result files created.")


if __name__ == "__main__":
    main()
