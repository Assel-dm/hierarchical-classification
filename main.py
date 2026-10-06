import gc
import argparse
import signal
from pathlib import Path
import pandas as pd
import torch
import torchvision
import timm
import config as cfg
from common.models import SpeciesClassifier, resolve_model_settings, build_transform, make_optimizer
from common.training import seed_everything, train_model, load_checkpoint, save_json, check_configuration
from common.evaluation import evaluate_model
from common.visualization import build_reports, plot_augmentation_preview
from common.data import build_loaders
from datasets.small_collemboles import load_small_collemboles
from datasets.inaturalist19 import load_inaturalist19, show_summary
from methods.mbm import build_loss, soft_label_statistics


def main():
    # 1. Config check
    stage_order = ["frozen", "partial", "deeper"]
    if cfg.STAGES != ["full"] and (not cfg.STAGES or cfg.stages != [stage for stage in stage_order if stage in cfg.STAGES]):
        raise ValueError("STAGES must be ['full'] or in the order frozen, partial, deeper")
    if not cfg.SEEDS or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in cfg.SEEDS):
        raise ValueError("SEEDS must contains integer in [0, 2**32)")
    if len(set(cfg.SEEDS)) != len(cfg.SEEDS) or len(set(cfg.EXPERIMENTS)) != len(cfg.EXPERIMENTS):
        raise ValueError("An experience or seed is repeeted")
    if not cfg.EXPERIMENTS or cfg.BATCH_SIZE < 1 or cfg.NUM_WORKERS < 0:
        raise ValueError("Check EXPERIMENTS, BATCH_SIZE and NUM_WORKERS")
    if Path(cfg.EXPERIMENT_NAME).name != cfg.EXPERIMENT_NAME or cfg.EXPERIMENT_NAME in {"", ".", ".."}:
        raise ValueError("EXPERIMENT_NAME must be a folder name")
    if cfg.AUGMENTATION not in {"standard", "autoaugment_original"}:
        raise ValueError("AUGMENTATION must be standard or autoaugment_original")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if cfg.DEVICE == "auto" else torch.device(cfg.DEVICE)
    if device.type not in {"cpu", "cuda"} or (device.type == "cuda" and not torch.cuda.is_available()):
        raise RuntimeError(f"Device unavailable : {device}")
    print("Device :", device, "| timm :", timm.__version__)

    # 2. Dataset and class representation
    if cfg.DATASET == "small_collemboles":
        data = load_small_collemboles(cfg.DATASET_ROOT)
    elif cfg.DATASET == "inaturalist19_h":
        data = load_inaturalist19(
            cfg.DATASET_ROOT,
            getattr(cfg, "INAT_HIERARCHY_FILE", None),
            getattr(cfg, "INAT_CATEGORIES_FILE", None),
        )
        show_summary(data)
    else:
        raise NotImplementedError(f"Dataset to be added main.py : {cfg.DATASET}")

    if cfg.METHOD != "mbm":
        raise NotImplementedError(f"Method to be added main.py : {cfg.METHOD}")

    for variant, parameter in cfg.EXPERIMENTS:
        # Check choices before training
        build_loss(
            variant,
            parameter,
            data,
            getattr(cfg, "HXE_NORMALIZE_WEIGHTS", False),
        )

    for stage in cfg.STAGES:
        spec = cfg.STAGE_SETTINGS[stage]
        if (
            spec["learning_rate"] <= 0
            or spec["max_epochs"] < 1
            or spec["patience"] < 1
        ):
            raise ValueError(f"Learning parameters invalid for {stage}")

    settings = resolve_model_settings(
        cfg.MODEL_KEY, cfg.IMAGE_SIZE, cfg.HEAD_DROPOUT, cfg.FREEZE_BATCH_NORM
    )
    campaign_dir = (
        Path(cfg.OUTPUT_ROOT).expanduser().resolve() / cfg.EXPERIMENT_NAME
    )

    campaign_config = {
        "dataset": data["name"], "taxonomy": data["taxonomy"].to_dict("records"),
        "split_counts": {split: len(samples) for split, samples in data["samples"].items()},
        "model": settings, "pretrained": cfg.PRETRAINED, "augmentation": cfg.AUGMENTATION,
        "batch_size": cfg.BATCH_SIZE, "method": cfg.METHOD, "loss_version": "mbm_v1",
        "torch": str(torch.__version__).split("+")[0], "torchvision": torchvision.__version__.split("+")[0],
        "timm": timm.__version__,
    }
    if data["name"] == "inaturalist19_h":
        campaign_config["loss_version"] = "mbm_seven_ranks_v1"
    if getattr(cfg, "HXE_NORMALIZE_WEIGHTS", False):
        campaign_config["hxe_normalize_weights"] = True
    
    check_configuration(campaign_dir / "campaign.json", campaign_config)
    save_json(campaign_dir / "runtime.json", {"dataset_root": str(data["root"]), "device": str(device),
                                            "num_workers": cfg.NUM_WORKERS})
    reports_dir = campaign_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    counts = data["taxonomy"].copy()
    for split, samples in data["samples"].items():
        counts[split] = pd.Series([label for _, label in samples]).value_counts().reindex(range(len(counts)), fill_value=0).to_numpy()
    counts.to_csv(reports_dir / "dataset_counts.csv", index=False)
    
    soft_settings = [
        soft_label_statistics(data["normalized_distances"], parameter)
        for variant, parameter in cfg.EXPERIMENTS
        if variant == "soft"
    ]
    if soft_settings:
        pd.DataFrame(soft_settings).to_csv(
            reports_dir / "soft_label_statistics.csv", index=False
        )
        for row in soft_settings:
            print(
                f"Soft Labels beta={row['beta']:g} : "
                f"mean mass for the real species={row['true_class_mass']:.3f}"
            )
    print("Results :", campaign_dir)
    print("Classes :", len(data["species"]), "| Images :", campaign_config["split_counts"])

    # 3. Augmentation preview
    if cfg.PREVIEW_AUGMENTATIONS:
        transforms = {name: build_transform(settings, True, name)
                      for name in ("standard", "autoaugment_original")}
        plot_augmentation_preview(data["samples"]["train"][0][0], transforms, settings,
                                  reports_dir / "augmentation_preview.png")

    # 4. Training campaign
    for seed in cfg.SEEDS:
        for variant, parameter in cfg.EXPERIMENTS:
            variant_name = variant if parameter is None else f"{variant}_{parameter}".replace(".", "p")
            experiment_dir = campaign_dir / "runs" / f"{variant_name}_seed{seed}"
            for stage in cfg.STAGES:
                spec = cfg.STAGE_SETTINGS[stage]
                run_dir = experiment_dir / stage
                metadata = {"campaign": cfg.EXPERIMENT_NAME, "model_key": cfg.MODEL_KEY,
                            "augmentation": cfg.AUGMENTATION, "variant": variant, "parameter": parameter,
                            "seed": seed, "stage": stage, "training": spec}
                check_configuration(run_dir / "metadata.json", metadata)
                print(f"\n{variant_name} | seed {seed} | {stage}")
                seed_everything(seed)
                new_frozen = stage == "frozen" and not (run_dir / "last.pt").exists() and not (run_dir / "complete.json").exists()
                model = SpeciesClassifier(settings, len(data["species"]),
                                          pretrained=cfg.PRETRAINED and cfg.RUN_TRAINING and new_frozen)
                model.set_stage(stage)
                model.to(device)
                criterion = build_loss(variant, parameter, data, getattr(cfg,"HXE_NORMALIZE_WEIGHTS", False),).to(device)
                loaders = build_loaders(data, settings, cfg.BATCH_SIZE, cfg.NUM_WORKERS,
                                        seed, cfg.AUGMENTATION, device)

                # The unfreeze start back from the best weights of the previous phase
                if cfg.RUN_TRAINING and stage != "frozen" and not (run_dir / "last.pt").exists() and not (run_dir / "complete.json").exists():
                    source_stage = stage_order[stage_order.index(stage) - 1]
                    source_dir = experiment_dir / source_stage
                    if not (source_dir / "complete.json").exists():
                        raise FileNotFoundError(f"First finish stage {source_stage} : {source_dir}")
                    source = load_checkpoint(source_dir / "best.pt")
                    expected = {**metadata, "stage": source_stage, "training": cfg.STAGE_SETTINGS[source_stage]}
                    if source["metadata"] != expected:
                        raise ValueError("Previous phase use a different configuration")
                    model.load_state_dict(source["model_state"])
                    print("Initialisation from :", source_dir / "best.pt")

                if cfg.RUN_TRAINING:
                    optimizer = make_optimizer(model, spec["learning_rate"])
                    train_model(model, criterion, optimizer, loaders, device, run_dir,
                                spec["max_epochs"], spec["patience"], metadata, cfg.RESUME_TRAINING)
                    del optimizer
                else:
                    saved = load_checkpoint(run_dir / "best.pt")
                    if saved["metadata"] != metadata:
                        raise ValueError("Best weights use a different configuration")
                    model.load_state_dict(saved["model_state"])

                # 5. Evaluation on validation or test 
                for split, enabled in (("validation", cfg.EVALUATE_VALIDATION), ("test", cfg.EVALUATE_TEST)):
                    if enabled:
                        scores = evaluate_model(model, loaders[split], device, data, run_dir,
                                                split, cfg.SAVE_PROBABILITIES)
                        print(f"{split} : accuracy={scores['top1_accuracy']:.3f}, "
                              f"F1 macro={scores['f1_macro']:.3f}, severity={scores['mistake_severity']:.3f}")
                del model, criterion, loaders
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    # 6. Graphs and comparison of results
    if cfg.MAKE_PLOTS:
        build_reports(campaign_dir)


def stop_requested(signum, frame):
    """Allows a restart after stopping the program"""
    raise KeyboardInterrupt


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    signal.signal(signal.SIGTERM, stop_requested)
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Restart with RESUME_TRAINING=True to resume")
        raise SystemExit(130)
