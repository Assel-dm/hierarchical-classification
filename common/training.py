"""Training loops"""
import json
import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch


def seed_everything(seed):
    """Reset random state"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def save_json(path, values):
    """Write configuration or result file"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(values, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def save_checkpoint(path, values):
    """Save checkpoints"""
    temporary = Path(path).with_suffix(".pt.tmp")
    torch.save(values, temporary)
    os.replace(temporary, path)


def load_checkpoint(path):
    """Read a checkpoint"""
    return torch.load(path, map_location="cpu", weights_only=False)


def check_configuration(path, configuration):
    """Avoid having 2 similar experiments"""
    path = Path(path)
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != configuration:
            raise ValueError(f"Different configuration {path.parent}. Choose a new EXPERIMENT_NAME")
    else:
        save_json(path, configuration)


def epoch_pass(model, loader, criterion, device, optimizer=None):
    """For one epoch compute loss, top-1 et top-5."""
    training = optimizer is not None
    model.train(training)
    total_loss = correct = top5 = count = 0
    with torch.enable_grad() if training else torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Unfinished loss; the interrupted epoch is not saved")
            if training:
                loss.backward()
                optimizer.step()
            n = len(targets)
            total_loss += loss.item() * n
            correct += (logits.argmax(1) == targets).sum().item()
            top5 += logits.topk(min(5, logits.shape[1]), dim=1).indices.eq(targets[:, None]).any(1).sum().item()
            count += n
    if count == 0:
        raise ValueError("Void split ")
    return {"loss": total_loss / count, "accuracy": correct / count, "top5_accuracy": top5 / count}


def train_model(model, criterion, optimizer, loaders, device, run_dir,
                max_epochs, patience, metadata, resume=True):
    """Train the phase and save its best weights"""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path, last_path = run_dir / "best.pt", run_dir / "last.pt"
    if not resume and (best_path.exists() or last_path.exists()):
        raise FileExistsError(f"Results already present in {run_dir}. Choose a new EXPERIMENT_NAME.")
    if (run_dir / "complete.json").exists():
        model.load_state_dict(load_checkpoint(best_path)["model_state"])
        print("Phase already finished :", run_dir)
        return pd.read_csv(run_dir / "history.csv")

    start_epoch, stale, best_loss, history = 0, 0, float("inf"), []
    if resume and last_path.exists():
        checkpoint = load_checkpoint(last_path)
        if checkpoint["metadata"] != metadata:
            raise ValueError("The checkpoint does not match the current configuration")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        stale, best_loss, history = checkpoint["stale"], checkpoint["best_loss"], checkpoint["history"]
        rng = checkpoint["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if rng["cuda"] is not None and device.type == "cuda":
            torch.cuda.set_rng_state_all(rng["cuda"])
        for split in ("train", "validation"):
            loaders[split].generator.set_state(rng[split])
        print(f"Restart at epoch {start_epoch + 1} : {run_dir.name}")

    for epoch in range(start_epoch, max_epochs):
        if stale >= patience:
            break
        train = epoch_pass(model, loaders["train"], criterion, device, optimizer)
        validation = epoch_pass(model, loaders["validation"], criterion, device)
        row = {"epoch": epoch + 1, **train, **{f"val_{key}": value for key, value in validation.items()}}
        history.append(row)
        if row["val_loss"] < best_loss:
            best_loss, stale = row["val_loss"], 0
            save_checkpoint(best_path, {"model_state": model.state_dict(), "epoch": epoch,
                                        "val_loss": best_loss, "metadata": metadata})
        else:
            stale += 1

        rng = {"python": random.getstate(), "numpy": np.random.get_state(),
               "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
               "train": loaders["train"].generator.get_state(),
               "validation": loaders["validation"].generator.get_state()}
        save_checkpoint(last_path, {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                                    "epoch": epoch, "best_loss": best_loss, "stale": stale,
                                    "history": history, "rng": rng, "metadata": metadata})
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        print(f"  Epoch {epoch + 1}/{max_epochs} | loss {row['loss']:.4f} | "
              f"val_loss {row['val_loss']:.4f} | val_accuracy {row['val_accuracy']:.4f}", flush=True)

    if not best_path.exists():
        raise RuntimeError("No best checkpoint available")
    frame = pd.DataFrame(history)
    frame.to_csv(run_dir / "history.csv", index=False)
    save_json(run_dir / "complete.json", {"epochs": len(history), "best_val_loss": best_loss})
    model.load_state_dict(load_checkpoint(best_path)["model_state"])
    return frame
