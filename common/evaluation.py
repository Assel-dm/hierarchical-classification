"""Evaluation of best weights and saving of metrics""" 
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from common.training import save_json


def collect_predictions(model, loader, device):
    """Recover labels and probabilities on all splits"""
    model.eval()
    labels, probabilities = [], []
    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images.to(device, non_blocking=True))
            labels.append(targets.numpy())
            probabilities.append(logits.softmax(dim=1).cpu().numpy())
    return np.concatenate(labels), np.concatenate(probabilities)


def compute_metrics(targets, probabilities, data):
    """Compute metrics"""
    n_classes = len(data["species"])
    if (probabilities.shape != (len(targets), n_classes) or not len(targets) or not np.isfinite(probabilities).all() or (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)):
        raise ValueError("Invalid probabilities or misaligned classes")
    predictions = probabilities.argmax(axis=1)
    mistakes = predictions != targets
    distances = data["distance_matrix"][targets, predictions]
    scores = {"top1_accuracy": float((~mistakes).mean()), "top1_error": float(mistakes.mean()),
              "species_accuracy": float((~mistakes).mean()),
              "mistake_severity": float(distances[mistakes].mean()) if mistakes.any() else 0.}
    ranking = np.argsort(probabilities, axis=1)[:, ::-1]
    for k in (1, 5, 20):
        top = predictions[:, None] if k == 1 else ranking[:, :min(k, n_classes)]
        scores[f"top{k}_accuracy"] = float((top == targets[:, None]).any(axis=1).mean())
        scores[f"avg_hierarchical_distance@{k}"] = float(data["distance_matrix"][targets[:, None], top].mean())
    for average in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(targets, predictions, labels=np.arange(n_classes), average=average, zero_division=0)
        scores.update({f"precision_{average}": float(p), f"recall_{average}": float(r), f"f1_{average}": float(f)})
    for rank, mapping in data["mappings"].items():
        true_ancestors = mapping[targets].argmax(axis=1)
        predicted_ancestors = (probabilities @ mapping).argmax(axis=1)
        scores[f"{rank}_accuracy"] = float((true_ancestors == predicted_ancestors).mean())
    return scores


def evaluate_model(model, loader, device, data, run_dir, split, save_probabilities=False):
    """Write scores, F1 per clasees, cinfusion and error per image""" 
    targets, probabilities = collect_predictions(model, loader, device)
    predictions = probabilities.argmax(axis=1)
    scores = compute_metrics(targets, probabilities, data)
    run_dir = Path(run_dir)
    save_json(run_dir / f"metrics_{split}.json", scores)

    precision, recall, f1, support = precision_recall_fscore_support(targets, predictions, labels=np.arange(len(data["species"])), zero_division=0)
    table = data["taxonomy"].copy()
    table.insert(0, "class_id", np.arange(len(table)))
    table["precision"], table["recall"], table["f1"], table["support"] = precision, recall, f1, support
    table.to_csv(run_dir / f"per_class_{split}.csv", index=False)
    table.sort_values("f1", ascending=False).to_csv(run_dir / f"per_class_ranked_{split}.csv", index=False)

    confusion = confusion_matrix(targets, predictions, labels=np.arange(len(table)))
    pd.DataFrame(confusion, index=data["species"], columns=data["species"]).to_csv(run_dir / f"confusion_{split}.csv")
    image_paths = [str(path.relative_to(data["root"])) for path, _ in loader.dataset.samples]
    pd.DataFrame({"image": image_paths, "true_id": targets, "predicted_id": predictions,
                  "true_species": np.asarray(data["species"])[targets],
                  "predicted_species": np.asarray(data["species"])[predictions],
                  "confidence": probabilities.max(axis=1),
                  "taxonomic_distance": data["distance_matrix"][targets, predictions]}).to_csv(
        run_dir / f"predictions_{split}.csv", index=False)
    if save_probabilities:
        np.savez_compressed(run_dir / f"probabilities_{split}.npz", targets=targets,
                            probabilities=probabilities, species=np.asarray(data["species"]))
    return scores
