"""Build CSV, PNG and HTML from checkpoints"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def save_figure(fig, path):
    """Save a figure"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_label(metadata):
    """Build readable label for figures"""
    names = {"ce": "CE", "soft": "Soft Labels", "hxe": "HXE"}
    label = names[metadata["variant"]]
    if metadata["parameter"] is not None:
        symbol = "β" if metadata["variant"] == "soft" else "α"
        label += f" {symbol}={metadata['parameter']:g}"
    return f"{label} · {metadata['stage']} · seed {metadata['seed']}"


def plot_learning_curves(run_dir, label):
    """Plot loss, accuracy and top-5 train/validation."""
    history_path = run_dir / "history.csv"
    if not history_path.exists():
        return
    history = pd.read_csv(history_path)
    for metric, title in (("loss", "Loss"), ("accuracy", "Species accuracy"), ("top5_accuracy", "Top-5 accuracy")):
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(history.epoch, history[metric], label="Train")
        ax.plot(history.epoch, history[f"val_{metric}"], "--", label="Validation")
        ax.set(title=label, xlabel="Epoch", ylabel=title)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=.2)
        ax.legend()
        save_figure(fig, run_dir / "figures" / f"learning_{metric}.png")


def plot_per_class(run_dir, split, label):
    """Print worst F1 and confusion per class"""
    table = pd.read_csv(run_dir / f"per_class_{split}.csv", dtype={"species": str})
    worst = table.sort_values("f1").head(25)
    fig, ax = plt.subplots(figsize=(9, max(4, len(worst) * .26)))
    ax.barh(worst.species, worst.f1)
    ax.invert_yaxis()
    ax.set(xlabel="F1", xlim=(0, 1), title=f"{label} · {split} · lowest F1")
    ax.grid(axis="x", alpha=.2)
    save_figure(fig, run_dir / "figures" / f"per_class_{split}.png")
    if len(table) <= 40:
        matrix = pd.read_csv(run_dir / f"confusion_{split}.csv", index_col=0)
        fig, ax = plt.subplots(figsize=(9, 8))
        image = ax.imshow(matrix.to_numpy(), cmap="Blues")
        ax.set_xticks(range(len(matrix)), matrix.columns, rotation=90, fontsize=7)
        ax.set_yticks(range(len(matrix)), matrix.index, fontsize=7)
        ax.set(xlabel="Predicted species", ylabel="True species", title=f"{label} · {split}")
        fig.colorbar(image, ax=ax, label="Images")
        save_figure(fig, run_dir / "figures" / f"confusion_{split}.png")


def plot_dataset_counts(path, output_dir):
    """Plot repartition of images through species and splits"""
    if not path.exists():
        return
    table = pd.read_csv(path, dtype={"species": str})
    table = table.sort_values("train", ascending=False)
    fig, ax = plt.subplots(figsize=(max(10, len(table) * .15), 5))
    bottom = np.zeros(len(table))
    for split in ("train", "validation", "test"):
        ax.bar(table.species, table[split], bottom=bottom, label=split)
        bottom += table[split].to_numpy()
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.set(ylabel="Images", title="Dataset distribution by species")
    ax.legend()
    save_figure(fig, output_dir / "dataset_distribution.png")


def plot_augmentation_preview(image_path, transforms, settings, output_path):
    """Compare different augmention fonr an image"""
    from PIL import Image
    with Image.open(image_path) as image:
        original = image.convert("RGB")
    mean = np.array(settings["mean"]).reshape(1, 1, 3)
    std = np.array(settings["std"]).reshape(1, 1, 3)
    fig, axes = plt.subplots(len(transforms), 5, figsize=(14, 3 * len(transforms)), squeeze=False)
    for row, (name, transform) in enumerate(transforms.items()):
        axes[row, 0].imshow(original)
        axes[row, 0].set_title(f"{name}\nOriginal")
        for col in range(1, 5):
            tensor = transform(original.copy())
            visible = np.clip(tensor.permute(1, 2, 0).numpy() * std + mean, 0, 1)
            axes[row, col].imshow(visible)
            axes[row, col].set_title(f"Example {col}")
    for ax in axes.flat:
        ax.axis("off")
    save_figure(fig, output_path)


def build_html(records, overall, seed_summary, output_path):
    """Build HTML report with Plotly"""
    import plotly.graph_objects as go
    from plotly.io import to_html
    figures = []
    learning = go.Figure()
    for record in records:
        history = record["history"]
        if history is None:
            continue
        for column, suffix, dash in (("loss", "train", "solid"), ("val_loss", "validation", "dash")):
            learning.add_trace(go.Scatter(x=history.epoch, y=history[column],
                                          name=f"{record['label']} · {suffix}",
                                          line=dict(dash=dash), visible=True if len(records) <= 4 else "legendonly"))
    learning.update_layout(title="Learning curves — loss", xaxis_title="Epoch", yaxis_title="Loss")
    figures.append(learning)

    tradeoff = go.Figure()
    for split in overall.split.unique() if not overall.empty else []:
        table = overall[overall.split == split]
        tradeoff.add_trace(go.Scatter(x=table.top1_error, y=table.mistake_severity,
                                     mode="markers", text=table.label, name=split,
                                     hovertemplate="%{text}<br>Error=%{x:.3f}<br>Severity=%{y:.3f}<extra></extra>"))
    tradeoff.update_layout(title="Hierarchical trade-off", xaxis_title="Top-1 error", yaxis_title="Mistake severity")
    figures.append(tradeoff)

    classes = go.Figure()
    class_labels = []
    for record in records:
        for split, table in record["classes"].items():
            class_labels.append(f"{record['label']} · {split}")
            classes.add_trace(go.Bar(x=table.species, y=table.f1, name=class_labels[-1],
                                     visible=len(class_labels) == 1))
    buttons = [dict(label=label, method="update", args=[{"visible": [i == j for j in range(len(class_labels))]}])
               for i, label in enumerate(class_labels)]
    classes.update_layout(title="F1 by species", yaxis=dict(title="F1", range=[0, 1]),
                          updatemenus=[dict(buttons=buttons, direction="down", x=0, y=1.2)] if buttons else [])
    figures.append(classes)
    blocks = [to_html(fig, full_html=False, include_plotlyjs=True if i == 0 else False)
              for i, fig in enumerate(figures)]
    html = ("<!doctype html><html><meta charset='utf-8'><title>MBM results</title>"
            "<style>body{font:15px system-ui;margin:24px}table{border-collapse:collapse}"
            "td,th{padding:7px;border:1px solid #ddd}.scroll{overflow:auto}</style>"
            "<h1>MBM results</h1><p>Offline report. Click legend entries to select learning curves.</p>"
            "<p>Loss magnitudes depend on the objective and its parameter; compare loss curves within the same objective.</p>"
            + "".join(blocks) + "<h2>Overall metrics</h2><div class='scroll'>"
            + overall.to_html(index=False, float_format=lambda x: f"{x:.4f}")
            + "</div><h2>Mean and standard deviation across seeds</h2><div class='scroll'>"
            + seed_summary.to_html(index=False, float_format=lambda x: f"{x:.4f}") + "</div></html>")
    output_path.write_text(html, encoding="utf-8")


def build_reports(campaign_dir):
    """Build graphics with available with available plots"""
    campaign_dir = Path(campaign_dir)
    if not campaign_dir.is_dir():
        raise FileNotFoundError(campaign_dir)
    reports_dir = campaign_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    records, rows, class_tables = [], [], []
    stage_order = {"frozen": 0, "partial": 1, "deeper": 2}
    paths = sorted((campaign_dir / "runs").glob("*/*/metadata.json"),
                   key=lambda path: (path.parent.parent.name, stage_order[path.parent.name]))
    for path in paths:
        run_dir = path.parent
        metadata = json.loads(path.read_text(encoding="utf-8"))
        label = run_label(metadata)
        plot_learning_curves(run_dir, label)
        history_path = run_dir / "history.csv"
        record = {"label": label, "history": pd.read_csv(history_path) if history_path.exists() else None, "classes": {}}
        for split in ("validation", "test"):
            metrics_path = run_dir / f"metrics_{split}.json"
            if not metrics_path.exists():
                continue
            scores = json.loads(metrics_path.read_text(encoding="utf-8"))
            rows.append({**{key: value for key, value in metadata.items() if key != "training"},
                         "label": label, "split": split, **scores})
            record["classes"][split] = pd.read_csv(run_dir / f"per_class_{split}.csv", dtype={"species": str})
            table = record["classes"][split].copy()
            for key, value in (("label", label), ("seed", metadata["seed"]), ("stage", metadata["stage"]),
                               ("variant", metadata["variant"]), ("parameter", metadata["parameter"]), ("split", split)):
                table[key] = np.nan if key == "parameter" and value is None else value
            class_tables.append(table)
            plot_per_class(run_dir, split, label)
        records.append(record)
    overall = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["label", "split"])
    overall.to_csv(reports_dir / "comparison.csv", index=False)
    if class_tables:
        pd.concat(class_tables, ignore_index=True).to_csv(reports_dir / "per_class_comparison.csv", index=False)
    summary = pd.DataFrame()
    if rows:
        groups = ["campaign", "model_key", "augmentation", "variant", "parameter", "stage", "split"]
        metrics = [column for column in overall.select_dtypes(include="number") if column not in {"seed", "parameter"}]
        grouped = overall.groupby(groups, dropna=False)
        summary = grouped[metrics].agg(["mean", "std"])
        summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
        summary["n_seeds"] = grouped.seed.nunique()
        summary = summary.reset_index()
        for split in overall.split.unique():
            table = overall[overall.split == split]
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes[0].bar(range(len(table)), table.top1_accuracy)
            axes[0].set_xticks(range(len(table)), table.label, rotation=90, fontsize=7)
            axes[0].set(title=f"Species accuracy · {split}", ylim=(0, 1))
            for variant in ("ce", "soft", "hxe"):
                selected = table[table.variant == variant]
                if not selected.empty:
                    axes[1].scatter(selected.top1_error, selected.mistake_severity, label=variant)
            axes[1].set(xlabel="Top-1 error", ylabel="Mistake severity", title="Hierarchical trade-off")
            axes[1].legend()
            axes[1].grid(alpha=.2)
            save_figure(fig, reports_dir / f"comparison_{split}.png")
            for metric in ("avg_hierarchical_distance@1", "avg_hierarchical_distance@5"):
                fig, ax = plt.subplots(figsize=(8, 5))
                for variant in ("ce", "soft", "hxe"):
                    selected = table[table.variant == variant]
                    if not selected.empty:
                        ax.scatter(selected.top1_error, selected[metric], label=variant)
                ax.set(xlabel="Top-1 error", ylabel=metric, title=f"Hierarchical distance · {split}")
                ax.legend()
                ax.grid(alpha=.2)
                save_figure(fig, reports_dir / f"{metric.replace('@', '_')}_{split}.png")
    summary.to_csv(reports_dir / "seed_summary.csv", index=False)
    plot_dataset_counts(reports_dir / "dataset_counts.csv", reports_dir)
    build_html(records, overall, summary, reports_dir / "report.html")
    print("Rapport :", reports_dir / "report.html")
    return overall


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", type=Path, help="File containing runs/ and campaign.json")
    build_reports(parser.parse_args().campaign_dir)
