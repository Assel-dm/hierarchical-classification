"""Build CSV, PNG and HTML from checkpoints"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter


def save_figure(fig, path):
    """Save a figure"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_label(metadata):
    """Build readable label for figures"""
    names = {"ce": "CE", "soft": "Soft Labels", "hxe": "HXE", "haf": "HAF", "haf_ce": "HAF CE", "haf_shc": "HAF CE + SHC", "haf_shc_margin": "HAF CE + SHC + margin", "haf_shc_gc": "HAF CE + SHC + GC"}
    label = names.get(metadata["variant"], metadata["variant"])
    if metadata["parameter"] is not None:
        symbol = "β" if metadata["variant"] == "soft" else "α"
        label += f" {symbol}={metadata['parameter']:g}"
    return f"{label} - {metadata['stage']} - seed {metadata['seed']}"


def plot_learning_curves(run_dir, label):
    """Plot loss, accuracy and top-5 train/validation"""
    history_path = run_dir / "history.csv"
    if not history_path.exists():
        return
    history = pd.read_csv(history_path)
    metrics = (("loss", "Loss"), ("accuracy", "Species accuracy"), ("top5_accuracy", "Top-5 accuracy"), ("ce", "HAF fine CE"), ("shc", "HAF soft hierarchical consistency"), ("margin", "HAF dissimilar-pair margin"), ("gc", "HAF geometric consistency (1 - cosine)"))
    for metric, title in metrics:
        fig, ax = plt.subplots(figsize=(9, 4))
        if metric in history :
            ax.plot(history.epoch, history[metric], label="Train")
        if f"val_{metric}" in history :
            ax.plot(history.epoch, history[f"val_{metric}"], "--", label="Validation")
        ax.set(title=label, xlabel="Epoch", ylabel=title)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=.2)
        ax.legend()
        save_figure(fig, run_dir / "figures" / f"learning_{metric}.png")
    
    components = ("ce", "shc", "margin", "gc")
    if any(key in history or f"val_{key}" in history for key in components):
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
        for ax, key in zip(axes.flat, components):
            for column, style, name in ((key, "-", "Train"), (f"val_{key}", "--", "Validation")):
                if column in history:
                    ax.plot(history.epoch, history[column], style, label=name)
            ax.set(title=key.upper(), xlabel="Epoch", ylabel="Loss component")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(alpha=.2)
            if ax.lines:
                ax.legend()
        fig.suptitle(label)
        fig.tight_layout()
        save_figure(fig, run_dir / "figures" / "learning_components.png")

    if "margin_active_ranks" in history:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, key, title in zip(axes, ("margin_active_ranks", "margin_pairs"), ("Ranks with dissimilar pairs per batch", "Selected pairs per batch (sum of ranks)")):
            ax.plot(history.epoch, history[key], label="Train")
            if f"val_{key}" in history:
                ax.plot(history.epoch, history[f"val_{key}"], "--", label="Validation")
            ax.set(title=title, xlabel="Epoch", ylabel="Mean per batch", ylim=(0, None))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(alpha=.2)
            ax.legend()
        fig.suptitle(label)
        fig.tight_layout()
        save_figure(fig, run_dir / "figures" / "learning_margin_diagnostics.png")


def plot_distance_histogram(run_dir, split, label):
    """Show both total-image and error-conditional distance distributions"""
    path = run_dir / f"distance_histogram_{split}.csv"
    if not path.exists():
        return
    table = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, selected, metric, title in (
        (axes[0], table, "fraction_of_images", "All images (distance 0 = correct)"),
        (axes[1], table[table.distance > 0], "fraction_of_errors", "Errors only"),
    ):
        ax.bar(selected.distance, selected[metric])
        ax.set(title=title, xlabel="LCA height", ylabel="Fraction", ylim=(0, 1))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.grid(axis="y", alpha=.2)
    fig.suptitle(f"{label} - {split}")
    fig.tight_layout()
    save_figure(fig, run_dir / "figures" / f"distance_histogram_{split}.png")



def plot_per_class(run_dir, split, label):
    """Print worst F1 and confusion per class"""
    table = pd.read_csv(run_dir / f"per_class_{split}.csv", dtype={"species": str})
    worst = table.sort_values("f1").head(25)
    fig, ax = plt.subplots(figsize=(9, max(4, len(worst) * .26)))
    labels = worst["species"]
    if "species_name" in worst :
        labels = worst["species_name"].fillna(worst["species"]) + " (" + worst["species"] + ")"
    ax.barh(labels, worst["f1"])
    ax.invert_yaxis()
    ax.set(xlabel="F1", xlim=(0, 1), title=f"{label} - {split} - lowest F1")
    ax.grid(axis="x", alpha=.2)
    save_figure(fig, run_dir / "figures" / f"per_class_{split}.png")
    if len(table) <= 40:
        matrix = pd.read_csv(run_dir / f"confusion_{split}.csv", index_col=0)
        fig, ax = plt.subplots(figsize=(9, 8))
        image = ax.imshow(matrix.to_numpy(), cmap="Blues")
        ax.set_xticks(range(len(matrix)), matrix.columns, rotation=90, fontsize=7)
        ax.set_yticks(range(len(matrix)), matrix.index, fontsize=7)
        ax.set(xlabel="Predicted species", ylabel="True species", title=f"{label} - {split}")
        fig.colorbar(image, ax=ax, label="Images")
        save_figure(fig, run_dir / "figures" / f"confusion_{split}.png")


def plot_dataset_counts(path, output_dir):
    """Plot repartition of images through species and splits"""
    if not path.exists():
        return
    table = pd.read_csv(path, dtype={"species": str})
    table = table.sort_values("train", ascending=False)
    large = len(table) > 40
    fig, ax = plt.subplots(figsize=(12 if large else max(10, len(table) * .15), 5))
    positions = np.arange(len(table)) if large else table.species
    bottom = np.zeros(len(table))
    for split in ("train", "validation", "test"):
        ax.bar(positions, table[split], bottom=bottom, label=split)
        bottom += table[split].to_numpy()
    if large :
        ax.set_xlabel("Species index, sorted by training frequency")
    else :
        ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.set(ylabel="Images", title="Dataset distribution by species")
    ax.legend()
    save_figure(fig, output_dir / "dataset_distribution.png")
    for rank in ("kingdom", "class", "order"):
        if rank not in table:
            continue

        column = f"{rank}_name" if f"{rank}_name" in table else rank
        grouped = (
            table.groupby(column)[["train", "validation", "test"]]
            .sum()
            .sort_values("train", ascending=False)
        )
        fig, ax = plt.subplots(figsize=(12, max(4, len(grouped) * .25)))
        bottom = np.zeros(len(grouped))

        for split in ("train", "validation", "test"):
            ax.barh(grouped.index, grouped[split], left=bottom, label=split)
            bottom += grouped[split].to_numpy()

        ax.invert_yaxis()
        ax.set(xlabel="Images", title=f"Dataset distribution by {rank}")
        ax.legend()
        save_figure(fig, output_dir / f"dataset_distribution_{rank}.png")


def plot_augmentation_preview(image_path, transforms, settings, output_path):
    """Compare different augmention for an image"""
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

def draw_tradeoff(ax, table, metric):
    """Plot parameter sweeps with seed means and automatic label placement"""
    from adjustText import adjust_text

    names = {"ce": "CE", "soft": "Soft Labels", "hxe": "HXE", "haf": "HAF"}
    colors = {
        "ce": "#252525", "soft": "#e28b20",
        "hxe": "#007d79", "haf": "#7756a8",
    }
    styles = {"frozen": ":", "partial": "--", "deeper": "-.", "full": "--"}

    groups = [
        key for key in ("model_key", "augmentation", "stage", "variant")
        if key in table
    ]
    varying = [
        key for key in groups
        if key != "variant" and table[key].nunique() > 1
    ]
    texts, target_x, target_y, avoid_x, avoid_y = [], [], [], [], []

    for _, selected in table.groupby(groups, dropna=False):
        selected = selected.dropna(subset=["top1_error", metric])
        if selected.empty:
            continue

        variant = selected.variant.iloc[0]
        label = names.get(variant, variant.upper().replace("_", " "))
        label += "".join(f" · {selected[key].iloc[0]}" for key in varying)
        color = colors.get(variant)

        points = selected.groupby("parameter", dropna=False)[
            ["top1_error", metric]
        ].agg(["mean", "std"]).sort_index()

        x = points[("top1_error", "mean")].to_numpy(dtype=float)
        y = points[(metric, "mean")].to_numpy(dtype=float)
        multiple_seeds = (
            selected.groupby("parameter", dropna=False).seed.nunique().max() > 1
        )

        ax.errorbar(
            x, y,
            xerr=points[("top1_error", "std")].fillna(0)
                 if multiple_seeds else None,
            yerr=points[(metric, "std")].fillna(0)
                 if multiple_seeds else None,
            fmt="*" if variant == "ce" else "o",
            color=color,
            linestyle=styles.get(selected.stage.iloc[0], "--")
                      if variant in ("soft", "hxe") else "none",
            markersize=10 if variant == "ce" else 7,
            capsize=3,
            linewidth=1.3,
            alpha=.9,
            label=label,
        )

        avoid_x.extend(x)
        avoid_y.extend(y)

        # Sample connecting lines so labels also avoid the curves.
        if variant in ("soft", "hxe"):
            for index in range(len(x) - 1):
                avoid_x.extend(np.linspace(x[index], x[index + 1], 25))
                avoid_y.extend(np.linspace(y[index], y[index + 1], 25))

        for parameter, xx, yy in zip(points.index, x, y):
            if variant in ("soft", "hxe") and pd.notna(parameter):
                text = f"{'β' if variant == 'soft' else 'α'}={parameter:g}"
            elif variant == "ce":
                text = "CE"
            else:
                continue

            texts.append(ax.text(
                xx, yy, text,
                fontsize=9, color=color, ha="center", va="center",
            ))
            target_x.append(xx)
            target_y.append(yy)

    ylabel = (
        "Mistake severity" if metric == "mistake_severity"
        else f"Average hierarchical distance @{metric.split('@')[-1]}"
    )
    ax.set(xlabel="Top-1 error", ylabel=ylabel)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.margins(x=.15, y=.18)
    ax.grid(alpha=.2)
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1))

    # Adjust labels after setting the final axes limits.
    if texts:
        adjust_text(
            texts,
            x=avoid_x, y=avoid_y,
            target_x=target_x, target_y=target_y,
            ax=ax,
            expand=(1.15, 1.3),
            force_text=(.25, .4),
            force_static=(.2, .3),
            iter_lim=400,
            min_arrow_len=5,
            arrowprops=dict(
                arrowstyle="-", color="#9aa4ad", lw=.6
            ),
        )


def build_html(records, overall, seed_summary, output_path):
    """Build HTML report with Plotly"""
    import base64
    from html import escape

    output_path = Path(output_path)
    reports_dir = output_path.parent
    campaign = reports_dir.parent.name

    def picture(path, caption):
        if not path.exists():
            return ""
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return (
            f'<figure><figcaption>{escape(caption)}</figcaption>'
            f'<img src="data:image/png;base64,{encoded}" '
            f'alt="{escape(caption)}" loading="lazy"></figure>'
        )

    def table(frame):
        if frame.empty:
            return "<p>No saved data available.</p>"
        return '<div class="table-wrap">' + frame.to_html(
            index=False,
            escape=True,
            na_rep="—",
            float_format=lambda value: f"{value:.4f}",
        ) + "</div>"

    parts = [
        f"<h1>{escape(campaign)}</h1>",
        "<p>Report generated from saved results. "
        '<nav><a href="#comparison">Comparisons</a> · '
        '<a href="#runs">Runs and species</a> · '
        '<a href="#metrics">Overall metrics</a> · '
        '<a href="#seeds">Across seeds</a> · '
        '<a href="#dataset">Dataset distribution</a></nav>',
        "<p>Lower values on both axes are preferable.</p>",
    ]

    if not overall.empty:
        missing = (
            overall["avg_hierarchical_distance@20"].isna().sum()
            if "avg_hierarchical_distance@20" in overall else len(overall)
        )
        if missing:
            parts.append(
                f'<p class="note">Distance @20 is missing for {missing} '
                "evaluated runs. Re-evaluate these runs to add this metric.</p>"
            )

        for split in overall.split.unique():
            parts.append(f"<h3>{escape(str(split).capitalize())}</h3>")
            parts.append(picture(
                reports_dir / f"comparison_{split}.png",
                "Top-1 error versus mistake severity",
            ))
            for k in (1, 5, 20):
                parts.append(picture(
                    reports_dir / f"avg_hierarchical_distance_{k}_{split}.png",
                    f"Top-1 error versus average hierarchical distance @{k}",
                ))
    else:
        parts.append(
            "<p>No saved evaluation results. "
            "Run an evaluation before comparing methods.</p>"
        )

    parts += [
        '<h2 id="runs">Learning curves and F1 by species</h2>',
        "<p>Expand a run to view its figures. Curves use the original history "
        "values without smoothing. Loss scales depend on the objective and its "
        "parameter; compare them within the same objective. Saved evaluation "
        "metrics refer to the selected checkpoint.</p>",
        "<p>HAF total = fine CE + SHC + margin + GC. GC uses 1 - cosine; "
        "the authors' -cosine convention differs by one constant per adjacent "
        "rank link and has the same gradients. Margin depends on the number of "
        "ranks with dissimilar pairs in each batch. Original validation ordering "
        "and checkpoint selection are preserved.</p>",
    ]

    for record in records:
        run_dir = record["run_dir"]
        parts.append(
            f'<details><summary>{escape(record["label"])}</summary>'
        )
        history = record["history"]
        if "HAF" in record["label"] and history is not None:
            columns = [prefix + key for prefix in ("", "val_") for key in ("ce", "shc", "margin", "gc")]
            if not all(key in history for key in columns):
                parts.append('<p class="note">This history does not contain all train/validation '
                             'HAF components. Missing epoch values cannot be reconstructed from final weights.</p>')
            else:
                for prefix, title in (("", "Train"), ("val_", "Validation")):
                    total = history[[prefix + key for key in ("ce", "shc", "margin", "gc")]].sum(axis=1)
                    if not np.allclose(total, history[prefix + "loss"], atol=1e-5, rtol=1e-5):
                        parts.append(f'<p class="note">{title} total loss does not match the sum of '
                                     'the four saved components. Check the history and loss version.</p>')

        for metric, title in (
            ("loss", "Loss"),
            ("accuracy", "Top-1 accuracy"),
            ("top5_accuracy", "Top-5 accuracy"),
            ("ce", "HAF CE"),
            ("shc", "HAF SHC"),
            ("margin", "HAF margin"),
            ("gc", "HAF GC"),
            ("components", "HAF loss components"),
            ("margin_diagnostics", "Dissimilar-pair availability"),
        ):
            parts.append(picture(
                run_dir / "figures" / f"learning_{metric}.png", title
            ))

        for split, classes in record["classes"].items():
            parts.append(picture(
                run_dir / "figures" / f"per_class_{split}.png",
                f"Lowest species F1 · {split}",
            ))
            parts.append(picture(
                run_dir / "figures" / f"confusion_{split}.png",
                f"Confusion matrix · {split}",
            ))
            columns = [
                key for key in (
                    "species", "species_name", "support",
                    "precision", "recall", "f1"
                )
                if key in classes
            ]
            parts.append(table(
                classes.sort_values("f1").head(25)[columns]
            ))

        parts.append("</details>")

    parts += [
        '<h2 id="metrics">Overall metrics</h2>',
        "<p>Historical rank_accuracy metrics retain probability-sum decoding. "
        "The explicit rank_accuracy_probability_sum columns are aliases; "
        "rank_accuracy_top1_ancestor uses the ancestor of the top-1 species, "
        "as in the paper's hierarchical evaluation.</p>",
        table(overall),
        '<h2 id="seeds">Mean and standard deviation across seeds</h2>',
        table(seed_summary),
        "<p>Single-seed standard deviations appear as “—”. "
        "Complete CSV files remain in runs/ and reports/.</p>",
        '<h2 id="dataset">Dataset distribution</h2>',
    ]

    for path in sorted(reports_dir.glob("dataset_distribution*.png")):
        parts.append(picture(
            path, path.stem.replace("_", " ").capitalize()
        ))
    parts.append(picture(
        reports_dir / "augmentation_preview.png", "Augmentation preview"
    ))

    css = """
    body{font:15px/1.5 system-ui,sans-serif;color:#203340;
         background:#f5f7f9;margin:0}
    main{max-width:1050px;margin:auto;padding:24px}
    h1{overflow-wrap:anywhere}h2{margin-top:32px}
    nav,a{color:#007d79}
    details,figure,.table-wrap{background:white;border:1px solid #dce4e8;
                              border-radius:8px}
    details{padding:16px;margin:12px 0}
    summary{cursor:pointer;font-weight:600}
    figure{margin:16px 0;padding:12px}
    figcaption{font-weight:600;margin-bottom:8px}
    img{display:block;width:100%;height:auto}
    .table-wrap{overflow:auto;max-height:480px}
    table{border-collapse:collapse;width:100%;font-size:13px}
    th,td{padding:8px;white-space:nowrap;border-bottom:1px solid #e2e8ec}
    th{position:sticky;top:0;background:#eef3f5}
    .note{padding:12px;background:#fff2d9}
    @media(max-width:600px){main{padding:12px}details{padding:10px}}
    """

    output_path.write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{escape(campaign)} · results</title><style>{css}</style></head>"
        "<body><main>" + "".join(parts) + "</main></body></html>",
        encoding="utf-8",
    )


def build_reports(campaign_dir):
    """Build graphics with available with available plots"""
    campaign_dir = Path(campaign_dir).expanduser()
    if not campaign_dir.is_dir():
        raise FileNotFoundError(campaign_dir)

    reports_dir = campaign_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    records, rows, class_tables = [], [], []
    stage_order = {"frozen": 0, "partial": 1, "deeper": 2, "full": 3}
    paths = sorted(
        (campaign_dir / "runs").glob("*/*/metadata.json"),
        key=lambda path: (
            path.parent.parent.name, stage_order[path.parent.name]
        ),
    )

    for path in paths:
        run_dir = path.parent
        metadata = json.loads(path.read_text(encoding="utf-8"))
        label = run_label(metadata)
        plot_learning_curves(run_dir, label)
        history_path = run_dir / "history.csv"
        record = {
            "run_dir": run_dir,
            "label": label,
            "history": pd.read_csv(history_path)
                       if history_path.exists() else None,
            "classes": {},
        }

        for split in ("validation", "test"):
            metrics_path = run_dir / f"metrics_{split}.json"
            if not metrics_path.exists():
                continue

            scores = json.loads(metrics_path.read_text(encoding="utf-8"))
            rows.append({
                **{key: value for key, value in metadata.items()
                   if key != "training"},
                "label": label, "split": split, **scores,
            })
            record["classes"][split] = pd.read_csv(
                run_dir / f"per_class_{split}.csv",
                dtype={"species": str},
            )
            table = record["classes"][split].copy()
            for key, value in (
                ("label", label), ("seed", metadata["seed"]),
                ("stage", metadata["stage"]),
                ("variant", metadata["variant"]),
                ("parameter", metadata["parameter"]), ("split", split),
            ):
                table[key] = (
                    np.nan if key == "parameter" and value is None else value
                )
            class_tables.append(table)
            plot_per_class(run_dir, split, label)
            plot_distance_histogram(run_dir, split, label)

        records.append(record)

    overall = (
        pd.DataFrame(rows) if rows
        else pd.DataFrame(columns=["label", "split"])
    )
    overall.to_csv(reports_dir / "comparison.csv", index=False)
    if class_tables:
        pd.concat(class_tables, ignore_index=True).to_csv(
            reports_dir / "per_class_comparison.csv", index=False
        )

    summary = pd.DataFrame()
    if rows:
        groups = [
            "campaign", "model_key", "augmentation",
            "variant", "parameter", "stage", "split",
        ]
        metrics = [
            column for column in overall.select_dtypes(include="number")
            if column not in {"seed", "parameter"}
        ]
        grouped = overall.groupby(groups, dropna=False)
        summary = grouped[metrics].agg(["mean", "std"])
        summary.columns = [
            f"{metric}_{stat}" for metric, stat in summary.columns
        ]
        summary["n_seeds"] = grouped.seed.nunique()
        summary = summary.reset_index()

        for split in overall.split.unique():
            table = overall[overall.split == split]
            fig, ax = plt.subplots(figsize=(9, 5.5))
            draw_tradeoff(ax, table, "mistake_severity")
            ax.set_title(f"Hierarchical trade-off · {split}")
            save_figure(fig, reports_dir / f"comparison_{split}.png")

            for k in (1, 5, 20):
                metric = f"avg_hierarchical_distance@{k}"
                if metric not in table or table[metric].notna().sum() == 0:
                    print(
                        f"Skipping {metric} · {split}: no saved values. "
                        "Re-evaluate to add this metric."
                    )
                    continue

                fig, ax = plt.subplots(figsize=(9, 5.5))
                draw_tradeoff(ax, table, metric)
                ax.set_title(f"Hierarchical distance @{k} · {split}")
                save_figure(
                    fig,
                    reports_dir / f"avg_hierarchical_distance_{k}_{split}.png",
                )

    summary.to_csv(reports_dir / "seed_summary.csv", index=False)
    plot_dataset_counts(reports_dir / "dataset_counts.csv", reports_dir)
    build_html(records, overall, summary, reports_dir / "report.html")
    print("Report:", reports_dir / "report.html")
    return overall


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", type=Path, help="File containing runs/ and campaign.json")
    build_reports(parser.parse_args().campaign_dir)
