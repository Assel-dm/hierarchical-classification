"""Read Small Collemboles and convert it in data usable for the common part""" 
from pathlib import Path
import pandas as pd
from common.data import build_hierarchy, build_loaders


def load_small_collemboles(root):
    """Return splits, classes, memberships matrix and distance"""
    root = Path(root).expanduser().resolve()
    taxonomy = pd.read_csv(root / "hierarchy.csv", dtype=str)
    ranks = ["order", "family", "genus", "species"]
    data = build_hierarchy(taxonomy[ranks], ranks)
    species = data["species"]
    species_to_idx = {name: i for i, name in enumerate(species)}

    samples, seen = {}, set()
    directories = {"train": "train", "validation": "val", "test": "test"}

    for split, directory in directories.items():
        folder = root / "dataset" / directory
        classes = sorted(p.name for p in folder.iterdir() if p.is_dir())
        if classes != species:
            raise ValueError(
                f"Classes of {folder} does not correspond to hierarchy.csv"
            )

        samples[split] = []
        for name in species:
            images = sorted(
                p
                for p in (folder / name).iterdir()
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            if not images:
                raise ValueError(f"No image pf {name} in {split}")

            for path in images:
                if path.resolve() in seen:
                    raise ValueError(f"Reused image in : {path}")
                seen.add(path.resolve())
                samples[split].append((path, species_to_idx[name]))

    data.update(name="small_collemboles", root=root, samples=samples)
    return data
