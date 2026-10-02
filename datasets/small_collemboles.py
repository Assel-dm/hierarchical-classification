"""Read Small Collemboles dataset and convert it in data usable for the common part""" 
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from common.models import build_transform

ImageFile.LOAD_TRUNCATED_IMAGES = True 


def load_small_collemboles(root):
    """Returns splits, classes, membership matrices and distances"""
    root = Path(root).expanduser().resolve()
    taxonomy = pd.read_csv(root / "hierarchy.csv", dtype=str)
    ranks = ["order", "family", "genus", "species"]
    if not set(ranks).issubset(taxonomy.columns):
        raise ValueError(f"hierarchy.csv doit contenir {ranks}")
    taxonomy = taxonomy[ranks].drop_duplicates()
    if taxonomy.empty or taxonomy.isna().any().any() or taxonomy.species.duplicated().any():
        raise ValueError("Empty taxonomy, incomplete or species with several parents")
    if taxonomy.map(lambda value: not value.strip()).any().any():
        raise ValueError("Empty class name in taxonomy")
    for child, parent in zip(ranks[1:], ranks[:-1]):
        if taxonomy.groupby(child)[parent].nunique().max() > 1:
            raise ValueError(f"{child} must have only one {parent}")

    taxonomy = taxonomy.sort_values("species").reset_index(drop=True)
    species = taxonomy.species.tolist()
    species_to_idx = {name: i for i, name in enumerate(species)}

    # A ligne per species, a column per ancestor
    mappings = {}
    for rank in ranks[:-1]:
        names = sorted(taxonomy[rank].unique())
        indices = {name: i for i, name in enumerate(names)}
        matrix = np.zeros((len(species), len(names)), dtype=np.float32)
        for i, ancestor in enumerate(taxonomy[rank]):
            matrix[i, indices[ancestor]] = 1
        mappings[rank] = matrix

    # Distance : 0, 1, 2, 3 or 4 depending on the common ancestor
    distances = np.zeros((len(species), len(species)), dtype=np.float32)
    for i in range(len(species)):
        for j in range(len(species)):
            if i == j:
                continue
            a, b = taxonomy.iloc[i], taxonomy.iloc[j]
            distances[i, j] = (1 if a.genus == b.genus else 2 if a.family == b.family
                               else 3 if a.order == b.order else 4)

    samples, seen = {}, set()
    directories = {"train": "train", "validation": "val", "test": "test"}
    for split, directory in directories.items():
        folder = root / "dataset" / directory
        classes = sorted(p.name for p in folder.iterdir() if p.is_dir())
        if classes != species:
            raise ValueError(f"Classes of {folder} does not corresponf of the taxonomy of hierarchy.csv")
        samples[split] = []
        for name in species:
            images = sorted(p for p in (folder / name).iterdir()
                            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
            if not images:
                raise ValueError(f"No images of {name} in {split}")
            for path in images:
                if path.resolve() in seen:
                    raise ValueError(f"Reused image in splits : {path}")
                seen.add(path.resolve())
                samples[split].append((path, species_to_idx[name]))

    return {"name": "small_collemboles", "root": root, "taxonomy": taxonomy,
            "species": species, "mappings": mappings, "distance_matrix": distances,
            "normalized_distances": distances / max(1, distances.max()), "samples": samples}


class SpeciesDataset(Dataset):
    """Load an image and return its species"""
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), int(label)


def seed_worker(worker_id):
    """Synchronise worker's randomness"""
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def build_loaders(data, settings, batch_size, num_workers, seed, augmentation, device):
    """Build data loader"""
    loaders = {}
    for split, samples in data["samples"].items():
        transform = build_transform(settings, training=split == "train", augmentation=augmentation)
        dataset = SpeciesDataset(samples, transform)
        loaders[split] = DataLoader(
            dataset, batch_size=batch_size, shuffle=split == "train",
            num_workers=num_workers, pin_memory=device.type == "cuda",
            generator=torch.Generator().manual_seed(seed), worker_init_fn=seed_worker,
            persistent_workers=num_workers > 0,
        )
    return loaders
