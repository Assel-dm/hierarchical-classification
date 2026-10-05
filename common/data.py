"""Convert a taxonomy in matrix and load images"""
import random
import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from common.models import build_transform

ImageFile.LOAD_TRUNCATED_IMAGES = True


def build_hierarchy(taxonomy, ranks):
    """Construct groups and taxonomic distance"""
    if not ranks or ranks[-1] != "species" or len(set(ranks)) != len(ranks):
        raise ValueError("RAnks must be unique and end with species")
    if not set(ranks).issubset(taxonomy.columns):
        raise ValueError(f"Taxonomy must contains {ranks}")

    taxonomy = taxonomy.drop_duplicates().copy()
    if taxonomy.empty or taxonomy[ranks].isna().any().any() or taxonomy.species.duplicated().any():
        raise ValueError("TEmpty or incomplete taxonomy, or species with multiples parents")
    if taxonomy[ranks].map(lambda value: not isinstance(value, str) or not value.strip()).any().any():
        raise ValueError("Taxonomic ids must be non empty strings")

    for child, parent in zip(ranks[1:], ranks[:-1]):
        if taxonomy.groupby(child)[parent].nunique().max() > 1:
            raise ValueError(f"Id of {child} must have only one {parent}")

    taxonomy = taxonomy.sort_values("species").reset_index(drop=True)
    species = taxonomy.species.tolist()

    # A ligne per species, a column per ancestor
    mappings = {}
    for rank in ranks[:-1]:
        names = sorted(taxonomy[rank].unique())
        indices = {name: i for i, name in enumerate(names)}
        matrix = np.zeros((len(species), len(names)), dtype=np.float32)
        for i, ancestor in enumerate(taxonomy[rank]):
            matrix[i, indices[ancestor]] = 1
        mappings[rank] = matrix

    # Height of the closest common ancester
    height = len(ranks)
    distances = np.full((len(species), len(species)), height, dtype=np.float32)
    for depth, rank in enumerate(ranks, start=1):
        ancestors = taxonomy[rank].to_numpy()
        same_ancestor = ancestors[:, None] == ancestors[None, :]
        distances[same_ancestor] = height - depth

    return {
        "taxonomy": taxonomy,
        "ranks": list(ranks),
        "species": species,
        "mappings": mappings,
        "distance_matrix": distances,
        "normalized_distances": distances / max(1, float(distances.max())),
    }


class SpeciesDataset(Dataset):
    """Load an RGB image and return its species ID"""
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
        transform = build_transform(
            settings,
            training=split == "train",
            augmentation=augmentation,
        )
        dataset = SpeciesDataset(samples, transform)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            generator=torch.Generator().manual_seed(seed),
            worker_init_fn=seed_worker,
            persistent_workers=num_workers > 0,
        )
    return loaders