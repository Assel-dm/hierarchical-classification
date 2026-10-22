"""Load prepared iNat2021 Insecta/Mammalia subsets (Mini or Full)"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from common.data import build_hierarchy


ALL_RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
DEFAULT_RANKS = ["order", "family", "genus", "species"]
SPLITS = ("train", "validation", "test")


def load_inat21(root, ranks=None, check_files=True):
    """Return the existing splits and the common MBM/HAF dataset interface"""
    root = Path(root).expanduser().resolve()
    ranks = list(DEFAULT_RANKS if ranks is None else ranks)
    if ranks not in (DEFAULT_RANKS, ALL_RANKS):
        raise ValueError("Use order-to-species or kingdom-to-species ranks")
    fingerprints = {}

    def read_json(filename):
        raw = (root / filename).read_bytes()
        fingerprints[filename] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    info = read_json("dataset_info.json")
    if info.get("image_paths_relative_to") != ".":
        raise ValueError("Image paths must be relative to DATASET_ROOT")
    if info.get("protocol") != "local-test":
        raise ValueError("Expected the prepared local-test protocol")
    training = info.get("training")
    if training not in {"mini", "full"}:
        raise ValueError("dataset_info.training must be mini or full")
    origins = info["split_origins"]
    expected_origin = "train_mini" if training == "mini" else "train"
    if origins != {"train": expected_origin, "validation": expected_origin, "test": "val"}:
        raise ValueError("Unexpected split origins for the local-test protocol")

    categories = read_json("taxonomy.json")
    if not isinstance(categories, list) or not categories:
        raise ValueError("taxonomy.json must contain a nonempty category list")
    indexed, rows = {}, []
    for category in categories:
        cid = category["id"]
        source_index = category["class_index"]
        if type(cid) is not int or cid < 0 or cid in indexed:
            raise ValueError(f"Invalid or duplicate category ID: {cid}")
        if type(source_index) is not int or source_index < 0:
            raise ValueError(f"Invalid class_index for category {cid}")
        fields = ALL_RANKS[:-1] + ["name", "image_dir_name"]
        if any(not isinstance(category.get(k), str) or not category[k].strip() for k in fields):
            raise ValueError(f"Incomplete taxonomy for category {cid}")
        indexed[cid] = category
        rows.append({
            **{rank: category[rank] for rank in ALL_RANKS[:-1]},
            "species": f"inat21_{cid:05d}",
            **{f"{rank}_name": category[rank] for rank in ALL_RANKS[:-1]},
            "species_name": category["name"],
            "category_id": cid,
            "source_class_index": source_index,
            "image_dir_name": category["image_dir_name"],
        })
    if sorted(c["class_index"] for c in categories) != list(range(len(categories))):
        raise ValueError("Source class_index values must be unique and contiguous")
    if info["species"] != len(categories):
        raise ValueError("Species count differs from dataset_info.json")
    groups = {c["class"] for c in categories}
    if len(groups) != 1 or not groups <= {"Insecta", "Mammalia"}:
        raise ValueError("Expected one subset: Insecta or Mammalia")
    group = next(iter(groups))
    data = build_hierarchy(pd.DataFrame(rows), ranks)
    data["taxonomy"]["model_class_index"] = np.arange(len(categories))
    category_to_index = {
        int(cid): i for i, cid in enumerate(data["taxonomy"]["category_id"])
    }

    samples, split_metadata = {}, {}
    seen_paths, seen_originals, seen_source_ids = set(), set(), set()
    category_fields = ["id", "class_index", "name", "image_dir_name"] + ALL_RANKS[:-1]
    for split in SPLITS:
        payload = read_json(f"{split}.json")
        meta = payload["info"]
        expected_meta = {
            "group": group, "split": split, "protocol": info["protocol"],
            "original_split": origins[split], "seed": info["seed"],
            "val_per_species": info["val_per_species"],
        }
        if any(meta.get(k) != v for k, v in expected_meta.items()):
            raise ValueError(f"Inconsistent preparation metadata in {split}.json")
        split_metadata[split] = meta
        split_categories = payload["categories"]
        if len(split_categories) != len(categories):
            raise ValueError(f"Incomplete categories in {split}.json")
        found_ids = set()
        for category in split_categories:
            cid = category["id"]
            if cid in found_ids or cid not in indexed:
                raise ValueError(f"Unknown or duplicate category {cid} in {split}")
            if any(category.get(k) != indexed[cid].get(k) for k in category_fields):
                raise ValueError(f"Category {cid} differs from taxonomy.json in {split}")
            found_ids.add(cid)

        labels = {}
        for annotation in payload["annotations"]:
            image_id, cid = annotation["image_id"], annotation["category_id"]
            if image_id in labels or cid not in indexed:
                raise ValueError(f"Duplicate annotation or unknown category in {split}: {image_id}")
            if annotation.get("class_index") != indexed[cid]["class_index"]:
                raise ValueError(f"Inconsistent class_index in {split}: {image_id}")
            labels[image_id] = cid

        current_samples = []
        for image in payload["images"]:
            image_id = image["id"]
            if image_id not in labels:
                raise ValueError(f"Duplicate image or missing annotation in {split}: {image_id}")
            cid = labels.pop(image_id)
            relative = PurePosixPath(image["file_name"])
            if (relative.is_absolute() or ".." in relative.parts or len(relative.parts) < 3
                    or relative.parts[:2] != (split, indexed[cid]["image_dir_name"])):
                raise ValueError(f"Invalid image path or class directory: {relative}")
            path = root.joinpath(*relative.parts)
            original = image["original_file_name"]
            original_path = PurePosixPath(original)
            if (original_path.is_absolute() or ".." in original_path.parts
                    or len(original_path.parts) < 3
                    or original_path.parts[:2] != (origins[split], indexed[cid]["image_dir_name"])
                    or original_path.name != relative.name):
                raise ValueError(f"Inconsistent original image path: {original}")
            # Official train and val can have independent ID namespaces.
            source_id = (origins[split], image_id)
            if (str(relative) in seen_paths or str(original_path) in seen_originals
                    or source_id in seen_source_ids):
                raise ValueError(f"Repeated image within/between splits: {relative}")
            seen_paths.add(str(relative))
            seen_originals.add(str(original_path))
            seen_source_ids.add(source_id)
            if check_files and not path.is_file():
                raise FileNotFoundError(f"Image missing: {path}")
            current_samples.append((path, category_to_index[cid]))

        if labels:
            raise ValueError(f"Annotations without images in {split}")
        if len(current_samples) != info["splits"][split]["images"]:
            raise ValueError(f"Image count differs from dataset_info.json: {split}")
        counts = np.bincount([label for _, label in current_samples], minlength=len(categories))
        if np.any(counts == 0):
            raise ValueError(f"Some species have no images in {split}")
        if split == "validation" and not np.all(counts == info["val_per_species"]):
            raise ValueError("Validation counts differ from val_per_species")
        samples[split] = current_samples
        del payload

    data.update(
        name=f"inat21_{group.lower()}_{training}", root=root, samples=samples,
        dataset_info=info, split_metadata=split_metadata,
        metadata_sha256=fingerprints, files_checked=bool(check_files),
    )
    return data


def show_summary(data):
    """Print ranks, counts and the provenance of each split"""
    print("Dataset:", data["name"])
    print("Root:", data["root"])
    for rank in data["ranks"]:
        print(f"{rank}: {data['taxonomy'][rank].nunique()} classes")
    for split, samples in data["samples"].items():
        counts = np.bincount([label for _, label in samples], minlength=len(data["species"]))
        print(f"{split}: {len(samples):,} images, {counts.min()}–{counts.max()} per species"
              f" (source: {data['dataset_info']['split_origins'][split]})")
    print("Maximum taxonomic distance:", float(data["distance_matrix"].max()))
    print("Image existence checked:", data["files_checked"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--metadata-only", action="store_true", help="Skip image existence checks")
    parser.add_argument("--all-ranks", action="store_true", help="Include kingdom, phylum and class")
    args = parser.parse_args()
    show_summary(load_inat21(args.root, ALL_RANKS if args.all_ranks else None,
                            check_files=not args.metadata_only))
