"""Read iNaturalist-19-H prepare with MBM splits and taxonomy"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from common.data import build_hierarchy

RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
IMAGE_EXTENSIONS = {".jpg"}


def find_metadata(root, filename, explicit_path=None, required=True):
    """Find a file in metadata"""
    if explicit_path is not None:
        path = Path(explicit_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    candidates = sorted((root / "metadata").rglob(filename))
    if (root / filename).is_file():
        candidates.append(root / filename)

    if len(candidates) > 1:
        raise ValueError(
            f"Several {filename} found change path in config.py"
        )
    if candidates:
        return candidates[0]
    if required:
        raise FileNotFoundError(
            f"{filename} absent of {root / 'metadata'}. "
            "Use file inaturalist19_isa.txt"
        )
    return None


def taxonomy_from_isa(path):
    """Construct the seven ranks from parent/children relationships via .isa file"""
    parents, nodes = {}, set()

    for number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip() or line.lstrip().startswith("#"):
            continue

        values = line.split()
        if len(values) != 2:
            raise ValueError(
                f"ISA line {number} : two ID parent children wanted"
            )

        parent, child = values
        if child in parents or child == parent:
            raise ValueError(
                f"Invalid ISA tree : welation with {child} repeated or multiple parents"
            )
        parents[child] = parent
        nodes.update(values)

    roots = nodes - parents.keys()
    if len(roots) != 1:
        raise ValueError("ISA tree must have only one root")

    root = roots.pop()
    leaves = sorted(nodes - set(parents.values()))
    rows = []

    for leaf in leaves:
        branch, visited, node = [], set(), leaf
        while node != root:
            if node in visited or node not in parents:
                raise ValueError(f"Cycle or branch disconnected in ISA : {leaf}")
            visited.add(node)
            branch.append(node)
            node = parents[node]

        branch.reverse()
        if len(branch) != len(RANKS):
            raise ValueError(
                f"{leaf} : {len(branch)} levels ; iNat-19-H must have seven"
            )
        if not leaf.startswith("nat") or not leaf[3:].isdigit():
            raise ValueError(f"Unwanted ISA leaf for iNat-19-H : {leaf}")

        rows.append(dict(zip(RANKS, branch)))

    if set(parents) != {node for row in rows for node in row.values()}:
        raise ValueError("The tree contains branches that are foreign to species")

    return pd.DataFrame(rows, columns=RANKS)


def add_scientific_names(taxonomy, categories_path):
    """Add readable names"""
    payload = json.loads(categories_path.read_text(encoding="utf-8"))
    categories = payload.get("categories") if isinstance(payload, dict) else payload

    if not isinstance(categories, list):
        raise ValueError("categories.json must contains a list of categories")

    indexed = {}
    for category in categories:
        label = f"nat{int(category['id']):04d}"
        if label in indexed:
            raise ValueError(f"Repeated category: {label}")
        indexed[label] = category

    for rank in RANKS:
        field = "name" if rank == "species" else rank
        names = [indexed[label][field] for label in taxonomy.species]
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError(f"Biologic name absent for {rank}")

        taxonomy[f"{rank}_name"] = names
        if taxonomy.groupby(rank)[f"{rank}_name"].nunique().max() > 1:
            raise ValueError(
                f"categories.json does not correspond to the tree at rank {rank}"
            )

    return taxonomy


def check_split_lists(root, species, samples):
    """Check the number of images per classes"""
    lists_root = root / "metadata" / "splits_inat19"
    if not lists_root.exists():
        return False

    seen = set()
    for split, directory in (
        ("train", "train"),
        ("validation", "val"),
        ("test", "test"),
    ):
        folder = lists_root / directory
        files = sorted(folder.glob("*.txt"))
        if sorted(path.stem for path in files) != species:
            raise ValueError(f"List of classes incomplete : {folder}")

        counts = np.bincount(
            [label for _, label in samples[split]],
            minlength=len(species),
        )
        for index, path in enumerate(files):
            entries = [
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(entries) != int(counts[index]):
                raise ValueError(
                    f"{split}/{species[index]} : {counts[index]} images, "
                    f"{len(entries)} entries in {path}"
                )

            for entry in entries:
                if entry in seen:
                    raise ValueError(
                        f"Repeated image: {entry}"
                    )
                seen.add(entry)

    return True


def load_inaturalist19(root, hierarchy_file=None, categories_file=None):
    """Return classes, hierarchy and prepared images"""
    root = Path(root).expanduser().resolve()
    hierarchy_file = find_metadata(root, "inaturalist19_isa.txt", hierarchy_file)
    categories_file = find_metadata(
        root, "categories.json", categories_file, required=False
    )

    taxonomy = taxonomy_from_isa(hierarchy_file)
    if categories_file is not None:
        taxonomy = add_scientific_names(taxonomy, categories_file)
    data = build_hierarchy(taxonomy, RANKS)

    species = data["species"]
    samples, seen = {}, set()

    for split, directory in (
        ("train", "train"),
        ("validation", "val"),
        ("test", "test"),
    ):
        folder = root / directory
        if not folder.is_dir():
            raise FileNotFoundError(f"Absent split  : {folder}")

        classes = sorted(path.name for path in folder.iterdir() if path.is_dir())
        if classes != species:
            missing = sorted(set(species) - set(classes))
            extra = sorted(set(classes) - set(species))
            raise ValueError(
                f"Classes different from the tree in {folder}. "
                f"Absent : {missing[:8]} ; extra : {extra[:8]}"
            )

        samples[split] = []
        for index, label in enumerate(species):
            images = sorted(
                path
                for path in (folder / label).rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not images:
                raise ValueError(f"No images in {folder / label}")

            for path in images:
                resolved = path.resolve()
                if resolved in seen:
                    raise ValueError(f"Reuses image between splits : {path}")
                seen.add(resolved)
                samples[split].append((path, index))

    lists_checked = check_split_lists(root, species, samples)
    data.update(
        name="inaturalist19_h",
        root=root,
        samples=samples,
        hierarchy_file=str(hierarchy_file),
        split_lists_checked=lists_checked,
    )
    return data


def show_summary(data):
    """Show a summary of the dataset"""
    print("Dataset :", data["root"])
    print("Tree :", data["hierarchy_file"])
    for rank in data["ranks"]:
        print(f"{rank} : {data['taxonomy'][rank].nunique()} classes")
    for split, samples in data["samples"].items():
        print(f"{split} : {len(samples)} images")
    print(
        "Number of images and splits verified :",
        data["split_lists_checked"],
    )
    print("Maximum distance :", float(data["distance_matrix"].max()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--hierarchy-file", type=Path)
    parser.add_argument("--categories-file", type=Path)
    args = parser.parse_args()
    show_summary(
        load_inaturalist19(args.root, args.hierarchy_file, args.categories_file)
    )