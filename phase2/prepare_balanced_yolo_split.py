from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path


LOCAL_PACKAGES = Path(__file__).resolve().parent / ".packages"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))

import yaml


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def read_class_counts(label_path: Path) -> Counter[int]:
    counts: Counter[int] = Counter()
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            counts[int(line.split()[0])] += 1
    return counts


def find_image(image_root: Path, stem: str) -> Path:
    for extension in IMAGE_EXTENSIONS:
        candidate = image_root / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Image for label {stem!r} was not found")


def select_balanced_images(
    dataset_root: Path,
    dominant_class_id: int,
    target_dominant_boxes: int,
    max_normal_images: int,
    seed: int,
) -> tuple[list[Path], Counter[int], dict[str, int]]:
    image_root = dataset_root / "images" / "train"
    label_root = dataset_root / "labels" / "train"
    minority_images: list[tuple[Path, Counter[int]]] = []
    dominant_only_images: list[tuple[Path, Counter[int]]] = []
    normal_images: list[Path] = []

    for label_path in sorted(label_root.glob("*.txt")):
        image_path = find_image(image_root, label_path.stem)
        counts = read_class_counts(label_path)
        if not counts:
            normal_images.append(image_path)
        elif any(class_id != dominant_class_id for class_id in counts):
            minority_images.append((image_path, counts))
        else:
            dominant_only_images.append((image_path, counts))

    selected: list[Path] = []
    selected_counts: Counter[int] = Counter()
    for image_path, counts in minority_images:
        selected.append(image_path)
        selected_counts.update(counts)

    rng = random.Random(seed)
    rng.shuffle(dominant_only_images)
    for image_path, counts in dominant_only_images:
        if selected_counts[dominant_class_id] >= target_dominant_boxes:
            break
        selected.append(image_path)
        selected_counts.update(counts)

    rng.shuffle(normal_images)
    selected.extend(normal_images[:max_normal_images])
    selected.sort()
    stats = {
        "minority_images": len(minority_images),
        "dominant_only_images_available": len(dominant_only_images),
        "normal_images_available": len(normal_images),
        "normal_images_selected": min(len(normal_images), max_normal_images),
        "total_images_selected": len(selected),
    }
    return selected, selected_counts, stats


def write_balanced_dataset(
    dataset_root: Path,
    dominant_class: str,
    target_dominant_boxes: int,
    max_normal_images: int,
    seed: int,
) -> tuple[Path, Path]:
    source_yaml = dataset_root / "data.yaml"
    document = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    names = document["names"]
    name_by_id = (
        {index: name for index, name in enumerate(names)}
        if isinstance(names, list)
        else {int(index): name for index, name in names.items()}
    )
    class_id_by_name = {name: class_id for class_id, name in name_by_id.items()}
    if dominant_class not in class_id_by_name:
        raise ValueError(f"Unknown dominant class: {dominant_class}")

    selected, counts, stats = select_balanced_images(
        dataset_root=dataset_root,
        dominant_class_id=class_id_by_name[dominant_class],
        target_dominant_boxes=target_dominant_boxes,
        max_normal_images=max_normal_images,
        seed=seed,
    )
    manifest_path = dataset_root / "balanced-train.txt"
    manifest_path.write_text(
        "\n".join(f"./{path.relative_to(dataset_root).as_posix()}" for path in selected)
        + "\n",
        encoding="utf-8",
    )

    balanced_yaml = dataset_root / "data-balanced.yaml"
    balanced_document = dict(document)
    balanced_document["train"] = manifest_path.name
    balanced_yaml.write_text(
        yaml.safe_dump(balanced_document, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    print(f"balanced train images: {stats['total_images_selected']}")
    print(f"normal images: {stats['normal_images_selected']}")
    print("boxes by class:")
    for class_id, name in sorted(name_by_id.items()):
        print(f"  {name}: {counts[class_id]}")
    print(f"manifest: {manifest_path}")
    print(f"dataset yaml: {balanced_yaml}")
    return manifest_path, balanced_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a class-balanced YOLO train list.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dominant-class", default="porosity")
    parser.add_argument("--target-dominant-boxes", type=int, default=4500)
    parser.add_argument("--max-normal-images", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    write_balanced_dataset(
        dataset_root=args.dataset_root.resolve(),
        dominant_class=args.dominant_class,
        target_dominant_boxes=args.target_dominant_boxes,
        max_normal_images=args.max_normal_images,
        seed=args.seed,
    )
