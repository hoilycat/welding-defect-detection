from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path


LOCAL_PACKAGES = Path(__file__).resolve().parent / ".packages"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))

import cv2
import numpy as np
import yaml


COLORS = [
    (64, 64, 230),
    (55, 180, 75),
    (230, 150, 40),
    (180, 70, 180),
    (40, 180, 220),
    (210, 110, 55),
]


def read_dataset_names(dataset_root: Path) -> dict[int, str]:
    with (dataset_root / "data.yaml").open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    names = document.get("names", {})
    if isinstance(names, list):
        return dict(enumerate(names))
    return {int(class_id): str(name) for class_id, name in names.items()}


def read_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    labels = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id, x_center, y_center, width, height = line.split()
        labels.append(
            (
                int(class_id),
                float(x_center),
                float(y_center),
                float(width),
                float(height),
            )
        )
    return labels


def draw_labels(
    image: np.ndarray,
    labels: list[tuple[int, float, float, float, float]],
    names: dict[int, str],
) -> np.ndarray:
    result = image.copy()
    image_height, image_width = result.shape[:2]
    for class_id, x_center, y_center, width, height in labels:
        x1 = round((x_center - width / 2) * image_width)
        y1 = round((y_center - height / 2) * image_height)
        x2 = round((x_center + width / 2) * image_width)
        y2 = round((y_center + height / 2) * image_height)
        color = COLORS[class_id % len(COLORS)]
        thickness = max(2, round(min(image_width, image_height) / 350))
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            result,
            names[class_id],
            (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            thickness,
            cv2.LINE_AA,
        )
    return result


def make_tile(image: np.ndarray, title: str, width: int = 480, height: int = 300) -> np.ndarray:
    header_height = 34
    canvas = np.full((height, width, 3), 28, dtype=np.uint8)
    available_height = height - header_height
    scale = min(width / image.shape[1], available_height / image.shape[0])
    resized_width = max(1, round(image.shape[1] * scale))
    resized_height = max(1, round(image.shape[0] * scale))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    x_offset = (width - resized_width) // 2
    y_offset = header_height + (available_height - resized_height) // 2
    canvas[y_offset : y_offset + resized_height, x_offset : x_offset + resized_width] = resized
    cv2.putText(
        canvas,
        title[:58],
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    return canvas


def create_preview(
    dataset_root: Path,
    output_path: Path,
    split: str,
    samples_per_class: int,
    normal_samples: int,
    seed: int,
) -> None:
    names = read_dataset_names(dataset_root)
    image_root = dataset_root / "images" / split
    label_root = dataset_root / "labels" / split
    candidates: dict[int, list[Path]] = defaultdict(list)
    normal_candidates: list[Path] = []

    for label_path in sorted(label_root.glob("*.txt")):
        labels = read_labels(label_path)
        if not labels:
            normal_candidates.append(label_path)
            continue
        for class_id in {label[0] for label in labels}:
            candidates[class_id].append(label_path)

    rng = random.Random(seed)
    selected: list[tuple[str, Path]] = []
    used: set[Path] = set()
    for class_id, class_name in names.items():
        options = candidates[class_id].copy()
        rng.shuffle(options)
        for label_path in options:
            if label_path in used:
                continue
            selected.append((class_name, label_path))
            used.add(label_path)
            if sum(name == class_name for name, _ in selected) >= samples_per_class:
                break

    rng.shuffle(normal_candidates)
    selected.extend(("normal", path) for path in normal_candidates[:normal_samples])

    tiles = []
    for expected_class, label_path in selected:
        image_path = image_root / f"{label_path.stem}.jpg"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        labels = read_labels(label_path)
        annotated = draw_labels(image, labels, names)
        tiles.append(make_tile(annotated, f"{expected_class} | {image_path.name}"))

    columns = 4
    rows = (len(tiles) + columns - 1) // columns
    blank = np.full_like(tiles[0], 28)
    while len(tiles) < rows * columns:
        tiles.append(blank.copy())
    contact_sheet = np.vstack(
        [np.hstack(tiles[row * columns : (row + 1) * columns]) for row in range(rows)]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), contact_sheet):
        raise OSError(f"Could not write preview: {output_path}")
    print(f"preview written to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a labeled YOLO dataset contact sheet.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--samples-per-class", type=int, default=3)
    parser.add_argument("--normal-samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    create_preview(
        dataset_root=arguments.dataset_root.resolve(),
        output_path=arguments.output.resolve(),
        split=arguments.split,
        samples_per_class=arguments.samples_per_class,
        normal_samples=arguments.normal_samples,
        seed=arguments.seed,
    )
