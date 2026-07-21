from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import shutil
import unicodedata
import zipfile


CLASS_NAMES = [
    "crack",
    "porosity",
    "lack_of_fusion",
    "slag_inclusion",
    "incomplete_penetration",
    "undercut",
]

CASE_TO_CLASS = {
    "crack": 0,
    "porosity": 1,
    "lack of fusion": 2,
    "slag inclusion": 3,
    "incomplete penetration": 4,
    "undercut": 5,
}


def normalized(value: str) -> str:
    return unicodedata.normalize("NFC", value).strip().lower()


def annotation_lines(payload: dict) -> tuple[list[str], Counter[str]]:
    image_data = payload.get("image_data", {})
    width = float(image_data.get("width", 0))
    height = float(image_data.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError("Label has invalid image width or height.")

    lines: list[str] = []
    counts: Counter[str] = Counter()
    for annotation in payload.get("annotations") or []:
        case_name = normalized(str(annotation.get("case", "")))
        if not case_name:
            continue
        if case_name not in CASE_TO_CLASS:
            counts[f"unknown:{case_name}"] += 1
            continue

        coordinate = annotation.get("coordinate") or {}
        xs = [float(value) for value in coordinate.get("x", [])]
        ys = [float(value) for value in coordinate.get("y", [])]
        if not xs or not ys or len(xs) != len(ys):
            counts[f"invalid:{case_name}"] += 1
            continue

        x_min = max(0.0, min(xs))
        x_max = min(width, max(xs))
        y_min = max(0.0, min(ys))
        y_max = min(height, max(ys))
        box_width = x_max - x_min
        box_height = y_max - y_min
        if box_width <= 0 or box_height <= 0:
            counts[f"invalid:{case_name}"] += 1
            continue

        x_center = ((x_min + x_max) / 2.0) / width
        y_center = ((y_min + y_max) / 2.0) / height
        normalized_width = box_width / width
        normalized_height = box_height / height
        class_id = CASE_TO_CLASS[case_name]
        lines.append(
            f"{class_id} {x_center:.8f} {y_center:.8f} "
            f"{normalized_width:.8f} {normalized_height:.8f}"
        )
        counts[CLASS_NAMES[class_id]] += 1

    return lines, counts


def choose_items(items: list, limit: int | None, seed: int) -> list:
    if not limit or len(items) <= limit:
        return sorted(items, key=lambda value: str(value))
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    return sorted(shuffled[:limit], key=lambda value: str(value))


def replace_file(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    if mode == "copy":
        shutil.copy2(source, target)
    else:
        target.symlink_to(source.resolve())


def write_sample(
    payload: dict,
    image_target: Path,
    label_target: Path,
    counts: Counter[str],
) -> None:
    lines, annotation_counts = annotation_lines(payload)
    label_target.parent.mkdir(parents=True, exist_ok=True)
    label_target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    counts.update(annotation_counts)
    counts["images"] += 1
    if not lines:
        counts["negative_images"] += 1
    if image_target.exists() or image_target.is_symlink():
        counts["images_ready"] += 1


def prepare_training(
    data_root: Path,
    output: Path,
    limit_per_folder: int | None,
    mode: str,
    seed: int,
) -> Counter[str]:
    image_root = data_root / "Training" / "01.원천데이터"
    label_root = data_root / "Training" / "02.라벨링데이터"
    counts: Counter[str] = Counter()

    for folder_index, label_dir in enumerate(sorted(label_root.iterdir())):
        if not label_dir.is_dir():
            continue
        image_dir = image_root / label_dir.name.replace("TL_", "TS_", 1)
        if not image_dir.is_dir():
            counts["missing_image_folders"] += 1
            continue

        label_paths = choose_items(
            list(label_dir.glob("*.json")), limit_per_folder, seed + folder_index
        )
        for label_path in label_paths:
            payload = json.loads(label_path.read_text(encoding="utf-8"))
            image_data = payload.get("image_data", {})
            stem = str(image_data.get("file_name") or label_path.stem)
            extension = str(image_data.get("format") or "jpg").lower().lstrip(".")
            image_source = image_dir / f"{stem}.{extension}"
            if not image_source.is_file():
                counts["missing_images"] += 1
                continue

            image_target = output / "images" / "train" / image_source.name
            label_target = output / "labels" / "train" / f"{stem}.txt"
            replace_file(image_source, image_target, mode)
            write_sample(payload, image_target, label_target, counts)

    return counts


def zip_members_by_name(archive: zipfile.ZipFile, suffix: str) -> dict[str, str]:
    return {
        Path(member).name: member
        for member in archive.namelist()
        if not member.endswith("/") and member.lower().endswith(suffix)
    }


def prepare_validation(
    data_root: Path,
    output: Path,
    limit_per_folder: int | None,
    seed: int,
) -> Counter[str]:
    image_root = data_root / "Validation" / "01.원천데이터"
    label_root = data_root / "Validation" / "02.라벨링데이터"
    counts: Counter[str] = Counter()

    for folder_index, label_zip_path in enumerate(sorted(label_root.glob("*.zip"))):
        image_zip_name = label_zip_path.name.replace("VL_", "VS_", 1)
        image_zip_path = image_root / image_zip_name
        if not image_zip_path.is_file():
            counts["missing_image_archives"] += 1
            continue

        with zipfile.ZipFile(label_zip_path) as label_zip, zipfile.ZipFile(
            image_zip_path
        ) as image_zip:
            label_members = zip_members_by_name(label_zip, ".json")
            selected_names = choose_items(
                list(label_members), limit_per_folder, seed + folder_index
            )
            image_members = zip_members_by_name(image_zip, ".jpg")

            for label_name in selected_names:
                payload = json.loads(label_zip.read(label_members[label_name]).decode("utf-8"))
                image_data = payload.get("image_data", {})
                stem = str(image_data.get("file_name") or Path(label_name).stem)
                extension = str(image_data.get("format") or "jpg").lower().lstrip(".")
                image_name = f"{stem}.{extension}"
                image_member = image_members.get(image_name)
                if image_member is None:
                    counts["missing_images"] += 1
                    continue

                image_target = output / "images" / "val" / image_name
                image_target.parent.mkdir(parents=True, exist_ok=True)
                image_target.write_bytes(image_zip.read(image_member))
                label_target = output / "labels" / "val" / f"{stem}.txt"
                write_sample(payload, image_target, label_target, counts)

    return counts


def write_dataset_yaml(output: Path) -> Path:
    yaml_path = output / "dataset.yaml"
    name_lines = "\n".join(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    yaml_path.write_text(
        f"path: {output.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "names:\n"
        f"{name_lines}\n",
        encoding="utf-8",
    )
    return yaml_path


def validate_roots(data_root: Path) -> None:
    required = [
        data_root / "Training" / "01.원천데이터",
        data_root / "Training" / "02.라벨링데이터",
        data_root / "Validation" / "01.원천데이터",
        data_root / "Validation" / "02.라벨링데이터",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing dataset paths:\n- " + "\n- ".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert RIAWELC polygon JSON labels into a YOLO detection dataset."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parent / "yolo_dataset"
    )
    parser.add_argument(
        "--limit-per-folder",
        type=int,
        help="Create a small balanced dataset by taking at most N files per source folder.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--image-mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="Training images can be symlinked to avoid duplicating the 41 GB source data.",
    )
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    validate_roots(data_root)
    output.mkdir(parents=True, exist_ok=True)

    train_counts = prepare_training(
        data_root, output, args.limit_per_folder, args.image_mode, args.seed
    )
    val_counts = prepare_validation(data_root, output, args.limit_per_folder, args.seed)
    yaml_path = write_dataset_yaml(output)

    summary = {
        "classes": CLASS_NAMES,
        "train": dict(sorted(train_counts.items())),
        "val": dict(sorted(val_counts.items())),
        "dataset_yaml": str(yaml_path),
    }
    summary_path = output / "conversion_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
