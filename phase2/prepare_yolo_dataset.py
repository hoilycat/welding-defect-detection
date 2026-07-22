from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


CLASS_NAMES = [
    "crack",
    "porosity",
    "lack_of_fusion",
    "slag_inclusion",
    "incomplete_penetration",
    "undercut",
]
CLASS_IDS = {name: index for index, name in enumerate(CLASS_NAMES)}
CLASS_NAMES_BY_TYPE = {
    "RT": ["crack", "porosity", "lack_of_fusion", "slag_inclusion"],
    "VT": ["porosity", "lack_of_fusion", "incomplete_penetration", "undercut"],
    "all": CLASS_NAMES,
}
CASE_ALIASES = {
    "crack": "crack",
    "porosity": "porosity",
    "lack of fusion": "lack_of_fusion",
    "slag inclusion": "slag_inclusion",
    "incomplete penetration": "incomplete_penetration",
    "lack of penetration": "incomplete_penetration",
    "undercut": "undercut",
}
SPLITS = {"Training": "train", "Validation": "val"}


@dataclass
class ConversionSummary:
    images: int = 0
    boxes: int = 0
    normal_images: int = 0
    skipped_images: int = 0
    invalid_boxes: int = 0
    invalid_box_examples: list[str] = field(default_factory=list)


class DegenerateBoxError(ValueError):
    pass


def _annotation_to_yolo(
    annotation: dict,
    width: int,
    height: int,
    fallback_case: str | None = None,
    class_ids: dict[str, int] | None = None,
) -> tuple[str, str] | None:
    if annotation.get("class") != "defect":
        return None

    source_case = str(annotation.get("case", "")).strip().lower()
    if not source_case and fallback_case:
        source_case = fallback_case
    class_name = CASE_ALIASES.get(source_case)
    if class_name is None:
        raise ValueError(f"Unknown defect case: {source_case!r}")

    coordinate = annotation.get("coordinate", {})
    xs = coordinate.get("x", [])
    ys = coordinate.get("y", [])
    if not xs or not ys or len(xs) != len(ys):
        raise ValueError("Polygon coordinates are empty or mismatched")

    x_min = max(0.0, min(float(value) for value in xs))
    x_max = min(float(width), max(float(value) for value in xs))
    y_min = max(0.0, min(float(value) for value in ys))
    y_max = min(float(height), max(float(value) for value in ys))
    if x_max <= x_min or y_max <= y_min:
        raise DegenerateBoxError("Polygon produces a zero-area bounding box")

    x_center = ((x_min + x_max) / 2.0) / width
    y_center = ((y_min + y_max) / 2.0) / height
    box_width = (x_max - x_min) / width
    box_height = (y_max - y_min) / height
    active_class_ids = class_ids or CLASS_IDS
    if class_name not in active_class_ids:
        raise ValueError(f"Defect case {class_name!r} is not valid for this inspection type")
    class_id = active_class_ids[class_name]
    line = (
        f"{class_id} {x_center:.6f} {y_center:.6f} "
        f"{box_width:.6f} {box_height:.6f}"
    )
    return class_name, line


def convert_split(
    source_root: Path,
    output_root: Path | None,
    source_split: str,
    target_split: str,
    inspection_type: str,
    dry_run: bool,
    limit_per_folder: int | None = None,
) -> tuple[ConversionSummary, Counter[str]]:
    split_root = source_root / source_split
    image_root = split_root / "01.원천데이터"
    annotation_root = split_root / "02.라벨링데이터"
    if not image_root.is_dir() or not annotation_root.is_dir():
        raise FileNotFoundError(f"Dataset folders are missing under {split_root}")

    summary = ConversionSummary()
    class_counts: Counter[str] = Counter()
    folder_counts: Counter[Path] = Counter()
    active_class_ids = {
        name: index for index, name in enumerate(CLASS_NAMES_BY_TYPE[inspection_type])
    }

    if not dry_run:
        assert output_root is not None
        output_images = output_root / "images" / target_split
        output_labels = output_root / "labels" / target_split
        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)

    for annotation_path in sorted(annotation_root.rglob("*.json")):
        if inspection_type != "all" and f"_{inspection_type}" not in annotation_path.parent.name:
            summary.skipped_images += 1
            continue
        if (
            limit_per_folder is not None
            and folder_counts[annotation_path.parent] >= limit_per_folder
        ):
            summary.skipped_images += 1
            continue
        folder_counts[annotation_path.parent] += 1

        with annotation_path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)

        current_type = str(document.get("info", {}).get("type", "")).upper()
        if inspection_type != "all" and current_type != inspection_type:
            summary.skipped_images += 1
            continue

        image_data = document.get("image_data", {})
        width = int(image_data.get("width", 0))
        height = int(image_data.get("height", 0))
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image dimensions in {annotation_path}")

        image_name = f"{image_data['file_name']}.{image_data['format']}"
        label_prefix = "TL_" if source_split == "Training" else "VL_"
        image_prefix = "TS_" if source_split == "Training" else "VS_"
        source_folder = annotation_path.parent.name
        image_folder = source_folder.replace(label_prefix, image_prefix, 1)
        image_path = image_root / image_folder / image_name
        if not image_path.is_file():
            raise FileNotFoundError(f"Image for {annotation_path.name} was not found: {image_name}")

        label_lines: list[str] = []
        annotation_cases = {
            str(case).strip().lower()
            for case in document.get("meta", {}).get("annotation_case", [])
            if str(case).strip().lower() in CASE_ALIASES
        }
        fallback_case = next(iter(annotation_cases)) if len(annotation_cases) == 1 else None
        for annotation_index, annotation in enumerate(document.get("annotations", [])):
            try:
                converted = _annotation_to_yolo(
                    annotation,
                    width,
                    height,
                    fallback_case=fallback_case,
                    class_ids=active_class_ids,
                )
            except DegenerateBoxError as error:
                summary.invalid_boxes += 1
                summary.invalid_box_examples.append(
                    f"{annotation_path} annotation #{annotation_index + 1}: {error}"
                )
                continue
            except ValueError as error:
                raise ValueError(
                    f"{annotation_path} annotation #{annotation_index + 1}: {error}"
                ) from error
            if converted is None:
                continue
            class_name, line = converted
            class_counts[class_name] += 1
            label_lines.append(line)

        summary.images += 1
        summary.boxes += len(label_lines)
        if not label_lines:
            summary.normal_images += 1

        if not dry_run:
            target_image = output_images / image_path.name
            target_label = output_labels / f"{image_path.stem}.txt"
            if target_image.exists() or target_label.exists():
                raise FileExistsError(f"Output already exists for {image_path.stem}")
            shutil.copy2(image_path, target_image)
            target_label.write_text("\n".join(label_lines), encoding="utf-8")

    return summary, class_counts


def write_data_yaml(output_root: Path, class_names: list[str]) -> None:
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(class_names))
    content = (
        f"path: {output_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        f"{names}\n"
    )
    (output_root / "data.yaml").write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the welding polygon JSON labels to Ultralytics YOLO boxes."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--inspection-type",
        choices=("all", "RT", "VT"),
        default="all",
        help="Keep all images or only one inspection modality.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and count files without writing the output dataset.",
    )
    parser.add_argument(
        "--limit-per-folder",
        type=int,
        help="Use at most this many images from each source category folder.",
    )
    args = parser.parse_args()
    if not args.dry_run and args.output_root is None:
        parser.error("--output-root is required unless --dry-run is used")
    if args.limit_per_folder is not None and args.limit_per_folder <= 0:
        parser.error("--limit-per-folder must be greater than zero")
    return args


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve() if args.output_root else None
    total_summary = ConversionSummary()
    total_counts: Counter[str] = Counter()

    for source_split, target_split in SPLITS.items():
        summary, counts = convert_split(
            source_root=source_root,
            output_root=output_root,
            source_split=source_split,
            target_split=target_split,
            inspection_type=args.inspection_type,
            dry_run=args.dry_run,
            limit_per_folder=args.limit_per_folder,
        )
        total_summary.images += summary.images
        total_summary.boxes += summary.boxes
        total_summary.normal_images += summary.normal_images
        total_summary.skipped_images += summary.skipped_images
        total_summary.invalid_boxes += summary.invalid_boxes
        total_summary.invalid_box_examples.extend(summary.invalid_box_examples)
        total_counts.update(counts)
        print(
            f"{target_split}: {summary.images} images, {summary.boxes} boxes, "
            f"{summary.normal_images} normal images, "
            f"{summary.invalid_boxes} invalid boxes skipped"
        )

    if not args.dry_run:
        assert output_root is not None
        write_data_yaml(output_root, CLASS_NAMES_BY_TYPE[args.inspection_type])

    print(f"inspection type: {args.inspection_type}")
    print(f"skipped by filter: {total_summary.skipped_images}")
    print(f"invalid boxes skipped: {total_summary.invalid_boxes}")
    for example in total_summary.invalid_box_examples[:10]:
        print(f"  warning: {example}")
    if len(total_summary.invalid_box_examples) > 10:
        print(f"  ... and {len(total_summary.invalid_box_examples) - 10} more")
    print("boxes by class:")
    for class_name in CLASS_NAMES_BY_TYPE[args.inspection_type]:
        print(f"  {class_name}: {total_counts[class_name]}")
    print("dry run complete" if args.dry_run else f"dataset written to: {output_root}")


if __name__ == "__main__":
    main()
