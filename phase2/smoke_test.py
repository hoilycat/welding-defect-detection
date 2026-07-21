from __future__ import annotations

import argparse
import sys
from pathlib import Path

LOCAL_PACKAGES = Path(__file__).resolve().parent / ".packages"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))

import cv2
import numpy as np

from vision import (
    detect_candidates_from_blackhat,
    detect_with_yolo,
    draw_detections,
    extract_features,
    preprocess_views,
    summarize,
)


def make_demo_weld_image() -> np.ndarray:
    image = np.full((280, 420, 3), 188, dtype=np.uint8)
    cv2.rectangle(image, (0, 108), (419, 174), (166, 166, 166), -1)
    cv2.line(image, (0, 134), (419, 148), (205, 205, 205), 16)
    cv2.line(image, (96, 142), (306, 150), (28, 28, 28), 5)
    cv2.circle(image, (318, 132), 10, (35, 35, 35), -1)
    cv2.ellipse(image, (235, 166), (36, 8), 8, 0, 360, (42, 42, 42), -1)
    noise = np.random.default_rng(42).normal(0, 5, image.shape).astype(np.int16)
    return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def write_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def run_smoke_test(save_dir: Path | None = None) -> str:
    image = make_demo_weld_image()
    views = preprocess_views(
        image,
        clahe_clip=3.0,
        blackhat_kernel=15,
        gradient_kernel=5,
        emboss_depth=4.5,
        sharpen_amount=0.8,
    )

    yolo_detections = detect_with_yolo(image, model_path="", confidence_threshold=0.25)
    detections = yolo_detections or detect_candidates_from_blackhat(
        views["blackhat"],
        min_area=180,
        threshold=80,
    )

    if not detections:
        raise AssertionError("Expected at least one OpenCV fallback candidate.")

    features = [extract_features(image, detection) for detection in detections]
    summary = summarize(detections, features)

    if save_dir:
        write_rgb(save_dir / "01_demo_input.png", image)
        write_rgb(save_dir / "02_detection_result.png", draw_detections(image, detections))
        for name in ("clahe", "blackhat", "gradient", "emboss"):
            write_rgb(save_dir / f"03_{name}.png", views[name])
        (save_dir / "summary.md").write_text(summary, encoding="utf-8")

    labels = ", ".join(detection.label for detection in detections)
    return f"OK: {len(detections)} fallback candidate(s) detected: {labels}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the YOLO-less Phase 2 demo path.")
    parser.add_argument("--save-dir", type=Path, help="Optional directory for demo output images.")
    args = parser.parse_args()
    print(run_smoke_test(args.save_dir))


if __name__ == "__main__":
    main()
