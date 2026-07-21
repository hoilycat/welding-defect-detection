from __future__ import annotations

import argparse
import os
from pathlib import Path

PHASE2_DIR = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(PHASE2_DIR / "yolo_config"))
os.environ.setdefault("MPLCONFIGDIR", str(PHASE2_DIR / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PHASE2_DIR / ".cache"))

import torch
from ultralytics import YOLO


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 on the converted weld dataset.")
    parser.add_argument(
        "--data",
        type=Path,
        default=PHASE2_DIR / "yolo_dataset" / "dataset.yaml",
    )
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--name", default="weld_yolov8n")
    parser.add_argument("--project", type=Path, default=PHASE2_DIR / "runs")
    args = parser.parse_args()

    data_path = args.data.expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_path}. Run prepare_yolo_dataset.py first."
        )

    device = choose_device(args.device)
    print(f"Training device: {device}")
    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        project=str(args.project.expanduser().resolve()),
        name=args.name,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
