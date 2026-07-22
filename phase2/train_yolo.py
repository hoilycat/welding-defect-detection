from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PHASE2_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PHASE2_ROOT.parent
LOCAL_PACKAGES = PHASE2_ROOT / ".packages"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / "Ultralytics"))

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO detector on a prepared dataset.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name", default="rt-pilot")
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot see an NVIDIA GPU.")

    model = YOLO(args.model)
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str((PROJECT_ROOT / "runs" / "detect").resolve()),
        name=args.name,
        exist_ok=False,
        seed=42,
        deterministic=True,
        patience=10,
        plots=True,
    )


if __name__ == "__main__":
    main()
