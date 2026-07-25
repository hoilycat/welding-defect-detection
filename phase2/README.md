# WeldVision Phase 2

Phase 2 is a local welding inspection assistant that combines separate RT/VT YOLOv8 detectors with OpenCV evidence views, feature analysis, risk scoring, and rule-based explanations.

## Current Status

- RT and VT inspection modes are available in the Gradio UI.
- Each mode selects its own model path and confidence threshold.
- RT classes: crack, porosity, lack of fusion, slag inclusion.
- VT classes: porosity, lack of fusion, incomplete penetration, undercut.
- The VT model reached a best training mAP50 of 0.847.

See the [final project report](../docs/final-project-report.md) for metrics and limitations, and the [demo guide](../docs/demo-guide.md) for the Korean user workflow.

## Run

Install dependencies:

```bash
pip install -r phase2/requirements.txt
```

Start the local app:

```bash
python phase2/gradio_app.py
```

Open `http://127.0.0.1:7860/` and select the inspection mode before uploading an image.

| Mode | Default model | Default confidence |
|---|---|---:|
| RT | `runs/detect/rt-v4-balanced/weights/best.pt` | 0.10 |
| VT | `runs/detect/vt-v1-balanced/weights/best.pt` | 0.25 |

The paths can be changed in the UI. If no valid model is available, the app can still show an OpenCV candidate, but that fallback is preprocessing evidence rather than a trained AI decision.

## Main Views

| View | Role |
|---|---|
| CLAHE | Enhances local contrast in faint areas |
| Black-hat | Highlights dark patterns relative to their surroundings |
| Gradient | Highlights boundaries and directional structure |
| Emboss | Highlights surface texture and uneven patterns |

The preprocessing sliders do not retrain or tune YOLO. `YOLO confidence` directly changes which model detections are displayed.

## Prepare a YOLO Dataset

The source dataset contains polygon JSON annotations. `prepare_yolo_dataset.py` validates the polygons and converts them to tight YOLO boxes, so the existing labels do not need to be redrawn in CVAT.

Validate without writing output:

```bash
python phase2/prepare_yolo_dataset.py --source-root "D:/path/to/1.데이터" --dry-run
```

Create one inspection-specific dataset:

```bash
python phase2/prepare_yolo_dataset.py --source-root "D:/path/to/1.데이터" --output-root "D:/path/to/yolo-vt" --inspection-type VT
```

Use `--inspection-type RT` or `--inspection-type VT`. Normal images receive empty label files, zero-area boxes are skipped, and classes outside the selected modality are excluded.

## Verification

Run unit tests:

```bash
python -m unittest discover -s phase2 -p "test_*.py" -v
```

Run the OpenCV fallback smoke test:

```bash
python phase2/smoke_test.py --save-dir phase2/demo_outputs
```

The fallback smoke test checks the non-YOLO processing path only. Reported model performance comes from the Ultralytics validation results, not from this smoke test.

## Important Limitations

- RT remains a pilot-quality model and is weaker than the VT model.
- VT undercut has lower recall than the other VT classes.
- Demo samples chosen from training folders verify the app flow, not independent field accuracy.
- Model weights are intentionally excluded from Git and must be supplied locally.
- The app is an inspection aid, not a safety certification or replacement for an inspector.
