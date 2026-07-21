# WeldVision Phase 2

YOLOv8 detection, OpenCV preprocessing, feature analysis, risk scoring, and rule-based cause inference are separated into small modules so the system can grow without throwing away the Stage 1 work.

## Goal

This stage changes the project from "defect classification only" into an inspection assistant:

- Detect defect location and class with YOLOv8.
- Show visual evidence with OpenCV preprocessing.
- Reuse Stage 1 features such as circularity, aspect ratio, brightness, and area.
- Explain risk, likely causes, and recommended actions.
- Present everything through a Gradio demo.

## Main Preprocessing Views

| View | Role |
|---|---|
| CLAHE | Local contrast enhancement for faint defects |
| Black-hat | Highlights dark defect candidates |
| Gradient | Highlights linearity, boundaries, and direction |
| Emboss | Highlights texture and uneven patterns |

Canny is intentionally not used as the core method because weld textures and noise can be over-detected as edges. It can be added later as a comparison-only view.

## UI Layout

```text
[Image upload] [Optional YOLO model path]

+----------------------+----------------------+
| Original             | Detection result     |
+----------------------+----------------------+

+-----------+-----------+-----------+-----------+
| CLAHE     | Black-hat | Gradient  | Emboss    |
+-----------+-----------+-----------+-----------+

[Sliders]
- Contrast clip limit
- Black-hat kernel size
- Gradient strength
- Emboss depth
- Sharpen amount

[Analysis]
- Defect type
- Confidence
- Risk score
- Circularity / aspect ratio / brightness / area
- Likely cause
- Recommended action
```

## Run

Install dependencies:

```bash
pip install -r phase2/requirements.txt
```

Run the demo:

```bash
python phase2/gradio_app.py
```

If you already have a trained YOLOv8 model, enter the model path in the UI, for example:

```text
runs/detect/train/weights/best.pt
```

If no model is provided, the app still runs with an OpenCV candidate detector so preprocessing and rule-based explanation can be demonstrated.

## YOLO-less Smoke Test

The current demo can be checked before a YOLOv8 model is trained or installed. The smoke test creates a synthetic weld-like image, runs the OpenCV fallback detector, extracts feature values, and writes demo output images if a save directory is provided.

```bash
python phase2/smoke_test.py --save-dir phase2/demo_outputs
```

Expected result:

```text
OK: 1 fallback candidate(s) detected: crack
```

The fallback result is a visual candidate, not a final AI judgment. It is intended to prove that the Phase 2 UI, preprocessing views, feature extraction, and rule-based explanation path work before the YOLOv8 detector is connected.
