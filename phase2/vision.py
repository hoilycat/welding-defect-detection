from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rules import explain_detection


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    source: str


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("Image is empty.")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def odd_kernel(value: int, minimum: int = 3) -> int:
    value = max(minimum, int(value))
    return value if value % 2 == 1 else value + 1


def preprocess_views(
    image_rgb: np.ndarray,
    clahe_clip: float,
    blackhat_kernel: int,
    gradient_kernel: int,
    emboss_depth: float,
    sharpen_amount: float,
) -> dict[str, np.ndarray]:
    bgr = ensure_bgr(image_rgb)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)

    blackhat_size = odd_kernel(blackhat_kernel)
    blackhat_kernel_mat = cv2.getStructuringElement(
        cv2.MORPH_RECT, (blackhat_size, blackhat_size)
    )
    blackhat = cv2.morphologyEx(clahe_img, cv2.MORPH_BLACKHAT, blackhat_kernel_mat)
    blackhat = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)

    grad_size = odd_kernel(gradient_kernel)
    grad_x = cv2.Scharr(clahe_img, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(clahe_img, cv2.CV_32F, 0, 1)
    gradient = cv2.magnitude(grad_x, grad_y)
    gradient = cv2.GaussianBlur(gradient, (grad_size, grad_size), 0)
    gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    depth = float(emboss_depth)
    emboss_kernel = np.array(
        [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32
    ) * depth
    emboss = cv2.filter2D(gray, -1, emboss_kernel) + 128
    emboss = np.clip(emboss, 0, 255).astype(np.uint8)

    if sharpen_amount > 0:
        blurred = cv2.GaussianBlur(clahe_img, (0, 0), sigmaX=1.2)
        sharpened = cv2.addWeighted(
            clahe_img, 1.0 + float(sharpen_amount), blurred, -float(sharpen_amount), 0
        )
    else:
        sharpened = clahe_img

    return {
        "gray": cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB),
        "clahe": cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB),
        "blackhat": cv2.cvtColor(blackhat.astype(np.uint8), cv2.COLOR_GRAY2RGB),
        "gradient": cv2.cvtColor(gradient, cv2.COLOR_GRAY2RGB),
        "emboss": cv2.cvtColor(emboss, cv2.COLOR_GRAY2RGB),
        "sharpened": cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB),
    }


def detect_with_yolo(
    image_rgb: np.ndarray,
    model_path: str | None,
    confidence_threshold: float,
    include_review_candidates: bool = False,
    review_threshold: float = 0.05,
) -> list[Detection]:
    if not model_path:
        return []

    path = Path(model_path)
    if not path.exists():
        return []

    config_dir = Path(__file__).resolve().parent / "yolo_config"
    config_dir.mkdir(exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))

    try:
        from ultralytics import YOLO
    except ImportError:
        return []

    model = YOLO(str(path))
    prediction_threshold = (
        min(float(confidence_threshold), float(review_threshold))
        if include_review_candidates
        else float(confidence_threshold)
    )
    # Ultralytics expects OpenCV-style BGR when a NumPy array is supplied.
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    results = model.predict(
        image_bgr,
        conf=prediction_threshold,
        iou=0.45,
        agnostic_nms=False,
        verbose=False,
    )
    if not results:
        return []

    detections: list[Detection] = []
    names: dict[int, str] = getattr(results[0], "names", {}) or {}
    boxes = getattr(results[0], "boxes", None)
    if boxes is None:
        return detections

    for box in boxes:
        xyxy = box.xyxy.cpu().numpy()[0].astype(int)
        conf = float(box.conf.cpu().numpy()[0])
        cls_id = int(box.cls.cpu().numpy()[0])
        label = names.get(cls_id, str(cls_id))
        detections.append(
            Detection(
                label=label,
                confidence=conf,
                bbox=(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])),
                source=(
                    "YOLOv8 review candidate"
                    if conf < float(confidence_threshold)
                    else "YOLOv8"
                ),
            )
        )
    return detections


def detect_candidates_from_blackhat(
    blackhat_rgb: np.ndarray, min_area: int, threshold: int
) -> list[Detection]:
    gray = cv2.cvtColor(blackhat_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(gray, int(threshold), 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections: list[Detection] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        bbox_area = max(1, w * h)
        extent = area / bbox_area
        if area < max(min_area, 120):
            continue
        if w < 5 or h < 5:
            continue
        if extent < 0.12:
            continue
        detections.append(
            Detection(
                label=infer_candidate_label(contour),
                confidence=0.0,
                bbox=(x, y, x + w, y + h),
                source="OpenCV candidate (rule)",
            )
        )
    detections.sort(key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]), reverse=True)
    return detections[:5]


def infer_candidate_label(contour: np.ndarray) -> str:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = 4.0 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0.0
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = max(w / max(h, 1), h / max(w, 1))

    if aspect_ratio >= 5.0 and area >= 80:
        return "crack"
    if circularity >= 0.70 and aspect_ratio <= 1.8 and 40 <= area <= 500:
        return "porosity"
    if aspect_ratio >= 2.5 and area >= 100:
        return "slag inclusion"
    return "candidate"


def extract_features(image_rgb: np.ndarray, detection: Detection) -> dict[str, float]:
    x1, y1, x2, y2 = detection.bbox
    h, w = image_rgb.shape[:2]
    x1, x2 = sorted((max(0, x1), min(w - 1, x2)))
    y1, y2 = sorted((max(0, y1), min(h - 1, y2)))
    roi = image_rgb[y1:y2, x1:x2]
    if roi.size == 0:
        return {
            "circularity": 0.0,
            "aspect_ratio": 0.0,
            "mean_brightness": 0.0,
            "std_brightness": 0.0,
            "area_ratio": 0.0,
        }

    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circularity = 0.0
    if contours:
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4.0 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0.0

    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    aspect_ratio = max(width / height, height / width)
    area_ratio = (width * height) / max(1, image_rgb.shape[0] * image_rgb.shape[1])

    return {
        "circularity": round(float(circularity), 3),
        "aspect_ratio": round(float(aspect_ratio), 3),
        "mean_brightness": round(float(np.mean(gray)), 2),
        "std_brightness": round(float(np.std(gray)), 2),
        "area_ratio": round(float(area_ratio), 5),
    }


def draw_detections(image_rgb: np.ndarray, detections: list[Detection]) -> np.ndarray:
    canvas = image_rgb.copy()
    colors = {
        "crack": (255, 40, 40),
        "porosity": (30, 190, 90),
        "lack of fusion": (40, 120, 255),
        "fusion": (40, 120, 255),
        "slag inclusion": (240, 180, 40),
        "undercut": (255, 120, 30),
        "candidate": (210, 210, 210),
    }

    for det in detections:
        rule = explain_detection(det.label)
        color = colors.get(rule.defect_type, (255, 255, 255))
        x1, y1, x2, y2 = det.bbox
        is_review_candidate = det.source == "YOLOv8 review candidate"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1 if is_review_candidate else 2)
        caption = f"{rule.defect_type}"
        if det.confidence > 0:
            caption += f" {det.confidence:.2f}"
        if is_review_candidate:
            caption += " review"
        cv2.putText(
            canvas,
            caption,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return canvas


def summarize(detections: list[Detection], features: list[dict[str, Any]]) -> str:
    if not detections:
        return "결함 후보가 검출되지 않았습니다. 어두운 영역 임계값을 낮추거나 학습된 YOLOv8 모델을 연결해 확인하세요."

    lines = ["### 검사 결과 요약"]
    if all(det.source.startswith("OpenCV candidate") for det in detections):
        lines.append(
            "> YOLOv8 모델이 연결되지 않아 OpenCV 전처리 기반 후보 모드로 표시 중입니다. 이 박스는 최종 AI 판정이 아니라, Black-hat에서 강조된 어두운 영역 후보입니다."
        )
    elif any(det.source == "YOLOv8 review candidate" for det in detections):
        lines.append(
            "> 얇은 박스의 `review` 표시는 기준 신뢰도보다 낮지만 모델이 포착한 검토 후보입니다. 최종 판정 전에 원본 영상과 함께 확인하세요."
        )
    for idx, det in enumerate(detections, start=1):
        rule = explain_detection(det.label)
        feat = features[idx - 1]
        confidence = f"{det.confidence:.2f}" if det.confidence > 0 else "N/A"
        lines.extend(
            [
                f"**{idx}. {rule.display_name}**",
                f"- 검출 방식 (Source): {det.source}",
                f"- 신뢰도 (Confidence): {confidence}",
                f"- 위험도 점수 (Risk): {rule.risk_score}",
                f"- 권장 조치 (Action): {rule.action}",
                f"- 추정 원인 (Likely Cause): {rule.likely_cause}",
                f"- 판단 근거 (Reason): {rule.reason}",
                f"- 특징값 (Features): 원형도 {feat['circularity']}, 종횡비 {feat['aspect_ratio']}, 평균 밝기 {feat['mean_brightness']}",
            ]
        )
    return "\n".join(lines)
