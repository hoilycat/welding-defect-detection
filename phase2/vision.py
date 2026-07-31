from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rules import explain_detection

logger = logging.getLogger(__name__)

# YOLO Model Cache
_MODEL_CACHE: dict[str, Any] = {}
_MAX_CACHE_SIZE = 4


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
    dark_threshold: int = 150,
    min_candidate_area: int = 180,
) -> dict[str, np.ndarray]:
    bgr = ensure_bgr(image_rgb)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)

    blackhat_size = odd_kernel(blackhat_kernel)
    blackhat_kernel_mat = cv2.getStructuringElement(
        cv2.MORPH_RECT, (blackhat_size, blackhat_size)
    )
    blackhat_raw = cv2.morphologyEx(clahe_img, cv2.MORPH_BLACKHAT, blackhat_kernel_mat)

    # GaussianBlur + NORM_MINMAX pipeline for noise reduction
    blurred = cv2.GaussianBlur(blackhat_raw, (3, 3), 0)
    blackhat_norm = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)

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
        blurred_sharp = cv2.GaussianBlur(clahe_img, (0, 0), sigmaX=1.2)
        sharpened = cv2.addWeighted(
            clahe_img, 1.0 + float(sharpen_amount), blurred_sharp, -float(sharpen_amount), 0
        )
    else:
        sharpened = clahe_img

    # Candidate mask & contour overlay with stats stamp
    blackhat_overlay = generate_blackhat_overlay(
        blackhat_norm,
        blackhat_size=blackhat_size,
        dark_threshold=int(dark_threshold),
        min_candidate_area=int(min_candidate_area),
    )

    return {
        "gray": cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB),
        "clahe": cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB),
        "blackhat": blackhat_overlay,
        "gradient": cv2.cvtColor(gradient, cv2.COLOR_GRAY2RGB),
        "emboss": cv2.cvtColor(emboss, cv2.COLOR_GRAY2RGB),
        "sharpened": cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB),
    }


def generate_blackhat_overlay(
    blackhat_norm: np.ndarray,
    blackhat_size: int,
    dark_threshold: int,
    min_candidate_area: int,
) -> np.ndarray:
    _, mask = cv2.threshold(blackhat_norm, dark_threshold, 255, cv2.THRESH_BINARY)

    opening_kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, opening_kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, opening_kernel)

    white_pct = (np.count_nonzero(mask_clean) / mask_clean.size) * 100.0
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    overlay = cv2.cvtColor(blackhat_norm, cv2.COLOR_GRAY2RGB)

    all_mask = np.zeros_like(blackhat_norm)
    valid_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if area >= 3:
            cv2.drawContours(all_mask, [c], -1, 255, -1)
        if area >= min_candidate_area:
            valid_contours.append(c)

    # Soft coral tint for raw thresholded mask
    overlay[all_mask > 0] = (
        overlay[all_mask > 0] * 0.65 + np.array([255, 80, 80]) * 0.35
    ).astype(np.uint8)

    # Bright lime green outlines for valid candidate contours
    cv2.drawContours(overlay, valid_contours, -1, (0, 255, 120), 2)

    # Stamp parameters and stats on header
    stamp = f"Thresh={dark_threshold} | MinArea={min_candidate_area} | White={white_pct:.1f}% | Candidates={len(valid_contours)}"
    cv2.rectangle(overlay, (5, 5), (min(overlay.shape[1] - 5, 520), 28), (20, 20, 20), -1)
    cv2.putText(
        overlay,
        stamp,
        (10, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (0, 255, 200),
        1,
        cv2.LINE_AA,
    )
    return overlay


def get_yolo_model(model_path_str: str) -> Any | None:
    """Load and cache YOLO model instances per path."""
    if not model_path_str:
        return None

    if model_path_str in _MODEL_CACHE:
        return _MODEL_CACHE[model_path_str]

    path = Path(model_path_str)
    if not path.is_absolute():
        project_root = Path(__file__).resolve().parents[1]
        path = project_root / path

    if not path.is_file():
        logger.warning(f"[YOLO Model] Model file not found: {model_path_str}")
        return None

    config_dir = Path(__file__).resolve().parent / "yolo_config"
    config_dir.mkdir(exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))

    try:
        from ultralytics import YOLO

        logger.info(f"[YOLO Model] Loading model weights into cache: {path}")
        model = YOLO(str(path))

        if len(_MODEL_CACHE) >= _MAX_CACHE_SIZE:
            oldest_key = next(iter(_MODEL_CACHE))
            del _MODEL_CACHE[oldest_key]

        _MODEL_CACHE[model_path_str] = model
        return model
    except Exception as e:
        logger.error(f"[YOLO Model Load Failure] Failed to load model '{model_path_str}': {e}")
        return None


def detect_with_yolo(
    image_rgb: np.ndarray,
    model_path: str | None,
    confidence_threshold: float,
    include_review_candidates: bool = False,
    review_threshold: float = 0.05,
) -> list[Detection]:
    if not model_path:
        return []

    model = get_yolo_model(model_path)
    if model is None:
        return []

    prediction_threshold = (
        min(float(confidence_threshold), float(review_threshold))
        if include_review_candidates
        else float(confidence_threshold)
    )

    try:
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        results = model.predict(
            image_bgr,
            conf=prediction_threshold,
            imgsz=960,
            iou=0.45,
            agnostic_nms=False,
            verbose=False,
        )
    except Exception as e:
        logger.error(f"[YOLO Inference Failure] Inference error for model '{model_path}': {e}")
        return []

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
    _, binary = cv2.threshold(gray, int(threshold), 255, cv2.THRESH_BINARY)
    opening_kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, opening_kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, opening_kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections: list[Detection] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        label = infer_candidate_label(w, h, area)
        detections.append(
            Detection(
                label=label,
                confidence=0.0,
                bbox=(x, y, x + w, y + h),
                source="OpenCV candidate",
            )
        )
    detections.sort(key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]), reverse=True)
    return detections[:10]


def infer_candidate_label(width: int, height: int, area: float) -> str:
    aspect_ratio = max(width / max(1, height), height / max(1, width))
    bounding_box_area = max(1, width * height)
    extent = area / bounding_box_area

    if aspect_ratio >= 5.0:
        return "crack"
    if extent >= 0.7 and aspect_ratio < 2.0:
        return "porosity"
    if aspect_ratio >= 2.5:
        return "slag inclusion"
    return "candidate"


def extract_features(image_rgb: np.ndarray, detection: Detection) -> dict[str, Any]:
    x1, y1, x2, y2 = detection.bbox
    h, w = image_rgb.shape[:2]

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return {
            "circularity": 0.0,
            "aspect_ratio": 0.0,
            "mean_brightness": 0.0,
            "std_brightness": 0.0,
            "area_ratio": 0.0,
        }

    roi = image_rgb[y1:y2, x1:x2]
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


def get_location_description(
    bbox: tuple[int, int, int, int], img_width: int, img_height: int
) -> str:
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

    col = "좌측" if cx < img_width / 3.0 else ("우측" if cx > img_width * 2.0 / 3.0 else "중앙")
    row = "상단" if cy < img_height / 3.0 else ("하단" if cy > img_height * 2.0 / 3.0 else "중앙")

    if row == "중앙" and col == "중앙":
        pos_name = "중앙"
    elif row == "중앙":
        pos_name = f"중앙 {col}"
    elif col == "중앙":
        pos_name = f"{row} 중앙"
    else:
        pos_name = f"{row} {col}"

    return f"이미지 {pos_name} (x1={x1}, y1={y1}, x2={x2}, y2={y2})"


def crop_detection_roi(
    image_rgb: np.ndarray, bbox: tuple[int, int, int, int], margin_pct: float = 0.30
) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    h, w = image_rgb.shape[:2]

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    side = max(bw, bh) * (1.0 + margin_pct)
    half = side / 2.0

    nx1 = max(0, int(round(cx - half)))
    ny1 = max(0, int(round(cy - half)))
    nx2 = min(w, int(round(cx + half)))
    ny2 = min(h, int(round(cy + half)))

    crop = image_rgb[ny1:ny2, nx1:nx2].copy()
    if crop.size == 0:
        crop = image_rgb[y1:y2, x1:x2].copy()
        nx1, ny1 = x1, y1

    rx1 = max(0, x1 - nx1)
    ry1 = max(0, y1 - ny1)
    rx2 = min(crop.shape[1], x2 - nx1)
    ry2 = min(crop.shape[0], y2 - ny1)

    crop_annotated = crop.copy()
    cv2.rectangle(
        crop_annotated,
        (rx1, ry1),
        (rx2, ry2),
        (255, 200, 0),
        2,
    )
    return crop_annotated


def draw_detections(image_rgb: np.ndarray, detections: list[Detection]) -> np.ndarray:
    canvas = image_rgb.copy()
    h, w = canvas.shape[:2]

    thickness = max(2, int(round(min(h, w) / 350.0)))
    font_scale = max(0.5, min(h, w) / 700.0)

    colors = {
        "crack": (255, 50, 50),
        "porosity": (40, 220, 100),
        "lack of fusion": (50, 140, 255),
        "fusion": (50, 140, 255),
        "slag inclusion": (255, 180, 0),
        "undercut": (255, 130, 40),
        "candidate": (220, 220, 220),
    }

    placed_label_rects: list[tuple[int, int, int, int]] = []

    def is_rect_overlapping(r1: tuple[int, int, int, int], r2: tuple[int, int, int, int]) -> bool:
        return not (r1[2] <= r2[0] or r1[0] >= r2[2] or r1[3] <= r2[1] or r1[1] >= r2[3])

    ext_slot_y = 30

    for idx, det in enumerate(detections, start=1):
        rule = explain_detection(det.label)
        color = colors.get(rule.defect_type, (255, 255, 255))
        x1, y1, x2, y2 = det.bbox
        is_review_candidate = det.source == "YOLOv8 review candidate"
        box_thickness = 1 if is_review_candidate else thickness

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, box_thickness)

        num_str = f"#{idx}"
        cv2.putText(
            canvas,
            num_str,
            (x1 + 3, max(y1 + 14, min(y2 - 2, y1 + 14))),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * 0.85,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            num_str,
            (x1 + 3, max(y1 + 14, min(y2 - 2, y1 + 14))),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * 0.85,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        caption = f"#{idx} {rule.defect_type}"
        if det.confidence > 0:
            caption += f" {det.confidence:.2f}"
        if is_review_candidate:
            caption += " rev"

        (txt_w, txt_h), baseline = cv2.getTextSize(
            caption, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )

        candidates_pos = [
            (x1, y1 - 4),                      # Above box
            (x1, y2 + txt_h + 4),              # Below box
            (x2 + 6, y1 + txt_h),              # Right of box
            (max(0, x1 - txt_w - 6), y1 - 4),  # Left of box
            (x1 + 25, y1 - 4),                 # Above shifted right
            (x1 + 25, y2 + txt_h + 4),         # Below shifted right
        ]

        chosen_rect = None
        chosen_pos = None

        for cx, cy in candidates_pos:
            rx1 = max(2, min(w - txt_w - 6, cx))
            ry1 = max(2, min(h - txt_h - baseline - 4, cy - txt_h - 2))
            rx2 = min(w - 2, rx1 + txt_w + 6)
            ry2 = min(h - 2, ry1 + txt_h + baseline + 4)

            rect = (rx1, ry1, rx2, ry2)

            if not any(is_rect_overlapping(rect, placed_r) for placed_r in placed_label_rects):
                chosen_rect = rect
                chosen_pos = (rx1 + 3, ry1 + txt_h + 2)
                break

        if chosen_rect is None:
            if ext_slot_y + txt_h + 10 > h:
                ext_slot_y = 30

            rx1 = max(2, w - txt_w - 10)
            ry1 = ext_slot_y
            rx2 = min(w - 2, rx1 + txt_w + 6)
            ry2 = min(h - 2, ry1 + txt_h + baseline + 4)

            chosen_rect = (rx1, ry1, rx2, ry2)
            chosen_pos = (rx1 + 3, ry1 + txt_h + 2)
            ext_slot_y += txt_h + baseline + 8

            box_cx, box_cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.line(
                canvas,
                (box_cx, box_cy),
                (rx1, (ry1 + ry2) // 2),
                color,
                1,
                cv2.LINE_AA,
            )

        placed_label_rects.append(chosen_rect)

        rx1, ry1, rx2, ry2 = chosen_rect
        sub_roi = canvas[ry1:ry2, rx1:rx2]
        if sub_roi.size > 0:
            dark_bg = np.zeros_like(sub_roi)
            cv2.addWeighted(sub_roi, 0.20, dark_bg, 0.80, 0, sub_roi)

        tx, ty = chosen_pos
        cv2.putText(
            canvas,
            caption,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            caption,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas


def summarize(detections: list[Detection], features: list[dict[str, Any]], image_shape: tuple[int, int] = (600, 800)) -> str:
    if not detections:
        return "결함 후보가 검출되지 않았습니다. 어두운 영역 임계값을 낮추거나 학습된 YOLOv8 모델을 연결해 확인하세요."

    total_count = len(detections)
    max_conf = max((d.confidence for d in detections if d.confidence > 0), default=0.0)
    max_risk = max(explain_detection(d.label).risk_score for d in detections)

    counts_by_type: dict[str, int] = {}
    for det in detections:
        rule = explain_detection(det.label)
        counts_by_type[rule.display_name] = counts_by_type.get(rule.display_name, 0) + 1

    counts_str = ", ".join(f"{name} {cnt}개" for name, cnt in counts_by_type.items())
    max_conf_str = f"{max_conf:.2f}" if max_conf > 0 else "N/A"

    header_lines = [
        "### 검사 결과 요약",
        "",
        f"- **전체 검출 개수**: {total_count}개",
        f"- **결함 종류별 개수**: {counts_str}",
        f"- **최고 신뢰도**: {max_conf_str}",
        f"- **최대 위험도**: {max_risk}점",
    ]

    if all(det.source.startswith("OpenCV candidate") for det in detections):
        header_lines.extend([
            "",
            "> **[알림: OpenCV 보조 후보 검출 모드]** 지정된 YOLOv8 모델 파일을 찾을 수 없어 OpenCV 전처리 기반 보조 후보 검출 모드로 구동되었습니다.",
            "> 이 박스는 최종 AI 판정이 아니라, Black-hat 전처리에서 강조된 어두운 영역 후보입니다.",
        ])
    elif any(det.source == "YOLOv8 review candidate" for det in detections):
        header_lines.extend([
            "",
            "> 얇은 박스의 `review` 표시는 기준 신뢰도보다 낮지만 모델이 포착한 검토 후보입니다. 최종 판정 전에 원본 영상과 함께 확인하세요.",
        ])

    header_text = "\n".join(header_lines)

    img_h, img_w = image_shape[:2]
    defect_blocks: list[str] = []
    for idx, det in enumerate(detections, start=1):
        rule = explain_detection(det.label)
        feat = features[idx - 1]
        confidence_str = f"{det.confidence:.2f}" if det.confidence > 0 else "N/A"

        source_label = (
            "YOLOv8"
            if "YOLO" in det.source
            else ("OpenCV 후보" if "OpenCV" in det.source else det.source)
        )
        if "review" in det.source:
            source_label += " (검토)"

        loc_desc = get_location_description(det.bbox, img_w, img_h)

        checkpoints_list = ""
        if rule.checkpoints:
            checkpoints_list = "\n" + "\n".join(f"  - {cp}" for cp in rule.checkpoints)

        block = f"""### {idx}. {rule.display_name}

- **검출 위치**: {loc_desc}
- **검출 방식**: {source_label}
- **신뢰도**: {confidence_str}
- **위험도**: {rule.risk_score}점
- **권장 조치**: {rule.action}
- **추정 원인**: {rule.likely_cause}
- **판단 근거**: {rule.reason}
- **권장 점검 항목**:{checkpoints_list}
- **특징값**:
  - 원형도: {feat['circularity']:.3f}
  - 종횡비: {feat['aspect_ratio']:.2f}
  - 평균 밝기: {feat['mean_brightness']:.1f}"""
        defect_blocks.append(block.strip())

    body_text = "\n\n---\n\n".join(defect_blocks)
    disclaimer = "\n\n---\n> **[안내]** 본 결과는 AI 검출 모델의 보조 해석이며, 최종 합격/불합격 판정은 비파괴검사(NDT) 자격 검사자의 최종 확인이 필요합니다."
    return f"{header_text}\n\n---\n\n{body_text}{disclaimer}"
