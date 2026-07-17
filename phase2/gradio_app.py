from __future__ import annotations

import sys
from pathlib import Path

LOCAL_PACKAGES = Path(__file__).resolve().parent / ".packages"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))

import pandas as pd

import gradio as gr

from rules import explain_detection
from vision import (
    detect_candidates_from_blackhat,
    detect_with_yolo,
    draw_detections,
    extract_features,
    preprocess_views,
    summarize,
)


def analyze_image(
    image,
    model_path,
    confidence_threshold,
    clahe_clip,
    blackhat_kernel,
    dark_threshold,
    min_candidate_area,
    gradient_kernel,
    emboss_depth,
    sharpen_amount,
):
    if image is None:
        empty = None
        return empty, empty, empty, empty, empty, empty, "먼저 이미지를 업로드하세요.", pd.DataFrame()

    views = preprocess_views(
        image,
        clahe_clip=clahe_clip,
        blackhat_kernel=blackhat_kernel,
        gradient_kernel=gradient_kernel,
        emboss_depth=emboss_depth,
        sharpen_amount=sharpen_amount,
    )

    detections = detect_with_yolo(image, model_path, confidence_threshold)
    if not detections:
        detections = detect_candidates_from_blackhat(
            views["blackhat"],
            min_area=int(min_candidate_area),
            threshold=int(dark_threshold),
        )

    annotated = draw_detections(image, detections)
    feature_rows = []
    for det in detections:
        feature = extract_features(image, det)
        rule = explain_detection(det.label)
        feature_rows.append(
            {
                "defect": rule.defect_type,
                "결함명 (Defect)": rule.display_name,
                "source": det.source,
                "검출 방식 (Source)": det.source,
                "confidence": round(det.confidence, 3) if det.confidence > 0 else None,
                "신뢰도 (Confidence)": round(det.confidence, 3) if det.confidence > 0 else None,
                "risk": rule.risk_score,
                "위험도 (Risk)": rule.risk_score,
                "circularity": feature["circularity"],
                "원형도 (Circularity)": feature["circularity"],
                "aspect_ratio": feature["aspect_ratio"],
                "종횡비 (Aspect Ratio)": feature["aspect_ratio"],
                "mean_brightness": feature["mean_brightness"],
                "평균 밝기 (Mean Brightness)": feature["mean_brightness"],
                "std_brightness": feature["std_brightness"],
                "area_ratio": feature["area_ratio"],
                "action": rule.action,
                "권장 조치 (Action)": rule.action,
            }
        )

    summary = summarize(detections, feature_rows)
    table_columns = [
        "결함명 (Defect)",
        "검출 방식 (Source)",
        "신뢰도 (Confidence)",
        "위험도 (Risk)",
        "원형도 (Circularity)",
        "종횡비 (Aspect Ratio)",
        "평균 밝기 (Mean Brightness)",
        "권장 조치 (Action)",
    ]
    return (
        image,
        annotated,
        views["clahe"],
        views["blackhat"],
        views["gradient"],
        views["emboss"],
        summary,
        pd.DataFrame(feature_rows, columns=table_columns),
    )


with gr.Blocks(title="WeldVision Phase 2 | 용접 결함 해석") as demo:
    gr.Markdown("# WeldVision Phase 2 | 용접 결함 검출 및 해석")
    gr.Markdown(
        "YOLOv8 검출 + OpenCV 전처리 근거 시각화 + 1단계 특징값 재사용 + 위험도/원인 추론"
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="용접 이미지 업로드 (Weld image)", type="numpy")
            model_path = gr.Textbox(
                label="YOLOv8 모델 경로",
                placeholder="선택 입력: runs/detect/train/weights/best.pt",
            )
            run_button = gr.Button("분석 시작", variant="primary")

        with gr.Column(scale=1):
            confidence_threshold = gr.Slider(
                0.05, 0.95, value=0.25, step=0.05, label="YOLO 신뢰도 임계값"
            )
            clahe_clip = gr.Slider(1.0, 8.0, value=3.0, step=0.2, label="국소 대비 강도 (CLAHE)")
            blackhat_kernel = gr.Slider(
                3, 41, value=15, step=2, label="어두운 결함 강조 크기 (Black-hat kernel)"
            )
            dark_threshold = gr.Slider(
                5, 250, value=150, step=5, label="어두운 후보 임계값 (Dark threshold)"
            )
            min_candidate_area = gr.Slider(
                10, 3000, value=180, step=10, label="최소 후보 면적 (Min candidate area)"
            )
            gradient_kernel = gr.Slider(3, 21, value=5, step=2, label="방향성 강조 부드러움 (Gradient)")
            emboss_depth = gr.Slider(1.0, 6.0, value=4.5, step=0.5, label="엠보싱 깊이 (Emboss)")
            sharpen_amount = gr.Slider(0.0, 3.0, value=0.8, step=0.1, label="샤프닝 강도")

    with gr.Row():
        original_output = gr.Image(label="원본 이미지")
        detection_output = gr.Image(label="검출 결과")

    gr.Markdown("## 전처리 근거 화면")
    with gr.Row():
        clahe_output = gr.Image(label="국소 대비 강화 (CLAHE)")
        blackhat_output = gr.Image(label="어두운 결함 강조 (Black-hat)")
    with gr.Row():
        gradient_output = gr.Image(label="방향성/경계 강조 (Gradient)")
        emboss_output = gr.Image(label="질감 강조 (Emboss)")

    gr.Markdown("## 분석 결과")
    summary_output = gr.Markdown()
    feature_table = gr.Dataframe(label="특징값 표")

    analysis_inputs = [
        image_input,
        model_path,
        confidence_threshold,
        clahe_clip,
        blackhat_kernel,
        dark_threshold,
        min_candidate_area,
        gradient_kernel,
        emboss_depth,
        sharpen_amount,
    ]
    analysis_outputs = [
        original_output,
        detection_output,
        clahe_output,
        blackhat_output,
        gradient_output,
        emboss_output,
        summary_output,
        feature_table,
    ]

    run_button.click(
        fn=analyze_image,
        inputs=analysis_inputs,
        outputs=analysis_outputs,
    )

    for live_component in [
        image_input,
        confidence_threshold,
        clahe_clip,
        blackhat_kernel,
        dark_threshold,
        min_candidate_area,
        gradient_kernel,
        emboss_depth,
        sharpen_amount,
    ]:
        live_component.change(
            fn=analyze_image,
            inputs=analysis_inputs,
            outputs=analysis_outputs,
            show_progress="hidden",
        )


if __name__ == "__main__":
    demo.launch()
