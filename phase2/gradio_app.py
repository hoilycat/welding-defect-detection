from __future__ import annotations

import sys
from pathlib import Path

LOCAL_PACKAGES = Path(__file__).resolve().parent / ".packages"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "runs"
    / "detect"
    / "rt-pilot-v2"
    / "weights"
    / "best.pt"
)

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


APP_CSS = """
.gradio-container {
    width: 100% !important;
    max-width: 1680px !important;
    padding: 12px !important;
    overflow-x: clip;
}
#dashboard-header h1 {
    margin-bottom: 4px;
}
#dashboard-shell {
    display: grid !important;
    grid-template-columns: minmax(390px, 0.9fr) minmax(0, 1.3fr);
    gap: 12px;
    align-items: start;
}
#dashboard-shell > div {
    min-width: 0 !important;
    width: auto !important;
}
#control-panel {
    position: sticky;
    top: 8px;
    padding: 10px;
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    background: var(--block-background-fill);
    overflow: visible;
}
#model-actions {
    display: block !important;
}
#model-actions > .form {
    align-items: end;
    gap: 8px;
}
#slider-grid > .form > div,
#evidence-grid > div {
    min-width: 0 !important;
    width: auto !important;
    margin: 0 !important;
    justify-self: stretch;
}
#slider-grid {
    display: block !important;
    margin-top: 8px;
}
#slider-grid > .form {
    display: grid !important;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 8px;
}
.compact-control {
    min-width: 0 !important;
    padding: 8px !important;
}
#evidence-grid {
    display: grid !important;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 8px;
}
.result-image img {
    max-height: 240px;
    object-fit: contain;
}
.evidence-image img {
    max-height: 160px;
    object-fit: contain;
}
#analysis-summary {
    max-height: 280px;
    overflow-y: auto;
    padding-right: 8px;
}
#feature-table {
    max-height: 300px;
    overflow: auto;
}
@media (max-width: 820px) {
    .gradio-container {
        padding: 10px !important;
    }
    #dashboard-shell {
        grid-template-columns: 1fr;
    }
    #control-panel {
        position: static;
    }
    .result-image img,
    .evidence-image img {
        max-height: 210px;
    }
}
"""


def analyze_image(
    image,
    model_path,
    confidence_threshold,
    include_review_candidates,
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

    detections = detect_with_yolo(
        image,
        model_path,
        confidence_threshold,
        include_review_candidates=include_review_candidates,
    )
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


with gr.Blocks(
    title="WeldVision Phase 2 | 용접 결함 해석",
    fill_width=True,
) as demo:
    with gr.Column(elem_id="dashboard-header"):
        gr.Markdown("# WeldVision Phase 2 | 용접 결함 검출 및 해석")
        gr.Markdown(
            "왼쪽에서 조건을 조절하면서 오른쪽의 검출·전처리·해석 결과를 한눈에 비교하세요."
        )

    with gr.Row(elem_id="dashboard-shell"):
        with gr.Column(elem_id="control-panel"):
            gr.Markdown("### 입력 및 분석 조건")
            image_input = gr.Image(
                label="용접 이미지 업로드 (Weld image)",
                type="numpy",
                height=180,
            )
            with gr.Row(elem_id="model-actions"):
                model_path = gr.Textbox(
                    label="YOLOv8 모델 경로",
                    value=str(DEFAULT_MODEL_PATH) if DEFAULT_MODEL_PATH.is_file() else "",
                    placeholder="선택 입력: runs/detect/train/weights/best.pt",
                    scale=3,
                )
                run_button = gr.Button("분석 시작", variant="primary", scale=1)

            include_review_candidates = gr.Checkbox(
                value=False,
                label="낮은 신뢰도 검토 후보도 표시 (필요할 때만)",
            )

            with gr.Row(elem_id="slider-grid"):
                confidence_threshold = gr.Slider(
                    0.05,
                    0.95,
                    value=0.15,
                    step=0.05,
                    label="YOLO 신뢰도",
                    elem_classes="compact-control",
                )
                clahe_clip = gr.Slider(
                    1.0,
                    8.0,
                    value=3.0,
                    step=0.2,
                    label="CLAHE 강도",
                    elem_classes="compact-control",
                )
                blackhat_kernel = gr.Slider(
                    3,
                    41,
                    value=15,
                    step=2,
                    label="Black-hat 크기",
                    elem_classes="compact-control",
                )
                dark_threshold = gr.Slider(
                    5,
                    250,
                    value=150,
                    step=5,
                    label="어두운 후보 임계값",
                    elem_classes="compact-control",
                )
                min_candidate_area = gr.Slider(
                    10,
                    3000,
                    value=180,
                    step=10,
                    label="최소 후보 면적",
                    elem_classes="compact-control",
                )
                gradient_kernel = gr.Slider(
                    3,
                    21,
                    value=5,
                    step=2,
                    label="Gradient 부드러움",
                    elem_classes="compact-control",
                )
                emboss_depth = gr.Slider(
                    1.0,
                    6.0,
                    value=4.5,
                    step=0.5,
                    label="Emboss 깊이",
                    elem_classes="compact-control",
                )
                sharpen_amount = gr.Slider(
                    0.0,
                    3.0,
                    value=0.8,
                    step=0.1,
                    label="샤프닝 강도",
                    elem_classes="compact-control",
                )

        with gr.Column():
            gr.Markdown("### 검출 결과")
            with gr.Row():
                original_output = gr.Image(
                    label="원본 이미지", height=240, elem_classes="result-image"
                )
                detection_output = gr.Image(
                    label="검출 결과", height=240, elem_classes="result-image"
                )

            gr.Markdown("### 전처리 근거 화면")
            with gr.Row(elem_id="evidence-grid"):
                clahe_output = gr.Image(
                    label="국소 대비 강화 (CLAHE)", height=160, elem_classes="evidence-image"
                )
                blackhat_output = gr.Image(
                    label="어두운 결함 강조 (Black-hat)",
                    height=160,
                    elem_classes="evidence-image",
                )
                gradient_output = gr.Image(
                    label="방향성/경계 강조 (Gradient)",
                    height=160,
                    elem_classes="evidence-image",
                )
                emboss_output = gr.Image(
                    label="질감 강조 (Emboss)", height=160, elem_classes="evidence-image"
                )

            gr.Markdown("### 분석 결과")
            summary_output = gr.Markdown(elem_id="analysis-summary")
            feature_table = gr.Dataframe(label="특징값 표", elem_id="feature-table")

    analysis_inputs = [
        image_input,
        model_path,
        confidence_threshold,
        include_review_candidates,
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
        include_review_candidates,
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
    demo.launch(css=APP_CSS)
