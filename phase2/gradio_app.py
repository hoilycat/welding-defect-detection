from __future__ import annotations

import sys
from pathlib import Path

LOCAL_PACKAGES = Path(__file__).resolve().parent / ".packages"
if (
    LOCAL_PACKAGES.exists()
    and sys.version_info[:2] == (3, 12)
    and sys.platform == "win32"
):
    sys.path.insert(0, str(LOCAL_PACKAGES))

DEFAULT_RT_MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "runs"
    / "detect"
    / "rt-v4-balanced"
    / "weights"
    / "best.pt"
)
DEFAULT_VT_MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "runs"
    / "detect"
    / "vt-v1-balanced"
    / "weights"
    / "best.pt"
)
MODEL_PATHS = {
    "RT": DEFAULT_RT_MODEL_PATH,
    "VT": DEFAULT_VT_MODEL_PATH,
}
MODEL_CONFIDENCE_THRESHOLDS = {
    "RT": 0.10,
    "VT": 0.25,
}

import pandas as pd
import gradio as gr

from rules import explain_detection
from vision import (
    crop_detection_roi,
    detect_candidates_from_blackhat,
    detect_with_yolo,
    draw_detections,
    extract_features,
    get_location_description,
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
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
#slider-grid > .form > div {
    background: #ffffff !important;
    border: 1px solid #eef0f3 !important;
    border-radius: 8px !important;
    padding: 4px 8px !important;
    box-shadow: none !important;
}
#evidence-grid {
    display: grid !important;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 8px;
}
.result-image:not(.modal) img {
    max-height: 480px;
    object-fit: contain;
}
.evidence-image:not(.modal) img {
    max-height: 200px;
    object-fit: contain;
}
.crop-gallery img {
    min-height: 340px !important;
    max-height: 400px !important;
    object-fit: contain !important;
    background: #111 !important;
    border-radius: 8px !important;
}
#analysis-summary {
    max-height: 280px;
    overflow-y: auto;
    padding-right: 8px;
}
#feature-table {
    max-height: fit-content !important;
    overflow-x: auto;
    font-size: 13px !important;
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
        max-height: 280px;
    }
}
"""


def model_path_for_inspection(inspection_type: str) -> str:
    path = MODEL_PATHS.get(inspection_type)
    if path and path.is_file():
        try:
            project_root = Path(__file__).resolve().parents[1]
            return str(path.relative_to(project_root)).replace("\\", "/")
        except Exception:
            return str(path)
    return str(path) if path else ""


def model_settings_for_inspection(inspection_type: str) -> tuple[str, float]:
    return (
        model_path_for_inspection(inspection_type),
        MODEL_CONFIDENCE_THRESHOLDS.get(inspection_type, 0.25),
    )


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
        user_guide = (
            "⚠️ **분석할 용접 이미지가 업로드되지 않았습니다.**\n\n"
            "좌측의 '용접 이미지 업로드' 영역에 분석할 용접 X-ray 또는 외관(VT) 사진을 드래그하거나 선택하여 업로드해 주세요."
        )
        return empty, empty, empty, empty, empty, empty, [], user_guide, pd.DataFrame()

    views = preprocess_views(
        image,
        clahe_clip=clahe_clip,
        blackhat_kernel=blackhat_kernel,
        gradient_kernel=gradient_kernel,
        emboss_depth=emboss_depth,
        sharpen_amount=sharpen_amount,
        dark_threshold=dark_threshold,
        min_candidate_area=min_candidate_area,
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
    img_h, img_w = image.shape[:2]

    feature_rows = []
    crop_tuples = []
    for idx, det in enumerate(detections, start=1):
        feature = extract_features(image, det)
        rule = explain_detection(det.label)
        loc_desc = get_location_description(det.bbox, img_w, img_h)
        crop_roi = crop_detection_roi(image, det.bbox, margin_pct=0.30)
        
        source_label = "YOLOv8" if "YOLO" in det.source else ("OpenCV 후보" if "OpenCV" in det.source else det.source)
        if "review" in det.source:
            source_label += " (검토)"

        conf_str = f"{det.confidence:.2f}" if det.confidence > 0 else "-"
        crop_tuples.append((crop_roi, f"#{idx}. {rule.display_name} ({conf_str})"))

        feature_rows.append(
            {
                "번호": f"#{idx}",
                "결함명": rule.display_name,
                "검출 방식": source_label,
                "신뢰도": conf_str,
                "위험도": f"{rule.risk_score}점",
                "검출 위치": loc_desc,
                "원형도": f"{feature['circularity']:.3f}",
                "종횡비": f"{feature['aspect_ratio']:.2f}",
                "평균 밝기": f"{feature['mean_brightness']:.1f}",
                "권장 조치": rule.action,
                "circularity": feature["circularity"],
                "aspect_ratio": feature["aspect_ratio"],
                "mean_brightness": feature["mean_brightness"],
            }
        )

    summary = summarize(detections, feature_rows, image.shape)
    table_columns = [
        "번호",
        "결함명",
        "검출 방식",
        "신뢰도",
        "위험도",
        "검출 위치",
        "원형도",
        "종횡비",
        "평균 밝기",
        "권장 조치",
    ]
    
    df_result = pd.DataFrame(feature_rows)
    if not df_result.empty and set(table_columns).issubset(df_result.columns):
        df_result = df_result[table_columns]
    else:
        df_result = pd.DataFrame(columns=table_columns)

    return (
        image,
        annotated,
        views["clahe"],
        views["blackhat"],
        views["gradient"],
        views["emboss"],
        crop_tuples,
        summary,
        df_result,
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
            inspection_type = gr.Radio(
                choices=[("방사선 검사 (RT)", "RT"), ("육안 검사 (VT)", "VT")],
                value="RT",
                label="검사 방식",
            )
            with gr.Row(elem_id="model-actions"):
                model_path = gr.Textbox(
                    label="YOLOv8 모델 경로",
                    value=model_path_for_inspection("RT"),
                    placeholder="선택 입력: runs/detect/train/weights/best.pt",
                    scale=3,
                )
                run_button = gr.Button("분석 시작", variant="primary", scale=1)

            include_review_candidates = gr.Checkbox(
                value=False,
                label="낮은 신뢰도 검토 후보도 표시 (필요할 때만)",
            )

            gr.Markdown(
                "**OpenCV 보조 후보 설정**  \n"
                "<small>YOLO 모델 미사용 또는 후보 표시 옵션 활성화 시 적용되며, YOLO 검출 결과에는 영향을 주지 않습니다.</small>"
            )

            with gr.Row(elem_id="slider-grid"):
                confidence_threshold = gr.Slider(
                    0.05,
                    0.95,
                    value=0.10,
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
                    label="원본 이미지", height=480, elem_classes="result-image"
                )
                detection_output = gr.Image(
                    label="검출 결과", height=480, elem_classes="result-image"
                )

            defect_crops_gallery = gr.Gallery(
                label="🔍 검출 결함 독립 확대 카드 (Cropped Defect Detail Cards)",
                columns=2,
                height=420,
                elem_classes="crop-gallery",
            )

            gr.Markdown("### 전처리 근거 화면")
            with gr.Row(elem_id="evidence-grid"):
                clahe_output = gr.Image(
                    label="국소 대비 강화 (CLAHE)", height=190, elem_classes="evidence-image"
                )
                blackhat_output = gr.Image(
                    label="어두운 결함 & OpenCV 후보 윤곽 (Black-hat & Mask)",
                    height=190,
                    elem_classes="evidence-image",
                )
                gradient_output = gr.Image(
                    label="방향성/경계 강조 (Gradient)",
                    height=190,
                    elem_classes="evidence-image",
                )
                emboss_output = gr.Image(
                    label="질감 강조 (Emboss)", height=190, elem_classes="evidence-image"
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
        defect_crops_gallery,
        summary_output,
        feature_table,
    ]

    run_button.click(
        fn=analyze_image,
        inputs=analysis_inputs,
        outputs=analysis_outputs,
    )

    inspection_type.change(
        fn=model_settings_for_inspection,
        inputs=inspection_type,
        outputs=[model_path, confidence_threshold],
    ).then(
        fn=analyze_image,
        inputs=analysis_inputs,
        outputs=analysis_outputs,
        show_progress="hidden",
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
