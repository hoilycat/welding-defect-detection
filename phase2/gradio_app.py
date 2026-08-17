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
    --wv-bg: #070b10;
    --wv-panel: #0e151d;
    --wv-panel-alt: #111b25;
    --wv-panel-soft: #15212d;
    --wv-line: #243342;
    --wv-line-strong: #385066;
    --wv-text: #edf5fa;
    --wv-muted: #8da0af;
    --wv-accent: #53d5c3;
    --wv-accent-soft: rgba(83, 213, 195, .12);
    --wv-warning: #ff9b52;
    --wv-danger: #ff5e69;
    --wv-input: #091018;
    --wv-image: #03070a;
    --wv-shadow: rgba(0, 0, 0, .34);
    --body-background-fill: var(--wv-bg);
    --background-fill-primary: var(--wv-bg);
    --background-fill-secondary: var(--wv-panel-alt);
    --block-background-fill: var(--wv-panel);
    --block-border-color: var(--wv-line);
    --border-color-primary: var(--wv-line);
    --input-background-fill: var(--wv-input);
    --button-primary-background-fill: #1c897d;
    --button-primary-background-fill-hover: #24a395;
    --button-primary-text-color: #f7fffd;
    --body-text-color: var(--wv-text);
    --body-text-color-subdued: var(--wv-muted);
    width: 100% !important;
    max-width: 1920px !important;
    min-height: 100vh;
    padding: 18px 22px 32px !important;
    overflow-x: clip;
    color: var(--wv-text);
    background: var(--wv-bg) !important;
}
.gradio-container::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    opacity: .22;
    background:
        linear-gradient(90deg, transparent 0 49.9%, rgba(83,213,195,.025) 50%, transparent 50.1%),
        repeating-linear-gradient(0deg, transparent 0 4px, rgba(255,255,255,.018) 5px);
}
#dashboard-header {
    position: relative;
    z-index: 1;
    margin-bottom: 14px;
}
#dashboard-title-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 24px;
    padding: 8px 8px 17px;
    border-bottom: 1px solid var(--wv-line);
}
#dashboard-header h1 {
    margin: 0 0 4px;
    color: var(--wv-text);
    font-size: clamp(28px, 3vw, 42px);
    letter-spacing: -.035em;
    white-space: nowrap;
}
#dashboard-header p {
    margin: 0;
    color: var(--wv-muted);
    font-size: 13px;
}
#dashboard-header .block,
#dashboard-header .prose,
.section-title,
.plain-markdown {
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}
.wv-console-meta {
    display: flex;
    flex: none;
    gap: 8px;
    align-items: center;
}
.wv-console-meta span {
    padding: 7px 10px;
    border: 1px solid var(--wv-line);
    border-radius: 6px;
    background: var(--wv-panel-alt);
    color: var(--wv-muted);
    font: 700 11px/1.2 ui-monospace, SFMono-Regular, Menlo, monospace;
    letter-spacing: .08em;
}
.wv-console-meta .status-ready {
    color: var(--wv-accent);
    border-color: rgba(83,213,195,.5);
    background: var(--wv-accent-soft);
}
#dashboard-shell {
    position: relative;
    z-index: 1;
    display: grid !important;
    grid-template-columns: minmax(285px, 310px) minmax(720px, 1fr) minmax(400px, 440px);
    gap: 14px;
    align-items: start;
}
#dashboard-shell > div {
    min-width: 0 !important;
    width: auto !important;
}
#control-panel,
#decision-panel {
    position: sticky;
    top: 10px;
    max-height: calc(100vh - 30px);
    overflow-y: auto;
}
#control-panel,
#viewer-panel,
#decision-panel {
    padding: 14px;
    border: 1px solid var(--wv-line);
    border-radius: 10px;
    background: var(--wv-panel);
    box-shadow: 0 14px 34px var(--wv-shadow);
}
#viewer-panel {
    background: #0a1118;
}
#decision-panel {
    background: var(--wv-panel-alt);
}
.section-title h3 {
    margin: 0 0 8px !important;
    color: var(--wv-text);
    font: 800 13px/1.3 ui-monospace, SFMono-Regular, Menlo, monospace;
    letter-spacing: .06em;
    text-transform: uppercase;
}
.section-title h3::before {
    content: "//";
    margin-right: 8px;
    color: var(--wv-accent);
}
#quick-controls {
    gap: 8px;
    align-items: end;
}
#quick-controls > div {
    min-width: 0 !important;
}
#tuning-grid > .form > div,
#evidence-grid > .form > div {
    min-width: 0 !important;
    width: auto !important;
    margin: 0 !important;
    justify-self: stretch;
}
#tuning-grid {
    display: grid !important;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 8px;
    margin-top: 4px;
}
#evidence-grid {
    display: grid !important;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 8px;
    margin-top: 4px;
}
#tuning-grid > .form,
#evidence-grid > .form {
    display: contents !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
#tuning-grid > .form > div {
    background: var(--wv-panel-alt) !important;
    border: 1px solid var(--wv-line) !important;
    border-radius: 7px !important;
    padding: 5px 7px !important;
    box-shadow: none !important;
}
button.primary {
    min-height: 44px;
    border: 1px solid var(--wv-accent) !important;
    border-radius: 7px !important;
    background: linear-gradient(135deg, #17766d, #25a394) !important;
    box-shadow: 0 8px 22px rgba(36,163,149,.2) !important;
    font-weight: 800 !important;
}
.block, .panel {
    border-radius: 7px !important;
}
.block {
    border-color: var(--wv-line) !important;
    background: var(--wv-panel-alt) !important;
}
input, textarea {
    color: var(--wv-text) !important;
    background: var(--wv-input) !important;
}
.primary-view:not(.modal) img {
    max-height: 750px;
    object-fit: contain;
    background: var(--wv-image);
}
.evidence-image:not(.modal) img {
    max-height: 175px;
    object-fit: contain;
    background: var(--wv-image);
}
.crop-gallery img {
    min-height: 120px !important;
    max-height: 150px !important;
    object-fit: contain !important;
    background: var(--wv-image) !important;
    border-radius: 5px !important;
}
#analysis-summary {
    min-height: 175px;
    max-height: 235px;
    overflow-y: auto;
    padding: 13px;
    border: 1px solid var(--wv-line-strong);
    border-left: 3px solid var(--wv-accent);
    border-radius: 6px;
    background: #0b131b;
    color: var(--wv-text);
}
#analysis-summary h3,
#analysis-summary h4 {
    color: var(--wv-accent);
}
#detail-panel {
    margin-top: 12px;
}
#feature-table {
    max-height: 340px !important;
    overflow-x: auto;
    font-size: 12px !important;
}
@media (max-width: 1400px) {
    #dashboard-shell {
        grid-template-columns: minmax(285px, 310px) minmax(0, 1fr);
    }
    #decision-panel {
        grid-column: 2;
        position: static;
        max-height: none;
    }
}
@media (max-width: 900px) {
    .gradio-container {
        padding: 10px !important;
    }
    #dashboard-title-row {
        align-items: flex-start;
        flex-direction: column;
    }
    #dashboard-header h1 {
        white-space: normal;
    }
    #dashboard-shell {
        grid-template-columns: 1fr;
    }
    #control-panel,
    #decision-panel {
        position: static;
        max-height: none;
    }
    #decision-panel { grid-column: auto; }
    .primary-view img,
    .evidence-image img {
        max-height: 360px;
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
        return (
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            [],
            user_guide,
            pd.DataFrame(),
        )

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
        views["sharpened"],
        crop_tuples,
        summary,
        df_result,
    )


with gr.Blocks(
    title="WeldVision Phase 2 | 용접 결함 해석",
    fill_width=True,
) as demo:
    with gr.Column(elem_id="dashboard-header"):
        with gr.Row(elem_id="dashboard-title-row"):
            with gr.Column():
                gr.Markdown("# WeldVision · Inspection Console")
                gr.Markdown(
                    "RT·VT 용접 결함을 검출하고 판정 근거와 조치 정보를 한 화면에서 검토합니다."
                )
            gr.HTML(
                """
                <div class="wv-console-meta" aria-label="검사 시스템 상태">
                  <span>YOLOv8</span>
                  <span>RT / VT</span>
                  <span class="status-ready">● SYSTEM READY</span>
                </div>
                """
            )

    with gr.Row(elem_id="dashboard-shell"):
        with gr.Column(elem_id="control-panel"):
            gr.Markdown("### Inspection Input", elem_classes="section-title")
            image_input = gr.Image(
                label="검사 이미지 업로드",
                type="numpy",
                height=170,
            )
            inspection_type = gr.Radio(
                choices=[("방사선 검사 (RT)", "RT"), ("육안 검사 (VT)", "VT")],
                value="RT",
                label="검사 방식",
            )

            with gr.Row(elem_id="quick-controls"):
                confidence_threshold = gr.Slider(
                    0.05,
                    0.95,
                    value=0.10,
                    step=0.05,
                    label="YOLO 신뢰도",
                    scale=2,
                )
                run_button = gr.Button("검사 실행", variant="primary", scale=1)

            with gr.Accordion("실시간 전처리 조정", open=False):
                gr.Markdown(
                    "<small>슬라이더를 조절하면 비교 화면이 즉시 갱신됩니다.</small>",
                    elem_classes="plain-markdown",
                )
                with gr.Row(elem_id="tuning-grid"):
                    clahe_clip = gr.Slider(
                        1.0, 8.0, value=3.0, step=0.2, label="CLAHE 강도"
                    )
                    blackhat_kernel = gr.Slider(
                        3, 41, value=15, step=2, label="Black-hat 크기"
                    )
                    dark_threshold = gr.Slider(
                        5, 250, value=150, step=5, label="어두운 후보 임계값"
                    )
                    min_candidate_area = gr.Slider(
                        10, 3000, value=180, step=10, label="최소 후보 면적"
                    )
                    gradient_kernel = gr.Slider(
                        3, 21, value=5, step=2, label="Gradient 부드러움"
                    )
                    emboss_depth = gr.Slider(
                        1.0, 6.0, value=4.5, step=0.5, label="Emboss 깊이"
                    )
                    sharpen_amount = gr.Slider(
                        0.0, 3.0, value=0.8, step=0.1, label="샤프닝 강도"
                    )

            with gr.Accordion("고급 분석 설정", open=False):
                model_path = gr.Textbox(
                    label="YOLOv8 모델 경로",
                    value=model_path_for_inspection("RT"),
                    placeholder="선택 입력: runs/detect/train/weights/best.pt",
                )
                include_review_candidates = gr.Checkbox(
                    value=False,
                    label="낮은 신뢰도 검토 후보도 표시",
                )

        with gr.Column(elem_id="viewer-panel"):
            gr.Markdown("### Live Inspection", elem_classes="section-title")
            detection_output = gr.Image(
                label="AI 검출 오버레이",
                height=760,
                elem_classes="primary-view",
            )

            with gr.Accordion("상세 측정값 및 권장 조치", open=False, elem_id="detail-panel"):
                feature_table = gr.Dataframe(
                    label="결함 특징값",
                    elem_id="feature-table",
                )

        with gr.Column(elem_id="decision-panel"):
            gr.Markdown("### Decision", elem_classes="section-title")
            summary_output = gr.Markdown(elem_id="analysis-summary")
            defect_crops_gallery = gr.Gallery(
                label="검출 영역 확대",
                columns=2,
                height=185,
                elem_classes="crop-gallery",
            )

            gr.Markdown("### Evidence Matrix", elem_classes="section-title")
            gr.Markdown(
                "<small>비교 이미지를 클릭하면 원본 크기로 확대됩니다.</small>",
                elem_classes="plain-markdown",
            )
            with gr.Row(elem_id="evidence-grid"):
                original_output = gr.Image(
                    label="원본",
                    height=180,
                    elem_classes="evidence-image",
                )
                clahe_output = gr.Image(
                    label="CLAHE",
                    height=180,
                    elem_classes="evidence-image",
                )
                blackhat_output = gr.Image(
                    label="Black-hat",
                    height=180,
                    elem_classes="evidence-image",
                )
                gradient_output = gr.Image(
                    label="Gradient",
                    height=180,
                    elem_classes="evidence-image",
                )
                emboss_output = gr.Image(
                    label="Emboss",
                    height=180,
                    elem_classes="evidence-image",
                )
                sharpen_output = gr.Image(
                    label="Sharpen",
                    height=180,
                    elem_classes="evidence-image",
                )

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
        sharpen_output,
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
