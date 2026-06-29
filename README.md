<div align="center">

# 🔥 WeldVision
### AI 기반 용접 결함 자동 검출 및 품질 분석 시스템

![Now](https://img.shields.io/badge/🔴%20현재-Stage%201%20C%2B%2B%20%7C%20OpenCV-red?style=for-the-badge)
![Next](https://img.shields.io/badge/🟠%20다음-Stage%202%20Python%20%7C%20YOLOv8-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/Day%209%20%2F%2010-In%20Progress-yellow?style=for-the-badge)

> C++ 고전비전으로 결함의 물리적 특성을 직접 이해하는 것부터 시작해,  
> 특징 분석 → YOLO 검출 → 위험도 추론까지 단계적으로 쌓아가는 프로젝트입니다.

</div>

---

## 🗺️ 전체 로드맵

```mermaid
flowchart LR
    subgraph S1["🔴 1단계 — C++ 고전비전 MVP (지금 ~ 2주)"]
        A[📷 X-ray 이미지 입력] --> B[전처리\nCLAHE · Blur]
        B --> C[결함 분리\nOtsu · Morphology]
        C --> D[특징 추출\n원형도 · 종횡비 · 밝기]
        D --> E[SVM 분류기]
        E --> F[✅ 정확도 + Confusion Matrix]
    end

    subgraph S2["🟠 2단계 — WeldVision AI 풀 시스템 (7/17 이후)"]
        G[📷 이미지 업로드] --> H[YOLOv8 검출\n결함 위치 · 종류]
        H --> I[1단계 특징분석 재활용\n원형도 · 종횡비 · 밝기]
        I --> J[위험도 스코어링\n+ 원인추론 룰]
        J --> K[🖥️ Gradio 데모\nHuggingFace Spaces]
    end

    D -- "특징 설계 자산 그대로 이어받음" --> I
    F -- "도메인 지식 축적" --> H

    style S1 fill:#2d1f1f,stroke:#ff4444,color:#fff
    style S2 fill:#2d1f0f,stroke:#ff8800,color:#fff
```

---

## 🔴 1단계 — C++ 고전비전 MVP

> **목표:** 용접 X-ray 이미지에서 결함을 이미지 처리 + SVM으로 자동 분류  
> **핵심:** 딥러닝 없이, CPU만으로, 전부 C++

### 왜 고전비전인가?
- 산업 현장 검사 시스템 상당수가 룰베이스 OpenCV C++ — 실무 직결
- 결함의 물리적 특성을 직접 수식으로 이해 → 2단계 원인추론의 씨앗

### 처리 파이프라인

```mermaid
flowchart TD
    A["📁 입력 PNG\n(224×224, 8bit)"] --> B

    B["① 전처리\nGrayscale → CLAHE → Median Blur"]
    B --> C["② 결함 분리\nOtsu Threshold → Morphology Open/Close"]
    C --> D["③ 특징 추출 ⭐\n면적 · 둘레 · 원형도 · 종횡비\n밝기 평균/표준편차 · blob 개수"]
    D --> E["④ SVM 분류\ncv::ml::SVM"]
    E --> F["⑤ 평가\n정확도 + Confusion Matrix"]

    G["🩻 결함 유형별 특징"]
    G --> G1["기공 Porosity\n→ 원형도 높음 🔴"]
    G --> G2["균열 Crack\n→ 종횡비 큼 🔴"]
    G --> G3["용입불량 LP\n→ 어두운 직선 🔴"]
    G --> G4["정상 ND\n→ 분산 낮음 ✅"]

    D -. "물리적 의미 매핑" .-> G

    style A fill:#1a1a2e,stroke:#ff4444,color:#fff
    style D fill:#2d1f1f,stroke:#ff8800,color:#fff
    style G fill:#1a1a1a,stroke:#ffcc00,color:#fff
```

### 데이터셋
- **RIAWELC** — 224×224 방사선 용접 이미지 24,407장, 4클래스
- LP(용입불량) / PO(기공) / CR(균열) / ND(정상)

### 진행 현황

| Day | 내용 | 상태 |
|-----|------|------|
| 1 | OpenCV C++ 환경설정 + 이미지 출력 | ✅ |
| 2~3 | 한글 경로 처리 + JSON 파싱 + 폴리곤 시각화 | ✅ |
| 4 | 전처리 파이프라인 (grayscale, blur, canny edge) | ✅ |
| 5 | 컨투어 검출 + 바운딩 박스 | ✅ |
| 6 | 특징 추출 (면적, 둘레, 가로세로비) | ✅ |
| 7 | 규칙 기반 분류기 + putText | ✅ |
| 8 | 배치 처리 (컨투어 디버깅 중) | ✅ |
| **9** | **GT 폴리곤 시각화 + 멀티뷰 (CLAHE·Canny·GT)** | **👈 여기까지** |
| 10 | SVM 학습 + 정확도 측정 | 🔜 |
| 11 | Confusion Matrix + 결과 정리 | 🔜 |
| 12 | README polish + 지원 완료 | 🔜 |

---

## 🟠 2단계 — WeldVision AI 풀 시스템

> **목표:** 검출 + 위험도 해석 + 대시보드  
> **핵심:** 1단계를 버리지 않고 두뇌로 재활용

### 위험도 스코어링

| 결함 종류 | 위험도 | 권장 조치 | 1단계 특징 연결 | 주요 원인 |
|-----------|--------|-----------|----------------|-----------|
| 균열 Crack | 🔴 100 | 즉시 재작업 | 종횡비 큼 | 냉각 속도 너무 빠름 |
| 용입불량 LP | 🔴 80 | 재검사 | 어두운 직선 띠 | 전류 너무 낮음 |
| 언더컷 | 🟠 60 | 보수 용접 | 가장자리 형상 | 전류 너무 높음 · 속도 빠름 |
| 기공 Porosity | 🟡 50 | 주의 관찰 | 원형도 높음 | 습기 · 가스 혼입 |
| 슬래그혼입 | 🟡 40 | 경미한 결함 | 밝기·텍스처 이상 | 이전 층 청소 미흡 |

### 2단계 흐름

```mermaid
flowchart LR
    A[📷 이미지 업로드] --> B["YOLOv8 검출\n'여기 결함, 종류=○○'"]
    B --> C["1단계 특징분석\n원형도 · 종횡비 · 밝기"]
    C --> D["위험도 스코어링\n+ 원인추론 룰베이스"]
    D --> E["🖥️ Gradio 데모\nHuggingFace Spaces"]

    style B fill:#2d1f0f,stroke:#ff8800,color:#fff
    style C fill:#2d1f1f,stroke:#ff4444,color:#fff
    style E fill:#0f2d1f,stroke:#44ff88,color:#fff
```

---

## 🚀 배포 계획 — HuggingFace Spaces + Gradio

> **목표:** C++ 결과 + YOLOv8 결과를 나란히 보여주는 웹 데모  
> **핵심:** C++을 버리지 않고, JSON 브리지로 Python과 연결

### 왜 HuggingFace Spaces인가?

HuggingFace Spaces는 Gradio 앱을 무료로 24/7 호스팅 — GPU 옵션도 제공합니다.  
C++ 결과 시각화를 Python과 연결하기 위해 C++ → JSON → Gradio 파이프라인으로 구성합니다.

### 아키텍처

```mermaid
flowchart TD
    subgraph LOCAL["💻 로컬 C++ (1단계 결과)"]
        A["📷 X-ray 이미지"] --> B["C++ 파이프라인\nCLAHE · 특징추출 · SVM"]
        B --> C["📄 result.json\n{defect_type, bbox, score, feature}"]
    end

    subgraph HF["☁️ HuggingFace Spaces (배포)"]
        D["📤 이미지 업로드\n(Gradio UI)"] --> E["YOLOv8 추론\nPython · Ultralytics"]
        E --> F["결과 병합\nC++ JSON + YOLO 결과"]
        C --> F
        F --> G["🖥️ Gradio 출력\n좌: C++ 분석 / 우: YOLO 검출"]
        G --> H["📊 위험도 스코어\n+ 원인 추론 텍스트"]
    end

    style LOCAL fill:#2d1f1f,stroke:#ff4444,color:#fff
    style HF fill:#1f1f2d,stroke:#8800ff,color:#fff
    style G fill:#0f2d1f,stroke:#44ff88,color:#fff
```

### C++ → JSON 출력 포맷 (설계 예정)

```json
{
  "filename": "KakaoTalk_Image_2025.jpg",
  "defects": [
    {
      "type": "crack",
      "bbox": [120, 340, 200, 380],
      "circularity": 0.21,
      "aspect_ratio": 4.5,
      "mean_brightness": 89.3,
      "svm_score": 0.91
    }
  ],
  "stage": "cpp_classical_vision"
}
```

### Gradio UI 설계

```python
import gradio as gr

with gr.Blocks(title="WeldVision", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔥 WeldVision — 용접 결함 검출 데모")
    gr.Markdown("X-ray 이미지를 업로드하면 C++ 분석 결과 + YOLOv8 검출 결과를 비교합니다.")

    with gr.Row():
        with gr.Column():
            img_input = gr.Image(label="📷 X-ray 이미지 업로드", type="filepath")
            run_btn = gr.Button("🔍 검출 시작", variant="primary")

        with gr.Column():
            cpp_output  = gr.Image(label="🔴 C++ 분석 (GT 폴리곤 + 특징)")
            yolo_output = gr.Image(label="🟠 YOLOv8 검출")

    with gr.Row():
        result_text = gr.Textbox(label="📊 결함 요약 + 위험도", lines=4)

    run_btn.click(fn=predict, inputs=img_input,
                  outputs=[cpp_output, yolo_output, result_text])
```

| 영역 | 내용 |
|------|------|
| 좌상단 | 이미지 업로드 + 검출 버튼 |
| 우상단 | C++ 분석 결과 / YOLOv8 검출 결과 나란히 |
| 하단 | 결함 요약 + 위험도 스코어 텍스트 |
| 추가 | 결함 원인 진단 ("이 결함은 이런 원인으로 생겼을 가능성이 높습니다") |

### 배포 단계

| 단계 | 내용 |
|------|------|
| ① | C++ SVM 완성 → JSON 출력 기능 추가 |
| ② | YOLOv8 파인튜닝 (용접 데이터셋 학습) |
| ③ | Gradio 앱 작성 (JSON 읽기 + YOLO 추론 + 시각화) |
| ④ | HuggingFace Spaces `gradio` SDK로 배포 |
| ⑤ | C++ 결과 / YOLO 결과 나란히 비교 데모 완성 |

---

## 🛠️ 기술 스택

| 단계 | 기술 |
|------|------|
| 1단계 | C++17, OpenCV, CMake, vcpkg, nlohmann_json |
| 2단계 | Python, YOLOv8 (Ultralytics), Plotly |
| 배포 | Gradio, HuggingFace Spaces, C++→JSON 브리지 |

## 📁 프로젝트 구조

```
welding-defect-detection/
├── src/
│   └── main.cpp          # 메인 처리 파이프라인
├── build/                # CMake 빌드 결과물
├── config.json           # 로컬 경로 설정 (git 제외)
├── config.json.example   # 경로 설정 예시
├── CMakeLists.txt        # 빌드 설정
├── run.bat               # Windows 실행 배치파일
└── README.md
```

## ⚙️ 빌드 및 실행

**1. config.json 생성**
```bash
cp config.json.example config.json
# config.json 열어서 본인 경로로 수정
```

**2. CMake 빌드 (Windows)**
```bash
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=D:/vcpkg/scripts/buildsystems/vcpkg.cmake \
         -DVCPKG_TARGET_TRIPLET=x64-windows \
         -DOpenCV_DIR="C:/Users/{사용자}/Downloads/opencv/build/x64/vc16/lib"
cmake --build . --config Release
```

**3. 실행**
```bash
# run.bat 더블클릭 또는
$env:PATH += ";C:/Users/{사용자}/Downloads/opencv/build/x64/vc16/bin"
./build/Release/main.exe
```

---

<div align="center">
  <sub>🔥 불똥처럼 — 작은 불꽃에서 시작해서 크게 번진다</sub>
</div>
