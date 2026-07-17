<div align="center">

# 🔥 WeldVision
### C++ OpenCV 기반 용접 결함 특징 추출 및 SVM 분류 실험

![Now](https://img.shields.io/badge/🔴%20현재-Stage%201%20C%2B%2B%20%7C%20OpenCV-red?style=for-the-badge)
![Next](https://img.shields.io/badge/🟠%20다음-Stage%202%20Python%20%7C%20YOLOv8-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/Stage%201-SVM%20Experiment-yellow?style=for-the-badge)

> C++ 고전비전으로 결함의 물리적 특성을 직접 이해하는 것부터 시작해,  
> 현재는 GT 폴리곤 라벨을 활용한 특징 추출과 SVM 4클래스 분류를 구현한 단계입니다.  
> YOLO 검출, 위험도 추론, Gradio 데모는 이후 확장 계획입니다.

</div>

---

## 🗺️ 전체 로드맵

```mermaid
flowchart LR
    subgraph S1["🔴 1단계 — C++ OpenCV + SVM 실험 (현재)"]
        A[📷 X-ray 이미지 + JSON 라벨 입력] --> B[전처리\nGrayscale · CLAHE]
        B --> C[GT 폴리곤 마스크 생성]
        C --> D[특징 추출\n원형도 · 종횡비 · 밝기 · 면적]
        D --> E[SVM 분류기]
        E --> F[✅ 정확도 + Confusion Matrix]
    end

    subgraph S2["🟠 2단계 — 검출/데모 확장 계획"]
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

## 🔴 1단계 — C++ OpenCV + SVM 실험

> **목표:** 용접 X-ray 이미지와 JSON 폴리곤 라벨에서 특징을 추출하고 SVM으로 결함 종류를 분류  
> **현재 범위:** 자동 검출이 아니라, 제공된 GT 폴리곤 라벨을 이용한 특징 기반 분류 실험

### 왜 고전비전인가?
- 산업 현장 검사 시스템 상당수가 룰베이스 OpenCV C++ — 실무 직결
- 결함의 물리적 특성을 직접 수식으로 이해 → 2단계 원인추론의 씨앗

### 처리 파이프라인

```mermaid
flowchart TD
    A["📁 입력 JPG + JSON 라벨"] --> B

    B["① 전처리\nGrayscale → CLAHE"]
    B --> C["② GT 폴리곤 마스크 생성\nJSON annotations 사용"]
    C --> D["③ 특징 추출 ⭐\n원형도 · 종횡비\n밝기 평균/표준편차 · 정규화 면적"]
    D --> E["④ SVM 분류\ncv::ml::SVM"]
    E --> F["⑤ 평가\n정확도 + Confusion Matrix"]

    G["🩻 결함 유형별 특징"]
    G --> G1["기공 Porosity\n→ 원형도 높음 🔴"]
    G --> G2["균열 Crack\n→ 종횡비 큼 🔴"]
    G --> G3["융합불량 Lack of fusion\n→ 형상/밝기 특징"]
    G --> G4["슬래그혼입 Slag inclusion\n→ 형상/밝기 특징"]

    D -. "물리적 의미 매핑" .-> G

    style A fill:#1a1a2e,stroke:#ff4444,color:#fff
    style D fill:#2d1f1f,stroke:#ff8800,color:#fff
    style G fill:#1a1a1a,stroke:#ffcc00,color:#fff
```

### 데이터셋
- 방사선 용접 이미지와 JSON 폴리곤 라벨을 사용합니다.
- 현재 코드의 학습 대상 4클래스: crack / porosity / lack of fusion / slag inclusion
- `normal`/`ND` 클래스는 현재 SVM 학습 코드에 포함되어 있지 않습니다.

### 진행 현황

| Day | 내용 | 상태 |
|-----|------|------|
| 1 | OpenCV C++ 환경설정 + 이미지 출력 | ✅ |
| 2~3 | 한글 경로 처리 + JSON 파싱 + 폴리곤 시각화 | ✅ |
| 4 | 전처리 파이프라인 (grayscale, CLAHE, Canny 시각화) | ✅ |
| 5 | GT 폴리곤 시각화 + 바운딩 박스/형상 실험 | ✅ |
| 6 | 특징 추출 (원형도, 종횡비, 밝기 통계, 정규화 면적) | ✅ |
| 7 | 규칙 기반 분류기 + putText | ✅ |
| 8 | 배치 처리 (컨투어 디버깅 중) | ✅ |
| 9 | GT 폴리곤 시각화 + 멀티뷰 (CLAHE·Canny·GT) | ✅ |
| **10** | **SVM 학습 + 정확도 86.2% (4클래스)** | **👈 여기까지** |
| 11 | Confusion Matrix 분석 + 클래스 불균형 개선 | 일부 구현 / 개선 예정 |
| 12 | README polish + 지원 완료 | 🔜 |

---

## 🟠 2단계 계획 — 검출/해석/데모 확장

> **목표:** 검출 + 위험도 해석 + 대시보드  
> **상태:** YOLOv8 학습/추론은 아직 예정 단계입니다. Gradio 기반 해석 대시보드는 `phase2-gradio-dashboard` 브랜치에서 1차 구현했습니다.

### 위험도 스코어링 아이디어

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

### 2단계 1차 구현 현황 — Gradio 해석 대시보드

> 구현 브랜치: `phase2-gradio-dashboard`
> 현재 main에는 2단계 계획과 구현 현황을 기록하고, 실제 Gradio 대시보드 코드는 별도 브랜치에서 관리 중입니다.

YOLOv8 학습 모델을 연결하기 전에, 먼저 검출 결과를 설명하고 시각화할 수 있는 Gradio 기반 해석 대시보드를 1차 구현했습니다.

현재 포함된 기능:

- 이미지 업로드
- 원본 이미지와 후보 검출 결과 비교
- OpenCV 전처리 근거 화면 제공: CLAHE, Black-hat, Gradient, Emboss
- 슬라이더 기반 자동 재분석
- 특징값 출력: Circularity, Aspect Ratio, Mean Brightness
- 결함별 위험도 점수 출력
- 추정 원인 및 권장 조치 출력
- 한글 UI + 영어 기술명 병기

현재 `YOLOv8 모델 경로`가 비어 있으면 OpenCV 기반 후보 검출 모드가 동작합니다. 이때 표시되는 박스는 최종 AI 판정이 아니라, Black-hat 전처리에서 강조된 어두운 영역을 확인하기 위한 보조 후보입니다.

다음 단계에서는 RIAWELC 데이터를 YOLO 형식으로 변환하고, YOLOv8 학습 후 `best.pt` 모델을 대시보드에 연결할 예정입니다.

---

## 🚀 배포 계획 — HuggingFace Spaces + Gradio

> **목표:** C++ 결과 + YOLOv8 결과를 나란히 보여주는 웹 데모  
> **상태:** 설계 단계입니다. 현재 구현된 배포 코드는 없습니다.

### 왜 HuggingFace Spaces인가?

HuggingFace Spaces는 Gradio 앱 호스팅과 GPU 옵션을 제공합니다.  
이후 C++ 결과 시각화를 Python과 연결하기 위해 C++ → JSON → Gradio 파이프라인을 구성할 예정입니다.

### 아키텍처 구상

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

### C++ → JSON 출력 포맷

현재 코드는 SVM 평가 결과를 `result.json`으로 저장합니다. 아래 포맷은 이후 이미지별 검출 결과와 연동하기 위한 설계 예시입니다.

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

### Gradio UI 설계 예시

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
| 2단계 계획 | Python, YOLOv8 (Ultralytics), Plotly |
| 배포 계획 | Gradio, HuggingFace Spaces, C++→JSON 브리지 |

## 📁 프로젝트 구조

```
welding-defect-detection/
├── src/
│   ├── main.cpp          # 메인 처리 파이프라인
│   └── visualize.py      # Python 라벨 시각화 실험 코드
├── config.json           # 로컬 경로 설정 (현재 저장소에 포함됨)
├── config.json.example   # 경로 설정 예시
├── CMakeLists.txt        # 빌드 설정
├── CMakeLists_win.txt    # 이전 Windows용 CMake 설정
├── run.bat               # Windows 실행 배치파일
└── README.md
```

## ⚙️ 빌드 및 실행

**1. config.json 생성**
```bash
cp config.json.example config.json
# config.json 열어서 본인 경로로 수정
```

> 참고: 현재 저장소에는 `config.json`도 함께 포함되어 있습니다. 다른 환경에서 실행하려면 `data_dir`, `label_dir`를 본인 데이터셋 경로로 수정해야 합니다.

**2. CMake 빌드 (Windows)**
```bash
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=D:/vcpkg/scripts/buildsystems/vcpkg.cmake \
         -DVCPKG_TARGET_TRIPLET=x64-windows \
         -DOpenCV_DIR="C:/Users/{사용자}/Downloads/opencv/build/x64/vc16/lib"
cmake --build . --config Release
```

> 참고: 현재 CMake 설정은 Windows/MSVC 기준 옵션(`/utf-8`)을 포함합니다. macOS/clang에서는 이 옵션 때문에 그대로 빌드되지 않을 수 있습니다.

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
