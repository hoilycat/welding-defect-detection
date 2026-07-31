#!/usr/bin/env bash

# 🔥 WeldVision Quality Inspection Dashboard Launcher for macOS / Linux

echo "======================================================================="
echo "               🔥 WeldVision Quality Inspection Dashboard"
echo "======================================================================="
echo ""

# 1. Detect Python executable
PY_CMD=""

if command -v python3 &>/dev/null; then
    PY_CMD="python3"
elif command -v python &>/dev/null; then
    PY_CMD="python"
else
    echo "[오류] Python 실행 환경을 찾을 수 없습니다."
    echo "Python 3.10 이상이 설치되어 있는지 확인해 주세요."
    exit 1
fi

echo "[정보] 감지된 Python 실행 명령어: $PY_CMD"
$PY_CMD -c "import sys; print('[정보] Python 버전:', sys.version)"
echo ""

# 2. Check required dependencies
echo "[정보] 필요 패키지(Gradio, OpenCV, Ultralytics) 상태 확인 중..."
$PY_CMD -c "import cv2, gradio, ultralytics, pandas; print('[성공] 모든 핵심 패키지가 정상적으로 로드되었습니다.')" &>/dev/null

if [ $? -ne 0 ]; then
    echo ""
    echo "[경고] 일부 필요 패키지가 현재 Python 환경에 설치되어 있지 않습니다."
    echo "아래 명령어를 실행하여 필수 패키지를 설치해 주세요:"
    echo ""
    echo "   pip3 install gradio ultralytics opencv-python pandas"
    echo ""
    read -p "계속 진행하시겠습니까? (y/n) " ans
    if [ "$ans" != "y" ]; then
        exit 1
    fi
fi

echo ""
echo "🚀 Gradio 대시보드를 시작합니다..."
echo "   - 브라우저가 열리지 않으면 http://127.0.0.1:7860 에 접속하세요."
echo "   - 종료하려면 Ctrl+C를 누르세요."
echo "======================================================================="
echo ""

$PY_CMD phase2/gradio_app.py
