@echo off
setlocal EnableDelayedExpansion

set PY_CMD=

rem Check py -3.12
py -3.12 -c "import sys" >nul 2>&1
if %errorlevel% equ 0 (
    set "PY_CMD=py -3.12"
    goto :PYTHON_FOUND
)

rem Check python
python -c "import sys" >nul 2>&1
if %errorlevel% equ 0 (
    set "PY_CMD=python"
    goto :PYTHON_FOUND
)

rem Check py
py -c "import sys" >nul 2>&1
if %errorlevel% equ 0 (
    set "PY_CMD=py"
    goto :PYTHON_FOUND
)

echo [ERROR] No Python installation found on system.
echo Please install Python 3.12 or ensure Python is available in PATH.
exit /b 1

:PYTHON_FOUND
echo ===================================================
echo Selected Python launcher: %PY_CMD%
for /f "delims=" %%v in ('%PY_CMD% --version 2^>^&1') do echo %%v
echo ===================================================

echo Checking required package imports (gradio, cv2, ultralytics, pandas, rules, vision)...
%PY_CMD% -c "import sys, pathlib; p2=pathlib.Path('phase2').resolve(); sys.path.insert(0, str(p2)); lp=p2/'.packages'; sys.path.insert(0, str(lp)) if lp.exists() and sys.version_info[:2]==(3,12) and sys.platform=='win32' else None; import gradio, cv2, ultralytics, pandas, rules, vision" >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to import one or more required packages.
    echo Please install required dependencies using:
    echo     pip install gradio ultralytics opencv-python pandas
    echo.
    exit /b 1
)

echo All package imports succeeded! Launching Gradio Web App...
echo ===================================================
%PY_CMD% phase2/gradio_app.py
