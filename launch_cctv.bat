@echo off
setlocal
echo =========================================================
echo   RetailVision CCTV Analytics  ^|  Production Launcher
echo =========================================================
echo.

:: ── Python check ──────────────────────────────────────────────────────────
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Install Python 3.9+ from https://python.org
    pause & exit /b 1
)

:: ── Virtual environment ───────────────────────────────────────────────────
if not exist ".venv\" (
    echo [INFO] Creating virtual environment...
    python -m venv .venv
)

echo [INFO] Activating environment...
call .venv\Scripts\activate.bat

:: ── Dependencies ─────────────────────────────────────────────────────────
echo [INFO] Installing runtime dependencies...
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

:: ── ONNX model export (one-time) ─────────────────────────────────────────
set MODEL=%~1
if "%MODEL%"=="" set MODEL=yolov8n.pt

:: Check if ONNX already exists
set ONNX=%MODEL:.pt=.onnx%
if not exist "%ONNX%" (
    echo [INFO] First run: exporting %MODEL% to ONNX format...
    echo        This will install ultralytics temporarily and only runs ONCE.
    pip install ultralytics --quiet
    python export_model.py --model %MODEL%
    if %errorlevel% neq 0 (
        echo [ERROR] Model export failed. Check that %MODEL% exists.
        pause & exit /b 1
    )
)

:: ── Clear stale bytecode ─────────────────────────────────────────────────
if exist "__pycache__\" rmdir /s /q __pycache__

:: ── Start server ─────────────────────────────────────────────────────────
echo.
echo [SUCCESS] Launching API server...
echo [INFO]    Open http://localhost:8000 in your browser.
echo.
python app.py

pause
endlocal
