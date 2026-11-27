@echo off
setlocal EnableExtensions
REM Optional standalone CLI helper: Koharu Local Cache (.safetensors) -> ONNX / GGUF Bridge.
REM Stages exports outside %LOCALAPPDATA% so Microsoft Store Python (WindowsApps)
REM MSIX AppData virtualization never hides files from native C++ Vellum.exe.

set "STAGE_DIR=%~dp0_koharu_stage"
set "TARGET_DIR=%LOCALAPPDATA%\ComicTranslator\models"

set "PYTHON_EXE="
if defined MANGA_PYTHON_EXE (
    if exist "%MANGA_PYTHON_EXE%" set "PYTHON_EXE=%MANGA_PYTHON_EXE%"
)
if not defined PYTHON_EXE (
    for /f "delims=" %%I in ('where python.exe 2^>nul') do (
        if not defined PYTHON_EXE set "PYTHON_EXE=%%I"
    )
)

if not defined PYTHON_EXE (
    echo [ERROR] Python 3 was not found on PATH (and MANGA_PYTHON_EXE is not set).
    echo         This converter is optional and only needed to convert local Koharu .safetensors weights to ONNX.
    echo         Install Python 3 with torch/safetensors/onnx/transformers/timm or download ONNX weights directly in Vellum.
    exit /b 1
)

mkdir "%TARGET_DIR%" 2>nul
mkdir "%STAGE_DIR%" 2>nul

"%PYTHON_EXE%" "%~dp0convert_koharu_to_onnx.py" --output-dir "%STAGE_DIR%" %*
if errorlevel 1 (
    echo [ERROR] Koharu ONNX conversion failed.
    exit /b 1
)

echo.
echo Moving exported models from "%STAGE_DIR%" to "%TARGET_DIR%" ...
robocopy "%STAGE_DIR%" "%TARGET_DIR%" /E /MOVE /NFL /NDL /NJH /NJS
set "RC=%ERRORLEVEL%"
if %RC% GEQ 8 (
    echo [ERROR] robocopy failed with exit code %RC%.
    exit /b %RC%
)

rmdir /S /Q "%STAGE_DIR%" 2>nul
echo [OK] All Koharu ONNX & GGUF models are ready in "%TARGET_DIR%".
exit /b 0
