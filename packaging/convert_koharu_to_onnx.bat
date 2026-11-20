@echo off
setlocal
REM One-time Koharu Local Cache -> ONNX / GGUF Bridge wrapper for Windows.
REM Stages exports outside %LOCALAPPDATA% so Microsoft Store Python (WindowsApps)
REM MSIX AppData virtualization never hides files from native C++ MangaFlowAI.exe.

set "REPO_ROOT=%~dp0.."
set "PYTHON_EXE=%REPO_ROOT%\backend\venv\Scripts\python.exe"
set "STAGE_DIR=%~dp0_koharu_stage"
set "TARGET_DIR=%LOCALAPPDATA%\ComicTranslator\models"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python virtualenv not found at "%PYTHON_EXE%".
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
