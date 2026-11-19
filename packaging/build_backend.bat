@echo off
setlocal EnableExtensions

cd /d "%~dp0\.."
set "ROOT_DIR=%CD%"

if not exist "%ROOT_DIR%\backend\venv\Scripts\python.exe" (
    echo [ERROR] Python virtual environment not found at backend\venv\Scripts\python.exe
    exit /b 1
)

"%ROOT_DIR%\backend\venv\Scripts\python.exe" "%ROOT_DIR%\packaging\build_backend.py"
if errorlevel 1 exit /b 1

endlocal
exit /b 0
