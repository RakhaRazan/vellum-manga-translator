@echo off
setlocal EnableExtensions

:: ============================================================================
:: Manga / Manhwa / Manhua Comic Translator - Development Launcher
:: Double-click to launch both the FastAPI Backend (port 8000)
:: and the React + Vite Frontend (port 5173).
:: ============================================================================

cd /d "%~dp0"
set "ROOT_DIR=%CD%"
set "NODE_DIR=%USERPROFILE%\.unsloth\node"

:: Prepend local Node.js directory to PATH (done outside parentheses so (x86) in PATH never breaks cmd parser)
if exist "%NODE_DIR%\node.exe" set "PATH=%NODE_DIR%;%PATH%"

echo ============================================================================
echo   MangaFlow AI - Local Manga / Manhwa / Manhua Comic Translator
echo ============================================================================
echo   Workspace : %ROOT_DIR%
echo   Backend   : http://localhost:8000
echo   Frontend  : http://127.0.0.1:5173
echo   LM Studio : http://localhost:1235/v1
echo ============================================================================
echo.

:: Verify Python virtual environment exists
if exist "%ROOT_DIR%\backend\venv\Scripts\python.exe" goto :check_node
echo [ERROR] Python virtual environment not found at backend\venv\Scripts\python.exe
if /i not "%~1"=="--no-pause" pause
exit /b 1

:check_node
where node >nul 2>nul
if not errorlevel 1 goto :start_backend
echo [ERROR] Node.js not found in PATH.
if /i not "%~1"=="--no-pause" pause
exit /b 1

:start_backend
:: 1. Launch FastAPI Backend on Port 8000 if not already listening
netstat -ano | findstr /R /C:":8000 .*LISTENING" >nul 2>nul
if not errorlevel 1 goto :backend_already_running
echo [1/2] Starting FastAPI Backend Server on http://localhost:8000 ...
start "MangaFlow FastAPI Backend (Port 8000)" /D "%ROOT_DIR%" cmd /k ""%ROOT_DIR%\backend\venv\Scripts\python.exe" backend\main.py"
goto :start_frontend

:backend_already_running
echo [1/2] FastAPI Backend Server is already running on port 8000.

:start_frontend
:: 2. Launch Vite Frontend Dev Server on Port 5173 if not already listening
netstat -ano | findstr /R /C:":5173 .*LISTENING" >nul 2>nul
if not errorlevel 1 goto :frontend_already_running
echo [2/2] Starting React + Vite Frontend on http://127.0.0.1:5173 ...
start "MangaFlow Vite Frontend (Port 5173)" /D "%ROOT_DIR%\frontend" cmd /k "npm.cmd run dev"
goto :done_launch

:frontend_already_running
echo [2/2] React + Vite Frontend is already running on port 5173.

:done_launch
echo.
echo [READY] Both Backend (http://localhost:8000) and Frontend (http://127.0.0.1:5173) are active!

if /i "%~1"=="--no-pause" goto :eof_clean
if /i not "%~1"=="--no-browser" (
    timeout /t 2 /nobreak >nul
    start "" "http://127.0.0.1:5173"
)
echo.
echo Press any key to close this launcher window...
pause >nul

:eof_clean
endlocal
exit /b 0
