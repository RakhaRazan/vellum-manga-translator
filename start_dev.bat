@echo off
setlocal EnableExtensions EnableDelayedExpansion

:: ============================================================================
:: Vellum - Pure C++17 + React Hot-Reload Development Launcher
:: Double-click to launch both the native C++17 HTTP Backend (`--headless`,
:: port 8000) and the React + Vite Dev Server (port 5173 with `/api` proxy).
:: ============================================================================

cd /d "%~dp0"
set "ROOT_DIR=%CD%"

:: Discover Node.js / npm on PATH or standard Windows install locations
where npm.cmd >nul 2>&1
if errorlevel 1 (
    if exist "%ProgramFiles%\nodejs\npm.cmd" (
        set "PATH=%ProgramFiles%\nodejs;!PATH!"
    )
)
where npm.cmd >nul 2>&1
if errorlevel 1 (
    if exist "%USERPROFILE%\.unsloth\node\npm.cmd" (
        set "PATH=%USERPROFILE%\.unsloth\node;!PATH!"
    )
)

:: Locate compiled native C++17 backend binary
set "BACKEND_EXE="
set "BACKEND_DIR="
if exist "%ROOT_DIR%\Vellum\Vellum.exe" (
    set "BACKEND_EXE=%ROOT_DIR%\Vellum\Vellum.exe"
    set "BACKEND_DIR=%ROOT_DIR%\Vellum"
) else if exist "%ROOT_DIR%\backend-cpp\build\Release\Vellum.exe" (
    set "BACKEND_EXE=%ROOT_DIR%\backend-cpp\build\Release\Vellum.exe"
    set "BACKEND_DIR=%ROOT_DIR%\backend-cpp\build\Release"
)

echo ============================================================================
echo   Vellum - Native C++17 + React Development Launcher
echo ============================================================================
echo   Workspace   : %ROOT_DIR%
echo   C++ Backend : http://127.0.0.1:8000 (--headless)
echo   React Vite  : http://127.0.0.1:5173
echo   LM Studio   : http://localhost:1234/v1
echo ============================================================================
echo.

if not defined BACKEND_EXE (
    echo [ERROR] Compiled native C++17 backend not found.
    echo         Please build it first by running: packaging\build_cpp_release.bat
    if /i not "%~1"=="--no-pause" pause
    exit /b 1
)

where npm.cmd >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js / npm was not found on PATH.
    if /i not "%~1"=="--no-pause" pause
    exit /b 1
)

:: 1. Launch Native C++17 Backend (`--headless`) on Port 8000 if not already listening
netstat -ano | findstr /R /C:":8000 .*LISTENING" >nul 2>&1
if not errorlevel 1 goto :backend_already_running
echo [1/2] Starting Native C++17 Backend on http://127.0.0.1:8000 ...
start "" /D "%BACKEND_DIR%" "%BACKEND_EXE%" --headless
timeout /t 1 /nobreak >nul
goto :start_frontend

:backend_already_running
echo [1/2] Native C++17 Backend is already running on port 8000.

:start_frontend
:: 2. Launch React + Vite HMR Frontend on Port 5173 if not already listening
netstat -ano | findstr /R /C:":5173 .*LISTENING" >nul 2>&1
if not errorlevel 1 goto :frontend_already_running
echo [2/2] Starting React + Vite Frontend on http://127.0.0.1:5173 ...
start "Vellum Vite Frontend (Port 5173)" /D "%ROOT_DIR%\frontend" cmd /k "npm.cmd run dev"
goto :done_launch

:frontend_already_running
echo [2/2] React + Vite Frontend is already running on port 5173.

:done_launch
echo.
echo [READY] Native C++17 Backend (http://127.0.0.1:8000) and Vite HMR (http://127.0.0.1:5173) are active!
echo         For standalone desktop mode, double-click Vellum.exe in the project root.

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
