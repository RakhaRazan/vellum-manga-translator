@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================================
REM Vellum - Standalone C++17 Desktop Release Builder (Phase 11.1)
REM
REM 1. Builds production React v2 UI (`frontend/dist`) via `npm run build`
REM 2. Configures & compiles `backend-cpp` in Release mode (MSVC /O2 /GL /LTCG)
REM 3. Stages `Vellum.exe`, ONNX Runtime DLLs, and `frontend/dist` into
REM    `Manga/Vellum_Runtime/` and installs the root one-click launcher
REM    `Manga/Vellum.exe`
REM 4. Packages `release/Vellum-Windows-x64.zip` for GitHub Releases
REM ============================================================================

pushd "%~dp0.." >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Failed to resolve repository root from "%~dp0..".
    exit /b 1
)
set "ROOT_DIR=%CD%"
popd >nul 2>&1

set "FRONTEND_DIR=%ROOT_DIR%\frontend"
set "FRONTEND_DIST=%FRONTEND_DIR%\dist"
set "CPP_DIR=%ROOT_DIR%\backend-cpp"
set "CPP_RELEASE_BIN=%CPP_DIR%\build\Release"
set "STAGE_DIR=%ROOT_DIR%\Vellum_Runtime"
set "ROOT_LAUNCHER=%ROOT_DIR%\Vellum.exe"
set "RELEASE_DIR=%ROOT_DIR%\release"
set "ZIP_PATH=%RELEASE_DIR%\Vellum-Windows-x64.zip"

echo ============================================================================
echo   Vellum - Standalone C++17 Desktop Release Builder
echo ============================================================================
echo   Project Root   : %ROOT_DIR%
echo   C++ Backend    : %CPP_DIR%
echo   Bundle Folder  : %STAGE_DIR%
echo   Root Launcher  : %ROOT_LAUNCHER%
echo   Release Zip    : %ZIP_PATH%
echo ============================================================================
echo.

REM ----------------------------------------------------------------------------
REM Tool Discovery: Node.js / npm
REM ----------------------------------------------------------------------------
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
where npm.cmd >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js / npm was not found on PATH.
    echo         Please install Node.js or add it to PATH and retry.
    exit /b 1
)

REM ----------------------------------------------------------------------------
REM Tool Discovery: CMake & Visual Studio Build Tools
REM ----------------------------------------------------------------------------
where cmake.exe >nul 2>&1
if errorlevel 1 (
    set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
    set "VS_INSTALL="
    if exist "!VSWHERE!" (
        for /f "usebackq delims=" %%I in (`"!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2^>nul`) do (
            set "VS_INSTALL=%%I"
        )
        if not defined VS_INSTALL (
            for /f "usebackq delims=" %%I in (`"!VSWHERE!" -latest -products * -property installationPath 2^>nul`) do (
                set "VS_INSTALL=%%I"
            )
        )
    )
    if defined VS_INSTALL (
        set "VS_CMAKE_BIN=!VS_INSTALL!\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"
        if exist "!VS_CMAKE_BIN!\cmake.exe" (
            set "PATH=!VS_CMAKE_BIN!;!PATH!"
        )
    )
)
where cmake.exe >nul 2>&1
if errorlevel 1 (
    if exist "%ProgramFiles%\CMake\bin\cmake.exe" (
        set "PATH=%ProgramFiles%\CMake\bin;!PATH!"
    )
)
where cmake.exe >nul 2>&1
if errorlevel 1 (
    echo [ERROR] cmake.exe was not found on PATH or inside Visual Studio Build Tools.
    echo         Please install CMake or Visual Studio C++ CMake Tools and retry.
    exit /b 1
)

REM ----------------------------------------------------------------------------
REM Step 1/4: Build React Frontend (`frontend/dist`)
REM ----------------------------------------------------------------------------
echo [1/4] Building production React frontend (`npm run build`) ...
pushd "%FRONTEND_DIR%"
call npm run build
set "STEP_ERR=!ERRORLEVEL!"
popd
if not "!STEP_ERR!"=="0" (
    echo [ERROR] Frontend build failed with exit code !STEP_ERR!.
    exit /b !STEP_ERR!
)
if not exist "%FRONTEND_DIST%\index.html" (
    echo [ERROR] Expected compiled frontend entry point not found: "%FRONTEND_DIST%\index.html"
    exit /b 1
)
echo       Frontend dist ready : %FRONTEND_DIST%
echo.

REM ----------------------------------------------------------------------------
REM Step 2/4: Configure & Compile `backend-cpp` in Release mode (/O2 /GL /LTCG)
REM ----------------------------------------------------------------------------
if not defined VELLUM_ORT_BACKEND (
    set "VELLUM_ORT_BACKEND=AUTO"
)
echo [2/4] Configuring and compiling backend-cpp (Release, MSVC /O2 /GL /LTCG, ORT=%VELLUM_ORT_BACKEND%) ...
pushd "%CPP_DIR%"
if exist "build\CMakeCache.txt" (
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DMANGAFLOW_ORT_BACKEND=%VELLUM_ORT_BACKEND%
    if errorlevel 1 (
        echo [INFO] Stale CMakeCache detected. Re-configuring cleanly with -A x64 ...
        del /f /q "build\CMakeCache.txt" 2>nul
        cmake -B build -A x64 -DCMAKE_BUILD_TYPE=Release -DMANGAFLOW_ORT_BACKEND=%VELLUM_ORT_BACKEND%
    )
) else (
    cmake -B build -A x64 -DCMAKE_BUILD_TYPE=Release -DMANGAFLOW_ORT_BACKEND=%VELLUM_ORT_BACKEND%
)
set "STEP_ERR=!ERRORLEVEL!"
if not "!STEP_ERR!"=="0" (
    popd
    echo [ERROR] CMake configure failed with exit code !STEP_ERR!.
    exit /b !STEP_ERR!
)

cmake --build build --config Release --parallel
set "STEP_ERR=!ERRORLEVEL!"
popd
if not "!STEP_ERR!"=="0" (
    echo [ERROR] CMake Release build failed with exit code !STEP_ERR!.
    exit /b !STEP_ERR!
)

if not exist "%CPP_RELEASE_BIN%\Vellum.exe" (
    echo [ERROR] Expected native binary not found: "%CPP_RELEASE_BIN%\Vellum.exe"
    exit /b 1
)
if not exist "%CPP_RELEASE_BIN%\Vellum-Launcher.exe" (
    echo [ERROR] Expected native launcher not found: "%CPP_RELEASE_BIN%\Vellum-Launcher.exe"
    exit /b 1
)
if not exist "%CPP_RELEASE_BIN%\onnxruntime.dll" (
    echo [ERROR] Expected ONNX Runtime DLL not found: "%CPP_RELEASE_BIN%\onnxruntime.dll"
    exit /b 1
)
echo       Compiled C++17 Release binaries ready : %CPP_RELEASE_BIN%
echo.

REM ----------------------------------------------------------------------------
REM Step 3/4: Stage `Manga/Vellum_Runtime/` and Root Launcher `Manga/Vellum.exe`
REM ----------------------------------------------------------------------------
echo [3/4] Staging standalone bundle (`Vellum_Runtime/`) and root launcher (`Vellum.exe`) ...
if exist "%STAGE_DIR%" (
    rmdir /s /q "%STAGE_DIR%" >nul 2>&1
    if exist "%STAGE_DIR%" (
        echo [ERROR] Could not clean "%STAGE_DIR%". Please close any running Vellum.exe and retry.
        exit /b 1
    )
)

mkdir "%STAGE_DIR%"
if errorlevel 1 (
    echo [ERROR] Failed to create staging directory "%STAGE_DIR%".
    exit /b 1
)
mkdir "%STAGE_DIR%\frontend\dist"
if errorlevel 1 (
    echo [ERROR] Failed to create frontend staging directory "%STAGE_DIR%\frontend\dist".
    exit /b 1
)

copy /y "%CPP_RELEASE_BIN%\Vellum.exe" "%STAGE_DIR%\Vellum.exe" >nul
if errorlevel 1 (
    echo [ERROR] Failed to copy Vellum.exe into "%STAGE_DIR%".
    exit /b 1
)

copy /y "%CPP_RELEASE_BIN%\*.dll" "%STAGE_DIR%\" >nul
if errorlevel 1 (
    echo [ERROR] Failed to copy ONNX Runtime DLLs into "%STAGE_DIR%".
    exit /b 1
)

if exist "%ROOT_DIR%\settings.json" (
    copy /y "%ROOT_DIR%\settings.json" "%STAGE_DIR%\settings.json" >nul
)

if exist "%ROOT_DIR%\fonts" (
    mkdir "%STAGE_DIR%\fonts" >nul 2>&1
    xcopy "%ROOT_DIR%\fonts\*" "%STAGE_DIR%\fonts\" /e /i /y /q >nul 2>&1
)

xcopy "%FRONTEND_DIST%\*" "%STAGE_DIR%\frontend\dist\" /e /i /y /q >nul
if errorlevel 1 (
    echo [ERROR] Failed to copy frontend/dist into "%STAGE_DIR%\frontend\dist".
    exit /b 1
)

copy /y "%CPP_RELEASE_BIN%\Vellum-Launcher.exe" "%ROOT_LAUNCHER%" >nul
if errorlevel 1 (
    echo [ERROR] Failed to install root one-click launcher "%ROOT_LAUNCHER%".
    echo         Please close any running Vellum.exe and retry.
    exit /b 1
)

if not exist "%STAGE_DIR%\Vellum.exe" (
    echo [ERROR] Staging verification failed: missing "%STAGE_DIR%\Vellum.exe"
    exit /b 1
)
if not exist "%STAGE_DIR%\onnxruntime.dll" (
    echo [ERROR] Staging verification failed: missing "%STAGE_DIR%\onnxruntime.dll"
    exit /b 1
)
if not exist "%STAGE_DIR%\frontend\dist\index.html" (
    echo [ERROR] Staging verification failed: missing "%STAGE_DIR%\frontend\dist\index.html"
    exit /b 1
)
if not exist "%ROOT_LAUNCHER%" (
    echo [ERROR] Staging verification failed: missing "%ROOT_LAUNCHER%"
    exit /b 1
)
echo       Staged standalone bundle : %STAGE_DIR%
echo       Installed root launcher  : %ROOT_LAUNCHER%
echo.

REM ----------------------------------------------------------------------------
REM Step 4/4: Package Clean GitHub Release Archive (`release/Vellum-Windows-x64.zip`)
REM ----------------------------------------------------------------------------
echo [4/4] Packaging GitHub Releases archive (`release/Vellum-Windows-x64.zip`) ...
if not exist "%RELEASE_DIR%" mkdir "%RELEASE_DIR%"
if exist "%ZIP_PATH%" del /f /q "%ZIP_PATH%"

set "MF_STAGE_DIR=%STAGE_DIR%"
set "MF_ROOT_LAUNCHER=%ROOT_LAUNCHER%"
set "MF_ZIP_PATH=%ZIP_PATH%"

powershell -NoProfile -ExecutionPolicy Bypass -Command "Compress-Archive -Path $env:MF_STAGE_DIR -DestinationPath $env:MF_ZIP_PATH -CompressionLevel Optimal -Force"
if errorlevel 1 (
    echo [ERROR] Failed to create release zip archive "%ZIP_PATH%".
    exit /b 1
)
if not exist "%ZIP_PATH%" (
    echo [ERROR] Expected release archive not found: "%ZIP_PATH%"
    exit /b 1
)

echo.
echo ============================================================================
echo   [SUCCESS] Vellum C++17 Standalone Release Ready.
echo ============================================================================
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$exe = Get-Item -LiteralPath (Join-Path $env:MF_STAGE_DIR 'Vellum.exe');" ^
    "$launcher = Get-Item -LiteralPath $env:MF_ROOT_LAUNCHER;" ^
    "$zip = Get-Item -LiteralPath $env:MF_ZIP_PATH;" ^
    "$dllBytes = (Get-ChildItem -LiteralPath $env:MF_STAGE_DIR -Filter '*.dll' | Measure-Object -Property Length -Sum).Sum;" ^
    "$stageBytes = (Get-ChildItem -LiteralPath $env:MF_STAGE_DIR -Recurse -File | Measure-Object -Property Length -Sum).Sum;" ^
    "Write-Host ('  Root Launcher     : {0} ({1:N1} KB)' -f $launcher.FullName, ($launcher.Length / 1KB));" ^
    "Write-Host ('  Native Binary     : {0} ({1:N2} MB)' -f $exe.FullName, ($exe.Length / 1MB));" ^
    "Write-Host ('  ONNX Runtime DLLs : {0:N2} MB' -f ($dllBytes / 1MB));" ^
    "Write-Host ('  Bundle Folder     : {0} ({1:N2} MB total)' -f $env:MF_STAGE_DIR, ($stageBytes / 1MB));" ^
    "Write-Host ('  GitHub Release Zip: {0} ({1:N2} MB)' -f $zip.FullName, ($zip.Length / 1MB));"
echo ============================================================================
echo.

endlocal
exit /b 0
