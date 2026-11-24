@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================================
REM Vellum v2.5 Studio Core - Dual-Core C++17 Release Builder (CUDA + DirectML)
REM
REM Usage:
REM   packaging\build_cpp_release.bat [AUTO|DUAL|CUDA|DML]
REM
REM   AUTO / DUAL : Builds BOTH CUDA core (Vellum_Runtime\Vellum_Core.exe)
REM                 AND lightweight DirectML core (Vellum_Runtime\dml\Vellum_Core_DML.exe)
REM                 sharing Vellum_Runtime\frontend\dist and Vellum_Runtime\fonts,
REM                 plus root Dual-Core Vellum.exe launcher.
REM                 -> release\Vellum-v2.5-Windows-x64-Universal.zip
REM   CUDA        : Builds CUDA core only
REM   DML         : Builds DirectML core only
REM ============================================================================

pushd "%~dp0.." >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Failed to resolve repository root from "%~dp0..".
    exit /b 1
)
set "ROOT_DIR=%CD%"
popd >nul 2>&1

set "VARIANT_ARG=%~1"
if not defined VARIANT_ARG set "VARIANT_ARG=AUTO"

set "BUILD_DML=0"
set "BUILD_CUDA=0"

if /I "%VARIANT_ARG%"=="CUDA" set "BUILD_CUDA=1"
if /I "%VARIANT_ARG%"=="DML" set "BUILD_DML=1"
if /I "%VARIANT_ARG%"=="UNIVERSAL" (
    set "BUILD_CUDA=1"
    set "BUILD_DML=1"
)
if /I "%VARIANT_ARG%"=="ALL" (
    set "BUILD_CUDA=1"
    set "BUILD_DML=1"
)
if /I "%VARIANT_ARG%"=="BOTH" (
    set "BUILD_CUDA=1"
    set "BUILD_DML=1"
)
if /I "%VARIANT_ARG%"=="DUAL" (
    set "BUILD_CUDA=1"
    set "BUILD_DML=1"
)
if /I "%VARIANT_ARG%"=="AUTO" (
    set "BUILD_CUDA=1"
    set "BUILD_DML=1"
)

if "%BUILD_DML%"=="0" if "%BUILD_CUDA%"=="0" (
    echo [ERROR] Unknown variant '%VARIANT_ARG%'. Use AUTO, DUAL, CUDA, or DML.
    exit /b 1
)

set "FRONTEND_DIR=%ROOT_DIR%\frontend"
set "FRONTEND_DIST=%FRONTEND_DIR%\dist"
set "CPP_DIR=%ROOT_DIR%\backend-cpp"
set "STAGE_DIR=%ROOT_DIR%\Vellum_Runtime"
set "ROOT_LAUNCHER=%ROOT_DIR%\Vellum.exe"
set "RELEASE_DIR=%ROOT_DIR%\release"
set "ZIP_NAME=Vellum-v2.5-Windows-x64-Universal.zip"
if /I "%VARIANT_ARG%"=="CUDA" set "ZIP_NAME=Vellum-v2.5-Windows-x64-CUDA.zip"
if /I "%VARIANT_ARG%"=="DML" set "ZIP_NAME=Vellum-v2.5-Windows-x64-DirectML.zip"
set "ZIP_PATH=%RELEASE_DIR%\%ZIP_NAME%"

echo ============================================================================
echo   Vellum v2.5 Studio Core - Dual-Core Release Builder
echo ============================================================================
echo   Project Root    : %ROOT_DIR%
echo   Target Mode     : %VARIANT_ARG% (CUDA=%BUILD_CUDA%, DirectML=%BUILD_DML%)
echo   Release Dir     : %RELEASE_DIR%
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
    exit /b 1
)

REM ----------------------------------------------------------------------------
REM Step 1: Build React Frontend (frontend\dist)
REM ----------------------------------------------------------------------------
echo [1/4] Building production React frontend ...
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

if not exist "%RELEASE_DIR%" mkdir "%RELEASE_DIR%"

REM ----------------------------------------------------------------------------
REM Step 2: Compile Selected C++ Cores
REM ----------------------------------------------------------------------------
set "CUDA_BIN_DIR=%CPP_DIR%\build-cuda\Release"
set "DML_BIN_DIR=%CPP_DIR%\build-dml\Release"
set "LAUNCHER_BIN="

if "%BUILD_CUDA%"=="1" (
    echo [2A/4] Ensuring NVIDIA CUDA 12 + cuDNN 9 redistributable DLLs are cached ...
    if not exist "%CPP_DIR%\build_deps\cuda_runtime\bin\cudnn64_9.dll" (
        python "%ROOT_DIR%\packaging\fetch_cuda_dlls.py"
        if errorlevel 1 (
            echo [ERROR] Failed to fetch NVIDIA CUDA 12 / cuDNN 9 runtime DLLs.
            exit /b 1
        )
    ) else (
        echo       CUDA 12 + cuDNN 9 DLLs ready : %CPP_DIR%\build_deps\cuda_runtime\bin
    )
    echo.
    call :CompileCore CUDA "build-cuda"
    if errorlevel 1 exit /b 1
    set "LAUNCHER_BIN=%CUDA_BIN_DIR%\Vellum-Launcher.exe"
)

if "%BUILD_DML%"=="1" (
    call :CompileCore DML "build-dml"
    if errorlevel 1 exit /b 1
    if not defined LAUNCHER_BIN set "LAUNCHER_BIN=%DML_BIN_DIR%\Vellum-Launcher.exe"
)

REM ----------------------------------------------------------------------------
REM Step 3: Stage Unified Dual-Core Vellum_Runtime + Root Launcher
REM ----------------------------------------------------------------------------
echo [3/4] Staging unified Vellum_Runtime directory ...
if exist "%STAGE_DIR%" (
    rmdir /s /q "%STAGE_DIR%" >nul 2>&1
    if exist "%STAGE_DIR%" (
        echo [ERROR] Could not clean "%STAGE_DIR%". Close any running Vellum.exe and retry.
        exit /b 1
    )
)

mkdir "%STAGE_DIR%"
if errorlevel 1 exit /b 1
mkdir "%STAGE_DIR%\frontend\dist"
if errorlevel 1 exit /b 1

if "%BUILD_CUDA%"=="1" (
    copy /y "%CUDA_BIN_DIR%\Vellum.exe" "%STAGE_DIR%\Vellum_Core.exe" >nul
    if errorlevel 1 exit /b 1
    copy /y "%CUDA_BIN_DIR%\*.dll" "%STAGE_DIR%\" >nul
    if errorlevel 1 exit /b 1
    if exist "%CPP_DIR%\build_deps\cuda_runtime\bin\*.dll" (
        copy /y "%CPP_DIR%\build_deps\cuda_runtime\bin\*.dll" "%STAGE_DIR%\" >nul
        if errorlevel 1 exit /b 1
    )
)

if "%BUILD_DML%"=="1" (
    mkdir "%STAGE_DIR%\dml"
    if errorlevel 1 exit /b 1
    copy /y "%DML_BIN_DIR%\Vellum.exe" "%STAGE_DIR%\dml\Vellum_Core_DML.exe" >nul
    if errorlevel 1 exit /b 1
    copy /y "%DML_BIN_DIR%\*.dll" "%STAGE_DIR%\dml\" >nul
    if errorlevel 1 exit /b 1
)

if exist "%ROOT_DIR%\fonts" (
    mkdir "%STAGE_DIR%\fonts" >nul 2>&1
    xcopy "%ROOT_DIR%\fonts\*" "%STAGE_DIR%\fonts\" /e /i /y /q >nul 2>&1
)

xcopy "%FRONTEND_DIST%\*" "%STAGE_DIR%\frontend\dist\" /e /i /y /q >nul
if errorlevel 1 exit /b 1

if not exist "%LAUNCHER_BIN%" (
    echo [ERROR] Missing compiled launcher binary: "%LAUNCHER_BIN%"
    exit /b 1
)
copy /y "%LAUNCHER_BIN%" "%ROOT_LAUNCHER%" >nul
if errorlevel 1 exit /b 1

REM ----------------------------------------------------------------------------
REM Step 4: Pre-Release Secret Guard & Portable Archive
REM ----------------------------------------------------------------------------
echo [4/4] Running Pre-Release Secret Guard and packaging %ZIP_NAME% ...
if exist "%STAGE_DIR%\settings.json" (
    echo [ERROR] Pre-Release Secret Guard aborted: settings.json found in "%STAGE_DIR%".
    exit /b 1
)

set "MF_STAGE_DIR=%STAGE_DIR%"
set "MF_ZIP_PATH=%ZIP_PATH%"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$stage = $env:MF_STAGE_DIR;" ^
    "$badFiles = Get-ChildItem -LiteralPath $stage -Recurse -File | Where-Object { $_.Name -ieq 'settings.json' -or $_.Name -ieq 'KOHARU_CLEANROOM_SPEC.md' };" ^
    "if ($badFiles) { Write-Error ('ABORT: Prohibited file found in release stage: ' + $badFiles[0].FullName); exit 1 };" ^
    "$badDirs = Get-ChildItem -LiteralPath $stage -Recurse -Directory | Where-Object { @('cache','projects_cache','models') -contains $_.Name.ToLowerInvariant() -or $_.Name -like '*.WebView2' };" ^
    "if ($badDirs) { Write-Error ('ABORT: Prohibited directory found in release stage: ' + $badDirs[0].FullName); exit 1 };" ^
    "$pattern = '(?i)(C:\\Users\\[^\\]+|C:/Users/[^/]+|C:\\FILE|C:/FILE|\bsk-(?:or-|ant-|proj-|[A-Za-z0-9]{12,})|AIza[0-9A-Za-z_-]{10,}|ghp_[A-Za-z0-9]{20,}|hauhaucs|Faithful & Uncensored|:1235\b)';" ^
    "$scanExts = @('.json','.js','.jsx','.css','.html','.txt','.md','.bat','.ps1','.cfg','.ini','.yaml','.yml');" ^
    "foreach ($f in (Get-ChildItem -LiteralPath $stage -Recurse -File)) {" ^
    "  if ($scanExts -contains $f.Extension.ToLowerInvariant()) {" ^
    "    $hits = Select-String -LiteralPath $f.FullName -Pattern $pattern -AllMatches;" ^
    "    if ($hits) { Write-Error ('ABORT: Secret or personal pattern detected in ' + $f.FullName + ' line ' + $hits[0].LineNumber); exit 1 }" ^
    "  }" ^
    "}"
if errorlevel 1 (
    echo [ERROR] Pre-Release Secret Guard failed for %ZIP_NAME%.
    exit /b 1
)

if exist "%ZIP_PATH%" del /f /q "%ZIP_PATH%"
powershell -NoProfile -ExecutionPolicy Bypass -Command "Compress-Archive -Path $env:MF_STAGE_DIR -DestinationPath $env:MF_ZIP_PATH -CompressionLevel Fastest -Force"
if errorlevel 1 exit /b 1
if not exist "%ZIP_PATH%" exit /b 1

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$zip = Get-Item -LiteralPath $env:MF_ZIP_PATH;" ^
    "$stageBytes = (Get-ChildItem -LiteralPath $env:MF_STAGE_DIR -Recurse -File | Measure-Object -Property Length -Sum).Sum;" ^
    "$dmlDir = Join-Path $env:MF_STAGE_DIR 'dml';" ^
    "$dmlBytes = if (Test-Path $dmlDir) { (Get-ChildItem -LiteralPath $dmlDir -Recurse -File | Measure-Object -Property Length -Sum).Sum } else { 0 };" ^
    "Write-Host ('  [DUAL-CORE] Stage: {0:N2} MB | DirectML Core: {1:N2} MB | Zip: {2} ({3:N2} MB)' -f ($stageBytes / 1MB), ($dmlBytes / 1MB), $zip.Name, ($zip.Length / 1MB));"

echo.
echo ============================================================================
echo   [SUCCESS] Vellum v2.5 Studio Core Dual-Core Release Build Complete.
echo ============================================================================
endlocal
exit /b 0

REM ============================================================================
REM Subroutine: CompileCore <ORT_BACKEND> <BUILD_SUBDIR>
REM ============================================================================
:CompileCore
set "ORT_BACKEND=%~1"
set "BUILD_SUBDIR=%~2"
set "BUILD_DIR=%CPP_DIR%\%BUILD_SUBDIR%"
set "BIN_DIR=%BUILD_DIR%\Release"

echo [2/4] Configuring and compiling %ORT_BACKEND% core in %BUILD_SUBDIR% ...
pushd "%CPP_DIR%"
cmake -B "%BUILD_SUBDIR%" -A x64 -DCMAKE_BUILD_TYPE=Release -DVELLUM_ORT_BACKEND=%ORT_BACKEND%
if errorlevel 1 (
    popd
    echo [ERROR] CMake configure failed for %ORT_BACKEND%.
    exit /b 1
)

cmake --build "%BUILD_SUBDIR%" --config Release --target MangaFlowAI MangaFlowCppLauncher test_psd_writer --parallel
set "STEP_ERR=!ERRORLEVEL!"
popd
if not "!STEP_ERR!"=="0" (
    echo [ERROR] CMake Release build failed for %ORT_BACKEND% with exit code !STEP_ERR!.
    exit /b !STEP_ERR!
)

if exist "%BIN_DIR%\test_psd_writer.exe" (
    echo       Running PSD writer unit tests: test_psd_writer.exe ...
    "%BIN_DIR%\test_psd_writer.exe"
    if errorlevel 1 (
        echo [ERROR] test_psd_writer.exe failed! Aborting release packaging.
        exit /b 1
    )
)

if not exist "%BIN_DIR%\Vellum.exe" (
    echo [ERROR] Missing "%BIN_DIR%\Vellum.exe"
    exit /b 1
)
if not exist "%BIN_DIR%\Vellum-Launcher.exe" (
    echo [ERROR] Missing "%BIN_DIR%\Vellum-Launcher.exe"
    exit /b 1
)
if not exist "%BIN_DIR%\onnxruntime.dll" (
    echo [ERROR] Missing "%BIN_DIR%\onnxruntime.dll"
    exit /b 1
)
exit /b 0
