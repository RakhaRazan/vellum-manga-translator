"""Build script to compile `backend/main.py` into a standalone `comic-backend.exe` binary via PyInstaller.

Usage:
    backend\\venv\\Scripts\\python.exe packaging/build_backend.py
"""

import os
from pathlib import Path
import shutil
import subprocess
import sys

PACKAGING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGING_DIR.parent
SPEC_PATH = PACKAGING_DIR / "comic-backend.spec"
DIST_DIR = PACKAGING_DIR / "dist"
BUILD_DIR = PACKAGING_DIR / "build"


def get_persistent_models_dir() -> Path:
    """Resolve `%LOCALAPPDATA%/ComicTranslator/models`."""
    local_app_data = os.environ.get("LOCALAPPDATA")
    base_dir = (
        Path(local_app_data).expanduser().resolve()
        if local_app_data
        else (Path.home() / "AppData" / "Local").resolve()
    )
    return base_dir / "ComicTranslator" / "models"


def seed_persistent_models_cache() -> None:
    """Copy already-downloaded local development models into `%LOCALAPPDATA%/ComicTranslator/models` if missing."""
    dev_models_dir = PROJECT_ROOT / "models"
    target_models_dir = get_persistent_models_dir()
    target_models_dir.mkdir(parents=True, exist_ok=True)

    if not dev_models_dir.is_dir():
        return

    copied_count = 0
    for src_file in dev_models_dir.rglob("*"):
        if not src_file.is_file():
            continue
        rel = src_file.relative_to(dev_models_dir)
        dst_file = target_models_dir / rel
        if not dst_file.is_file() or dst_file.stat().st_size != src_file.stat().st_size:
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            _ = shutil.copy2(src_file, dst_file)
            copied_count += 1

    print(
        f"[MODELS] Persistent offline directory ready: {target_models_dir} "
        + f"(synced {copied_count} new file(s))"
    )


def build_executable() -> int:
    """Run PyInstaller using `packaging/comic-backend.spec` to produce `comic-backend.exe`."""
    print("=== Building Standalone FastAPI Sidecar (comic-backend.exe) ===")
    print(f"  Project Root : {PROJECT_ROOT}")
    print(f"  Spec File    : {SPEC_PATH}")
    print(f"  Output Dir   : {DIST_DIR}")

    seed_persistent_models_cache()

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--distpath",
        str(DIST_DIR),
        "--workpath",
        str(BUILD_DIR),
        str(SPEC_PATH),
    ]
    if "--clean" in sys.argv:
        cmd.insert(4, "--clean")

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False)
    if result.returncode != 0:
        print(
            f"[ERROR] PyInstaller build failed with exit code {result.returncode}.",
            file=sys.stderr,
        )
        return result.returncode

    built_exe = DIST_DIR / "comic-backend.exe"
    if not built_exe.is_file():
        print(f"[ERROR] Expected binary not found at {built_exe}", file=sys.stderr)
        return 1

    root_packaging_exe = PACKAGING_DIR / "comic-backend.exe"
    try:
        _ = shutil.copy2(built_exe, root_packaging_exe)
    except OSError as exc:
        print(f"[WARN] Could not copy binary to {root_packaging_exe}: {exc}")

    tauri_binaries_dir = PROJECT_ROOT / "frontend" / "src-tauri" / "binaries"
    tauri_binaries_dir.mkdir(parents=True, exist_ok=True)
    tauri_sidecar_exe = tauri_binaries_dir / "comic-backend-x86_64-pc-windows-msvc.exe"
    try:
        _ = shutil.copy2(built_exe, tauri_sidecar_exe)
        print(f"[SUCCESS] Tauri sidecar binary staged : {tauri_sidecar_exe}")
    except OSError as exc:
        print(f"[WARN] Could not copy sidecar binary to {tauri_sidecar_exe}: {exc}")

    tauri_release_dir = PROJECT_ROOT / "frontend" / "src-tauri" / "target" / "release"
    if tauri_release_dir.is_dir():
        for release_name in (
            "comic-backend.exe",
            "comic-backend-x86_64-pc-windows-msvc.exe",
        ):
            target_release_exe = tauri_release_dir / release_name
            try:
                _ = shutil.copy2(built_exe, target_release_exe)
                print(f"[SUCCESS] Updated Tauri release binary: {target_release_exe}")
            except OSError as exc:
                print(f"[WARN] Could not update {target_release_exe}: {exc}")

    size_mb = built_exe.stat().st_size / (1024 * 1024)
    print(f"\n[SUCCESS] Built standalone executable: {built_exe} ({size_mb:.1f} MB)")
    print(f"[SUCCESS] Copy available at          : {root_packaging_exe}")
    return 0


if __name__ == "__main__":
    raise SystemExit(build_executable())
