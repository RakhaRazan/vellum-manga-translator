"""Build MangaFlow AI as an instant-launch standalone Windows desktop app (`--onedir`, `--noconsole`).

Steps performed automatically:
1. Runs `npm run build` inside `frontend/` to compile the production React UI (`frontend/dist/`).
2. Ensures persistent runtime directories under `%LOCALAPPDATA%\\ComicTranslator\\`
   (`models/`, `settings.json`, `fonts/`) and seeds offline weights/fonts if present locally.
3. Packages `backend/desktop_app.py`, `frontend/dist`, ONNX Runtime, RapidOCR, Manga-OCR,
   `pywebview`, `img2pdf`/`pikepdf`, and all v2 backend modules via PyInstaller
   (`packaging/MangaFlow.spec`, `--onedir` + `--noconsole`) into `Manga/MangaFlow_App/`.
4. Compiles a tiny, zero-lag native Windows GUI launcher at `Manga/MangaFlow.exe`
   (no console flash, embedded icon) that launches `MangaFlow_App/MangaFlow.exe`.

Usage:
    backend\\venv\\Scripts\\python.exe packaging/build_standalone.py [--clean]
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PACKAGING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGING_DIR.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
FRONTEND_DIST_DIR = FRONTEND_DIR / "dist"
SPEC_PATH = PACKAGING_DIR / "MangaFlow.spec"
WORK_DIR = PACKAGING_DIR / "build" / "MangaFlow_standalone"
APP_BUNDLE_DIR = PROJECT_ROOT / "MangaFlow_App"
APP_MAIN_EXE = APP_BUNDLE_DIR / "MangaFlow.exe"
ROOT_LAUNCHER_EXE = PROJECT_ROOT / "MangaFlow.exe"
APP_ICON_PATH = PACKAGING_DIR / "icon.ico"
LEGACY_ICON_PATH = PACKAGING_DIR / "tauri" / "icons" / "icon.ico"


_LAUNCHER_CS_SOURCE = r"""using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Windows.Forms;

internal static class MangaFlowLauncher
{
    private static string QuoteArg(string arg)
    {
        if (string.IsNullOrEmpty(arg))
        {
            return "\"\"";
        }
        bool needsQuotes = false;
        foreach (char c in arg)
        {
            if (char.IsWhiteSpace(c) || c == '"')
            {
                needsQuotes = true;
                break;
            }
        }
        if (!needsQuotes)
        {
            return arg;
        }
        var sb = new StringBuilder();
        sb.Append('"');
        int backslashes = 0;
        foreach (char c in arg)
        {
            if (c == '\\')
            {
                backslashes++;
            }
            else if (c == '"')
            {
                sb.Append('\\', backslashes * 2 + 1);
                sb.Append('"');
                backslashes = 0;
            }
            else
            {
                if (backslashes > 0)
                {
                    sb.Append('\\', backslashes);
                    backslashes = 0;
                }
                sb.Append(c);
            }
        }
        if (backslashes > 0)
        {
            sb.Append('\\', backslashes * 2);
        }
        sb.Append('"');
        return sb.ToString();
    }

    [STAThread]
    private static int Main(string[] args)
    {
        string rootDir = AppDomain.CurrentDomain.BaseDirectory;
        string appDir = Path.Combine(rootDir, "MangaFlow_App");
        string targetExe = Path.Combine(appDir, "MangaFlow.exe");

        if (!File.Exists(targetExe))
        {
            MessageBox.Show(
                "MangaFlow AI application binary was not found at:\n"
                    + targetExe
                    + "\n\nPlease rebuild using packaging\\build_standalone.py.",
                "MangaFlow AI",
                MessageBoxButtons.OK,
                MessageBoxIcon.Error
            );
            return 1;
        }

        var argBuilder = new StringBuilder();
        for (int i = 0; i < args.Length; i++)
        {
            if (i > 0)
            {
                argBuilder.Append(' ');
            }
            argBuilder.Append(QuoteArg(args[i]));
        }

        var psi = new ProcessStartInfo
        {
            FileName = targetExe,
            Arguments = argBuilder.ToString(),
            WorkingDirectory = appDir,
            UseShellExecute = false
        };

        try
        {
            Process.Start(psi);
            return 0;
        }
        catch (Exception ex)
        {
            MessageBox.Show(
                "Failed to launch MangaFlow AI:\n" + ex.Message,
                "MangaFlow AI",
                MessageBoxButtons.OK,
                MessageBoxIcon.Error
            );
            return 1;
        }
    }
}
"""


def get_persistent_appdata_dir() -> Path:
    """Resolve `%LOCALAPPDATA%\\ComicTranslator`."""
    local_app_data = os.environ.get("LOCALAPPDATA")
    base_dir = (
        Path(local_app_data).expanduser().resolve()
        if local_app_data
        else (Path.home() / "AppData" / "Local").resolve()
    )
    return base_dir / "ComicTranslator"


def _sync_directory(src_dir: Path, dst_dir: Path) -> int:
    """Copy new or size-changed files from `src_dir` into `dst_dir`."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    if not src_dir.is_dir():
        return 0

    copied = 0
    for src_file in src_dir.rglob("*"):
        if not src_file.is_file():
            continue
        rel = src_file.relative_to(src_dir)
        dst_file = dst_dir / rel
        if not dst_file.is_file() or dst_file.stat().st_size != src_file.stat().st_size:
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            _ = shutil.copy2(src_file, dst_file)
            copied += 1
    return copied


def seed_persistent_appdata() -> Path:
    """Ensure `%LOCALAPPDATA%\\ComicTranslator\\{models,fonts,settings.json}` are ready."""
    appdata_root = get_persistent_appdata_dir()
    models_dir = appdata_root / "models"
    fonts_dir = appdata_root / "fonts"
    settings_file = appdata_root / "settings.json"

    appdata_root.mkdir(parents=True, exist_ok=True)
    models_copied = _sync_directory(PROJECT_ROOT / "models", models_dir)
    fonts_copied = _sync_directory(PROJECT_ROOT / "fonts", fonts_dir)

    dev_settings = PROJECT_ROOT / "backend" / "settings.json"
    seeded_settings = False
    if dev_settings.is_file() and not settings_file.is_file():
        _ = shutil.copy2(dev_settings, settings_file)
        seeded_settings = True

    print(f"[2/4] Persistent AppData ready : {appdata_root}")
    print(f"      - models/       : {models_dir} (synced {models_copied} file(s))")
    print(f"      - fonts/        : {fonts_dir} (synced {fonts_copied} file(s))")
    print(
        f"      - settings.json : {settings_file} "
        + ("(seeded from dev)" if seeded_settings else "(preserved existing / default)")
    )
    return appdata_root


def _resolve_npm_executable() -> tuple[str, dict[str, str]]:
    """Locate `npm.cmd` / `npm` and return `(npm_path, env)` with Node on PATH."""
    env = os.environ.copy()
    unsloth_node_dir = Path.home() / ".unsloth" / "node"
    if (unsloth_node_dir / "node.exe").is_file():
        env["PATH"] = f"{unsloth_node_dir}{os.pathsep}{env.get('PATH', '')}"

    for candidate in ("npm.cmd", "npm"):
        found = shutil.which(candidate, path=env.get("PATH"))
        if found:
            return found, env

    raise FileNotFoundError(
        "Node.js / npm was not found on PATH (also checked %USERPROFILE%\\.unsloth\\node)."
    )


def build_frontend() -> None:
    """Run `npm run build` inside `frontend/` and verify `frontend/dist/index.html`."""
    print("[1/4] Building production React frontend (`npm run build`) ...")
    npm_exe, env = _resolve_npm_executable()
    result = subprocess.run(
        [npm_exe, "run", "build"],
        cwd=str(FRONTEND_DIR),
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Frontend build failed with exit code {result.returncode}.")

    index_html = FRONTEND_DIST_DIR / "index.html"
    if not index_html.is_file():
        raise FileNotFoundError(f"Expected compiled frontend entry point at {index_html}")
    print(f"      Frontend bundle ready    : {FRONTEND_DIST_DIR}")


def ensure_app_icon() -> Path | None:
    """Ensure `packaging/icon.ico` exists for embedding into the executables."""
    if APP_ICON_PATH.is_file():
        return APP_ICON_PATH
    if LEGACY_ICON_PATH.is_file():
        _ = shutil.copy2(LEGACY_ICON_PATH, APP_ICON_PATH)
        return APP_ICON_PATH
    return None


def build_onedir_bundle(clean: bool = False) -> None:
    """Run PyInstaller with `packaging/MangaFlow.spec` to build `Manga/MangaFlow_App/`."""
    print("[3/4] Packaging MangaFlow AI via PyInstaller (`--onedir`, `--noconsole`) ...")
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--distpath",
        str(PROJECT_ROOT),
        "--workpath",
        str(WORK_DIR),
        str(SPEC_PATH),
    ]
    if clean:
        cmd.insert(4, "--clean")

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"PyInstaller build failed with exit code {result.returncode}.")

    if not APP_MAIN_EXE.is_file():
        raise FileNotFoundError(f"Expected standalone app binary at {APP_MAIN_EXE}")

    bundled_index = APP_BUNDLE_DIR / "_internal" / "frontend" / "dist" / "index.html"
    if not bundled_index.is_file():
        raise FileNotFoundError(f"Expected bundled React index at {bundled_index}")

    print(f"      Standalone bundle ready  : {APP_BUNDLE_DIR}")


def _find_csc_compiler() -> Path:
    """Locate the built-in Windows .NET Framework C# compiler (`csc.exe`)."""
    windir = Path(os.environ.get("WINDIR", r"C:\Windows"))
    candidates = [
        windir / "Microsoft.NET" / "Framework64" / "v4.0.30319" / "csc.exe",
        windir / "Microsoft.NET" / "Framework" / "v4.0.30319" / "csc.exe",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    found = shutil.which("csc.exe")
    if found:
        return Path(found)
    raise FileNotFoundError(
        "Windows C# compiler (`csc.exe`) not found under %WINDIR%\\Microsoft.NET\\Framework64\\v4.0.30319."
    )


def build_root_launcher(icon_path: Path | None) -> None:
    """Compile the instant-start, zero-console native launcher at `Manga/MangaFlow.exe`."""
    print("[4/4] Compiling root one-click native launcher (`Manga/MangaFlow.exe`) ...")
    csc_exe = _find_csc_compiler()
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    launcher_cs = WORK_DIR / "MangaFlowLauncher.cs"
    _ = launcher_cs.write_text(_LAUNCHER_CS_SOURCE, encoding="utf-8")

    cmd = [
        str(csc_exe),
        "/nologo",
        "/target:winexe",
        "/optimize+",
        "/platform:anycpu",
        "/reference:System.Windows.Forms.dll",
        f"/out:{ROOT_LAUNCHER_EXE}",
    ]
    if icon_path is not None and icon_path.is_file():
        cmd.append(f"/win32icon:{icon_path}")
    cmd.append(str(launcher_cs))

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Root launcher compilation failed with exit code {result.returncode}.")

    if not ROOT_LAUNCHER_EXE.is_file():
        raise FileNotFoundError(f"Expected root launcher at {ROOT_LAUNCHER_EXE}")


def _dir_size_mb(directory: Path) -> float:
    total_bytes = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
    return total_bytes / (1024 * 1024)


def main() -> int:
    """Execute the full standalone build pipeline."""
    started = time.monotonic()
    clean = "--clean" in sys.argv

    print("======================================================================")
    print("  MangaFlow AI - Standalone Native Desktop Builder (--onedir)")
    print("======================================================================")
    print(f"  Project Root : {PROJECT_ROOT}")
    print(f"  Spec File    : {SPEC_PATH}")
    print(f"  Bundle Dir   : {APP_BUNDLE_DIR}")
    print(f"  Launcher Exe : {ROOT_LAUNCHER_EXE}")
    print("======================================================================\n")

    try:
        build_frontend()
        appdata_dir = seed_persistent_appdata()
        icon_path = ensure_app_icon()
        build_onedir_bundle(clean=clean)
        build_root_launcher(icon_path)
    except Exception as exc:
        print(f"\n[ERROR] Standalone build failed: {exc}", file=sys.stderr)
        return 1

    elapsed = time.monotonic() - started
    bundle_mb = _dir_size_mb(APP_BUNDLE_DIR)
    launcher_kb = ROOT_LAUNCHER_EXE.stat().st_size / 1024.0

    print("\n======================================================================")
    print(f"  [SUCCESS] Standalone build completed in {elapsed:.1f}s")
    print(f"  One-Click Launcher : {ROOT_LAUNCHER_EXE} ({launcher_kb:.1f} KB)")
    print(f"  App Bundle Folder  : {APP_BUNDLE_DIR} ({bundle_mb:.1f} MB)")
    print(f"  Persistent Data    : {appdata_dir}")
    print("======================================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
