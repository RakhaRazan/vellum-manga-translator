# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller `--onedir` + `--noconsole` Specification for MangaFlow AI.

Bundles `backend/desktop_app.py`, the compiled React frontend (`frontend/dist`),
FastAPI/Uvicorn, `pywebview` (Edge WebView2 + `pythonnet`/`clr_loader`),
ONNX Runtime, RapidOCR (`rapidocr_onnxruntime`), Japanese Manga-OCR
(`manga_ocr`, `fugashi`, `unidic_lite`, trimmed `transformers`, `torch`),
`img2pdf` + `pikepdf`, and all v2 backend modules (`providers.py`, `exporter.py`,
`font_manager.py`, etc.) into an instant-launch `--onedir` bundle:

    Manga/MangaFlow_App/MangaFlow.exe
    Manga/MangaFlow_App/_internal/...

At runtime (`sys.frozen == True`), persistent user data resolves to:
    - `%LOCALAPPDATA%\\ComicTranslator\\models\\`
    - `%LOCALAPPDATA%\\ComicTranslator\\settings.json`
    - `%LOCALAPPDATA%\\ComicTranslator\\fonts\\`
"""

from pathlib import Path
import sys

from PyInstaller.utils.hooks import (
    PY_DYLIB_PATTERNS,
    collect_all,
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)
import pikepdf
import transformers

SPEC_DIR = Path(SPECPATH).resolve()
PROJECT_ROOT = (
    SPEC_DIR.parents[1]
    if SPEC_DIR.name.lower() == "pyinstaller"
    else SPEC_DIR.parent
)
ENTRY_SCRIPT = str(PROJECT_ROOT / "backend" / "desktop_app.py")

# Resolve application icon (packaging/icon.ico preferred, fallback to existing icon.ico)
_ICON_CANDIDATES = (
    PROJECT_ROOT / "packaging" / "icon.ico",
    PROJECT_ROOT / "packaging" / "tauri" / "icons" / "icon.ico",
)
ICON_PATH: str | None = next(
    (str(p) for p in _ICON_CANDIDATES if p.is_file()),
    None,
)

# Require compiled frontend bundle (`frontend/dist/index.html`)
FRONTEND_DIST_DIR = PROJECT_ROOT / "frontend" / "dist"
if not (FRONTEND_DIST_DIR / "index.html").is_file():
    raise FileNotFoundError(
        f"Compiled frontend bundle not found at {FRONTEND_DIST_DIR / 'index.html'}. "
        "Run `npm run build` inside `frontend/` (or run `packaging/build_standalone.py`) first."
    )

datas: list[tuple[str, str]] = [
    (str(FRONTEND_DIST_DIR), "frontend/dist"),
]
binaries: list[tuple[str, str]] = []

# Include Windows-compatible Python stdlib modules so dynamic imports inside
# torch/transformers/uvicorn never fail at runtime.
_STDLIB_EXCLUDE = {
    "antigravity",
    "this",
    "tkinter",
    "idlelib",
    "turtle",
    "turtledemo",
    "test",
    "venv",
    "ensurepip",
    "lib2to3",
    "distutils",
    "curses",
    "crypt",
    "fcntl",
    "grp",
    "nis",
    "ossaudiodev",
    "posix",
    "pwd",
    "pty",
    "readline",
    "resource",
    "spwd",
    "syslog",
    "termios",
    "tty",
}

hiddenimports: list[str] = [
    *[
        m
        for m in sys.stdlib_module_names
        if not m.startswith("_") and m not in _STDLIB_EXCLUDE
    ],
    # Core v2 backend & desktop host modules
    "backend",
    "backend.desktop_app",
    "backend.main",
    "backend.core",
    "backend.core.model_manager",
    "backend.core.detector",
    "backend.core.ocr",
    "backend.core.inpainter",
    "backend.core.translator",
    "backend.core.providers",
    "backend.core.exporter",
    "backend.core.font_manager",
    # FastAPI, Starlette & Uvicorn runtime modules
    "uvicorn",
    "uvicorn.config",
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "fastapi",
    "fastapi.staticfiles",
    "starlette",
    "starlette.staticfiles",
    "pydantic",
    "pydantic_core",
    "multipart",
    "python_multipart",
    # Native pywebview desktop host (Windows Edge WebView2 + WinForms + pythonnet)
    "webview",
    "webview.platforms.winforms",
    "webview.platforms.edgechromium",
    "pythonnet",
    "clr",
    "clr_loader",
    "cffi",
    "_cffi_backend",
    "pycparser",
    "bottle",
    "proxy_tools",
    # Export stack (.cbz / .pdf)
    "img2pdf",
    "pikepdf",
    "pikepdf._core",
    "pikepdf._methods",
    "pikepdf.models",
    # CV, OCR & LLM clients
    "onnxruntime",
    "rapidocr_onnxruntime",
    "manga_ocr",
    "manga_ocr.ocr",
    "fugashi",
    "unidic_lite",
    "jaconv",
    "cv2",
    "numpy",
    "PIL",
    "PIL.Image",
    "PIL.ImageFont",
    "pyclipper",
    "shapely",
    "six",
    "openai",
    "httpx",
    "httpcore",
    "anyio",
    "sniffio",
    "h11",
    "distro",
    "jiter",
    "requests",
    "urllib3",
    "idna",
    "certifi",
    "charset_normalizer",
    "regex",
    "tqdm",
    "packaging",
    "filelock",
    "fsspec",
    "yaml",
    "loguru",
    "win32_setctime",
    "colorama",
    "jinja2",
    "markupsafe",
    "typing_extensions",
    "typing_inspection",
    "annotated_types",
    "annotated_doc",
    "click",
    # Common stdlib submodules imported dynamically
    "asyncio.base_events",
    "asyncio.base_futures",
    "asyncio.base_subprocess",
    "asyncio.base_tasks",
    "asyncio.constants",
    "asyncio.coroutines",
    "asyncio.events",
    "asyncio.exceptions",
    "asyncio.format_helpers",
    "asyncio.futures",
    "asyncio.locks",
    "asyncio.log",
    "asyncio.mixins",
    "asyncio.proactor_events",
    "asyncio.protocols",
    "asyncio.queues",
    "asyncio.runners",
    "asyncio.selector_events",
    "asyncio.sslproto",
    "asyncio.staggered",
    "asyncio.streams",
    "asyncio.subprocess",
    "asyncio.taskgroups",
    "asyncio.tasks",
    "asyncio.threads",
    "asyncio.timeouts",
    "asyncio.transports",
    "asyncio.trsock",
    "asyncio.windows_events",
    "asyncio.windows_utils",
    "collections.abc",
    "concurrent.futures",
    "concurrent.futures._base",
    "concurrent.futures.process",
    "concurrent.futures.thread",
    "ctypes._endian",
    "ctypes.util",
    "ctypes.wintypes",
    "email._encoded_words",
    "email._parseaddr",
    "email._policybase",
    "email.base64mime",
    "email.charset",
    "email.encoders",
    "email.errors",
    "email.feedparser",
    "email.header",
    "email.iterators",
    "email.message",
    "email.parser",
    "email.quoprimime",
    "email.utils",
    "html.entities",
    "html.parser",
    "http.client",
    "http.cookiejar",
    "http.cookies",
    "http.server",
    "importlib.abc",
    "importlib.machinery",
    "importlib.metadata",
    "importlib.readers",
    "importlib.resources",
    "importlib.resources.abc",
    "importlib.resources.readers",
    "importlib.util",
    "json.decoder",
    "json.encoder",
    "json.scanner",
    "logging.config",
    "logging.handlers",
    "multiprocessing.connection",
    "multiprocessing.context",
    "multiprocessing.dummy",
    "multiprocessing.pool",
    "multiprocessing.process",
    "multiprocessing.queues",
    "multiprocessing.reduction",
    "multiprocessing.resource_sharer",
    "multiprocessing.resource_tracker",
    "multiprocessing.shared_memory",
    "multiprocessing.spawn",
    "multiprocessing.synchronize",
    "multiprocessing.util",
    "unittest.case",
    "unittest.loader",
    "unittest.main",
    "unittest.mock",
    "unittest.result",
    "unittest.runner",
    "unittest.signals",
    "unittest.suite",
    "unittest.util",
    "urllib.error",
    "urllib.parse",
    "urllib.request",
    "urllib.response",
    "urllib.robotparser",
    "xml.dom",
    "xml.dom.minidom",
    "xml.etree.ElementPath",
    "xml.etree.ElementTree",
    "xml.parsers.expat",
    "xml.sax",
    "xml.sax.handler",
    "xml.sax.saxutils",
    "xml.sax.xmlreader",
]

# 1. Collect ONNX Runtime, RapidOCR, Manga-OCR, Fugashi, pythonnet, clr_loader
for pkg_name in (
    "onnxruntime",
    "rapidocr_onnxruntime",
    "manga_ocr",
    "fugashi",
    "pythonnet",
    "clr_loader",
):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg_name)
    datas.extend(pkg_datas)
    binaries.extend(pkg_binaries)
    hiddenimports.extend(pkg_hidden)

# 2. Collect pywebview (keeping Windows Edge WebView2 / WinForms assets, skipping mobile/macOS/Linux platforms)
_WEBVIEW_SKIP_PLATFORMS = ("android", "cocoa", "gtk", "qt", "cef")
wv_datas, wv_binaries, wv_hidden = collect_all("webview")
for src, dst in wv_datas:
    norm_dst = dst.replace("\\", "/").lower()
    if any(f"webview/platforms/{p}" in norm_dst for p in _WEBVIEW_SKIP_PLATFORMS):
        continue
    datas.append((src, dst))
binaries.extend(wv_binaries)
hiddenimports.extend(
    m
    for m in wv_hidden
    if not any(m.startswith(f"webview.platforms.{p}") for p in _WEBVIEW_SKIP_PLATFORMS)
)

# 3. Collect pikepdf + sibling `pikepdf.libs` native DLLs required by `img2pdf` for lossless PDF export
pike_datas, pike_binaries, pike_hidden = collect_all("pikepdf")
datas.extend(pike_datas)
binaries.extend(pike_binaries)
binaries.extend(
    collect_dynamic_libs("pikepdf", search_patterns=PY_DYLIB_PATTERNS + ["*.pyd"])
)
hiddenimports.extend(m for m in pike_hidden if not m.startswith("pikepdf.pdfa"))

pikepdf_libs_dir = Path(pikepdf.__file__).resolve().parents[1] / "pikepdf.libs"
if pikepdf_libs_dir.is_dir():
    for dll_file in pikepdf_libs_dir.glob("*.dll"):
        binaries.append((str(dll_file), "pikepdf.libs"))

datas.extend(collect_data_files("unidic_lite"))
hiddenimports.extend(collect_submodules("unidic_lite"))
hiddenimports.extend(collect_submodules("backend"))

# 4. Collect PyTorch and companion packages directly (.py + .dll/.pyd) for fast builds
for direct_pkg in (
    "torch",
    "torchgen",
    "functorch",
    "sympy",
    "mpmath",
    "networkx",
    "safetensors",
    "tokenizers",
    "huggingface_hub",
    "tqdm",
    "PIL",
    "jinja2",
    "markupsafe",
    "fsspec",
    "filelock",
    "yaml",
    "loguru",
    "jaconv",
    "requests",
    "urllib3",
    "packaging",
    "regex",
):
    datas.extend(
        collect_data_files(
            direct_pkg,
            include_py_files=True,
            excludes=[
                "**/*.h",
                "**/*.hpp",
                "**/*.cuh",
                "**/*.lib",
                "**/*.cpp",
                "**/*.pyi",
                "**/*.cmake",
                "**/include/**",
            ],
        )
    )
    binaries.extend(
        collect_dynamic_libs(
            direct_pkg,
            search_patterns=PY_DYLIB_PATTERNS + ["*.pyd"],
        )
    )

# 5. Collect trimmed `transformers` (ViT + BERT Japanese + VisionEncoderDecoder needed by manga-ocr)
tf_root = Path(transformers.__file__).parent
keep_tf_models = {
    "auto",
    "bert",
    "bert_japanese",
    "encoder_decoder",
    "vision_encoder_decoder",
    "vit",
    "deprecated",
    "__pycache__",
}
for src_path, dst_dir in collect_data_files(
    "transformers",
    include_py_files=True,
    excludes=["**/*.pyi"],
):
    rel_parts = Path(src_path).relative_to(tf_root).parts
    if (
        len(rel_parts) >= 3
        and rel_parts[0] == "models"
        and rel_parts[1] not in keep_tf_models
    ):
        continue
    datas.append((src_path, dst_dir))

# 6. Copy dist-info metadata queried at runtime by transformers, huggingface_hub, img2pdf, and pywebview
#    NOTE: "pillow" is REQUIRED so transformers.utils.import_utils.is_vision_available() returns True.
for meta_pkg in (
    "pillow",
    "torch",
    "transformers",
    "huggingface_hub",
    "tokenizers",
    "safetensors",
    "tqdm",
    "regex",
    "requests",
    "packaging",
    "filelock",
    "numpy",
    "PyYAML",
    "fsspec",
    "manga-ocr",
    "fugashi",
    "unidic-lite",
    "jaconv",
    "rapidocr-onnxruntime",
    "onnxruntime",
    "opencv-python",
    "opencv-python-headless",
    "pyclipper",
    "shapely",
    "sympy",
    "mpmath",
    "networkx",
    "Jinja2",
    "MarkupSafe",
    "loguru",
    "openai",
    "httpx",
    "httpcore",
    "anyio",
    "urllib3",
    "certifi",
    "charset-normalizer",
    "idna",
    "six",
    "typing_extensions",
    "colorama",
    "fastapi",
    "starlette",
    "uvicorn",
    "pydantic",
    "pydantic_core",
    "python-multipart",
    "img2pdf",
    "pikepdf",
    "pywebview",
    "pythonnet",
    "clr-loader",
    "cffi",
    "pycparser",
    "bottle",
    "proxy_tools",
):
    try:
        datas.extend(copy_metadata(meta_pkg))
    except Exception:
        pass

a = Analysis(
    [ENTRY_SCRIPT],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=list(dict.fromkeys(hiddenimports)),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Heavy unused scientific/notebook/GUI/test frameworks
        "matplotlib",
        "scipy",
        "tkinter",
        "pytest",
        "IPython",
        "ipython",
        "notebook",
        "jupyter",
        "jupyter_client",
        "jupyter_core",
        "ipykernel",
        "ipywidgets",
        "pandas",
        "sklearn",
        # Non-Windows pywebview backends
        "PyQt5",
        "PyQt6",
        "PySide2",
        "PySide6",
        "qtpy",
        "gi",
        "objc",
        "AppKit",
        "Foundation",
        "WebKit",
        "android",
        "jnius",
        # Collected directly into `_internal/` above for fast builds & minimal size
        "torch",
        "torchgen",
        "functorch",
        "transformers",
        "sympy",
        "mpmath",
        "networkx",
        "safetensors",
        "tokenizers",
        "huggingface_hub",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MangaFlow",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=ICON_PATH,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="MangaFlow_App",
)
