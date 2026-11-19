# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller Specification for `comic-backend.exe`.

Bundles `backend/main.py` along with ONNX Runtime, RapidOCR (`rapidocr_onnxruntime`),
Japanese Manga-OCR (`manga_ocr`, `fugashi`, `unidic_lite`, `transformers`, `torch`),
and all `backend.core` modules into a single standalone Windows executable (`comic-backend.exe`).

At runtime, when `getattr(sys, "frozen", False)` is True, `backend.core.model_manager`
resolves the persistent `models/` directory to `%LOCALAPPDATA%/ComicTranslator/models`.
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
import transformers

SPEC_DIR = Path(SPECPATH).resolve()
PROJECT_ROOT = (
    SPEC_DIR.parents[1]
    if SPEC_DIR.name.lower() == "pyinstaller"
    else SPEC_DIR.parent
)
ENTRY_SCRIPT = str(PROJECT_ROOT / "backend" / "main.py")

datas: list[tuple[str, str]] = []
binaries: list[tuple[str, str]] = []

# Include all Windows-compatible Python standard library top-level modules so dynamically
# imported stdlib modules inside torch/transformers (e.g. graphlib, filecmp, pickletools) are always present.
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
    # Core backend modules
    "backend",
    "backend.main",
    "backend.core",
    "backend.core.model_manager",
    "backend.core.detector",
    "backend.core.ocr",
    "backend.core.inpainter",
    "backend.core.translator",
    # FastAPI & Uvicorn runtime modules
    "uvicorn",
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
    "starlette",
    "pydantic",
    "pydantic_core",
    "multipart",
    "python_multipart",
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

# 1. Collect ONNX Runtime, RapidOCR, Manga-OCR, Fugashi, and Unidic-Lite assets & binaries
for pkg_name in ("onnxruntime", "rapidocr_onnxruntime", "manga_ocr", "fugashi"):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg_name)
    datas.extend(pkg_datas)
    binaries.extend(pkg_binaries)
    hiddenimports.extend(pkg_hidden)

datas.extend(collect_data_files("unidic_lite"))
hiddenimports.extend(collect_submodules("unidic_lite"))
hiddenimports.extend(collect_submodules("backend"))

# Bundle compiled production frontend (`frontend/dist`) if available so standalone exe serves UI + API
frontend_dist_dir = PROJECT_ROOT / "frontend" / "dist"
if frontend_dist_dir.is_dir():
    datas.append((str(frontend_dist_dir), "frontend_dist"))

# 2. Collect PyTorch and its companion packages (.py files + .dll/.pyd dynamic libraries) directly
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

# 3. Collect Transformers (.py files for ViT + BERT Japanese + VisionEncoderDecoder needed by manga-ocr)
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
    if len(rel_parts) >= 3 and rel_parts[0] == "models" and rel_parts[1] not in keep_tf_models:
        continue
    datas.append((src_path, dst_dir))

# 4. Copy dist-info package metadata queried at runtime by transformers & huggingface_hub
#    NOTE: "pillow" is REQUIRED so transformers.utils.import_utils.is_vision_available() returns True!
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
        "pytest",
        "tkinter",
        "matplotlib",
        "scipy",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="comic-backend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
