"""Model Manager for verifying and downloading required ONNX and OCR weights from Hugging Face Hub.

Manages local storage in the root `models/` directory:
1. Comic Text Detector ONNX (`models/comictextdetector.pt.onnx`)
2. LaMa Inpainting ONNX (`models/lama_fp32.onnx`)
3. Korean & Chinese PP-OCRv4 ONNX recognition/detection weights (`models/rapidocr/*.onnx`)
4. Japanese `kha-white/manga-ocr-base` weights and tokenizer (`models/manga-ocr-base/*`)

Once downloaded, all status checks and path resolutions run 100% offline with zero network calls.
"""

import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
import shutil
import sys
from typing import Literal, NotRequired, Protocol, TypedDict, cast

import onnxruntime as ort  # pyright: ignore[reportMissingTypeStubs]
import requests
from huggingface_hub import hf_hub_url
from huggingface_hub.utils._headers import (
    build_hf_headers,  # pyright: ignore[reportUnknownVariableType]
)


DownloadStatus = Literal[
    "skipped",
    "starting",
    "downloading",
    "file_completed",
    "completed",
]


class ModelGroupMeta(TypedDict):
    """Static metadata for a logical model group."""

    name: str
    size_mb: int


class ModelGroupStatus(TypedDict):
    """Status entry for a single model group."""

    installed: bool
    size_mb: int
    downloaded_mb: float
    files: list[str]
    missing_files: list[str]


class ModelStatusResponse(TypedDict):
    """Complete model verification response matching SPEC.md Section 5.4."""

    all_ready: bool
    execution_provider: str
    models_dir: str
    models: dict[str, ModelGroupStatus]


class RapidOcrPaths(TypedDict):
    """Resolved ONNX file paths for RapidOCR initialization."""

    det_model_path: Path
    rec_model_path: Path
    cls_model_path: Path


class DownloadProgressEvent(TypedDict):
    """Real-time progress payload yielded during model downloads."""

    status: DownloadStatus
    model_key: str
    file: str | None
    file_percent: float
    overall_percent: float
    downloaded_bytes: int
    total_bytes: int
    offline: bool
    message: str
    repo_id: NotRequired[str]
    file_index: NotRequired[int]
    file_count: NotRequired[int]
    file_downloaded_bytes: NotRequired[int]
    file_total_bytes: NotRequired[int]


@dataclass(frozen=True, slots=True)
class ModelFileSpec:
    """Specification for a single required model file on Hugging Face Hub."""

    model_key: str
    repo_id: str
    remote_filename: str
    local_rel_path: str
    expected_bytes: int
    min_valid_bytes: int


# Verified active Hugging Face Hub repositories and exact filenames
MODEL_FILE_SPECS: list[ModelFileSpec] = [
    # 1. Comic Text Detector ONNX (~90.28 MB)
    ModelFileSpec(
        model_key="comic_text_detector",
        repo_id="mayocream/comic-text-detector-onnx",
        remote_filename="comic-text-detector.onnx",
        local_rel_path="comictextdetector.pt.onnx",
        expected_bytes=94_662_990,
        min_valid_bytes=50_000_000,
    ),
    # 2. LaMa Inpainting ONNX (~198.41 MB)
    ModelFileSpec(
        model_key="lama_inpaint",
        repo_id="Carve/LaMa-ONNX",
        remote_filename="lama_fp32.onnx",
        local_rel_path="lama_fp32.onnx",
        expected_bytes=208_049_867,
        min_valid_bytes=100_000_000,
    ),
    # 3. Korean & Chinese PP-OCRv4 ONNX weights for rapidocr_onnxruntime (~40.70 MB total)
    ModelFileSpec(
        model_key="rapid_ocr_ko_zh",
        repo_id="pitapo/rapidocr",
        remote_filename="onnx/PP-OCRv4/det/ch_PP-OCRv4_det_infer.onnx",
        local_rel_path="rapidocr/ch_PP-OCRv4_det_infer.onnx",
        expected_bytes=4_747_796,
        min_valid_bytes=2_000_000,
    ),
    ModelFileSpec(
        model_key="rapid_ocr_ko_zh",
        repo_id="pitapo/rapidocr",
        remote_filename="onnx/PP-OCRv4/det/Multilingual_PP-OCRv3_det_infer.onnx",
        local_rel_path="rapidocr/Multilingual_PP-OCRv3_det_infer.onnx",
        expected_bytes=2_423_876,
        min_valid_bytes=1_000_000,
    ),
    ModelFileSpec(
        model_key="rapid_ocr_ko_zh",
        repo_id="pitapo/rapidocr",
        remote_filename="onnx/PP-OCRv4/rec/ch_PP-OCRv4_rec_infer.onnx",
        local_rel_path="rapidocr/ch_PP-OCRv4_rec_infer.onnx",
        expected_bytes=10_852_784,
        min_valid_bytes=5_000_000,
    ),
    ModelFileSpec(
        model_key="rapid_ocr_ko_zh",
        repo_id="pitapo/rapidocr",
        remote_filename="onnx/PP-OCRv4/rec/korean_PP-OCRv4_rec_infer.onnx",
        local_rel_path="rapidocr/korean_PP-OCRv4_rec_infer.onnx",
        expected_bytes=24_065_974,
        min_valid_bytes=10_000_000,
    ),
    ModelFileSpec(
        model_key="rapid_ocr_ko_zh",
        repo_id="pitapo/rapidocr",
        remote_filename="onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_infer.onnx",
        local_rel_path="rapidocr/ch_ppocr_mobile_v2.0_cls_infer.onnx",
        expected_bytes=585_540,
        min_valid_bytes=100_000,
    ),
    # 4. Japanese manga-ocr (kha-white/manga-ocr-base) local pre-cache (~423.65 MB total)
    ModelFileSpec(
        model_key="manga_ocr",
        repo_id="kha-white/manga-ocr-base",
        remote_filename="config.json",
        local_rel_path="manga-ocr-base/config.json",
        expected_bytes=75_610,
        min_valid_bytes=100,
    ),
    ModelFileSpec(
        model_key="manga_ocr",
        repo_id="kha-white/manga-ocr-base",
        remote_filename="preprocessor_config.json",
        local_rel_path="manga-ocr-base/preprocessor_config.json",
        expected_bytes=228,
        min_valid_bytes=50,
    ),
    ModelFileSpec(
        model_key="manga_ocr",
        repo_id="kha-white/manga-ocr-base",
        remote_filename="special_tokens_map.json",
        local_rel_path="manga-ocr-base/special_tokens_map.json",
        expected_bytes=112,
        min_valid_bytes=20,
    ),
    ModelFileSpec(
        model_key="manga_ocr",
        repo_id="kha-white/manga-ocr-base",
        remote_filename="tokenizer_config.json",
        local_rel_path="manga-ocr-base/tokenizer_config.json",
        expected_bytes=486,
        min_valid_bytes=50,
    ),
    ModelFileSpec(
        model_key="manga_ocr",
        repo_id="kha-white/manga-ocr-base",
        remote_filename="vocab.txt",
        local_rel_path="manga-ocr-base/vocab.txt",
        expected_bytes=24_130,
        min_valid_bytes=1_000,
    ),
    ModelFileSpec(
        model_key="manga_ocr",
        repo_id="kha-white/manga-ocr-base",
        remote_filename="pytorch_model.bin",
        local_rel_path="manga-ocr-base/pytorch_model.bin",
        expected_bytes=444_135_113,
        min_valid_bytes=400_000_000,
    ),
]

MODEL_GROUPS_META: dict[str, ModelGroupMeta] = {
    "comic_text_detector": {
        "name": "Comic Text Detector (ONNX)",
        "size_mb": 90,
    },
    "manga_ocr": {
        "name": "Manga OCR Base (Japanese)",
        "size_mb": 424,
    },
    "rapid_ocr_ko_zh": {
        "name": "RapidOCR PP-OCRv4 (Korean & Chinese ONNX)",
        "size_mb": 41,
    },
    "lama_inpaint": {
        "name": "LaMa Inpainting (ONNX)",
        "size_mb": 198,
    },
}


def is_frozen_executable() -> bool:
    """Return True when running inside a PyInstaller frozen executable (`comic-backend.exe`)."""
    return bool(getattr(sys, "frozen", False))


def get_default_models_dir() -> Path:
    """Resolve the persistent `models/` directory.

    Priority:
    1. `MANGA_MODELS_DIR` environment variable (if set).
    2. `%LOCALAPPDATA%/ComicTranslator/models` when running as a frozen PyInstaller executable
       so weights downloaded on first launch persist permanently for offline use.
    3. `<workspace_root>/models` when running from Python source during development.
    """
    env_dir = os.environ.get("MANGA_MODELS_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()

    if is_frozen_executable():
        local_app_data = os.environ.get("LOCALAPPDATA")
        base_dir = (
            Path(local_app_data).expanduser().resolve()
            if local_app_data
            else (Path.home() / "AppData" / "Local").resolve()
        )
        return base_dir / "ComicTranslator" / "models"

    workspace_root = Path(__file__).resolve().parents[2]
    return workspace_root / "models"


GPU_EXECUTION_PROVIDERS: tuple[str, ...] = (
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
)


def get_available_execution_providers() -> list[str]:
    """Return all ONNX Runtime execution providers registered in the current environment."""
    try:
        providers: list[str] = list(
            ort.get_available_providers()  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        )
        return providers if providers else ["CPUExecutionProvider"]
    except Exception:
        return ["CPUExecutionProvider"]


def get_preferred_execution_providers(force_cpu: bool = False) -> list[str]:
    """Return prioritized ONNX Runtime execution providers (`CUDA` -> `DirectML` -> `CPU`).

    Auto-detects GPU acceleration (`CUDAExecutionProvider` or `DmlExecutionProvider`
    via `onnxruntime-directml` when installed) and always ends with `CPUExecutionProvider`
    as a safe fallback.
    """
    if force_cpu or os.environ.get("MANGA_FORCE_CPU") == "1":
        return ["CPUExecutionProvider"]

    available = get_available_execution_providers()
    ordered: list[str] = []
    for gpu_provider in GPU_EXECUTION_PROVIDERS:
        if gpu_provider in available:
            ordered.append(gpu_provider)
    ordered.append("CPUExecutionProvider")
    return ordered


def get_execution_provider(force_cpu: bool = False) -> str:
    """Return the primary ONNX Runtime execution provider available on this machine."""
    return get_preferred_execution_providers(force_cpu=force_cpu)[0]


class _OrtSessionOptions(Protocol):
    """Typed protocol for `onnxruntime.SessionOptions` (pybind11 extension without stubs)."""

    graph_optimization_level: object
    enable_mem_pattern: bool
    execution_mode: object


class _OrtSessionOptionsFactory(Protocol):
    """Constructor protocol for `onnxruntime.SessionOptions`."""

    def __call__(self) -> _OrtSessionOptions: ...


class _OrtInferenceSessionFactory(Protocol):
    """Constructor protocol for `onnxruntime.InferenceSession`."""

    def __call__(
        self,
        path_or_bytes: str,
        sess_options: _OrtSessionOptions | None = None,
        providers: Sequence[str] | None = None,
    ) -> ort.InferenceSession: ...


def _build_session_options(use_dml: bool = False) -> _OrtSessionOptions:
    """Create ONNX Runtime SessionOptions tuned for RAM efficiency and DirectML compatibility."""
    session_options_cls = cast(
        _OrtSessionOptionsFactory,
        getattr(ort, "SessionOptions"),
    )
    sess_options = session_options_cls()

    opt_level_enum = cast(object, getattr(ort, "GraphOptimizationLevel"))
    sess_options.graph_optimization_level = cast(
        object,
        getattr(opt_level_enum, "ORT_ENABLE_ALL"),
    )

    if use_dml:
        # DirectML EP requires sequential execution and mem pattern disabled for stability
        sess_options.enable_mem_pattern = False
        exec_mode_enum = cast(object, getattr(ort, "ExecutionMode"))
        sess_options.execution_mode = cast(
            object,
            getattr(exec_mode_enum, "ORT_SEQUENTIAL"),
        )

    return sess_options


def create_onnx_session(
    model_path: Path | str,
    force_cpu: bool = False,
) -> tuple[ort.InferenceSession, str]:
    """Create an ONNX Runtime `InferenceSession` with automatic CUDA/DirectML -> CPU fallback.

    Returns `(session, active_provider)` where `active_provider` reflects the actual primary
    execution provider initialized on the created session (`session.get_providers()[0]`).
    """
    path_str = str(model_path)
    preferred = get_preferred_execution_providers(force_cpu=force_cpu)
    use_dml = "DmlExecutionProvider" in preferred
    session_cls = cast(
        _OrtInferenceSessionFactory,
        getattr(ort, "InferenceSession"),
    )

    try:
        sess_options = _build_session_options(use_dml=use_dml)
        session = session_cls(
            path_str,
            sess_options=sess_options,
            providers=preferred,
        )
        active_providers: list[str] = list(session.get_providers())
        active = active_providers[0] if active_providers else "CPUExecutionProvider"
        return session, active
    except Exception:
        if preferred == ["CPUExecutionProvider"]:
            raise
        cpu_options = _build_session_options(use_dml=False)
        session = session_cls(
            path_str,
            sess_options=cpu_options,
            providers=["CPUExecutionProvider"],
        )
        return session, "CPUExecutionProvider"


class ModelManager:
    """Manages local verification, path resolution, and streaming downloads of AI model weights."""

    models_dir: Path

    def __init__(self, models_dir: Path | str | None = None) -> None:
        self.models_dir = (
            Path(models_dir).expanduser().resolve()
            if models_dir is not None
            else get_default_models_dir()
        )
        self.models_dir.mkdir(parents=True, exist_ok=True)
        if models_dir is None and is_frozen_executable():
            self._seed_persistent_dir_from_local_models()

    def _seed_persistent_dir_from_local_models(self) -> None:
        """If running as a frozen executable and a sibling/workspace `models/` folder exists,
        seed any missing weights into `%LOCALAPPDATA%/ComicTranslator/models` so offline weights
        are immediately available without re-downloading.
        """
        if self.is_all_ready():
            return

        exe_path = Path(sys.executable).resolve()
        candidate_dirs = [
            exe_path.parent / "models",
            exe_path.parents[1] / "models" if len(exe_path.parents) > 1 else None,
            exe_path.parents[2] / "models" if len(exe_path.parents) > 2 else None,
        ]
        for candidate in candidate_dirs:
            if candidate is None or not candidate.is_dir() or candidate == self.models_dir:
                continue
            for spec in self.get_missing_specs():
                src_file = candidate / spec.local_rel_path
                if src_file.is_file() and src_file.stat().st_size >= spec.min_valid_bytes:
                    dst_file = self.resolve_path(spec.local_rel_path)
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        _ = shutil.copy2(src_file, dst_file)
                    except OSError:
                        pass
            if self.is_all_ready():
                break

    def resolve_path(self, local_rel_path: str) -> Path:
        """Resolve a model file's relative path inside the root `models/` directory."""
        return self.models_dir / local_rel_path

    def is_file_installed(self, spec: ModelFileSpec) -> bool:
        """Check offline whether a specific model file exists and meets its minimum valid byte size."""
        target = self.resolve_path(spec.local_rel_path)
        if not target.is_file():
            return False
        try:
            return target.stat().st_size >= spec.min_valid_bytes
        except OSError:
            return False

    def get_missing_specs(self) -> list[ModelFileSpec]:
        """Return the list of model file specs that are missing or incomplete on disk."""
        return [spec for spec in MODEL_FILE_SPECS if not self.is_file_installed(spec)]

    def is_all_ready(self) -> bool:
        """Return True if all required model files are present and valid on disk (100% offline check)."""
        return len(self.get_missing_specs()) == 0

    def get_status(self) -> ModelStatusResponse:
        """Return detailed offline status for all model groups matching SPEC.md Section 5.4."""
        groups_status: dict[str, ModelGroupStatus] = {}
        all_ready = True

        for group_key, meta in MODEL_GROUPS_META.items():
            group_specs = [s for s in MODEL_FILE_SPECS if s.model_key == group_key]
            missing = [s.local_rel_path for s in group_specs if not self.is_file_installed(s)]
            installed = len(missing) == 0
            if not installed:
                all_ready = False

            actual_bytes = sum(
                self.resolve_path(s.local_rel_path).stat().st_size
                for s in group_specs
                if self.is_file_installed(s)
            )
            groups_status[group_key] = {
                "installed": installed,
                "size_mb": meta["size_mb"],
                "downloaded_mb": round(actual_bytes / (1024 * 1024), 2),
                "files": [s.local_rel_path for s in group_specs],
                "missing_files": missing,
            }

        return {
            "all_ready": all_ready,
            "execution_provider": get_execution_provider(),
            "models_dir": str(self.models_dir),
            "models": groups_status,
        }

    def download_missing_models(
        self,
        chunk_size: int = 256 * 1024,
        timeout: int = 60,
    ) -> Iterator[DownloadProgressEvent]:
        """Verify local models and download any missing weights from Hugging Face Hub."""
        missing_specs = self.get_missing_specs()
        if not missing_specs:
            yield {
                "status": "skipped",
                "model_key": "all",
                "file": None,
                "file_percent": 100.0,
                "overall_percent": 100.0,
                "downloaded_bytes": 0,
                "total_bytes": 0,
                "offline": True,
                "message": "All required models are already present in models/. Running 100% offline.",
            }
            return

        total_expected_bytes = sum(s.expected_bytes for s in missing_specs)
        cumulative_downloaded = 0
        headers: dict[str, str] = build_hf_headers()

        yield {
            "status": "starting",
            "model_key": "all",
            "file": None,
            "file_percent": 0.0,
            "overall_percent": 0.0,
            "downloaded_bytes": 0,
            "total_bytes": total_expected_bytes,
            "offline": False,
            "message": (
                f"Downloading {len(missing_specs)} missing model file(s) "
                + f"(~{total_expected_bytes / (1024 * 1024):.1f} MB)..."
            ),
        }

        with requests.Session() as session:
            for idx, spec in enumerate(missing_specs, start=1):
                target_path = self.resolve_path(spec.local_rel_path)
                target_path.parent.mkdir(parents=True, exist_ok=True)
                temp_path = target_path.with_suffix(target_path.suffix + ".part")

                url = hf_hub_url(
                    repo_id=spec.repo_id,
                    filename=spec.remote_filename,
                )

                with session.get(
                    url,
                    headers=headers,
                    stream=True,
                    timeout=timeout,
                    allow_redirects=True,
                ) as response:
                    response.raise_for_status()
                    content_length_header = response.headers.get("Content-Length")
                    file_total_bytes = (
                        int(content_length_header)
                        if content_length_header and content_length_header.isdigit()
                        else spec.expected_bytes
                    )

                    if file_total_bytes != spec.expected_bytes:
                        total_expected_bytes += file_total_bytes - spec.expected_bytes

                    file_downloaded = 0
                    last_reported_pct = -1.0

                    with open(temp_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if not chunk:
                                continue
                            _ = f.write(chunk)
                            chunk_len = len(chunk)
                            file_downloaded += chunk_len
                            cumulative_downloaded += chunk_len

                            file_pct = (
                                min(100.0, (file_downloaded / file_total_bytes) * 100.0)
                                if file_total_bytes > 0
                                else 100.0
                            )
                            overall_pct = (
                                min(100.0, (cumulative_downloaded / total_expected_bytes) * 100.0)
                                if total_expected_bytes > 0
                                else 100.0
                            )

                            if (
                                file_pct - last_reported_pct >= 1.0
                                or file_downloaded >= file_total_bytes
                            ):
                                last_reported_pct = file_pct
                                yield {
                                    "status": "downloading",
                                    "model_key": spec.model_key,
                                    "repo_id": spec.repo_id,
                                    "file": spec.local_rel_path,
                                    "file_index": idx,
                                    "file_count": len(missing_specs),
                                    "file_downloaded_bytes": file_downloaded,
                                    "file_total_bytes": file_total_bytes,
                                    "file_percent": round(file_pct, 1),
                                    "overall_percent": round(overall_pct, 1),
                                    "downloaded_bytes": cumulative_downloaded,
                                    "total_bytes": total_expected_bytes,
                                    "offline": False,
                                    "message": (
                                        f"[{idx}/{len(missing_specs)}] Downloading {spec.local_rel_path}: "
                                        + f"{file_pct:.1f}% ({file_downloaded / (1024 * 1024):.1f}/"
                                        + f"{file_total_bytes / (1024 * 1024):.1f} MB)"
                                    ),
                                }

                actual_size = temp_path.stat().st_size
                if actual_size < spec.min_valid_bytes:
                    temp_path.unlink(missing_ok=True)
                    raise RuntimeError(
                        f"Downloaded file {spec.local_rel_path} is incomplete "
                        + f"({actual_size} bytes < {spec.min_valid_bytes} min bytes)."
                    )

                os.replace(temp_path, target_path)

                yield {
                    "status": "file_completed",
                    "model_key": spec.model_key,
                    "repo_id": spec.repo_id,
                    "file": spec.local_rel_path,
                    "file_index": idx,
                    "file_count": len(missing_specs),
                    "file_percent": 100.0,
                    "overall_percent": round(
                        min(
                            100.0,
                            (cumulative_downloaded / max(1, total_expected_bytes)) * 100.0,
                        ),
                        1,
                    ),
                    "downloaded_bytes": cumulative_downloaded,
                    "total_bytes": total_expected_bytes,
                    "offline": False,
                    "message": (
                        f"Saved {spec.local_rel_path} ({actual_size / (1024 * 1024):.2f} MB)"
                    ),
                }

        yield {
            "status": "completed",
            "model_key": "all",
            "file": None,
            "file_percent": 100.0,
            "overall_percent": 100.0,
            "downloaded_bytes": cumulative_downloaded,
            "total_bytes": total_expected_bytes,
            "offline": False,
            "message": "All required models downloaded and verified in models/.",
        }

    def get_detector_path(self) -> Path:
        """Return the path to `models/comictextdetector.pt.onnx`."""
        return self.resolve_path("comictextdetector.pt.onnx")

    def get_lama_path(self) -> Path:
        """Return the path to `models/lama_fp32.onnx`."""
        return self.resolve_path("lama_fp32.onnx")

    def get_manga_ocr_dir(self) -> Path:
        """Return the local directory `models/manga-ocr-base` containing `kha-white/manga-ocr-base`."""
        return self.resolve_path("manga-ocr-base")

    def get_rapidocr_paths(self, language: str = "zh") -> RapidOcrPaths:
        """Return local ONNX model paths (`det_model_path`, `rec_model_path`, `cls_model_path`) for RapidOCR."""
        lang = language.lower().strip()
        if lang == "ko":
            return {
                "det_model_path": self.resolve_path(
                    "rapidocr/Multilingual_PP-OCRv3_det_infer.onnx"
                ),
                "rec_model_path": self.resolve_path("rapidocr/korean_PP-OCRv4_rec_infer.onnx"),
                "cls_model_path": self.resolve_path(
                    "rapidocr/ch_ppocr_mobile_v2.0_cls_infer.onnx"
                ),
            }
        return {
            "det_model_path": self.resolve_path("rapidocr/ch_PP-OCRv4_det_infer.onnx"),
            "rec_model_path": self.resolve_path("rapidocr/ch_PP-OCRv4_rec_infer.onnx"),
            "cls_model_path": self.resolve_path("rapidocr/ch_ppocr_mobile_v2.0_cls_infer.onnx"),
        }


__all__ = [
    "DownloadProgressEvent",
    "DownloadStatus",
    "GPU_EXECUTION_PROVIDERS",
    "MODEL_FILE_SPECS",
    "MODEL_GROUPS_META",
    "ModelFileSpec",
    "ModelGroupMeta",
    "ModelGroupStatus",
    "ModelManager",
    "ModelStatusResponse",
    "RapidOcrPaths",
    "create_onnx_session",
    "get_available_execution_providers",
    "get_default_models_dir",
    "get_execution_provider",
    "get_preferred_execution_providers",
    "is_frozen_executable",
]
