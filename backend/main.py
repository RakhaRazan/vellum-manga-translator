"""FastAPI Backend Entry Point for Manga/Manhwa/Manhua Comic Translator (v2).

Implements SPEC.md v2 API contracts + Multi-Provider Engine & Persistent Settings:
- `GET /api/providers` & `POST /api/providers/test`: Multi-provider catalog (`lmstudio`, `ollama`, `openai`, `gemini`, `anthropic`, `deepseek`, `openrouter`, `custom`) and live credential/model verification.
- `GET /api/languages`: Supported source (`ja`/`ko`/`zh`) and 16 target languages.
- `GET /api/settings` & `POST /api/settings`: Persistent configuration stored at `%LOCALAPPDATA%/ComicTranslator/settings.json` (frozen) or `backend/settings.json` (dev).
- `GET /api/models/status`: Verify local ONNX & OCR model weights in `models/` + LM Studio status.
- `GET /api/models/download` (and `/api/models/download-stream`): SSE stream of real-time download progress.
- `GET /api/lmstudio/status` (and `/api/health`): Ping LM Studio and report connection status, execution provider, active LLM provider, and active model.
- `GET /api/cache/images/{image_id}.png`: Serve cached inpainted/mask PNG images via `FileResponse` (zero Base64 lag).
- `POST /api/stage/detect` (and `/api/pipeline/detect`): Single-stage Comic Text Detector returning sorted bounding boxes, reading order (`rtl`/`ltr`), and cached binary mask.
- `POST /api/stage/ocr` (and `/api/pipeline/ocr`): Single-stage Multi-Language OCR (`manga-ocr` for `ja`, `RapidOCR` PP-OCRv4 for `ko`/`zh`) on selected bboxes.
- `POST /api/stage/inpaint` (and `/api/pipeline/inpaint`): Single-stage tile-based LaMa inpainting on selected bboxes/mask returning cached PNG URL.
- `POST /api/stage/translate`: Single-stage context-aware LLM translation accepting text blocks, optional page image (JSON or multipart), and full provider/sampling/reasoning/vision/glossary config.
- `POST /api/process-page` (and `/api/pipeline/process-page`): Full sequential pipeline (Detect -> Reading-Order Sort -> OCR -> LaMa Tile Inpaint -> Context-Aware LLM Translate) with per-stage ONNX/OCR session unloading.
- `POST /api/retranslate` (and `/api/pipeline/translate`): Re-translate existing dialogue blocks across any of the 16 target languages without re-running CV/OCR.
"""

import base64
from collections.abc import Iterator, Mapping, Sequence
import gc
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import sys
import tempfile
import traceback
from typing import Annotated, Literal, NotRequired, TypedDict, cast, override
import uuid

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field, ValidationError
from starlette.datastructures import UploadFile as StarletteUploadFile
from starlette.types import Scope
import uvicorn

# Guard against broken Windows registry MIME associations (e.g. `.js` mapped to `text/plain`)
# that would prevent WebView2 from executing ES module scripts served by FastAPI.
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("image/svg+xml", ".svg")
mimetypes.add_type("font/woff2", ".woff2")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.detector import (  # noqa: E402
    ComicTextDetector,
    DetectedBubble,
    ImageDimensions,
    NDArrayU8,
    TextDirection,
    load_rgb_image,
)
from backend.core.exporter import (  # noqa: E402
    ArchiveImportError,
    CACHEABLE_IMAGE_EXTENSIONS,
    ComicExportError,
    ExportImageFormat,
    ExportPageInput,
    build_cbz_bytes,
    build_pdf_bytes,
    extract_archive_pages,
    sanitize_export_filename,
)
from backend.core.font_manager import (  # noqa: E402
    FontCategory,
    FontDownloadResponseDict,
    FontEntryDict,
    FontListResponseDict,
    FontManager,
    FontValidationError,
)
from backend.core.inpainter import LamaInpainter, load_mask_image  # noqa: E402
from backend.core.model_manager import (  # noqa: E402
    ModelGroupStatus,
    ModelManager,
    get_available_execution_providers,
    get_execution_provider,
)
from backend.core.ocr import (  # noqa: E402
    OCRRouter,
    SourceLanguage,
    normalize_language,
)
from backend.core.providers import (  # noqa: E402
    AccentColorOption,
    AppSettingsDict,
    ExecutionModeOption,
    PROVIDER_PRESETS,
    ProviderId,
    ProviderInfoDict,
    ProviderProfileDict,
    ProviderTestResult,
    ReadingOrderOption,
    ReasoningEffort,
    SUPPORTED_REASONING_EFFORTS,
    SettingsStore,
    TargetLanguage,
    TargetLanguageInfoDict,
    TextAlignOption,
    ThemeOption,
    UiDensityOption,
    format_display_model_name,
    list_provider_presets,
    list_target_languages,
    normalize_provider_base_url,
    normalize_provider_id,
    normalize_reasoning_effort,
    normalize_target_language,
    parse_glossary,
    resolve_provider_api_key,
    test_provider_connection,
)
from backend.core.translator import (  # noqa: E402
    ComicTranslator,
    LMStudioConnectionError,
    LMStudioTranslationError,
    TranslatedDialogueItem,
    encode_image_for_vision,
    normalize_source_language,
)

TextAlign = Literal["left", "center", "right"]
ReadingOrder = Literal["rtl", "ltr"]


class BubbleStyleDict(TypedDict):
    """Typography and canvas styling attributes matching SPEC.md Section 5.1."""

    font_family: str
    font_size: int
    color: str
    stroke_color: str
    stroke_width: int
    align: TextAlign
    line_height: float


class ProcessedBlockItem(TypedDict):
    """Processed speech bubble block returned by `/api/process-page` and `/api/retranslate`."""

    id: str
    order: int
    bbox: list[int]
    polygon: list[list[int]]
    original_text: str
    source_text: str
    translated_text: str
    font_size: int
    color: str
    align: TextAlign
    confidence: float
    text_direction: TextDirection
    style: BubbleStyleDict


class LMStudioEndpointStatus(TypedDict):
    """Status payload returned by `GET /api/lmstudio/status`."""

    connected: bool
    endpoint: str
    active_model: str | None
    resolved_model_id: str | None
    preferred_model: str
    available_models: list[str]
    error: NotRequired[str]


class CombinedModelsStatusResponse(TypedDict):
    """Status payload returned by `GET /api/models/status` matching SPEC.md Section 5.5."""

    all_ready: bool
    execution_provider: str
    models_dir: str
    models: dict[str, ModelGroupStatus]
    lm_studio: LMStudioEndpointStatus


class ProcessPageResponse(TypedDict):
    """Complete page processing response returned by `POST /api/process-page`."""

    page_id: str
    dimensions: ImageDimensions
    source_lang: SourceLanguage
    target_lang: TargetLanguage
    source_language: SourceLanguage
    target_language: TargetLanguage
    provider: ProviderId
    model: str
    vision_used: bool
    reasoning_effort: ReasoningEffort
    execution_provider: str
    tiles_inpainted: int
    cleaned_image_id: str
    cleaned_image_url: str
    inpainted_image_url: str
    cleaned_image: str
    inpainted_image: str
    mask_id: NotRequired[str]
    mask_url: NotRequired[str]
    blocks: list[ProcessedBlockItem]
    bubbles: list[ProcessedBlockItem]


class RetranslateResponse(TypedDict):
    """Response payload returned by `POST /api/retranslate`."""

    provider: ProviderId
    model: str
    vision_used: bool
    reasoning_effort: ReasoningEffort
    source_lang: SourceLanguage
    target_lang: TargetLanguage
    source_language: SourceLanguage
    target_language: TargetLanguage
    translations: dict[str, str]
    dialogues: list[TranslatedDialogueItem]
    blocks: list[ProcessedBlockItem]
    bubbles: list[ProcessedBlockItem]


class RetranslateBlockModel(BaseModel):
    """Pydantic input model for a single text block in `POST /api/retranslate`."""

    id: str
    bbox: list[int] = Field(default_factory=lambda: [0, 0, 160, 80])
    polygon: list[list[int]] | None = None
    order: int | None = None
    original_text: str | None = None
    source_text: str | None = None
    text: str | None = None
    translated_text: str | None = None
    font_size: int | None = None
    color: str = "#000000"
    align: TextAlign = "center"
    confidence: float = 1.0
    text_direction: TextDirection = "horizontal"


class TranslationConfigModel(BaseModel):
    """Optional multi-provider LLM translation overrides accepted by translate/retranslate/process-page."""

    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=0, le=500)
    max_tokens: int | None = Field(default=None, ge=64, le=32768)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    reasoning_effort: str | None = None
    timeout: float | None = Field(default=None, ge=5.0, le=300.0)
    custom_instructions: str | None = None
    glossary: dict[str, str] | list[dict[str, str]] | str | None = None
    use_vision: bool | None = None
    vision_max_side: int | None = Field(default=None, ge=256, le=2048)


class RetranslateRequest(BaseModel):
    """Pydantic request body for `POST /api/retranslate`."""

    source_lang: str = "ja"
    target_lang: str = "id"
    source_language: str | None = None
    target_language: str | None = None
    blocks: list[RetranslateBlockModel] = Field(default_factory=list)
    bubbles: list[RetranslateBlockModel] = Field(default_factory=list)
    dialogues: list[RetranslateBlockModel] = Field(default_factory=list)
    image_url: str | None = None
    image_base64: str | None = None
    custom_instructions: str | None = None
    glossary: dict[str, str] | list[dict[str, str]] | str | None = None
    use_vision: bool | None = None
    config: TranslationConfigModel | None = None


class ProviderTestRequest(BaseModel):
    """Request body for `POST /api/providers/test`."""

    provider: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    timeout: float = Field(default=10.0, ge=1.0, le=60.0)


class ProviderProfilePatchModel(BaseModel):
    """Per-provider saved profile patch in `POST /api/settings`."""

    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None


class SettingsUpdateRequest(BaseModel):
    """Partial or full settings payload accepted by `POST /api/settings`."""

    provider: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    timeout: float | None = Field(default=None, ge=5.0, le=300.0)
    source_lang: str | None = None
    target_lang: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=0, le=500)
    max_tokens: int | None = Field(default=None, ge=64, le=32768)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    reasoning_effort: str | None = None
    strip_thinking: bool | None = None
    use_vision: bool | None = None
    vision_max_side: int | None = Field(default=None, ge=256, le=2048)
    custom_instructions: str | None = None
    glossary: dict[str, str] | list[dict[str, str]] | str | None = None
    providers: dict[str, ProviderProfilePatchModel] | None = None
    # Tab 1: Appearance
    theme: str | None = None
    accent_color: str | None = None
    ui_density: str | None = None
    # Tab 2: Pipeline
    execution_mode: str | None = None
    conf_threshold: float | None = Field(default=None, ge=0.05, le=0.95)
    nms_threshold: float | None = Field(default=None, ge=0.05, le=0.95)
    mask_threshold: float | None = Field(default=None, ge=0.05, le=0.95)
    bbox_padding: int | None = Field(default=None, ge=0, le=64)
    lama_tile_size: int | None = Field(default=None, ge=256, le=1024)
    reading_order: str | None = None
    auto_unload_vram: bool | None = None
    refine_mask: bool | None = None
    # Tab 5: Typesetting
    font_family: str | None = None
    font_fallback_chain: list[str] | None = None
    auto_fit_font_size: bool | None = None
    default_font_size: int | None = Field(default=None, ge=8, le=96)
    default_stroke_width: float | None = Field(default=None, ge=0.0, le=16.0)
    default_stroke_color: str | None = None
    default_text_color: str | None = None
    default_line_height: float | None = Field(default=None, ge=0.8, le=2.5)
    default_align: str | None = None
    all_caps: bool | None = None


class SettingsResponse(TypedDict):
    """Response payload returned by `GET /api/settings` and `POST /api/settings`."""

    provider: ProviderId
    base_url: str
    api_key: str
    model: str
    timeout: float
    source_lang: str
    target_lang: TargetLanguage
    temperature: float
    top_p: float | None
    top_k: int | None
    max_tokens: int
    frequency_penalty: float | None
    reasoning_effort: ReasoningEffort
    strip_thinking: bool
    use_vision: bool
    vision_max_side: int
    custom_instructions: str
    glossary: dict[str, str]
    providers: dict[str, ProviderProfileDict]
    theme: ThemeOption
    accent_color: AccentColorOption
    ui_density: UiDensityOption
    execution_mode: ExecutionModeOption
    conf_threshold: float
    nms_threshold: float
    mask_threshold: float
    bbox_padding: int
    lama_tile_size: int
    reading_order: ReadingOrderOption
    auto_unload_vram: bool
    refine_mask: bool
    font_family: str
    font_fallback_chain: list[str]
    auto_fit_font_size: bool
    default_font_size: int
    default_stroke_width: float
    default_stroke_color: str
    default_text_color: str
    default_line_height: float
    default_align: TextAlignOption
    all_caps: bool
    settings_path: str
    available_providers: list[ProviderInfoDict]
    supported_target_languages: list[TargetLanguageInfoDict]
    supported_reasoning_efforts: list[ReasoningEffort]


class StageDetectResponse(TypedDict):
    """Response payload returned by `POST /api/stage/detect`."""

    page_id: str
    dimensions: ImageDimensions
    source_lang: SourceLanguage
    source_language: SourceLanguage
    reading_order: ReadingOrder
    execution_provider: str
    mask_id: str
    mask_url: str
    mask: str
    mask_base64: NotRequired[str]
    ordered_ids: list[str]
    bubbles: list[DetectedBubble]
    blocks: list[DetectedBubble]


class StageOcrBlockItem(TypedDict):
    """Single OCR-extracted text block returned by `POST /api/stage/ocr`."""

    id: str
    order: int
    bbox: list[int]
    polygon: list[list[int]]
    original_text: str
    source_text: str
    confidence: float
    text_direction: TextDirection


class StageOcrResponse(TypedDict):
    """Response payload returned by `POST /api/stage/ocr`."""

    source_lang: SourceLanguage
    source_language: SourceLanguage
    engine: str
    count: int
    texts: list[str]
    blocks: list[StageOcrBlockItem]
    bubbles: list[StageOcrBlockItem]


class StageInpaintResponse(TypedDict):
    """Response payload returned by `POST /api/stage/inpaint`."""

    dimensions: ImageDimensions
    execution_provider: str
    tiles_inpainted: int
    cleaned_image_id: str
    cleaned_image_url: str
    inpainted_image_url: str
    cleaned_image: str
    inpainted_image: str


class StageTranslateRequest(BaseModel):
    """JSON request body for `POST /api/stage/translate`."""

    source_lang: str = "ja"
    target_lang: str = "id"
    source_language: str | None = None
    target_language: str | None = None
    blocks: list[RetranslateBlockModel] = Field(default_factory=list)
    bubbles: list[RetranslateBlockModel] = Field(default_factory=list)
    dialogues: list[RetranslateBlockModel] = Field(default_factory=list)
    image_url: str | None = None
    image_base64: str | None = None
    config: TranslationConfigModel | None = None


class StageTranslateResponse(TypedDict):
    """Response payload returned by `POST /api/stage/translate`."""

    provider: ProviderId
    model: str
    vision_used: bool
    reasoning_effort: ReasoningEffort
    source_lang: SourceLanguage
    target_lang: TargetLanguage
    source_language: SourceLanguage
    target_language: TargetLanguage
    has_page_image: bool
    page_image_dimensions: ImageDimensions | None
    translations: dict[str, str]
    dialogues: list[TranslatedDialogueItem]
    blocks: list[ProcessedBlockItem]
    bubbles: list[ProcessedBlockItem]


class ImportedArchivePageResponse(TypedDict):
    """Single extracted comic page returned by `POST /api/import/archive`."""

    index: int
    filename: str
    archive_path: str
    image_id: str
    image_url: str
    dimensions: ImageDimensions
    size_bytes: int


class ImportArchiveResponse(TypedDict):
    """Response payload returned by `POST /api/import/archive`."""

    archive_name: str
    total_pages: int
    pages: list[ImportedArchivePageResponse]


class ExportPageItemModel(BaseModel):
    """Single page image specification in JSON `POST /api/export/cbz` or `/api/export/pdf`."""

    filename: str | None = None
    image: str | None = None
    url: str | None = None
    image_url: str | None = None
    image_id: str | None = None


class ExportPagesRequest(BaseModel):
    """JSON request body for `POST /api/export/cbz` and `POST /api/export/pdf`."""

    filename: str | None = None
    title: str | None = None
    image_format: ExportImageFormat = "original"
    jpeg_quality: int = Field(default=95, ge=50, le=100)
    pages: list[str | ExportPageItemModel] = Field(default_factory=list)
    images: list[str | ExportPageItemModel] = Field(default_factory=list)


def _json_loads(raw: str) -> object:
    """Parse a JSON string into a typed `object`."""
    res: object = json.loads(raw)  # pyright: ignore[reportAny]
    return res


def _coerce_int(val: object, field_name: str) -> int:
    """Safely convert a JSON numeric scalar into an `int`."""
    if isinstance(val, bool):
        raise ValueError(f"Expected integer for '{field_name}', got boolean.")
    if isinstance(val, (int, float)):
        return int(round(val))
    if isinstance(val, str):
        return int(round(float(val)))
    raise ValueError(f"Expected integer for '{field_name}', got {val!r}.")


def _coerce_float(val: object, default: float) -> float:
    """Safely convert a JSON numeric scalar into a `float`."""
    if isinstance(val, bool):
        return default
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val)
        except ValueError:
            return default
    return default


def _parse_bool_flag(raw: object, default: bool = False) -> bool:
    """Parse common form boolean values (`true`/`1`/`yes` vs `false`/`0`/`no`)."""
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return default
    text = str(raw).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    return default


def _parse_bbox_coords(raw_box: object) -> list[int]:
    """Validate and normalize a `[x1, y1, x2, y2]` bounding box array."""
    if not isinstance(raw_box, Sequence) or isinstance(raw_box, (str, bytes)):
        raise ValueError(f"Expected 4-element bbox array [x1, y1, x2, y2], got {raw_box!r}")
    if len(raw_box) < 4:
        raise ValueError(f"Expected at least 4 coordinates in bbox, got {raw_box!r}")
    x1 = _coerce_int(raw_box[0], "x1")
    y1 = _coerce_int(raw_box[1], "y1")
    x2 = _coerce_int(raw_box[2], "x2")
    y2 = _coerce_int(raw_box[3], "y2")
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def _parse_stage_bubbles_json(raw_json: str) -> list[DetectedBubble]:
    """Parse JSON array of `[x1, y1, x2, y2]` lists or bubble dicts into `list[DetectedBubble]`."""
    try:
        parsed = _json_loads(raw_json)
    except ValueError as exc:
        raise ValueError(f"Invalid JSON in bboxes/blocks field: {exc}") from exc

    if isinstance(parsed, Mapping):
        parsed_map = cast(Mapping[str, object], parsed)
        for key in ("bboxes", "blocks", "bubbles"):
            candidate = parsed_map.get(key)
            if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
                parsed = candidate
                break

    if not isinstance(parsed, Sequence) or isinstance(parsed, (str, bytes)):
        raise ValueError("Expected a JSON array of bounding boxes or bubble objects.")

    bubbles: list[DetectedBubble] = []
    for idx, entry in enumerate(cast(Sequence[object], parsed), start=1):
        if isinstance(entry, Mapping):
            entry_map = cast(Mapping[str, object], entry)
            bbox = _parse_bbox_coords(entry_map.get("bbox"))
            raw_id = entry_map.get("id")
            bid = (
                str(raw_id).strip()
                if raw_id is not None and str(raw_id).strip()
                else f"bubble_{idx:03d}"
            )
            raw_order = entry_map.get("order")
            order_num = _coerce_int(raw_order, "order") if raw_order is not None else idx
            confidence = _coerce_float(entry_map.get("confidence"), 1.0)

            raw_dir = str(entry_map.get("text_direction", "")).strip().lower()
            if raw_dir == "vertical":
                direction: TextDirection = "vertical"
            elif raw_dir == "horizontal":
                direction = "horizontal"
            else:
                bw = max(1, bbox[2] - bbox[0])
                bh = max(1, bbox[3] - bbox[1])
                direction = "vertical" if bh > bw * 1.2 else "horizontal"

            raw_poly = entry_map.get("polygon")
            polygon: list[list[int]] = []
            if isinstance(raw_poly, Sequence) and not isinstance(raw_poly, (str, bytes)):
                for pt in raw_poly:
                    if isinstance(pt, Sequence) and not isinstance(pt, (str, bytes)):
                        if len(pt) >= 2:
                            polygon.append(
                                [_coerce_int(pt[0], "px"), _coerce_int(pt[1], "py")]
                            )
            if len(polygon) < 4:
                polygon = _bbox_to_polygon(bbox)

            bubbles.append(
                {
                    "id": bid,
                    "order": order_num,
                    "bbox": bbox,
                    "polygon": polygon,
                    "confidence": confidence,
                    "text_direction": direction,
                }
            )
        else:
            bbox = _parse_bbox_coords(entry)
            bw = max(1, bbox[2] - bbox[0])
            bh = max(1, bbox[3] - bbox[1])
            inferred_dir: TextDirection = "vertical" if bh > bw * 1.2 else "horizontal"
            bubbles.append(
                {
                    "id": f"bubble_{idx:03d}",
                    "order": idx,
                    "bbox": bbox,
                    "polygon": _bbox_to_polygon(bbox),
                    "confidence": 1.0,
                    "text_direction": inferred_dir,
                }
            )

    return bubbles


def _encode_page_image_for_vlm(rgb_image: NDArrayU8, max_side: int = 1024) -> str:
    """Resize an RGB comic page to `<= max_side` px and encode as a compact JPEG data URI."""
    return encode_image_for_vision(rgb_image, max_side=max_side)


def _estimate_font_size(bbox: Sequence[int], text: str) -> int:
    """Estimate an initial canvas font size (14-28 px) from bubble dimensions and character count."""
    if len(bbox) < 4:
        return 18
    bw = max(24, int(bbox[2]) - int(bbox[0]))
    bh = max(24, int(bbox[3]) - int(bbox[1]))
    usable_area = (bw * 0.82) * (bh * 0.78)
    char_count = max(1, len(text.strip()))
    estimated = int(math.sqrt(usable_area / (char_count * 0.85)))
    return max(14, min(28, estimated))


def _build_default_style(
    font_size: int,
    color: str = "#000000",
    align: TextAlign = "center",
) -> BubbleStyleDict:
    """Create a default `BubbleStyleDict` matching SPEC.md Section 5.1."""
    return {
        "font_family": "Anime Ace",
        "font_size": font_size,
        "color": color,
        "stroke_color": "#ffffff",
        "stroke_width": 2,
        "align": align,
        "line_height": 1.2,
    }


def _bbox_to_polygon(bbox: Sequence[int]) -> list[list[int]]:
    """Convert `[x1, y1, x2, y2]` into a 4-point clockwise polygon."""
    if len(bbox) < 4:
        return [[0, 0], [100, 0], [100, 50], [0, 50]]
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


MAX_CACHED_IMAGES = 500
_SAFE_IMAGE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_CACHE_EXT_TO_MEDIA_TYPE: dict[str, str] = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
}


def get_image_cache_dir() -> Path:
    """Resolve and ensure the local temporary cache directory for processed & imported images."""
    env_cache = os.environ.get("MANGA_CACHE_DIR")
    cache_dir = (
        Path(env_cache).expanduser().resolve()
        if env_cache
        else (Path(tempfile.gettempdir()) / "mangaflow_cache" / "images").resolve()
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _prune_image_cache(cache_dir: Path, max_files: int = MAX_CACHED_IMAGES) -> None:
    """Evict the oldest cached image files once the cache folder exceeds `max_files`."""
    try:
        cached_files = [
            p
            for p in cache_dir.iterdir()
            if p.is_file() and p.suffix.lower().lstrip(".") in CACHEABLE_IMAGE_EXTENSIONS
        ]
        if len(cached_files) <= max_files:
            return
        cached_files.sort(key=lambda p: p.stat().st_mtime)
        for stale in cached_files[: len(cached_files) - max_files]:
            try:
                stale.unlink(missing_ok=True)
            except OSError:
                pass
    except OSError:
        pass


def save_inpainted_image_to_cache(rgb_image: NDArrayU8) -> tuple[str, str]:
    """Save an inpainted RGB image array as PNG in the local cache and return `(image_id, url)`."""
    cache_dir = get_image_cache_dir()
    image_id = uuid.uuid4().hex
    final_path = cache_dir / f"{image_id}.png"
    tmp_path = cache_dir / f"{image_id}.png.tmp"

    try:
        Image.fromarray(rgb_image).save(tmp_path, format="PNG")
        os.replace(tmp_path, final_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    _prune_image_cache(cache_dir)
    return image_id, f"/api/cache/images/{image_id}.png"


def save_mask_image_to_cache(mask: NDArrayU8) -> tuple[str, str]:
    """Save a binary `(H, W)` uint8 text mask as PNG in the local cache and return `(mask_id, url)`."""
    cache_dir = get_image_cache_dir()
    mask_id = f"mask_{uuid.uuid4().hex}"
    final_path = cache_dir / f"{mask_id}.png"
    tmp_path = cache_dir / f"{mask_id}.png.tmp"

    try:
        Image.fromarray(mask).save(tmp_path, format="PNG")
        os.replace(tmp_path, final_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    _prune_image_cache(cache_dir)
    return mask_id, f"/api/cache/images/{mask_id}.png"


def resolve_cached_image_path(id_or_url: str) -> Path:
    """Resolve a cached image ID or `/api/cache/images/{id}.{ext}` URL to a verified local path."""
    cleaned = id_or_url.strip()
    prefix = "/api/cache/images/"
    if prefix in cleaned:
        cleaned = cleaned.split(prefix, 1)[1]
    if "?" in cleaned:
        cleaned = cleaned.split("?", 1)[0]

    requested_ext: str | None = None
    if "." in cleaned:
        stem, suffix = cleaned.rsplit(".", 1)
        lower_ext = suffix.lower()
        if lower_ext not in CACHEABLE_IMAGE_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported cached image extension: {suffix!r}",
            )
        cleaned = stem
        requested_ext = lower_ext

    if not _SAFE_IMAGE_ID_RE.fullmatch(cleaned):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid cached image identifier: {id_or_url!r}",
        )

    cache_root = get_image_cache_dir().resolve()
    exts_to_check = (
        (requested_ext,)
        if requested_ext is not None
        else ("png", "jpg", "jpeg", "webp")
    )
    for ext in exts_to_check:
        target_path = (cache_root / f"{cleaned}.{ext}").resolve()
        if target_path.parent == cache_root and target_path.is_file():
            return target_path

    raise HTTPException(
        status_code=404,
        detail=f"Cached image '{cleaned}' not found.",
    )


def _build_stage_translator(config: TranslationConfigModel | None) -> ComicTranslator:
    """Return the shared `ComicTranslator` or a request-scoped instance with multi-provider/control overrides."""
    base = pipeline_service.translator
    if config is None:
        return base

    has_override = any(
        val is not None
        for val in (
            config.provider,
            config.model,
            config.base_url,
            config.api_key,
            config.temperature,
            config.top_p,
            config.top_k,
            config.max_tokens,
            config.frequency_penalty,
            config.reasoning_effort,
            config.use_vision,
            config.vision_max_side,
            config.custom_instructions,
            config.glossary,
            config.timeout,
        )
    )
    if not has_override:
        return base

    try:
        target_provider: ProviderId = (
            normalize_provider_id(config.provider)
            if config.provider is not None and config.provider.strip()
            else base.provider
        )
        target_reasoning: ReasoningEffort = (
            normalize_reasoning_effort(config.reasoning_effort)
            if config.reasoning_effort is not None and config.reasoning_effort.strip()
            else base.reasoning_effort
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    switching_provider = target_provider != base.provider
    settings = pipeline_service.get_settings()
    saved_profile = settings["providers"].get(target_provider)
    preset = PROVIDER_PRESETS[target_provider]

    if config.base_url is not None and config.base_url.strip():
        resolved_url = normalize_provider_base_url(target_provider, config.base_url)
    elif switching_provider:
        resolved_url = normalize_provider_base_url(
            target_provider,
            saved_profile["base_url"] if saved_profile else preset.default_base_url,
        )
    else:
        resolved_url = base.base_url

    if config.api_key is not None and config.api_key.strip():
        resolved_key = config.api_key.strip()
    elif switching_provider:
        resolved_key = resolve_provider_api_key(
            target_provider,
            saved_profile["api_key"] if saved_profile else preset.default_api_key,
        )
    else:
        resolved_key = base.api_key

    if config.model is not None and config.model.strip():
        resolved_model = config.model.strip()
    elif switching_provider:
        resolved_model = (
            saved_profile["model"] if saved_profile and saved_profile["model"] else preset.default_model
        )
    else:
        resolved_model = base.preferred_model

    merged_glossary: dict[str, str] = dict(base.glossary)
    if config.glossary is not None:
        merged_glossary.update(parse_glossary(config.glossary))

    return ComicTranslator(
        provider=target_provider,
        base_url=resolved_url,
        api_key=resolved_key,
        preferred_model=resolved_model,
        temperature=config.temperature if config.temperature is not None else base.temperature,
        top_p=config.top_p if config.top_p is not None else base.top_p,
        top_k=config.top_k if config.top_k is not None else base.top_k,
        max_tokens=config.max_tokens if config.max_tokens is not None else base.max_tokens,
        frequency_penalty=(
            config.frequency_penalty
            if config.frequency_penalty is not None
            else base.frequency_penalty
        ),
        reasoning_effort=target_reasoning,
        use_vision=config.use_vision if config.use_vision is not None else base.use_vision,
        vision_max_side=(
            config.vision_max_side
            if config.vision_max_side is not None
            else base.vision_max_side
        ),
        custom_instructions=(
            config.custom_instructions
            if config.custom_instructions is not None
            else base.custom_instructions
        ),
        glossary=merged_glossary,
        timeout=config.timeout if config.timeout is not None else base.timeout,
    )


def _format_settings_response(settings: AppSettingsDict) -> SettingsResponse:
    """Attach read-only path, provider catalog, and language metadata to persisted settings."""
    return {
        "provider": settings["provider"],
        "base_url": settings["base_url"],
        "api_key": settings["api_key"],
        "model": settings["model"],
        "timeout": settings["timeout"],
        "source_lang": settings["source_lang"],
        "target_lang": settings["target_lang"],
        "temperature": settings["temperature"],
        "top_p": settings["top_p"],
        "top_k": settings["top_k"],
        "max_tokens": settings["max_tokens"],
        "frequency_penalty": settings["frequency_penalty"],
        "reasoning_effort": settings["reasoning_effort"],
        "strip_thinking": settings["strip_thinking"],
        "use_vision": settings["use_vision"],
        "vision_max_side": settings["vision_max_side"],
        "custom_instructions": settings["custom_instructions"],
        "glossary": dict(settings["glossary"]),
        "providers": settings["providers"],
        "theme": settings["theme"],
        "accent_color": settings["accent_color"],
        "ui_density": settings["ui_density"],
        "execution_mode": settings["execution_mode"],
        "conf_threshold": settings["conf_threshold"],
        "nms_threshold": settings["nms_threshold"],
        "mask_threshold": settings["mask_threshold"],
        "bbox_padding": settings["bbox_padding"],
        "lama_tile_size": settings["lama_tile_size"],
        "reading_order": settings["reading_order"],
        "auto_unload_vram": settings["auto_unload_vram"],
        "refine_mask": settings["refine_mask"],
        "font_family": settings["font_family"],
        "font_fallback_chain": list(settings["font_fallback_chain"]),
        "auto_fit_font_size": settings["auto_fit_font_size"],
        "default_font_size": settings["default_font_size"],
        "default_stroke_width": settings["default_stroke_width"],
        "default_stroke_color": settings["default_stroke_color"],
        "default_text_color": settings["default_text_color"],
        "default_line_height": settings["default_line_height"],
        "default_align": settings["default_align"],
        "all_caps": settings["all_caps"],
        "settings_path": str(pipeline_service.settings_store.path),
        "available_providers": list_provider_presets(),
        "supported_target_languages": list_target_languages(),
        "supported_reasoning_efforts": list(SUPPORTED_REASONING_EFFORTS),
    }


class PipelineService:
    """Manages lazy initialization, persistent settings, fonts, and per-page unloading of CV/OCR/Inpainting/LLM engines."""

    model_manager: ModelManager
    settings_store: SettingsStore
    font_manager: FontManager
    translator: ComicTranslator
    _detector: ComicTextDetector | None
    _ocr_router: OCRRouter | None
    _inpainter: LamaInpainter | None

    def __init__(self) -> None:
        self.model_manager = ModelManager()
        self.settings_store = SettingsStore()
        self.font_manager = FontManager()
        initial_settings = self.settings_store.load()
        self.translator = ComicTranslator.from_settings(initial_settings)
        self._detector = None
        self._ocr_router = None
        self._inpainter = None

    def get_settings(self, force_reload: bool = False) -> AppSettingsDict:
        """Return the current persisted application settings."""
        return self.settings_store.load(force_reload=force_reload)

    def update_settings(self, patch: Mapping[str, object]) -> AppSettingsDict:
        """Validate, atomically persist, and hot-reload the active `ComicTranslator`."""
        updated = self.settings_store.update(patch)
        self.translator = ComicTranslator.from_settings(updated)
        return updated

    def get_detector(self) -> ComicTextDetector:
        """Lazily initialize `ComicTextDetector`."""
        if self._detector is None:
            self._detector = ComicTextDetector(model_manager=self.model_manager)
        return self._detector

    def get_ocr_router(self) -> OCRRouter:
        """Lazily initialize `OCRRouter`."""
        if self._ocr_router is None:
            self._ocr_router = OCRRouter(model_manager=self.model_manager)
        return self._ocr_router

    def get_inpainter(self) -> LamaInpainter:
        """Lazily initialize `LamaInpainter`."""
        if self._inpainter is None:
            self._inpainter = LamaInpainter(model_manager=self.model_manager)
        return self._inpainter

    def unload_all(self) -> None:
        """Unload all active ONNX and OCR sessions to free RAM/VRAM after a page completes."""
        if self._detector is not None:
            self._detector.unload()
        if self._ocr_router is not None:
            self._ocr_router.unload()
        if self._inpainter is not None:
            self._inpainter.unload()
        _ = gc.collect()

    def get_lmstudio_status(self) -> LMStudioEndpointStatus:
        """Ping LM Studio (`http://localhost:1235/v1/models`) and format status with display & resolved model names."""
        if self.translator.provider == "lmstudio":
            lm_translator = self.translator
        else:
            settings = self.get_settings()
            lm_profile = settings["providers"].get("lmstudio")
            lm_preset = PROVIDER_PRESETS["lmstudio"]
            lm_translator = ComicTranslator(
                provider="lmstudio",
                base_url=(
                    lm_profile["base_url"] if lm_profile else lm_preset.default_base_url
                ),
                api_key=(
                    lm_profile["api_key"] if lm_profile else lm_preset.default_api_key
                ),
                preferred_model=(
                    lm_profile["model"] if lm_profile else lm_preset.default_model
                ),
                timeout=5.0,
            )

        raw = lm_translator.get_status()
        resolved_id = raw["active_model"]
        display_model = (
            format_display_model_name(
                provider="lmstudio",
                resolved_model_id=resolved_id,
                preferred_model=lm_translator.preferred_model,
            )
            if raw["connected"]
            else None
        )

        status: LMStudioEndpointStatus = {
            "connected": raw["connected"],
            "endpoint": raw["endpoint"],
            "active_model": display_model,
            "resolved_model_id": resolved_id,
            "preferred_model": lm_translator.preferred_model,
            "available_models": raw["available_models"],
        }
        err_msg = raw.get("error")
        if err_msg:
            status["error"] = err_msg
        return status

    def get_active_provider_summary(self) -> dict[str, object]:
        """Return lightweight active LLM provider configuration summary for `/api/health`."""
        settings = self.get_settings()
        return {
            "provider": settings["provider"],
            "provider_name": PROVIDER_PRESETS[settings["provider"]].name,
            "endpoint": settings["base_url"],
            "model": settings["model"],
            "use_vision": settings["use_vision"],
            "reasoning_effort": settings["reasoning_effort"],
            "target_lang": settings["target_lang"],
        }


pipeline_service = PipelineService()

app = FastAPI(
    title="Manga/Manhwa/Manhua Comic Translator API",
    description="Local offline-first Comic Text Detection, Multi-Language OCR, LaMa Inpainting, and Multi-Provider LLM Translation.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def api_health() -> dict[str, object]:
    """System health check reporting server uptime, ONNX execution provider, active LLM provider, and LM Studio status."""
    model_status = pipeline_service.model_manager.get_status()
    lm_status = pipeline_service.get_lmstudio_status()
    return {
        "status": "ok",
        "execution_provider": get_execution_provider(),
        "available_providers": get_available_execution_providers(),
        "models_ready": model_status["all_ready"],
        "active_provider": pipeline_service.get_active_provider_summary(),
        "lm_studio": lm_status,
    }


@app.get("/api/providers")
def api_list_providers() -> dict[str, object]:
    """Return all 8 supported Koharu-style LLM provider presets, reasoning options, and target languages."""
    settings = pipeline_service.get_settings()
    return {
        "active_provider": settings["provider"],
        "providers": list_provider_presets(),
        "supported_reasoning_efforts": list(SUPPORTED_REASONING_EFFORTS),
        "target_languages": list_target_languages(),
    }


@app.post("/api/providers/test")
def api_test_provider(payload: ProviderTestRequest | None = None) -> ProviderTestResult:
    """Verify provider credentials/connectivity and fetch available models from the selected provider."""
    req = payload if payload is not None else ProviderTestRequest()
    settings = pipeline_service.get_settings()

    try:
        prov_id = (
            normalize_provider_id(req.provider)
            if req.provider is not None and req.provider.strip()
            else settings["provider"]
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    saved_profile = settings["providers"].get(prov_id)
    preset = PROVIDER_PRESETS[prov_id]

    if req.base_url is not None and req.base_url.strip():
        base_url = req.base_url.strip()
    elif prov_id == settings["provider"] and settings["base_url"]:
        base_url = settings["base_url"]
    elif saved_profile and saved_profile["base_url"]:
        base_url = saved_profile["base_url"]
    else:
        base_url = preset.default_base_url

    if req.api_key is not None:
        api_key = req.api_key.strip()
    elif prov_id == settings["provider"]:
        api_key = settings["api_key"]
    elif saved_profile:
        api_key = saved_profile["api_key"]
    else:
        api_key = preset.default_api_key

    if req.model is not None and req.model.strip():
        preferred_model = req.model.strip()
    elif prov_id == settings["provider"] and settings["model"]:
        preferred_model = settings["model"]
    elif saved_profile and saved_profile["model"]:
        preferred_model = saved_profile["model"]
    else:
        preferred_model = preset.default_model

    return test_provider_connection(
        provider=prov_id,
        base_url=base_url,
        api_key=api_key,
        preferred_model=preferred_model,
        timeout=req.timeout,
    )


@app.get("/api/languages")
def api_list_languages() -> dict[str, object]:
    """Return supported source comic languages (`ja`/`ko`/`zh`) and all 16 target translation languages."""
    return {
        "source_languages": [
            {"code": "ja", "name": "Japanese (Manga)", "native_name": "日本語", "reading_order": "rtl"},
            {"code": "ko", "name": "Korean (Manhwa)", "native_name": "한국어", "reading_order": "ltr"},
            {"code": "zh", "name": "Chinese (Manhua)", "native_name": "中文", "reading_order": "ltr"},
        ],
        "target_languages": list_target_languages(),
    }


@app.get("/api/settings")
def api_get_settings() -> SettingsResponse:
    """Return persistent configuration loaded from `%LOCALAPPDATA%/ComicTranslator/settings.json` (frozen) or `backend/settings.json` (dev)."""
    settings = pipeline_service.get_settings()
    return _format_settings_response(settings)


@app.post("/api/settings")
def api_save_settings(payload: SettingsUpdateRequest) -> SettingsResponse:
    """Validate, atomically persist, and hot-reload application settings."""
    raw_patch = payload.model_dump(exclude_unset=True)
    try:
        updated = pipeline_service.update_settings(raw_patch)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to persist settings to disk: {exc}",
        ) from exc
    return _format_settings_response(updated)


@app.get("/api/cache/images/{image_id}.{ext}")
def api_get_cached_image(image_id: str, ext: str) -> FileResponse:
    """Serve a cached inpainted, mask, or imported page image directly via FileResponse."""
    lower_ext = ext.lower()
    media_type = _CACHE_EXT_TO_MEDIA_TYPE.get(lower_ext)
    if media_type is None or not _SAFE_IMAGE_ID_RE.fullmatch(image_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid cached image identifier.",
        )

    cache_root = get_image_cache_dir().resolve()
    target_path = (cache_root / f"{image_id}.{lower_ext}").resolve()
    if target_path.parent != cache_root or not target_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Cached image '{image_id}.{lower_ext}' not found.",
        )

    return FileResponse(
        path=str(target_path),
        media_type=media_type,
        filename=f"{image_id}.{lower_ext}",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.get("/api/models/status")
def api_models_status() -> CombinedModelsStatusResponse:
    """Check whether all local CV/OCR models in `models/` are downloaded and ready."""
    cv_status = pipeline_service.model_manager.get_status()
    lm_status = pipeline_service.get_lmstudio_status()
    return {
        "all_ready": cv_status["all_ready"],
        "execution_provider": cv_status["execution_provider"],
        "models_dir": cv_status["models_dir"],
        "models": cv_status["models"],
        "lm_studio": lm_status,
    }


def _sse_download_generator() -> Iterator[str]:
    """Yield Server-Sent Events (SSE) JSON lines from `ModelManager.download_missing_models()`."""
    try:
        for event in pipeline_service.model_manager.download_missing_models():
            payload = json.dumps(event, ensure_ascii=False)
            yield f"data: {payload}\n\n"
    except Exception as exc:
        err_payload = json.dumps(
            {
                "status": "error",
                "model_key": "all",
                "file": None,
                "file_percent": 0.0,
                "overall_percent": 0.0,
                "downloaded_bytes": 0,
                "total_bytes": 0,
                "offline": False,
                "message": f"Model download error: {exc}",
            },
            ensure_ascii=False,
        )
        yield f"data: {err_payload}\n\n"


@app.get("/api/models/download")
@app.get("/api/models/download-stream")
def api_models_download() -> StreamingResponse:
    """SSE endpoint streaming real-time model download progress from `model_manager.py`."""
    return StreamingResponse(
        _sse_download_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/lmstudio/status")
def api_lmstudio_status() -> LMStudioEndpointStatus:
    """Ping `http://localhost:1235/v1/models` and return connection status and active model name."""
    return pipeline_service.get_lmstudio_status()


@app.post("/api/stage/detect")
@app.post("/api/pipeline/detect")
async def api_stage_detect(
    image: Annotated[UploadFile | None, File()] = None,
    file: Annotated[UploadFile | None, File()] = None,
    source_lang: Annotated[str, Form()] = "ja",
    conf_threshold: Annotated[float | None, Form()] = None,
    nms_threshold: Annotated[float | None, Form()] = None,
    mask_threshold: Annotated[float | None, Form()] = None,
    include_mask_base64: Annotated[bool, Form()] = False,
) -> StageDetectResponse:
    """Stage 1 (Detect): Return reading-order sorted bounding boxes and binary text mask."""
    upload = image if image is not None else file
    if upload is None:
        raise HTTPException(
            status_code=400,
            detail="Missing uploaded image file. Provide multipart form field 'image' or 'file'.",
        )

    try:
        src_lang = normalize_language(source_lang)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    raw_bytes = await upload.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image file is empty.")

    try:
        rgb_image = load_rgb_image(raw_bytes)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to decode uploaded image: {exc}",
        ) from exc

    if not pipeline_service.model_manager.get_detector_path().is_file():
        raise HTTPException(
            status_code=503,
            detail="Comic Text Detector ONNX weights are missing. Call GET /api/models/download first.",
        )

    detector = pipeline_service.get_detector()
    try:
        det_result = detector.detect(
            rgb_image,
            source_language=src_lang,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            mask_threshold=mask_threshold,
            auto_unload=True,
        )
        mask_id, mask_url = save_mask_image_to_cache(det_result["mask"])
        mask_b64 = (
            ComicTextDetector.mask_to_base64(det_result["mask"])
            if include_mask_base64
            else None
        )
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Text detection failed ({type(exc).__name__}): {exc}",
        ) from exc
    finally:
        detector.unload()

    bubbles = det_result["bubbles"]
    reading_order: ReadingOrder = "rtl" if src_lang == "ja" else "ltr"
    page_name = Path(upload.filename or "page_001").stem or "page_001"
    response: StageDetectResponse = {
        "page_id": page_name,
        "dimensions": det_result["dimensions"],
        "source_lang": src_lang,
        "source_language": src_lang,
        "reading_order": reading_order,
        "execution_provider": det_result["execution_provider"],
        "mask_id": mask_id,
        "mask_url": mask_url,
        "mask": mask_b64 if mask_b64 is not None else mask_url,
        "ordered_ids": [b["id"] for b in bubbles],
        "bubbles": bubbles,
        "blocks": bubbles,
    }
    if mask_b64 is not None:
        response["mask_base64"] = mask_b64
    return response


@app.post("/api/stage/ocr")
@app.post("/api/pipeline/ocr")
async def api_stage_ocr(
    image: Annotated[UploadFile | None, File()] = None,
    file: Annotated[UploadFile | None, File()] = None,
    bboxes: Annotated[str | None, Form()] = None,
    blocks: Annotated[str | None, Form()] = None,
    bubbles: Annotated[str | None, Form()] = None,
    source_lang: Annotated[str, Form()] = "ja",
) -> StageOcrResponse:
    """Stage 2 (OCR): Extract text for selected bounding boxes using `manga-ocr` (`ja`) or `RapidOCR` (`ko`/`zh`)."""
    upload = image if image is not None else file
    if upload is None:
        raise HTTPException(
            status_code=400,
            detail="Missing uploaded image file. Provide multipart form field 'image' or 'file'.",
        )

    try:
        src_lang = normalize_language(source_lang)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    raw_boxes_json = (
        bboxes if bboxes is not None else (blocks if blocks is not None else bubbles)
    )
    if raw_boxes_json is None or not raw_boxes_json.strip():
        raise HTTPException(
            status_code=400,
            detail="Missing selected bounding boxes. Provide JSON array in form field 'bboxes' (or 'blocks'/'bubbles').",
        )

    try:
        parsed_bubbles = _parse_stage_bubbles_json(raw_boxes_json)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    raw_bytes = await upload.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image file is empty.")

    try:
        rgb_image = load_rgb_image(raw_bytes)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to decode uploaded image: {exc}",
        ) from exc

    if src_lang == "ja":
        if not (pipeline_service.model_manager.get_manga_ocr_dir() / "config.json").is_file():
            raise HTTPException(
                status_code=503,
                detail="Japanese manga-ocr weights are missing. Call GET /api/models/download first.",
            )
    else:
        rapid_paths = pipeline_service.model_manager.get_rapidocr_paths(language=src_lang)
        required_rapid_files: tuple[Path, ...] = (
            rapid_paths["det_model_path"],
            rapid_paths["rec_model_path"],
            rapid_paths["cls_model_path"],
        )
        if not all(p.is_file() for p in required_rapid_files):
            raise HTTPException(
                status_code=503,
                detail=f"RapidOCR ({src_lang}) ONNX weights are missing. Call GET /api/models/download first.",
            )

    ocr_router = pipeline_service.get_ocr_router()
    try:
        ocr_results = ocr_router.recognize_bubbles(
            image=rgb_image,
            bubbles=parsed_bubbles,
            source_lang=src_lang,
            auto_unload=True,
        )
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"OCR ({src_lang}) failed ({type(exc).__name__}): {exc}",
        ) from exc
    finally:
        ocr_router.unload()

    stage_blocks: list[StageOcrBlockItem] = [
        {
            "id": b["id"],
            "order": b["order"],
            "bbox": list(b["bbox"]),
            "polygon": [list(pt) for pt in b["polygon"]],
            "original_text": b["source_text"],
            "source_text": b["source_text"],
            "confidence": b["confidence"],
            "text_direction": b["text_direction"],
        }
        for b in ocr_results
    ]
    return {
        "source_lang": src_lang,
        "source_language": src_lang,
        "engine": "manga-ocr" if src_lang == "ja" else "rapidocr",
        "count": len(stage_blocks),
        "texts": [b["source_text"] for b in stage_blocks],
        "blocks": stage_blocks,
        "bubbles": stage_blocks,
    }


@app.post("/api/stage/inpaint")
@app.post("/api/pipeline/inpaint")
async def api_stage_inpaint(
    image: Annotated[UploadFile | None, File()] = None,
    file: Annotated[UploadFile | None, File()] = None,
    mask: Annotated[UploadFile | None, File()] = None,
    mask_id: Annotated[str | None, Form()] = None,
    mask_url: Annotated[str | None, Form()] = None,
    bboxes: Annotated[str | None, Form()] = None,
    blocks: Annotated[str | None, Form()] = None,
    bubbles: Annotated[str | None, Form()] = None,
    tile_padding: Annotated[int | None, Form()] = None,
    refine_mask: Annotated[bool, Form()] = True,
    include_base64: Annotated[bool, Form()] = False,
) -> StageInpaintResponse:
    """Stage 3 (Inpaint): Erase selected bboxes and/or binary mask using tile-based LaMa ONNX and return a cached image URL."""
    upload = image if image is not None else file
    if upload is None:
        raise HTTPException(
            status_code=400,
            detail="Missing uploaded image file. Provide multipart form field 'image' or 'file'.",
        )

    raw_bytes = await upload.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image file is empty.")

    try:
        rgb_image = load_rgb_image(raw_bytes)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to decode uploaded image: {exc}",
        ) from exc

    orig_h, orig_w = int(rgb_image.shape[0]), int(rgb_image.shape[1])

    raw_boxes_json = (
        bboxes if bboxes is not None else (blocks if blocks is not None else bubbles)
    )
    selected_bboxes: list[list[int]] | None = None
    if raw_boxes_json is not None and raw_boxes_json.strip():
        try:
            parsed_bubbles = _parse_stage_bubbles_json(raw_boxes_json)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        selected_bboxes = [b["bbox"] for b in parsed_bubbles]

    # Resolve mask from uploaded file, cached mask_id/mask_url, or synthesize from selected bboxes
    working_mask: NDArrayU8 | None = None
    has_explicit_mask = False

    if mask is not None:
        mask_bytes = await mask.read()
        if not mask_bytes:
            raise HTTPException(status_code=400, detail="Uploaded mask file is empty.")
        try:
            working_mask = load_mask_image(mask_bytes, target_hw=(orig_h, orig_w))
            has_explicit_mask = True
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to decode uploaded mask image: {exc}",
            ) from exc
    elif mask_id or mask_url:
        cached_mask_path = resolve_cached_image_path(mask_id or mask_url or "")
        working_mask = load_mask_image(cached_mask_path, target_hw=(orig_h, orig_w))
        has_explicit_mask = True

    if working_mask is None:
        if not selected_bboxes:
            raise HTTPException(
                status_code=400,
                detail="Provide at least one of 'mask', 'mask_id', 'mask_url', or 'bboxes'.",
            )
        # Synthesize binary mask covering the user-selected bounding boxes
        working_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        for box in selected_bboxes:
            x1 = max(0, min(orig_w, box[0]))
            y1 = max(0, min(orig_h, box[1]))
            x2 = max(0, min(orig_w, box[2]))
            y2 = max(0, min(orig_h, box[3]))
            if x2 > x1 and y2 > y1:
                working_mask[y1:y2, x1:x2] = 255
    elif selected_bboxes is not None:
        # Restrict provided full-page mask strictly to the selected bboxes
        roi_mask: NDArrayU8 = np.zeros((orig_h, orig_w), dtype=np.uint8)
        pad = 8
        for box in selected_bboxes:
            x1 = max(0, min(orig_w, box[0] - pad))
            y1 = max(0, min(orig_h, box[1] - pad))
            x2 = max(0, min(orig_w, box[2] + pad))
            y2 = max(0, min(orig_h, box[3] + pad))
            if x2 > x1 and y2 > y1:
                roi_mask[y1:y2, x1:x2] = 255
        working_mask = np.asarray(
            np.where((working_mask > 0) & (roi_mask > 0), 255, 0),
            dtype=np.uint8,
        )

    if not pipeline_service.model_manager.get_lama_path().is_file():
        raise HTTPException(
            status_code=503,
            detail="LaMa Inpainting ONNX weights are missing. Call GET /api/models/download first.",
        )

    inpainter = pipeline_service.get_inpainter()
    try:
        inpaint_res = inpainter.inpaint_with_metadata(
            image=rgb_image,
            mask=working_mask,
            bboxes=selected_bboxes,
            tile_padding=tile_padding,
            refine_mask=(refine_mask and has_explicit_mask),
            auto_unload=True,
        )
        cleaned_image_id, cleaned_image_url = save_inpainted_image_to_cache(
            inpaint_res["image"]
        )
        cleaned_image_payload = (
            LamaInpainter.image_to_base64(inpaint_res["image"])
            if include_base64
            else cleaned_image_url
        )
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"LaMa inpainting failed ({type(exc).__name__}): {exc}",
        ) from exc
    finally:
        inpainter.unload()

    return {
        "dimensions": {"width": orig_w, "height": orig_h},
        "execution_provider": inpaint_res["execution_provider"],
        "tiles_inpainted": inpaint_res["tiles_processed"],
        "cleaned_image_id": cleaned_image_id,
        "cleaned_image_url": cleaned_image_url,
        "inpainted_image_url": cleaned_image_url,
        "cleaned_image": cleaned_image_payload,
        "inpainted_image": cleaned_image_payload,
    }


@app.post("/api/stage/translate")
async def api_stage_translate(request: Request) -> StageTranslateResponse:
    """Stage 4 (Translate): Translate dialogue blocks with optional page image and multi-provider translation config.

    Accepts either:
    - `application/json`: `StageTranslateRequest` (`blocks`, `source_lang`, `target_lang`, optional `image_url`/`image_base64`, optional `config`).
    - `multipart/form-data`: `blocks`/`bubbles`/`dialogues` (JSON string), optional `image`/`file` upload, `source_lang`, `target_lang`, and optional `config` (JSON string) or flat override fields (`provider`, `model`, `base_url`, `api_key`, `temperature`, `top_p`, `top_k`, `max_tokens`, `frequency_penalty`, `reasoning_effort`, `timeout`, `custom_instructions`, `glossary`, `use_vision`, `vision_max_side`).
    """
    content_type = request.headers.get("content-type", "").lower()

    payload_model: StageTranslateRequest
    uploaded_image_bytes: bytes | None = None

    if "multipart/form-data" in content_type:
        form = await request.form()

        raw_blocks_field: object = (
            form.get("blocks")
            if form.get("blocks") is not None
            else (
                form.get("bubbles")
                if form.get("bubbles") is not None
                else form.get("dialogues")
            )
        )
        parsed_blocks_list: list[object] = []
        if isinstance(raw_blocks_field, str) and raw_blocks_field.strip():
            try:
                loaded_blocks = _json_loads(raw_blocks_field)
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid JSON in 'blocks' form field: {exc}",
                ) from exc
            if not isinstance(loaded_blocks, list):
                raise HTTPException(
                    status_code=400,
                    detail="'blocks' form field must be a JSON array.",
                )
            parsed_blocks_list = cast(list[object], loaded_blocks)

        config_dict: dict[str, object] = {}
        raw_config_field = form.get("config")
        if isinstance(raw_config_field, str) and raw_config_field.strip():
            try:
                loaded_cfg = _json_loads(raw_config_field)
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid JSON in 'config' form field: {exc}",
                ) from exc
            if not isinstance(loaded_cfg, Mapping):
                raise HTTPException(
                    status_code=400,
                    detail="'config' form field must be a JSON object.",
                )
            for k, v in cast(Mapping[str, object], loaded_cfg).items():
                config_dict[str(k)] = v

        # Allow flat form overrides alongside or instead of JSON `config`
        for str_key in (
            "provider",
            "model",
            "base_url",
            "api_key",
            "reasoning_effort",
            "custom_instructions",
        ):
            val = form.get(str_key)
            if isinstance(val, str) and val.strip():
                config_dict[str_key] = val.strip()
        for num_key in ("temperature", "top_p", "frequency_penalty", "timeout"):
            val = form.get(num_key)
            if isinstance(val, str) and val.strip():
                try:
                    config_dict[num_key] = float(val)
                except ValueError as exc:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid float for '{num_key}': {val!r}",
                    ) from exc
        for int_key in ("top_k", "max_tokens", "vision_max_side"):
            raw_int_val = form.get(int_key)
            if isinstance(raw_int_val, str) and raw_int_val.strip():
                try:
                    config_dict[int_key] = int(raw_int_val)
                except ValueError as exc:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid integer for '{int_key}': {raw_int_val!r}",
                    ) from exc
        raw_glossary = form.get("glossary")
        if isinstance(raw_glossary, str) and raw_glossary.strip():
            config_dict["glossary"] = parse_glossary(raw_glossary)
        raw_use_vision = form.get("use_vision")
        if raw_use_vision is not None:
            config_dict["use_vision"] = _parse_bool_flag(raw_use_vision, default=False)

        raw_img_field = form.get("image") or form.get("file")
        if isinstance(raw_img_field, StarletteUploadFile):
            uploaded_image_bytes = await raw_img_field.read()

        raw_src_form = form.get("source_language") or form.get("source_lang") or "ja"
        raw_tgt_form = form.get("target_language") or form.get("target_lang") or "id"
        raw_img_url = form.get("image_url")
        raw_img_b64 = form.get("image_base64")

        try:
            payload_model = StageTranslateRequest.model_validate(
                {
                    "source_lang": str(raw_src_form),
                    "target_lang": str(raw_tgt_form),
                    "blocks": parsed_blocks_list,
                    "image_url": str(raw_img_url) if isinstance(raw_img_url, str) else None,
                    "image_base64": str(raw_img_b64) if isinstance(raw_img_b64, str) else None,
                    "config": config_dict if config_dict else None,
                }
            )
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
    else:
        raw_body = await request.body()
        if not raw_body:
            raise HTTPException(status_code=400, detail="Empty request body.")
        try:
            body_obj = _json_loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid JSON request body: {exc}",
            ) from exc
        try:
            payload_model = StageTranslateRequest.model_validate(body_obj)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc

    raw_src = payload_model.source_language or payload_model.source_lang
    raw_tgt = payload_model.target_language or payload_model.target_lang
    try:
        src_lang = normalize_source_language(raw_src)
        tgt_lang = normalize_target_language(raw_tgt)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Resolve optional page image (multipart upload, cached image_url, or data URI / base64)
    page_rgb: NDArrayU8 | None = None
    if uploaded_image_bytes:
        try:
            page_rgb = load_rgb_image(uploaded_image_bytes)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to decode attached page image: {exc}",
            ) from exc
    elif payload_model.image_url:
        cached_path = resolve_cached_image_path(payload_model.image_url)
        page_rgb = load_rgb_image(cached_path)
    elif payload_model.image_base64:
        b64_str = payload_model.image_base64.strip()
        if "," in b64_str and b64_str.lower().startswith("data:"):
            b64_str = b64_str.split(",", 1)[1]
        try:
            decoded_img = base64.b64decode(b64_str)
            page_rgb = load_rgb_image(decoded_img)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to decode 'image_base64': {exc}",
            ) from exc

    translator = _build_stage_translator(payload_model.config)

    page_image_dims: ImageDimensions | None = None
    page_image_data_uri: str | None = None
    if page_rgb is not None:
        page_image_dims = {
            "width": int(page_rgb.shape[1]),
            "height": int(page_rgb.shape[0]),
        }
        if translator.use_vision:
            page_image_data_uri = _encode_page_image_for_vlm(
                page_rgb,
                max_side=translator.vision_max_side,
            )

    input_items = (
        payload_model.blocks
        if payload_model.blocks
        else (payload_model.bubbles if payload_model.bubbles else payload_model.dialogues)
    )
    raw_blocks: list[dict[str, object]] = []
    for item in input_items:
        src_text = (
            item.original_text
            if item.original_text is not None
            else (item.source_text if item.source_text is not None else (item.text or ""))
        )
        raw_blocks.append({"id": item.id, "source_text": src_text})

    try:
        translation_res = translator.translate_blocks(
            blocks=raw_blocks,
            source_lang=src_lang,
            target_lang=tgt_lang,
            page_image_data_uri=page_image_data_uri,
            use_vision=translator.use_vision,
        )
    except LMStudioConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LMStudioTranslationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    trans_map = translation_res["translations"]
    translated_blocks: list[ProcessedBlockItem] = []

    for idx, item in enumerate(input_items, start=1):
        orig_text = (
            item.original_text
            if item.original_text is not None
            else (item.source_text if item.source_text is not None else (item.text or ""))
        )
        new_trans = trans_map.get(item.id, orig_text)
        bbox = list(item.bbox) if len(item.bbox) >= 4 else [0, 0, 160, 80]
        polygon = (
            [list(pt) for pt in item.polygon]
            if item.polygon is not None
            else _bbox_to_polygon(bbox)
        )
        fsize = (
            item.font_size
            if item.font_size is not None and item.font_size > 0
            else _estimate_font_size(bbox, new_trans or orig_text)
        )
        order_num = item.order if item.order is not None else idx

        translated_blocks.append(
            {
                "id": item.id,
                "order": order_num,
                "bbox": bbox,
                "polygon": polygon,
                "original_text": orig_text,
                "source_text": orig_text,
                "translated_text": new_trans,
                "font_size": fsize,
                "color": item.color,
                "align": item.align,
                "confidence": item.confidence,
                "text_direction": item.text_direction,
                "style": _build_default_style(
                    font_size=fsize,
                    color=item.color,
                    align=item.align,
                ),
            }
        )

    return {
        "provider": translation_res["provider"],
        "model": translation_res["model"],
        "vision_used": translation_res["vision_used"],
        "reasoning_effort": translation_res["reasoning_effort"],
        "source_lang": src_lang,
        "target_lang": tgt_lang,
        "source_language": src_lang,
        "target_language": tgt_lang,
        "has_page_image": page_rgb is not None,
        "page_image_dimensions": page_image_dims,
        "translations": trans_map,
        "dialogues": translation_res["dialogues"],
        "blocks": translated_blocks,
        "bubbles": translated_blocks,
    }


@app.post("/api/process-page")
@app.post("/api/pipeline/process-page")
async def api_process_page(
    image: Annotated[UploadFile | None, File()] = None,
    file: Annotated[UploadFile | None, File()] = None,
    source_lang: Annotated[str, Form()] = "ja",
    target_lang: Annotated[str, Form()] = "id",
    include_base64: Annotated[bool, Form()] = False,
    include_base64_query: Annotated[bool, Query(alias="include_base64")] = False,
    use_vision: Annotated[bool | None, Form()] = None,
    reasoning_effort: Annotated[str | None, Form()] = None,
    custom_instructions: Annotated[str | None, Form()] = None,
    glossary: Annotated[str | None, Form()] = None,
    provider: Annotated[str | None, Form()] = None,
    model: Annotated[str | None, Form()] = None,
    temperature: Annotated[float | None, Form()] = None,
    top_p: Annotated[float | None, Form()] = None,
    top_k: Annotated[int | None, Form()] = None,
    max_tokens: Annotated[int | None, Form()] = None,
    frequency_penalty: Annotated[float | None, Form()] = None,
    config: Annotated[str | None, Form()] = None,
) -> ProcessPageResponse:
    """Run full page pipeline: Detector -> OCR -> LaMa Inpainter -> Multi-Provider LLM Translator.

    Accepts multipart form-data with `image` (or `file`), `source_lang` (`ja`, `ko`, `zh`),
    `target_lang` (any of the 16 supported target languages), optional `include_base64` (`false` by default),
    and optional translation control overrides (falling back to persisted `settings.json`).
    """
    upload = image if image is not None else file
    if upload is None:
        raise HTTPException(
            status_code=400,
            detail="Missing uploaded image file. Provide multipart form field 'image' or 'file'.",
        )

    try:
        src_lang = normalize_language(source_lang)
        tgt_lang = normalize_target_language(target_lang)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Build optional per-request translation config overriding persistent settings
    override_dict: dict[str, object] = {}
    if config and config.strip():
        try:
            loaded_cfg = _json_loads(config)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid JSON in 'config' form field: {exc}",
            ) from exc
        if not isinstance(loaded_cfg, Mapping):
            raise HTTPException(
                status_code=400,
                detail="'config' form field must be a JSON object.",
            )
        for k, v in cast(Mapping[str, object], loaded_cfg).items():
            override_dict[str(k)] = v

    if provider is not None and provider.strip():
        override_dict["provider"] = provider.strip()
    if model is not None and model.strip():
        override_dict["model"] = model.strip()
    if temperature is not None:
        override_dict["temperature"] = temperature
    if top_p is not None:
        override_dict["top_p"] = top_p
    if top_k is not None:
        override_dict["top_k"] = top_k
    if max_tokens is not None:
        override_dict["max_tokens"] = max_tokens
    if frequency_penalty is not None:
        override_dict["frequency_penalty"] = frequency_penalty
    if reasoning_effort is not None and reasoning_effort.strip():
        override_dict["reasoning_effort"] = reasoning_effort.strip()
    if use_vision is not None:
        override_dict["use_vision"] = use_vision
    if custom_instructions is not None:
        override_dict["custom_instructions"] = custom_instructions
    if glossary is not None and glossary.strip():
        override_dict["glossary"] = parse_glossary(glossary)

    stage_config: TranslationConfigModel | None = None
    if override_dict:
        try:
            stage_config = TranslationConfigModel.model_validate(override_dict)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc

    translator = _build_stage_translator(stage_config)

    raw_bytes = await upload.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image file is empty.")

    try:
        rgb_image = load_rgb_image(raw_bytes)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to decode uploaded image: {exc}",
        ) from exc

    if not pipeline_service.model_manager.is_all_ready():
        missing = [
            s.local_rel_path
            for s in pipeline_service.model_manager.get_missing_specs()
        ]
        raise HTTPException(
            status_code=503,
            detail=f"Required local CV/OCR models are missing: {missing}. Call GET /api/models/download first.",
        )

    should_include_base64 = include_base64 or include_base64_query

    try:
        # Stage 1: Comic Text Detection (Bounding Boxes + Reading-Order Sort + Binary Text Mask)
        detector = pipeline_service.get_detector()
        try:
            det_result = detector.detect(
                rgb_image,
                source_language=src_lang,
                auto_unload=True,
            )
            detected_bubbles = det_result["bubbles"]
            text_mask = det_result["mask"]
            dimensions = det_result["dimensions"]
            mask_id, mask_url = save_mask_image_to_cache(text_mask)
        except HTTPException:
            raise
        except Exception as exc:
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Text detection failed ({type(exc).__name__}): {exc}",
            ) from exc
        finally:
            detector.unload()

        # Stage 2: Multi-Language OCR Router (manga-ocr for ja, RapidOCR PP-OCRv4 for ko/zh)
        ocr_router = pipeline_service.get_ocr_router()
        try:
            ocr_bubbles = ocr_router.recognize_bubbles(
                image=rgb_image,
                bubbles=detected_bubbles,
                source_lang=src_lang,
                auto_unload=True,
            )
        except HTTPException:
            raise
        except Exception as exc:
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"OCR ({src_lang}) failed ({type(exc).__name__}): {exc}",
            ) from exc
        finally:
            ocr_router.unload()

        # Stage 3: Tile-Based LaMa Inpainting (512x512 ROI patches)
        inpainter = pipeline_service.get_inpainter()
        try:
            bboxes = [b["bbox"] for b in detected_bubbles]
            inpaint_res = inpainter.inpaint_with_metadata(
                image=rgb_image,
                mask=text_mask,
                bboxes=bboxes,
                auto_unload=True,
            )
            cleaned_image_id, cleaned_image_url = save_inpainted_image_to_cache(
                inpaint_res["image"]
            )
            cleaned_image_payload = (
                LamaInpainter.image_to_base64(inpaint_res["image"])
                if should_include_base64
                else cleaned_image_url
            )
        except HTTPException:
            raise
        except Exception as exc:
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"LaMa inpainting failed ({type(exc).__name__}): {exc}",
            ) from exc
        finally:
            inpainter.unload()
    finally:
        # Ensure zero ONNX/OCR sessions remain resident after each page (even on error)
        pipeline_service.unload_all()

    # Stage 4: Context-Aware Full-Page Translation via active Multi-Provider LLM Engine
    dialogue_inputs: list[dict[str, object]] = [
        {"id": b["id"], "source_text": b["source_text"]}
        for b in ocr_bubbles
    ]
    page_image_data_uri = (
        _encode_page_image_for_vlm(rgb_image, max_side=translator.vision_max_side)
        if translator.use_vision
        else None
    )
    try:
        translation_res = translator.translate_blocks(
            blocks=dialogue_inputs,
            source_lang=src_lang,
            target_lang=tgt_lang,
            page_image_data_uri=page_image_data_uri,
            use_vision=translator.use_vision,
        )
    except LMStudioConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LMStudioTranslationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    trans_map = translation_res["translations"]
    active_model = translation_res["model"]

    processed_blocks: list[ProcessedBlockItem] = []
    for b in ocr_bubbles:
        bid = b["id"]
        orig_text = b["source_text"]
        trans_text = trans_map.get(bid, orig_text)
        fsize = _estimate_font_size(b["bbox"], trans_text or orig_text)
        default_color = "#000000"
        default_align: TextAlign = "center"

        processed_blocks.append(
            {
                "id": bid,
                "order": b["order"],
                "bbox": list(b["bbox"]),
                "polygon": [list(pt) for pt in b["polygon"]],
                "original_text": orig_text,
                "source_text": orig_text,
                "translated_text": trans_text,
                "font_size": fsize,
                "color": default_color,
                "align": default_align,
                "confidence": b["confidence"],
                "text_direction": b["text_direction"],
                "style": _build_default_style(
                    font_size=fsize,
                    color=default_color,
                    align=default_align,
                ),
            }
        )

    page_name = Path(upload.filename or "page_001").stem or "page_001"
    return {
        "page_id": page_name,
        "dimensions": dimensions,
        "source_lang": src_lang,
        "target_lang": tgt_lang,
        "source_language": src_lang,
        "target_language": tgt_lang,
        "provider": translation_res["provider"],
        "model": active_model,
        "vision_used": translation_res["vision_used"],
        "reasoning_effort": translation_res["reasoning_effort"],
        "execution_provider": det_result["execution_provider"],
        "tiles_inpainted": inpaint_res["tiles_processed"],
        "cleaned_image_id": cleaned_image_id,
        "cleaned_image_url": cleaned_image_url,
        "inpainted_image_url": cleaned_image_url,
        "cleaned_image": cleaned_image_payload,
        "inpainted_image": cleaned_image_payload,
        "mask_id": mask_id,
        "mask_url": mask_url,
        "blocks": processed_blocks,
        "bubbles": processed_blocks,
    }


@app.post("/api/retranslate")
@app.post("/api/pipeline/translate")
def api_retranslate(payload: RetranslateRequest) -> RetranslateResponse:
    """Re-translate existing text blocks across any supported target language or provider config."""
    raw_src = payload.source_language or payload.source_lang
    raw_tgt = payload.target_language or payload.target_lang
    try:
        src_lang = normalize_source_language(raw_src)
        tgt_lang = normalize_target_language(raw_tgt)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    input_items = (
        payload.blocks
        if payload.blocks
        else (payload.bubbles if payload.bubbles else payload.dialogues)
    )

    raw_blocks: list[dict[str, object]] = []
    for item in input_items:
        src_text = (
            item.original_text
            if item.original_text is not None
            else (item.source_text if item.source_text is not None else (item.text or ""))
        )
        raw_blocks.append(
            {
                "id": item.id,
                "source_text": src_text,
            }
        )

    merged_cfg: TranslationConfigModel | None = payload.config
    if any(
        val is not None
        for val in (payload.custom_instructions, payload.glossary, payload.use_vision)
    ):
        cfg_data = merged_cfg.model_dump() if merged_cfg is not None else {}
        if payload.custom_instructions is not None:
            cfg_data["custom_instructions"] = payload.custom_instructions
        if payload.glossary is not None:
            cfg_data["glossary"] = payload.glossary
        if payload.use_vision is not None:
            cfg_data["use_vision"] = payload.use_vision
        merged_cfg = TranslationConfigModel.model_validate(cfg_data)

    translator = _build_stage_translator(merged_cfg)

    page_image_data_uri: str | None = None
    if translator.use_vision:
        page_rgb: NDArrayU8 | None = None
        if payload.image_url:
            cached_path = resolve_cached_image_path(payload.image_url)
            page_rgb = load_rgb_image(cached_path)
        elif payload.image_base64:
            b64_str = payload.image_base64.strip()
            if "," in b64_str and b64_str.lower().startswith("data:"):
                b64_str = b64_str.split(",", 1)[1]
            try:
                page_rgb = load_rgb_image(base64.b64decode(b64_str))
            except Exception as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to decode 'image_base64': {exc}",
                ) from exc
        if page_rgb is not None:
            page_image_data_uri = _encode_page_image_for_vlm(
                page_rgb,
                max_side=translator.vision_max_side,
            )

    try:
        translation_res = translator.translate_blocks(
            blocks=raw_blocks,
            source_lang=src_lang,
            target_lang=tgt_lang,
            page_image_data_uri=page_image_data_uri,
            use_vision=translator.use_vision,
        )
    except LMStudioConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LMStudioTranslationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    trans_map = translation_res["translations"]
    retranslated_blocks: list[ProcessedBlockItem] = []

    for idx, item in enumerate(input_items, start=1):
        orig_text = (
            item.original_text
            if item.original_text is not None
            else (item.source_text if item.source_text is not None else (item.text or ""))
        )
        new_trans = trans_map.get(item.id, orig_text)
        bbox = list(item.bbox) if len(item.bbox) >= 4 else [0, 0, 160, 80]
        polygon = (
            [list(pt) for pt in item.polygon]
            if item.polygon is not None
            else _bbox_to_polygon(bbox)
        )
        fsize = (
            item.font_size
            if item.font_size is not None and item.font_size > 0
            else _estimate_font_size(bbox, new_trans or orig_text)
        )
        order_num = item.order if item.order is not None else idx

        retranslated_blocks.append(
            {
                "id": item.id,
                "order": order_num,
                "bbox": bbox,
                "polygon": polygon,
                "original_text": orig_text,
                "source_text": orig_text,
                "translated_text": new_trans,
                "font_size": fsize,
                "color": item.color,
                "align": item.align,
                "confidence": item.confidence,
                "text_direction": item.text_direction,
                "style": _build_default_style(
                    font_size=fsize,
                    color=item.color,
                    align=item.align,
                ),
            }
        )

    return {
        "provider": translation_res["provider"],
        "model": translation_res["model"],
        "vision_used": translation_res["vision_used"],
        "reasoning_effort": translation_res["reasoning_effort"],
        "source_lang": src_lang,
        "target_lang": tgt_lang,
        "source_language": src_lang,
        "target_language": tgt_lang,
        "translations": trans_map,
        "dialogues": translation_res["dialogues"],
        "blocks": retranslated_blocks,
        "bubbles": retranslated_blocks,
    }


def _decode_export_image_reference(ref: str, page_index: int) -> bytes:
    """Resolve an export page image from a `data:image/...;base64,...` URI, raw base64, or cached URL."""
    cleaned = ref.strip()
    if not cleaned:
        raise HTTPException(
            status_code=400,
            detail=f"Page {page_index} image reference is empty.",
        )

    if cleaned.startswith("/api/cache/images/") or _SAFE_IMAGE_ID_RE.fullmatch(
        cleaned.rsplit(".", 1)[0]
    ):
        cached_path = resolve_cached_image_path(cleaned)
        return cached_path.read_bytes()

    b64_payload = cleaned
    if b64_payload.lower().startswith("data:") and "," in b64_payload:
        b64_payload = b64_payload.split(",", 1)[1]

    try:
        return base64.b64decode(b64_payload, validate=True)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Page {page_index} contains invalid base64 or image URL: {exc}",
        ) from exc


def _normalize_export_image_format(raw: str | None) -> ExportImageFormat:
    """Validate optional `image_format` (`'original'`, `'png'`, or `'jpg'`)."""
    cleaned = (raw or "original").strip().lower()
    if cleaned in ("original", "auto", ""):
        return "original"
    if cleaned == "png":
        return "png"
    if cleaned in ("jpg", "jpeg"):
        return "jpg"
    raise HTTPException(
        status_code=400,
        detail=f"Unsupported image_format {raw!r}. Expected 'original', 'png', or 'jpg'.",
    )


async def _parse_export_pages_request(
    request: Request,
) -> tuple[list[ExportPageInput], str | None, ExportImageFormat, int]:
    """Parse an ordered list of composited page images from `multipart/form-data` or `application/json`."""
    content_type = request.headers.get("content-type", "").lower()

    if "multipart/form-data" in content_type:
        form = await request.form()
        raw_uploads: list[StarletteUploadFile] = []
        for field_name in ("images", "files", "pages", "image", "file"):
            for entry in form.getlist(field_name):
                if isinstance(entry, StarletteUploadFile):
                    raw_uploads.append(entry)

        if not raw_uploads:
            raise HTTPException(
                status_code=400,
                detail="No page images uploaded. Provide ordered multipart file fields named 'images' or 'files'.",
            )

        pages: list[ExportPageInput] = []
        for idx, upload in enumerate(raw_uploads, start=1):
            img_bytes = await upload.read()
            if not img_bytes:
                raise HTTPException(
                    status_code=400,
                    detail=f"Uploaded page image #{idx} ({upload.filename or 'unnamed'}) is empty.",
                )
            pages.append(
                ExportPageInput(
                    image_bytes=img_bytes,
                    filename=upload.filename,
                )
            )

        raw_name = form.get("filename") or form.get("title")
        export_name = str(raw_name) if isinstance(raw_name, str) and raw_name.strip() else None
        raw_fmt = form.get("image_format")
        image_format = _normalize_export_image_format(
            str(raw_fmt) if isinstance(raw_fmt, str) else None
        )
        raw_quality = form.get("jpeg_quality")
        jpeg_quality = 95
        if isinstance(raw_quality, str) and raw_quality.strip():
            try:
                jpeg_quality = max(50, min(100, int(raw_quality)))
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid integer for 'jpeg_quality': {raw_quality!r}",
                ) from exc

        return pages, export_name, image_format, jpeg_quality

    raw_body = await request.body()
    if not raw_body:
        raise HTTPException(status_code=400, detail="Empty export request body.")
    try:
        body_obj = _json_loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid JSON export request body: {exc}",
        ) from exc

    try:
        parsed_req = ExportPagesRequest.model_validate(body_obj)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    raw_items = parsed_req.pages if parsed_req.pages else parsed_req.images
    if not raw_items:
        raise HTTPException(
            status_code=400,
            detail="Provide a non-empty ordered list in 'pages' or 'images'.",
        )

    pages_from_json: list[ExportPageInput] = []
    for idx, item in enumerate(raw_items, start=1):
        if isinstance(item, str):
            img_bytes = _decode_export_image_reference(item, page_index=idx)
            pages_from_json.append(ExportPageInput(image_bytes=img_bytes))
        else:
            ref = item.image or item.url or item.image_url or item.image_id
            if not ref:
                raise HTTPException(
                    status_code=400,
                    detail=f"Page {idx} object must include 'image', 'url', 'image_url', or 'image_id'.",
                )
            img_bytes = _decode_export_image_reference(ref, page_index=idx)
            pages_from_json.append(
                ExportPageInput(
                    image_bytes=img_bytes,
                    filename=item.filename,
                )
            )

    export_name = parsed_req.filename or parsed_req.title
    return pages_from_json, export_name, parsed_req.image_format, parsed_req.jpeg_quality


@app.post("/api/import/archive")
async def api_import_archive(
    archive: Annotated[UploadFile | None, File()] = None,
    file: Annotated[UploadFile | None, File()] = None,
) -> ImportArchiveResponse:
    """Extract comic page images from an uploaded `.cbz` or `.zip` archive in natural page order."""
    upload = archive if archive is not None else file
    if upload is None:
        raise HTTPException(
            status_code=400,
            detail="Missing uploaded archive file. Provide multipart form field 'archive' or 'file'.",
        )

    archive_name = Path((upload.filename or "comic.cbz").replace("\\", "/")).name
    suffix = Path(archive_name).suffix.lower()
    if suffix and suffix not in (".cbz", ".zip"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported archive format '{suffix}'. Upload a .cbz or .zip archive.",
        )

    raw_bytes = await upload.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded archive file is empty.")

    cache_dir = get_image_cache_dir()
    try:
        extracted = extract_archive_pages(
            archive_bytes=raw_bytes,
            output_dir=cache_dir,
        )
    except ArchiveImportError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Archive import failed ({type(exc).__name__}): {exc}",
        ) from exc

    _prune_image_cache(cache_dir)

    response_pages: list[ImportedArchivePageResponse] = [
        {
            "index": p["index"],
            "filename": p["filename"],
            "archive_path": p["archive_path"],
            "image_id": p["image_id"],
            "image_url": f"/api/cache/images/{p['image_id']}.{p['extension']}",
            "dimensions": {
                "width": p["dimensions"]["width"],
                "height": p["dimensions"]["height"],
            },
            "size_bytes": p["size_bytes"],
        }
        for p in extracted
    ]

    return {
        "archive_name": archive_name,
        "total_pages": len(response_pages),
        "pages": response_pages,
    }


@app.post("/api/export/cbz")
async def api_export_cbz(request: Request) -> Response:
    """Package an ordered list of composited page images into a downloadable `.cbz` archive."""
    pages, export_name, image_format, jpeg_quality = await _parse_export_pages_request(
        request
    )
    try:
        cbz_bytes = build_cbz_bytes(
            pages=pages,
            image_format=image_format,
            jpeg_quality=jpeg_quality,
        )
    except ComicExportError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"CBZ export failed ({type(exc).__name__}): {exc}",
        ) from exc

    download_name = sanitize_export_filename(export_name, extension="cbz")
    return Response(
        content=cbz_bytes,
        media_type="application/vnd.comicbook+zip",
        headers={
            "Content-Disposition": f'attachment; filename="{download_name}"',
            "X-Page-Count": str(len(pages)),
        },
    )


@app.post("/api/export/pdf")
async def api_export_pdf(request: Request) -> Response:
    """Package an ordered list of composited page images into a downloadable `.pdf` document."""
    pages, export_name, _image_format, _jpeg_quality = await _parse_export_pages_request(
        request
    )
    try:
        pdf_bytes = build_pdf_bytes(pages=pages)
    except ComicExportError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"PDF export failed ({type(exc).__name__}): {exc}",
        ) from exc

    download_name = sanitize_export_filename(export_name, extension="pdf")
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{download_name}"',
            "X-Page-Count": str(len(pages)),
        },
    )


class FontDownloadRequest(BaseModel):
    """JSON request body for `POST /api/fonts/download`."""

    fonts: list[str] = Field(default_factory=list)
    font: str | None = None
    family: str | None = None
    weight: str | None = None
    overwrite: bool = False


class FontUploadResponse(TypedDict):
    """Response payload returned by `POST /api/fonts/upload`."""

    status: Literal["ok"]
    filename: str
    family: str
    category: FontCategory
    url: str
    font: FontEntryDict


@app.get("/api/fonts/list")
def api_fonts_list(
    category: Annotated[str | None, Query()] = None,
    include_catalog: Annotated[bool, Query()] = True,
    refresh: Annotated[bool, Query()] = False,
) -> FontListResponseDict:
    """List Windows system fonts, local custom/downloaded fonts, and the 129-font Google Fonts catalog."""
    resolved_cat: FontCategory | None = None
    if category is not None and category.strip():
        cleaned_cat = category.strip().lower()
        if cleaned_cat == "system":
            resolved_cat = "system"
        elif cleaned_cat == "custom":
            resolved_cat = "custom"
        elif cleaned_cat == "downloaded":
            resolved_cat = "downloaded"
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid font category {category!r}. Expected 'system', 'custom', or 'downloaded'.",
            )

    return pipeline_service.font_manager.list_fonts(
        category=resolved_cat,
        include_catalog=include_catalog,
        force_refresh=refresh,
    )


@app.post("/api/fonts/download")
def api_fonts_download(payload: FontDownloadRequest) -> FontDownloadResponseDict:
    """Download selected Google Fonts from the catalog into local `fonts/downloaded/` for offline use."""
    requested: list[str] = [f.strip() for f in payload.fonts if f and f.strip()]
    if payload.font and payload.font.strip():
        requested.append(payload.font.strip())
    if payload.family and payload.family.strip():
        requested.append(payload.family.strip())

    deduped = list(dict.fromkeys(requested))
    if not deduped:
        raise HTTPException(
            status_code=400,
            detail="Provide at least one font name or catalog ID in 'fonts', 'font', or 'family'.",
        )

    return pipeline_service.font_manager.download_catalog_fonts(
        requested_fonts=deduped,
        weight=payload.weight,
        overwrite=payload.overwrite,
    )


@app.post("/api/fonts/upload")
async def api_fonts_upload(
    file: Annotated[UploadFile | None, File()] = None,
    font: Annotated[UploadFile | None, File()] = None,
    family: Annotated[str | None, Form()] = None,
    overwrite: Annotated[bool, Form()] = True,
) -> FontUploadResponse:
    """Upload a custom `.ttf`/`.otf`/`.woff2`/`.woff` comic font into the local `fonts/` directory."""
    upload = file if file is not None else font
    if upload is None:
        raise HTTPException(
            status_code=400,
            detail="Missing uploaded font file. Provide multipart form field 'file' or 'font'.",
        )

    filename = (upload.filename or "").strip()
    if not filename:
        raise HTTPException(
            status_code=400,
            detail="Uploaded font file must include a filename (e.g. 'CCWildWords.ttf').",
        )

    raw_bytes = await upload.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded font file is empty.")

    try:
        entry = pipeline_service.font_manager.save_uploaded_font(
            filename=filename,
            raw_bytes=raw_bytes,
            override_family=family,
            overwrite=overwrite,
        )
    except FontValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded font: {exc}",
        ) from exc

    return {
        "status": "ok",
        "filename": entry["filename"],
        "family": entry["family"],
        "category": entry["category"],
        "url": entry["url"],
        "font": entry,
    }


@app.get("/api/fonts/file/{font_name}")
def api_fonts_file(font_name: str) -> FileResponse:
    """Serve a local or Windows system font binary cleanly for dynamic frontend `@font-face` loading."""
    try:
        font_path, media_type = pipeline_service.font_manager.resolve_font_file(font_name)
    except FontValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return FileResponse(
        path=str(font_path),
        media_type=media_type,
        filename=font_path.name,
        headers={
            "Cache-Control": "public, max-age=86400",
            "Access-Control-Allow-Origin": "*",
        },
    )


class _FrontendStaticFiles(StaticFiles):
    """Serve compiled React static assets with `no-cache` on HTML entry points."""

    @override
    async def get_response(self, path: str, scope: Scope) -> Response:
        response = await super().get_response(path, scope)
        content_type = response.headers.get("content-type", "")
        if "text/html" in content_type.lower():
            response.headers["Cache-Control"] = "no-cache"
        return response


def resolve_frontend_dist() -> Path | None:
    """Resolve the compiled `frontend/dist` directory in dev or inside PyInstaller `_MEIPASS`."""
    candidates: list[Path] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if isinstance(meipass, str) and meipass:
        meipass_root = Path(meipass)
        candidates.append(meipass_root / "frontend" / "dist")
        candidates.append(meipass_root / "frontend_dist")
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        candidates.append(exe_dir / "frontend" / "dist")
        candidates.append(exe_dir / "frontend_dist")
    candidates.append(PROJECT_ROOT / "frontend" / "dist")
    for cand in candidates:
        if cand.is_dir() and (cand / "index.html").is_file():
            return cand
    return None


_resolve_frontend_dist = resolve_frontend_dist


def mount_frontend_dist(target_app: FastAPI = app) -> Path | None:
    """Mount the compiled React production bundle at `/` after all `/api/*` routes."""
    dist_dir = resolve_frontend_dist()
    if dist_dir is None:
        return None
    already_mounted = any(
        getattr(route, "name", None) == "frontend" for route in target_app.routes
    )
    if not already_mounted:
        target_app.mount(
            "/",
            _FrontendStaticFiles(directory=str(dist_dir), html=True),
            name="frontend",
        )
    return dist_dir


_FRONTEND_DIST_DIR = mount_frontend_dist(app)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
