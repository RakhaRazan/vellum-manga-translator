"""FastAPI Backend Entry Point for Manga/Manhwa/Manhua Comic Translator.

Implements SPEC.md Section 4 & Section 5 API contracts with CORS enabled:
- `GET /api/models/status`: Verify local ONNX & OCR model weights in `models/` + LM Studio status.
- `GET /api/models/download` (and `/api/models/download-stream`): SSE stream of real-time download progress.
- `GET /api/lmstudio/status` (and `/api/health`): Ping `http://localhost:1235/v1/models` and report connection status and active model (`Gemma 4 E2B Uncensored HauhauCS Aggressive Q4 K P`).
- `POST /api/process-page` (and `/api/pipeline/process-page`): Full sequential pipeline (Detect -> Reading-Order Sort -> OCR -> LaMa Tile Inpaint -> Context-Aware LLM Translate) returning cleaned base64 PNG + structured block array (`id`, `bbox`, `original_text`, `translated_text`, `font_size`, `color`, `align`).
- `POST /api/retranslate` (and `/api/pipeline/translate`): Re-translate existing dialogue blocks when switching target languages (`id` / `en`) or re-running LLM translation without re-running CV/OCR.
"""

from collections.abc import Iterator, Sequence
import json
import math
from pathlib import Path
import sys
import traceback
from typing import Annotated, Literal, NotRequired, TypedDict

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.detector import (  # noqa: E402
    ComicTextDetector,
    ImageDimensions,
    TextDirection,
    load_rgb_image,
)
from backend.core.inpainter import LamaInpainter  # noqa: E402
from backend.core.model_manager import (  # noqa: E402
    ModelGroupStatus,
    ModelManager,
    get_execution_provider,
)
from backend.core.ocr import (  # noqa: E402
    OCRRouter,
    SourceLanguage,
    normalize_language,
)
from backend.core.translator import (  # noqa: E402
    ComicTranslator,
    DEFAULT_LM_STUDIO_MODEL,
    LMStudioConnectionError,
    LMStudioTranslationError,
    TargetLanguage,
    TranslatedDialogueItem,
    normalize_source_language,
    normalize_target_language,
)

TextAlign = Literal["left", "center", "right"]


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
    """Status payload returned by `GET /api/models/status` matching SPEC.md Section 5.4."""

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
    model: str
    execution_provider: str
    tiles_inpainted: int
    cleaned_image: str
    inpainted_image: str
    blocks: list[ProcessedBlockItem]
    bubbles: list[ProcessedBlockItem]


class RetranslateResponse(TypedDict):
    """Response payload returned by `POST /api/retranslate`."""

    model: str
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


class RetranslateRequest(BaseModel):
    """Pydantic request body for `POST /api/retranslate`."""

    source_lang: str = "ja"
    target_lang: str = "id"
    source_language: str | None = None
    target_language: str | None = None
    blocks: list[RetranslateBlockModel] = Field(default_factory=list)
    bubbles: list[RetranslateBlockModel] = Field(default_factory=list)
    dialogues: list[RetranslateBlockModel] = Field(default_factory=list)


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


class PipelineService:
    """Manages lazy initialization of CV, OCR, Inpainting, and LLM Translation engines."""

    model_manager: ModelManager
    translator: ComicTranslator
    _detector: ComicTextDetector | None
    _ocr_router: OCRRouter | None
    _inpainter: LamaInpainter | None

    def __init__(self) -> None:
        self.model_manager = ModelManager()
        self.translator = ComicTranslator()
        self._detector = None
        self._ocr_router = None
        self._inpainter = None

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

    def get_lmstudio_status(self) -> LMStudioEndpointStatus:
        """Ping LM Studio (`http://localhost:1235/v1/models`) and format status with display & resolved model names."""
        raw = self.translator.get_status()
        resolved_id = raw["active_model"]

        # When the resolved model ID matches our target Gemma 4 E2B model, expose the canonical
        # target name "Gemma 4 E2B Uncensored HauhauCS Aggressive Q4 K P" as `active_model`
        # and keep the exact LM Studio slug in `resolved_model_id`.
        display_model: str | None = None
        if raw["connected"] and resolved_id is not None:
            if "gemma-4-e2b-uncensored-hauhaucs-aggressive" in resolved_id.lower():
                display_model = DEFAULT_LM_STUDIO_MODEL
            else:
                display_model = resolved_id

        status: LMStudioEndpointStatus = {
            "connected": raw["connected"],
            "endpoint": raw["endpoint"],
            "active_model": display_model,
            "resolved_model_id": resolved_id,
            "preferred_model": self.translator.preferred_model,
            "available_models": raw["available_models"],
        }
        err_msg = raw.get("error")
        if err_msg:
            status["error"] = err_msg
        return status


pipeline_service = PipelineService()

app = FastAPI(
    title="Manga/Manhwa/Manhua Comic Translator API",
    description="Local offline-first Comic Text Detection, Multi-Language OCR, LaMa Inpainting, and LM Studio LLM Translation.",
    version="1.0.0",
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
    """System health check reporting server uptime, ONNX execution provider, and LM Studio status."""
    model_status = pipeline_service.model_manager.get_status()
    lm_status = pipeline_service.get_lmstudio_status()
    return {
        "status": "ok",
        "execution_provider": get_execution_provider(),
        "models_ready": model_status["all_ready"],
        "lm_studio": lm_status,
    }


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


@app.post("/api/process-page")
@app.post("/api/pipeline/process-page")
async def api_process_page(
    image: Annotated[UploadFile | None, File()] = None,
    file: Annotated[UploadFile | None, File()] = None,
    source_lang: Annotated[str, Form()] = "ja",
    target_lang: Annotated[str, Form()] = "id",
) -> ProcessPageResponse:
    """Run full page pipeline: Detector -> OCR -> LaMa Inpainter -> LM Studio Translator.

    Accepts multipart form-data with `image` (or `file`), `source_lang` (`ja`, `ko`, `zh`),
    and `target_lang` (`id`, `en`).
    Returns cleaned base64 PNG image + JSON array of text blocks with `id`, `bbox`,
    `original_text`, `translated_text`, `font_size`, `color`, and `align`.
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

    # Stage 1: Comic Text Detection (Bounding Boxes + Reading-Order Sort + Binary Text Mask)
    try:
        detector = pipeline_service.get_detector()
        det_result = detector.detect(rgb_image, source_language=src_lang)
        detected_bubbles = det_result["bubbles"]
        text_mask = det_result["mask"]
        dimensions = det_result["dimensions"]
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Text detection failed ({type(exc).__name__}): {exc}",
        ) from exc

    # Stage 2: Multi-Language OCR Router (manga-ocr for ja, RapidOCR PP-OCRv4 for ko/zh)
    try:
        ocr_router = pipeline_service.get_ocr_router()
        ocr_bubbles = ocr_router.recognize_bubbles(
            image=rgb_image,
            bubbles=detected_bubbles,
            source_lang=src_lang,
        )
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"OCR ({src_lang}) failed ({type(exc).__name__}): {exc}",
        ) from exc

    # Stage 3: Tile-Based LaMa Inpainting (512x512 ROI patches)
    try:
        inpainter = pipeline_service.get_inpainter()
        bboxes = [b["bbox"] for b in detected_bubbles]
        inpaint_res = inpainter.inpaint_with_metadata(
            image=rgb_image,
            mask=text_mask,
            bboxes=bboxes,
        )
        cleaned_data_uri = LamaInpainter.image_to_base64(inpaint_res["image"])
        # Free LaMa ONNX session VRAM after inpainting per SPEC.md Section 6
        inpainter.unload()
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"LaMa inpainting failed ({type(exc).__name__}): {exc}",
        ) from exc

    # Stage 4: Context-Aware Full-Page Translation via LM Studio
    dialogue_inputs: list[dict[str, object]] = [
        {"id": b["id"], "source_text": b["source_text"]}
        for b in ocr_bubbles
    ]
    try:
        translation_res = pipeline_service.translator.translate_blocks(
            blocks=dialogue_inputs,
            source_lang=src_lang,
            target_lang=tgt_lang,
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
        "model": active_model,
        "execution_provider": det_result["execution_provider"],
        "tiles_inpainted": inpaint_res["tiles_processed"],
        "cleaned_image": cleaned_data_uri,
        "inpainted_image": cleaned_data_uri,
        "blocks": processed_blocks,
        "bubbles": processed_blocks,
    }


@app.post("/api/retranslate")
@app.post("/api/pipeline/translate")
def api_retranslate(payload: RetranslateRequest) -> RetranslateResponse:
    """Re-translate existing text blocks when switching target languages or re-running translation."""
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

    try:
        translation_res = pipeline_service.translator.translate_blocks(
            blocks=raw_blocks,
            source_lang=src_lang,
            target_lang=tgt_lang,
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
        "model": translation_res["model"],
        "source_lang": src_lang,
        "target_lang": tgt_lang,
        "source_language": src_lang,
        "target_language": tgt_lang,
        "translations": trans_map,
        "dialogues": translation_res["dialogues"],
        "blocks": retranslated_blocks,
        "bubbles": retranslated_blocks,
    }


def _resolve_frontend_dist() -> Path | None:
    """Resolve the compiled `frontend/dist` directory when running from source or frozen executable."""
    meipass = getattr(sys, "_MEIPASS", None)
    candidates: list[Path] = []
    if isinstance(meipass, str) and meipass:
        candidates.append(Path(meipass) / "frontend_dist")
    candidates.append(PROJECT_ROOT / "frontend" / "dist")
    for cand in candidates:
        if cand.is_dir() and (cand / "index.html").is_file():
            return cand
    return None


_FRONTEND_DIST_DIR = _resolve_frontend_dist()
if _FRONTEND_DIST_DIR is not None:
    app.mount(
        "/",
        StaticFiles(directory=str(_FRONTEND_DIST_DIR), html=True),
        name="frontend",
    )


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
