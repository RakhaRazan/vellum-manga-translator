"""Comic Archive Importer (.cbz / .zip) and Multi-Page Exporter (.cbz / .pdf).

Provides:
- `natural_sort_key`: Natural numeric sorting (`page_2.jpg` before `page_10.jpg`) across nested archive folders.
- `extract_archive_pages`: Safe extraction of comic page images from `.cbz` / `.zip` archives (with zip-slip,
  zip-bomb, hidden/macOS artifact filtering, and lossless image caching).
- `build_cbz_bytes`: Package an ordered list of composited comic page images into a standard `.cbz` archive.
- `build_pdf_bytes`: Package an ordered list of composited comic page images into a lossless multi-page `.pdf`
  using lightweight `img2pdf` (automatically flattening alpha channels and converting WebP/BMP as needed).
"""

from collections.abc import Sequence
from dataclasses import dataclass
import importlib
import io
import os
from pathlib import Path
import re
from typing import Literal, Protocol, TypedDict, cast
import uuid
import zipfile

from PIL import Image, UnidentifiedImageError

SUPPORTED_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
)
CACHEABLE_IMAGE_EXTENSIONS: frozenset[str] = frozenset({"png", "jpg", "jpeg", "webp"})

MAX_ARCHIVE_PAGES = 500
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024  # 1 GiB safety guard
MAX_SINGLE_IMAGE_BYTES = 64 * 1024 * 1024  # 64 MiB per page safety guard

ExportImageFormat = Literal["original", "png", "jpg"]

_NATURAL_CHUNK_RE = re.compile(r"(\d+)")
_UNSAFE_FILENAME_RE = re.compile(r'[^A-Za-z0-9._ -]+')


class ArchiveImportError(ValueError):
    """Raised when an uploaded `.cbz` / `.zip` archive is invalid, unsafe, or contains no images."""


class ComicExportError(ValueError):
    """Raised when `.cbz` or `.pdf` export fails due to invalid or missing input images."""


class PageDimensions(TypedDict):
    """Image width and height in pixels."""

    width: int
    height: int


class ExtractedArchivePage(TypedDict):
    """Metadata for a single extracted comic page image saved to the local cache."""

    index: int
    filename: str
    archive_path: str
    image_id: str
    extension: str
    saved_path: str
    dimensions: PageDimensions
    size_bytes: int


@dataclass(frozen=True, slots=True)
class ExportPageInput:
    """Single composited page image passed to `build_cbz_bytes` or `build_pdf_bytes`."""

    image_bytes: bytes
    filename: str | None = None


class _Img2PdfConvert(Protocol):
    """Typed callable protocol for `img2pdf.convert`."""

    def __call__(self, images: Sequence[bytes]) -> bytes | None: ...


NaturalToken = tuple[int, int] | tuple[int, str]


def natural_sort_key(path_str: str) -> tuple[tuple[NaturalToken, ...], ...]:
    """Compute a natural sort key (`page_2.jpg` < `page_10.jpg`) across directory segments."""
    normalized = path_str.replace("\\", "/").strip("/")
    segments = normalized.split("/")
    key_segments: list[tuple[NaturalToken, ...]] = []
    for seg in segments:
        tokens: list[NaturalToken] = []
        for part in _NATURAL_CHUNK_RE.split(seg.lower()):
            if not part:
                continue
            if part.isdigit():
                tokens.append((0, int(part)))
            else:
                tokens.append((1, part))
        key_segments.append(tuple(tokens))
    return tuple(key_segments)


def sanitize_export_filename(
    name: str | None,
    extension: Literal["cbz", "pdf"],
    default_stem: str = "translated_comic",
) -> str:
    """Sanitize a user-provided export filename and guarantee the requested `.cbz` / `.pdf` suffix."""
    raw = (name or "").strip()
    if raw:
        raw = Path(raw.replace("\\", "/")).name
    for suffix in (".cbz", ".pdf", ".zip"):
        if raw.lower().endswith(suffix):
            raw = raw[: -len(suffix)]
            break
    cleaned = _UNSAFE_FILENAME_RE.sub("_", raw).strip(" ._")
    if not cleaned:
        cleaned = default_stem
    return f"{cleaned[:100]}.{extension}"


def _sanitize_page_stem(filename: str | None) -> str:
    """Sanitize an optional page filename stem for inclusion inside a `.cbz` archive."""
    if not filename:
        return ""
    stem = Path(filename.replace("\\", "/")).stem
    cleaned = _UNSAFE_FILENAME_RE.sub("_", stem).strip(" ._")
    return cleaned[:64]


def _is_safe_archive_member(member_name: str) -> bool:
    """Return True if a ZIP entry is a safe, non-hidden image path."""
    normalized = member_name.replace("\\", "/")
    if not normalized or normalized.startswith("/") or normalized.endswith("/"):
        return False

    parts = [p for p in normalized.split("/") if p]
    if not parts:
        return False

    for part in parts:
        if part in (".", "..") or part == "__MACOSX" or part.startswith("."):
            return False

    ext = Path(parts[-1]).suffix.lower()
    return ext in SUPPORTED_IMAGE_EXTENSIONS


def _inspect_and_normalize_extracted_image(
    raw_bytes: bytes,
    source_name: str,
) -> tuple[bytes, str, int, int]:
    """Validate image bytes and return `(output_bytes, ext_without_dot, width, height)`."""
    if not raw_bytes:
        raise ArchiveImportError(f"Archive entry '{source_name}' is empty.")
    if len(raw_bytes) > MAX_SINGLE_IMAGE_BYTES:
        raise ArchiveImportError(
            f"Archive entry '{source_name}' exceeds maximum page size ({MAX_SINGLE_IMAGE_BYTES // (1024 * 1024)} MiB)."
        )

    try:
        with Image.open(io.BytesIO(raw_bytes)) as pil_img:
            width, height = pil_img.size
            fmt = (pil_img.format or "").upper()
            # Force full pixel decode check so corrupt files fail cleanly
            pil_img.load()

            if width <= 0 or height <= 0:
                raise ArchiveImportError(
                    f"Archive entry '{source_name}' has invalid dimensions ({width}x{height})."
                )

            if fmt == "JPEG":
                return raw_bytes, "jpg", width, height
            if fmt == "PNG":
                return raw_bytes, "png", width, height
            if fmt == "WEBP":
                return raw_bytes, "webp", width, height

            # Convert BMP / TIFF / other Pillow-readable formats into lossless PNG
            rgb_img = pil_img.convert("RGB")
            out_buf = io.BytesIO()
            rgb_img.save(out_buf, format="PNG")
            return out_buf.getvalue(), "png", width, height
    except ArchiveImportError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ArchiveImportError(
            f"Archive entry '{source_name}' is not a valid image: {exc}"
        ) from exc


def extract_archive_pages(
    archive_bytes: bytes,
    output_dir: Path,
    max_pages: int = MAX_ARCHIVE_PAGES,
) -> list[ExtractedArchivePage]:
    """Extract comic page images from a `.cbz` or `.zip` archive in natural page order.

    Saves each valid page image atomically into `output_dir` and returns ordered metadata.
    """
    if not archive_bytes:
        raise ArchiveImportError("Uploaded archive file is empty.")

    output_dir.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO(archive_bytes)

    try:
        zf = zipfile.ZipFile(buffer, mode="r")
    except zipfile.BadZipFile as exc:
        raise ArchiveImportError(
            "Uploaded file is not a valid .cbz or .zip archive."
        ) from exc

    extracted_pages: list[ExtractedArchivePage] = []

    with zf:
        candidate_infos: list[zipfile.ZipInfo] = [
            info
            for info in zf.infolist()
            if not info.is_dir() and _is_safe_archive_member(info.filename)
        ]

        if not candidate_infos:
            raise ArchiveImportError(
                "No supported comic page images (.png, .jpg, .jpeg, .webp, .bmp) found inside the archive."
            )

        candidate_infos.sort(key=lambda info: natural_sort_key(info.filename))

        if len(candidate_infos) > max_pages:
            raise ArchiveImportError(
                f"Archive contains {len(candidate_infos)} images, exceeding limit of {max_pages} pages."
            )

        total_uncompressed = sum(max(0, info.file_size) for info in candidate_infos)
        if total_uncompressed > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
            raise ArchiveImportError(
                "Archive uncompressed size exceeds the 1 GiB safety limit."
            )

        cumulative_bytes = 0
        for idx, info in enumerate(candidate_infos, start=1):
            try:
                raw_entry = zf.read(info)
            except (zipfile.BadZipFile, RuntimeError, OSError) as exc:
                raise ArchiveImportError(
                    f"Failed to read '{info.filename}' from archive: {exc}"
                ) from exc

            cumulative_bytes += len(raw_entry)
            if cumulative_bytes > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
                raise ArchiveImportError(
                    "Archive extracted payload exceeds the 1 GiB safety limit."
                )

            norm_bytes, ext, width, height = _inspect_and_normalize_extracted_image(
                raw_bytes=raw_entry,
                source_name=info.filename,
            )

            image_id = f"imp_{uuid.uuid4().hex}"
            final_path = (output_dir / f"{image_id}.{ext}").resolve()
            tmp_path = output_dir / f"{image_id}.{ext}.tmp"

            try:
                _ = tmp_path.write_bytes(norm_bytes)
                os.replace(tmp_path, final_path)
            except Exception:
                tmp_path.unlink(missing_ok=True)
                raise

            clean_member = info.filename.replace("\\", "/")
            extracted_pages.append(
                {
                    "index": idx,
                    "filename": Path(clean_member).name,
                    "archive_path": clean_member,
                    "image_id": image_id,
                    "extension": ext,
                    "saved_path": str(final_path),
                    "dimensions": {"width": width, "height": height},
                    "size_bytes": len(norm_bytes),
                }
            )

    return extracted_pages


def _flatten_to_rgb(pil_img: Image.Image) -> Image.Image:
    """Convert any PIL image (including RGBA/LA/palette transparency) onto an opaque white RGB canvas."""
    if pil_img.mode == "RGB":
        return pil_img
    has_alpha = (
        pil_img.mode in ("RGBA", "LA", "PA")
        or ("transparency" in pil_img.info)
    )
    if has_alpha:
        rgba = pil_img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return pil_img.convert("RGB")


def _encode_page_for_cbz(
    raw_bytes: bytes,
    page_index: int,
    image_format: ExportImageFormat = "original",
    jpeg_quality: int = 95,
) -> tuple[bytes, str]:
    """Validate and optionally transcode a single page image for `.cbz` packaging."""
    if not raw_bytes:
        raise ComicExportError(f"Page {page_index} image payload is empty.")
    if len(raw_bytes) > MAX_SINGLE_IMAGE_BYTES:
        raise ComicExportError(f"Page {page_index} exceeds maximum allowed image size.")

    try:
        with Image.open(io.BytesIO(raw_bytes)) as pil_img:
            fmt = (pil_img.format or "").upper()
            pil_img.load()

            if image_format == "original":
                if fmt == "JPEG":
                    return raw_bytes, "jpg"
                if fmt == "PNG":
                    return raw_bytes, "png"
                if fmt == "WEBP":
                    return raw_bytes, "webp"
                out_buf = io.BytesIO()
                _flatten_to_rgb(pil_img).save(out_buf, format="PNG")
                return out_buf.getvalue(), "png"

            if image_format == "jpg":
                out_buf = io.BytesIO()
                _flatten_to_rgb(pil_img).save(
                    out_buf,
                    format="JPEG",
                    quality=max(50, min(100, jpeg_quality)),
                    optimize=True,
                )
                return out_buf.getvalue(), "jpg"

            # image_format == "png"
            out_buf = io.BytesIO()
            if fmt == "PNG":
                return raw_bytes, "png"
            pil_img.convert("RGB").save(out_buf, format="PNG")
            return out_buf.getvalue(), "png"
    except ComicExportError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ComicExportError(
            f"Page {page_index} is not a valid image: {exc}"
        ) from exc


def build_cbz_bytes(
    pages: Sequence[ExportPageInput],
    image_format: ExportImageFormat = "original",
    jpeg_quality: int = 95,
) -> bytes:
    """Package an ordered sequence of composited page images into a downloadable `.cbz` archive."""
    if not pages:
        raise ComicExportError("At least one page image is required to export a .cbz archive.")
    if len(pages) > MAX_ARCHIVE_PAGES:
        raise ComicExportError(
            f"Cannot export {len(pages)} pages; maximum is {MAX_ARCHIVE_PAGES}."
        )

    pad_width = max(3, len(str(len(pages))))
    out_buffer = io.BytesIO()

    with zipfile.ZipFile(out_buffer, mode="w", compression=zipfile.ZIP_STORED) as zf:
        for idx, page in enumerate(pages, start=1):
            encoded_bytes, ext = _encode_page_for_cbz(
                raw_bytes=page.image_bytes,
                page_index=idx,
                image_format=image_format,
                jpeg_quality=jpeg_quality,
            )
            stem = _sanitize_page_stem(page.filename)
            prefix = f"{idx:0{pad_width}d}"
            entry_name = (
                f"{prefix}_{stem}.{ext}"
                if stem and not stem.startswith(prefix)
                else f"{prefix}.{ext}"
            )
            zf.writestr(entry_name, encoded_bytes)

    return out_buffer.getvalue()


def _prepare_image_bytes_for_img2pdf(raw_bytes: bytes, page_index: int) -> bytes:
    """Return `img2pdf`-compatible lossless image bytes (passing JPEG/opaque-PNG through directly)."""
    if not raw_bytes:
        raise ComicExportError(f"Page {page_index} image payload is empty.")
    if len(raw_bytes) > MAX_SINGLE_IMAGE_BYTES:
        raise ComicExportError(f"Page {page_index} exceeds maximum allowed image size.")

    try:
        with Image.open(io.BytesIO(raw_bytes)) as pil_img:
            fmt = (pil_img.format or "").upper()
            mode = pil_img.mode
            has_transparency = "transparency" in pil_img.info
            pil_img.load()

            # Lossless direct passthrough for standard JPEG and opaque PNG
            if fmt == "JPEG" and mode in ("RGB", "L", "CMYK"):
                return raw_bytes
            if fmt == "PNG" and mode in ("RGB", "L", "1") and not has_transparency:
                return raw_bytes

            # Flatten RGBA/LA/transparent PNG or transcode WebP/BMP into opaque lossless PNG
            rgb_img = _flatten_to_rgb(pil_img)
            out_buf = io.BytesIO()
            rgb_img.save(out_buf, format="PNG")
            return out_buf.getvalue()
    except ComicExportError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ComicExportError(
            f"Page {page_index} is not a valid image: {exc}"
        ) from exc


def build_pdf_bytes(pages: Sequence[ExportPageInput]) -> bytes:
    """Package an ordered sequence of composited page images into a multi-page `.pdf` via `img2pdf`."""
    if not pages:
        raise ComicExportError("At least one page image is required to export a .pdf document.")
    if len(pages) > MAX_ARCHIVE_PAGES:
        raise ComicExportError(
            f"Cannot export {len(pages)} pages; maximum is {MAX_ARCHIVE_PAGES}."
        )

    prepared_pages: list[bytes] = [
        _prepare_image_bytes_for_img2pdf(page.image_bytes, page_index=idx)
        for idx, page in enumerate(pages, start=1)
    ]

    try:
        img2pdf_mod = importlib.import_module("img2pdf")
        convert_fn = cast(_Img2PdfConvert, getattr(img2pdf_mod, "convert"))
        pdf_bytes = convert_fn(prepared_pages)
    except Exception as exc:
        raise ComicExportError(f"PDF generation via img2pdf failed: {exc}") from exc

    if not pdf_bytes or not pdf_bytes.startswith(b"%PDF-"):
        raise ComicExportError("img2pdf returned an invalid PDF payload.")
    return pdf_bytes


__all__ = [
    "ArchiveImportError",
    "CACHEABLE_IMAGE_EXTENSIONS",
    "ComicExportError",
    "ExportImageFormat",
    "ExportPageInput",
    "ExtractedArchivePage",
    "MAX_ARCHIVE_PAGES",
    "PageDimensions",
    "SUPPORTED_IMAGE_EXTENSIONS",
    "build_cbz_bytes",
    "build_pdf_bytes",
    "extract_archive_pages",
    "natural_sort_key",
    "sanitize_export_filename",
]
