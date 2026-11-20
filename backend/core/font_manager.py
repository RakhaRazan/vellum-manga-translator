"""Local, System, and Google Fonts Manager for Manga/Manhwa/Manhua Lettering.

Features:
- Scans Windows system font directories (`C:\\Windows\\Fonts`, `%LOCALAPPDATA%\\Microsoft\\Windows\\Fonts`)
  plus a persistent local `fonts/` folder (`%LOCALAPPDATA%/ComicTranslator/fonts` when frozen,
  `<project_root>/fonts` in dev), classifying fonts into `system`, `custom`, and `downloaded`.
- Extracts true font family, subfamily, full name, weight, and italic flags via a fast pure-Python
  OpenType/TrueType SFNT `name` table parser (with `PIL.ImageFont` fallback and zero `fontTools` dependency).
- Provides a curated catalog of 129 free comic, SFX/action, handwriting, webtoon, and CJK
  (Japanese, Korean, Chinese) Google Fonts that can be downloaded on demand as standalone offline `.ttf` files.
- Validates and stores uploaded custom `.ttf`/`.otf`/`.woff2`/`.woff` comic fonts (e.g. `CCWildWords.ttf`).
- Safely resolves font files for `GET /api/fonts/file/{font_name}` with strict path-traversal guards.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sys
import threading
from typing import Literal, NotRequired, TypedDict, cast
from urllib.parse import quote

from PIL import ImageFont
import requests

FontCategory = Literal["system", "custom", "downloaded"]
FontFormat = Literal["ttf", "otf", "woff2", "woff"]
CatalogStyleCategory = Literal[
    "comic",
    "sfx",
    "handwriting",
    "cjk_ja",
    "cjk_ko",
    "cjk_zh",
    "webtoon",
]

SUPPORTED_FONT_EXTENSIONS: frozenset[str] = frozenset({".ttf", ".otf", ".woff2", ".woff"})

FONT_EXT_TO_MEDIA_TYPE: dict[str, str] = {
    ".ttf": "font/ttf",
    ".otf": "font/otf",
    ".woff2": "font/woff2",
    ".woff": "font/woff",
}

FONT_EXT_TO_FORMAT: dict[str, FontFormat] = {
    ".ttf": "ttf",
    ".otf": "otf",
    ".woff2": "woff2",
    ".woff": "woff",
}

MAX_FONT_FILE_BYTES = 50 * 1024 * 1024  # 50 MB safety cap (accommodates large CJK fonts)
GOOGLE_FONTS_CSS2_URL = "https://fonts.googleapis.com/css2"
# Non-WOFF2 User-Agent instructs Google Fonts CSS2 to return a single unsubsetted `.ttf`/`.otf` URL
_GOOGLE_FONTS_TTF_USER_AGENT = "Mozilla/4.0"

_SFNT_MAGICS: tuple[bytes, ...] = (
    b"\x00\x01\x00\x00",  # TrueType
    b"OTTO",              # OpenType CFF
    b"true",              # Apple TrueType
    b"typ1",              # PostScript Type 1 SFNT
)
_WOFF_MAGICS: tuple[bytes, ...] = (b"wOFF", b"wOF2")
_URL_IN_CSS_RE = re.compile(r"url\((https://[^)'\"]+)\)", flags=re.IGNORECASE)
_SAFE_FILENAME_CHARS_RE = re.compile(r"[^A-Za-z0-9._ -]+")


class FontError(RuntimeError):
    """Base error raised for font validation, upload, or download failures."""


class FontValidationError(FontError):
    """Raised when an uploaded or downloaded font file is invalid or unsupported."""


class FontDownloadError(FontError):
    """Raised when fetching a font from Google Fonts fails."""


@dataclass(frozen=True, slots=True)
class CatalogFontSpec:
    """Curated free Google Font entry suitable for comic/manga/manhwa/manhua lettering."""

    id: str
    family: str
    style_category: CatalogStyleCategory
    scripts: tuple[str, ...]
    description: str


class FontEntryDict(TypedDict):
    """Metadata for a single installed font file returned by `GET /api/fonts/list`."""

    id: str
    family: str
    subfamily: str
    full_name: str
    filename: str
    category: FontCategory
    format: FontFormat
    weight: int
    italic: bool
    size_bytes: int
    url: str


class FontVariantDict(TypedDict):
    """Single style/weight variant belonging to a font family group."""

    subfamily: str
    filename: str
    format: FontFormat
    weight: int
    italic: bool
    url: str


class FontFamilyGroupDict(TypedDict):
    """Deduplicated family-level grouping for frontend font selectors."""

    family: str
    category: FontCategory
    default_file: str
    url: str
    variants: list[FontVariantDict]


class CatalogFontDict(TypedDict):
    """Serializable curated Google Font catalog item with live installed state."""

    id: str
    family: str
    style_category: CatalogStyleCategory
    scripts: list[str]
    description: str
    installed: bool
    filename: str | None
    url: str | None


class FontCountsDict(TypedDict):
    """Summary counts returned by `GET /api/fonts/list`."""

    total: int
    families: int
    system: int
    custom: int
    downloaded: int
    catalog: int


class FontListResponseDict(TypedDict):
    """Full response payload for `GET /api/fonts/list`."""

    fonts_dir: str
    system_font_dirs: list[str]
    counts: FontCountsDict
    fonts: list[FontEntryDict]
    families: list[FontFamilyGroupDict]
    catalog: list[CatalogFontDict]


class FontDownloadItemResult(TypedDict):
    """Outcome for a single font requested in `POST /api/fonts/download`."""

    requested: str
    family: str
    status: Literal["downloaded", "skipped", "failed"]
    filename: str | None
    url: str | None
    size_bytes: int
    font: NotRequired[FontEntryDict]
    error: NotRequired[str]


class FontDownloadResponseDict(TypedDict):
    """Response payload for `POST /api/fonts/download`."""

    downloaded_count: int
    skipped_count: int
    failed_count: int
    results: list[FontDownloadItemResult]


class _FontMetadata(TypedDict):
    """Internal extracted OpenType/TrueType naming & style metadata."""

    family: str
    subfamily: str
    full_name: str
    weight: int
    italic: bool


class _DownloadedManifestEntry(TypedDict):
    """Metadata persisted inside `fonts/downloaded/manifest.json`."""

    family: str
    catalog_id: str | None
    source_url: str


def _slugify_font_id(family: str) -> str:
    """Convert a Google Font family name into a stable lowercase kebab-case ID."""
    cleaned = re.sub(r"[^a-z0-9]+", "-", family.lower()).strip("-")
    return cleaned or "font"


def _make_catalog(
    items: Sequence[tuple[str, CatalogStyleCategory, tuple[str, ...], str]],
) -> tuple[CatalogFontSpec, ...]:
    return tuple(
        CatalogFontSpec(
            id=_slugify_font_id(family),
            family=family,
            style_category=cat,
            scripts=scripts,
            description=desc,
        )
        for family, cat, scripts, desc in items
    )


# 129 curated, verified free Google Fonts tailored for Manga, Manhwa, Manhua, and Webtoons
CURATED_GOOGLE_FONTS: tuple[CatalogFontSpec, ...] = _make_catalog(
    (
        # --- Comic & Manga Dialogue (25) ---
        ("Comic Neue", "comic", ("latin",), "Clean classic comic book speech bubble font"),
        ("Bangers", "comic", ("latin",), "Bold comic book impact and shouting font"),
        ("Patrick Hand", "comic", ("latin", "vietnamese"), "Warm hand-lettered dialogue bubbles"),
        ("Patrick Hand SC", "comic", ("latin", "vietnamese"), "Small-caps manga lettering style"),
        ("Architects Daughter", "comic", ("latin",), "Crisp hand-drawn comic dialogue"),
        ("Kalam", "comic", ("latin", "devanagari"), "Expressive hand-lettered comic script"),
        ("Neucha", "comic", ("latin", "cyrillic"), "Casual comic speech bubble lettering"),
        ("Dekko", "comic", ("latin", "devanagari"), "Clean friendly comic speech font"),
        ("Short Stack", "comic", ("latin",), "Quirky rounded indie comic dialogue"),
        ("Boogaloo", "comic", ("latin",), "Retro bouncy cartoon and comic lettering"),
        ("Fredoka", "comic", ("latin", "hebrew"), "Smooth rounded modern comic dialogue"),
        ("Bubblegum Sans", "comic", ("latin",), "Playful cartoon speech bubbles"),
        ("Luckiest Guy", "comic", ("latin",), "Heavy bold cartoon shout and title"),
        ("Permanent Marker", "comic", ("latin",), "Thick felt-tip marker shout and emphasis"),
        ("Chewy", "comic", ("latin",), "Bouncy comedic dialogue and bubbles"),
        ("Sniglet", "comic", ("latin",), "Soft rounded comedy speech bubbles"),
        ("Baloo 2", "comic", ("latin", "vietnamese"), "Rounded friendly webtoon and comic font"),
        ("Changa One", "comic", ("latin",), "Punchy heavy comic emphasis"),
        ("Carter One", "comic", ("latin",), "Expressive bold comic dialogue"),
        ("Lilita One", "comic", ("latin",), "Chunky bold comic shout and title"),
        ("Titan One", "comic", ("latin",), "Ultra-bold impact bubble lettering"),
        ("Freckle Face", "comic", ("latin",), "Fun whimsical indie comic font"),
        ("Coming Soon", "comic", ("latin",), "Neat classic comic strip lettering"),
        ("Walter Turncoat", "comic", ("latin",), "Sketchy hand-inked comic dialogue"),
        ("Schoolbell", "comic", ("latin",), "Light casual slice-of-life comic"),
        # --- Action, Horror & SFX Display (22) ---
        ("Anton", "sfx", ("latin", "vietnamese"), "Tall heavy shout and sound effect"),
        ("Bebas Neue", "sfx", ("latin",), "Condensed bold narration and action SFX"),
        ("Bungee", "sfx", ("latin", "vietnamese"), "Blocky heavy comic sound effects"),
        ("Bungee Shade", "sfx", ("latin", "vietnamese"), "3D shadowed comic title and SFX"),
        ("Creepster", "sfx", ("latin",), "Spooky horror manga title and SFX"),
        ("Nosifer", "sfx", ("latin",), "Dripping horror sound effects"),
        ("Eater", "sfx", ("latin",), "Jagged monster and thriller SFX"),
        ("Butcherman", "sfx", ("latin",), "Gritty dark horror shout font"),
        ("Shojumaru", "sfx", ("latin",), "Japanese brush-inspired action display"),
        ("Rampart One", "sfx", ("latin", "japanese"), "3D outlined manga SFX and chapter title"),
        ("Black Ops One", "sfx", ("latin",), "Stencil military and mecha SFX"),
        ("Russo One", "sfx", ("latin", "cyrillic"), "Heavy sci-fi and battle shout"),
        ("Oswald", "sfx", ("latin", "cyrillic", "vietnamese"), "Narrow dramatic narration and shout"),
        ("Teko", "sfx", ("latin", "devanagari"), "Tall bold condensed action SFX"),
        ("Staatliches", "sfx", ("latin",), "Bold graphic caption and shout"),
        ("Righteous", "sfx", ("latin",), "Stylized sci-fi comic display"),
        ("Press Start 2P", "sfx", ("latin", "cyrillic"), "Retro 8-bit RPG game UI box"),
        ("Silkscreen", "sfx", ("latin",), "Crisp pixel status window for isekai manhwa"),
        ("Orbitron", "sfx", ("latin",), "Futuristic cyberpunk HUD and system box"),
        ("Audiowide", "sfx", ("latin",), "Mecha and sci-fi narration box"),
        ("Rubik Mono One", "sfx", ("latin", "cyrillic"), "Ultra-heavy impact BOOM sound effect"),
        ("Sigmar One", "sfx", ("latin", "vietnamese"), "Heavy rounded cartoon impact SFX"),
        # --- Handwriting, Whispers & Margin Notes (24) ---
        ("Caveat", "handwriting", ("latin", "cyrillic"), "Natural handwritten inner monologue"),
        ("Caveat Brush", "handwriting", ("latin",), "Expressive brush margin notes"),
        ("Indie Flower", "handwriting", ("latin",), "Airy slice-of-life handwritten notes"),
        ("Shadows Into Light", "handwriting", ("latin",), "Delicate handwritten whisper text"),
        ("Shadows Into Light Two", "handwriting", ("latin",), "Clean readable handwritten dialogue"),
        ("Amatic SC", "handwriting", ("latin", "cyrillic", "vietnamese"), "Tall sketchy manga aside and whisper"),
        ("Annie Use Your Telescope", "handwriting", ("latin",), "Cute tall handwritten background note"),
        ("Gloria Hallelujah", "handwriting", ("latin",), "Expressive comic strip handwriting"),
        ("Gochi Hand", "handwriting", ("latin",), "Casual teenager journal handwriting"),
        ("Handlee", "handwriting", ("latin",), "Neat rounded handwritten notes"),
        ("Just Another Hand", "handwriting", ("latin",), "Narrow margin scribbles and whispers"),
        ("Klee One", "handwriting", ("japanese", "latin"), "Neat textbook script supporting Japanese"),
        ("Nanum Pen Script", "handwriting", ("korean", "latin"), "Expressive pen script supporting Korean"),
        ("Nanum Brush Script", "handwriting", ("korean", "latin"), "Dynamic brush script supporting Korean"),
        ("Reenie Beanie", "handwriting", ("latin",), "Soft handwritten diary and whisper"),
        ("Rock Salt", "handwriting", ("latin",), "Rough gritty scratchy scrawl"),
        ("Covered By Your Grace", "handwriting", ("latin",), "Casual felt-tip pen side note"),
        ("Homemade Apple", "handwriting", ("latin",), "Flowing cursive letter and flashback note"),
        ("Nothing You Could Do", "handwriting", ("latin",), "Light sketchy margin commentary"),
        ("Sue Ellen Francisco", "handwriting", ("latin",), "Tall whimsical handwritten aside"),
        ("Pangolin", "handwriting", ("latin", "cyrillic", "vietnamese"), "Comic handwritten hybrid with Cyrillic"),
        ("Marck Script", "handwriting", ("latin", "cyrillic"), "Flowing handwritten script with Cyrillic"),
        ("Mali", "handwriting", ("thai", "latin", "vietnamese"), "Playful handwritten comic font with Thai"),
        ("Sriracha", "handwriting", ("thai", "latin", "vietnamese"), "Expressive comic script with Thai"),
        # --- Japanese Manga CJK (18) ---
        ("Noto Sans JP", "cjk_ja", ("japanese", "latin"), "Clean universal Japanese manga gothic"),
        ("Noto Serif JP", "cjk_ja", ("japanese", "latin"), "Dramatic narration and horror Japanese mincho"),
        ("Kosugi Maru", "cjk_ja", ("japanese", "latin"), "Classic rounded Japanese manga bubble font"),
        ("Kosugi", "cjk_ja", ("japanese", "latin"), "Clean Japanese gothic speech lettering"),
        ("M PLUS Rounded 1c", "cjk_ja", ("japanese", "latin"), "Modern rounded Japanese dialogue font"),
        ("M PLUS 1p", "cjk_ja", ("japanese", "latin"), "Crisp modern Japanese comic sans"),
        ("Zen Maru Gothic", "cjk_ja", ("japanese", "latin"), "Soft friendly rounded Japanese bubbles"),
        ("Zen Kaku Gothic New", "cjk_ja", ("japanese", "latin"), "Modern clean manga speech gothic"),
        ("Zen Kurenaido", "cjk_ja", ("japanese", "latin"), "Handwritten Japanese manga margin notes"),
        ("Yomogi", "cjk_ja", ("japanese", "latin"), "Cute handwritten Japanese background text"),
        ("Yusei Magic", "cjk_ja", ("japanese", "latin"), "Bold marker-style Japanese manga font"),
        ("Hachi Maru Pop", "cjk_ja", ("japanese", "latin"), "Playful bubbly pop Japanese lettering"),
        ("Dela Gothic One", "cjk_ja", ("japanese", "latin"), "Ultra-heavy Japanese manga SFX and shout"),
        ("Reggae One", "cjk_ja", ("japanese", "latin"), "Spiky shonen action Japanese SFX"),
        ("RocknRoll One", "cjk_ja", ("japanese", "latin"), "Energetic pop action Japanese display"),
        ("DotGothic16", "cjk_ja", ("japanese", "latin"), "Pixel RPG status window Japanese font"),
        ("Shippori Mincho", "cjk_ja", ("japanese", "latin"), "Literary classic Japanese narration mincho"),
        ("Kaisei Opti", "cjk_ja", ("japanese", "latin"), "Stylized Japanese book and narration serif"),
        # --- Korean Manhwa CJK (17) ---
        ("Noto Sans KR", "cjk_ko", ("korean", "latin"), "Clean universal Korean webtoon gothic"),
        ("Noto Serif KR", "cjk_ko", ("korean", "latin"), "Dramatic narration Korean serif"),
        ("Jua", "cjk_ko", ("korean", "latin"), "Playful rounded Korean comic dialogue"),
        ("Do Hyeon", "cjk_ko", ("korean", "latin"), "Bold impactful Korean manhwa shout"),
        ("Black Han Sans", "cjk_ko", ("korean", "latin"), "Ultra-heavy Korean battle SFX"),
        ("Gaegu", "cjk_ko", ("korean", "latin"), "Cute handwritten Korean manhwa notes"),
        ("Hi Melody", "cjk_ko", ("korean", "latin"), "Casual handwritten Korean speech"),
        ("Gamja Flower", "cjk_ko", ("korean", "latin"), "Playful quirky Korean comic lettering"),
        ("Poor Story", "cjk_ko", ("korean", "latin"), "Charming indie Korean webtoon font"),
        ("Cute Font", "cjk_ko", ("korean", "latin"), "Tall cute Korean comic dialogue"),
        ("Yeon Sung", "cjk_ko", ("korean", "latin"), "Traditional Korean brush-style dialogue"),
        ("Stylish", "cjk_ko", ("korean", "latin"), "Clean expressive Korean webtoon lettering"),
        ("Song Myung", "cjk_ko", ("korean", "latin"), "Classic literary Korean narration serif"),
        ("Gowun Dodum", "cjk_ko", ("korean", "latin"), "Soft readable Korean webtoon sans"),
        ("Gowun Batang", "cjk_ko", ("korean", "latin"), "Warm literary Korean narration"),
        ("Dongle", "cjk_ko", ("korean", "latin"), "Soft rounded bubbly Korean dialogue"),
        ("Single Day", "cjk_ko", ("korean", "latin"), "Expressive hand-drawn Korean lettering"),
        # --- Chinese Manhua CJK (11) ---
        ("Noto Sans SC", "cjk_zh", ("chinese-simplified", "latin"), "Clean Simplified Chinese manhua sans"),
        ("Noto Sans TC", "cjk_zh", ("chinese-traditional", "latin"), "Clean Traditional Chinese manhua sans"),
        ("Noto Serif SC", "cjk_zh", ("chinese-simplified", "latin"), "Classic Simplified Chinese narration"),
        ("Noto Serif TC", "cjk_zh", ("chinese-traditional", "latin"), "Classic Traditional Chinese narration"),
        ("ZCOOL KuaiLe", "cjk_zh", ("chinese-simplified", "latin"), "Fun rounded comic Simplified Chinese"),
        ("ZCOOL QingKe HuangYou", "cjk_zh", ("chinese-simplified", "latin"), "Sharp modern comic Chinese display"),
        ("ZCOOL XiaoWei", "cjk_zh", ("chinese-simplified", "latin"), "Elegant literary Chinese manhua font"),
        ("Ma Shan Zheng", "cjk_zh", ("chinese-simplified", "latin"), "Bold martial-arts brush calligraphy"),
        ("Zhi Mang Xing", "cjk_zh", ("chinese-simplified", "latin"), "Flowing wuxia/xianxia brush script"),
        ("Liu Jian Mao Cao", "cjk_zh", ("chinese-simplified", "latin"), "Wild cursive wuxia action SFX"),
        ("Long Cang", "cjk_zh", ("chinese-simplified", "latin"), "Expressive Chinese handwritten notes"),
        # --- Clean Webtoon, Narration & Multilingual (12) ---
        ("Inter", "webtoon", ("latin", "cyrillic", "vietnamese"), "Clean modern webtoon dialogue sans"),
        ("Poppins", "webtoon", ("latin", "devanagari"), "Geometric clean comic speech sans"),
        ("Nunito", "webtoon", ("latin", "cyrillic", "vietnamese"), "Friendly rounded webtoon lettering"),
        ("Quicksand", "webtoon", ("latin", "vietnamese"), "Light airy geometric dialogue"),
        ("Rubik", "webtoon", ("latin", "cyrillic", "arabic", "hebrew"), "Versatile rounded multilingual sans"),
        ("Plus Jakarta Sans", "webtoon", ("latin", "vietnamese"), "Crisp Indonesian and SEA webtoon sans"),
        ("Lexend", "webtoon", ("latin", "vietnamese"), "High-readability speech bubble sans"),
        ("Merriweather", "webtoon", ("latin", "cyrillic", "vietnamese"), "Classic comic narration box serif"),
        ("Playfair Display", "webtoon", ("latin", "cyrillic", "vietnamese"), "Elegant fantasy and romance narration"),
        ("Cairo", "webtoon", ("arabic", "latin"), "Clean modern Arabic comic lettering"),
        ("Tajawal", "webtoon", ("arabic", "latin"), "Friendly Arabic speech bubble font"),
        ("Kanit", "webtoon", ("thai", "latin", "vietnamese"), "Clean modern Thai and Latin webtoon sans"),
    )
)

_CATALOG_BY_ID: dict[str, CatalogFontSpec] = {spec.id: spec for spec in CURATED_GOOGLE_FONTS}
_CATALOG_BY_FAMILY_LOWER: dict[str, CatalogFontSpec] = {
    spec.family.lower(): spec for spec in CURATED_GOOGLE_FONTS
}


def _json_loads(raw: str) -> object:
    """Parse a JSON string into a typed `object`."""
    res: object = json.loads(raw)  # pyright: ignore[reportAny]
    return res


def is_frozen_executable() -> bool:
    """Return True when running inside a PyInstaller frozen executable."""
    return bool(getattr(sys, "frozen", False))


def get_fonts_dir() -> Path:
    """Resolve the persistent local `fonts/` directory.

    Priority:
    1. `MANGA_FONTS_DIR` environment variable (optional override).
    2. `%LOCALAPPDATA%/ComicTranslator/fonts` when frozen as a Windows executable.
    3. `<project_root>/fonts` during development.
    """
    env_dir = os.environ.get("MANGA_FONTS_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()

    if is_frozen_executable():
        local_app_data = os.environ.get("LOCALAPPDATA")
        base_dir = (
            Path(local_app_data).expanduser().resolve()
            if local_app_data
            else (Path.home() / "AppData" / "Local").resolve()
        )
        return base_dir / "ComicTranslator" / "fonts"

    workspace_root = Path(__file__).resolve().parents[2]
    return workspace_root / "fonts"


def get_windows_system_font_dirs() -> list[Path]:
    """Return existing Windows system and per-user font directories."""
    candidates: list[Path] = []

    windir = os.environ.get("WINDIR") or r"C:\Windows"
    candidates.append(Path(windir) / "Fonts")

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates.append(Path(local_app_data) / "Microsoft" / "Windows" / "Fonts")
    else:
        candidates.append(Path.home() / "AppData" / "Local" / "Microsoft" / "Windows" / "Fonts")

    existing: list[Path] = []
    seen: set[str] = set()
    for d in candidates:
        try:
            resolved = d.resolve()
        except OSError:
            continue
        key = str(resolved).lower()
        if key not in seen and resolved.is_dir():
            seen.add(key)
            existing.append(resolved)
    return existing


def _infer_weight_and_italic(subfamily: str, stem: str) -> tuple[int, bool]:
    """Infer CSS numeric font-weight (100-900) and italic boolean from subfamily/filename."""
    text = f"{subfamily} {stem}".lower()
    is_italic = any(tok in text for tok in ("italic", "oblique", "slanted"))
    if not is_italic and re.search(r"(?:^|[-_ ])i$", stem.lower()):
        is_italic = True

    if any(tok in text for tok in ("black", "heavy", "ultra", "extrablack")):
        weight = 900
    elif any(tok in text for tok in ("extrabold", "extra bold", "ultrabold")):
        weight = 800
    elif any(tok in text for tok in ("semibold", "semi bold", "demibold", "demi bold")):
        weight = 600
    elif "bold" in text or re.search(r"bd(?:i|it)?$", stem.lower()):
        weight = 700
    elif "medium" in text:
        weight = 500
    elif any(tok in text for tok in ("extralight", "extra light", "ultralight")):
        weight = 200
    elif "thin" in text or "hairline" in text:
        weight = 100
    elif "light" in text:
        weight = 300
    else:
        weight = 400

    return weight, is_italic


def _decode_sfnt_name_bytes(
    raw: bytes,
    platform_id: int,
    encoding_id: int,
) -> str | None:
    """Decode an OpenType `name` table record string cleanly."""
    if not raw:
        return None
    try:
        if platform_id == 3 or platform_id == 0:
            # Windows / Unicode platform strings are UTF-16-BE
            return raw.decode("utf-16-be", errors="ignore").replace("\x00", "").strip()
        if platform_id == 1 and encoding_id == 0:
            return raw.decode("mac_roman", errors="ignore").replace("\x00", "").strip()
        return raw.decode("latin-1", errors="ignore").replace("\x00", "").strip()
    except Exception:
        return None


def _u16_be(buf: bytes, offset: int) -> int:
    """Read an unsigned 16-bit big-endian integer from `buf` at `offset`."""
    return int.from_bytes(buf[offset : offset + 2], "big")


def _u32_be(buf: bytes, offset: int) -> int:
    """Read an unsigned 32-bit big-endian integer from `buf` at `offset`."""
    return int.from_bytes(buf[offset : offset + 4], "big")


def _parse_sfnt_name_table(path: Path) -> tuple[str | None, str | None, str | None]:
    """Fast pure-Python OpenType/TrueType `name` table parser returning `(family, subfamily, full_name)`."""
    try:
        with path.open("rb") as fh:
            header = fh.read(12)
            if len(header) < 12:
                return None, None, None
            sfnt_magic = header[:4]
            if sfnt_magic not in _SFNT_MAGICS:
                return None, None, None
            num_tables = _u16_be(header, 4)
            if num_tables == 0 or num_tables > 256:
                return None, None, None

            table_dir = fh.read(num_tables * 16)
            if len(table_dir) < num_tables * 16:
                return None, None, None

            name_offset = 0
            name_length = 0
            for idx in range(num_tables):
                rec_start = idx * 16
                tag = table_dir[rec_start : rec_start + 4]
                if tag == b"name":
                    name_offset = _u32_be(table_dir, rec_start + 8)
                    name_length = _u32_be(table_dir, rec_start + 12)
                    break

            if name_offset == 0 or name_length < 6 or name_length > 2_000_000:
                return None, None, None

            fh.seek(name_offset)
            name_data = fh.read(name_length)
            if len(name_data) < 6:
                return None, None, None

            count = _u16_be(name_data, 2)
            string_offset = _u16_be(name_data, 4)
            if count == 0 or 6 + count * 12 > len(name_data):
                return None, None, None

            # Track best (priority, value) per target nameID:
            # 1=Family, 2=Subfamily, 4=FullName, 16=TypographicFamily, 17=TypographicSubfamily
            best: dict[int, tuple[int, str]] = {}
            target_ids = {1, 2, 4, 16, 17}

            for idx in range(count):
                rec_pos = 6 + idx * 12
                platform_id = _u16_be(name_data, rec_pos)
                encoding_id = _u16_be(name_data, rec_pos + 2)
                language_id = _u16_be(name_data, rec_pos + 4)
                name_id = _u16_be(name_data, rec_pos + 6)
                length = _u16_be(name_data, rec_pos + 8)
                offset = _u16_be(name_data, rec_pos + 10)
                if name_id not in target_ids or length == 0:
                    continue
                str_start = string_offset + offset
                str_end = str_start + length
                if str_end > len(name_data):
                    continue

                decoded = _decode_sfnt_name_bytes(
                    name_data[str_start:str_end], platform_id, encoding_id
                )
                if not decoded:
                    continue

                # Prefer Windows en-US (3, 0x0409) > Windows other > Unicode (0) > Mac English (1, 0) > other
                if platform_id == 3 and language_id == 0x0409:
                    priority = 100
                elif platform_id == 3:
                    priority = 80
                elif platform_id == 0:
                    priority = 70
                elif platform_id == 1 and language_id == 0:
                    priority = 60
                else:
                    priority = 40

                prev = best.get(name_id)
                if prev is None or priority > prev[0]:
                    best[name_id] = (priority, decoded)

            family = (
                best[16][1]
                if 16 in best
                else (best[1][1] if 1 in best else None)
            )
            subfamily = (
                best[17][1]
                if 17 in best
                else (best[2][1] if 2 in best else None)
            )
            full_name = best[4][1] if 4 in best else None
            return family, subfamily, full_name
    except OSError:
        return None, None, None


def extract_font_metadata(
    path: Path,
    override_family: str | None = None,
) -> _FontMetadata:
    """Extract font family, subfamily, full name, weight, and italic info from a font file."""
    family, subfamily, full_name = _parse_sfnt_name_table(path)

    # Fallback to Pillow FreeType for unusual `.ttf`/`.otf` tables
    if not family and path.suffix.lower() in (".ttf", ".otf"):
        try:
            pil_font = ImageFont.truetype(str(path), size=12)
            pil_fam, pil_sub = pil_font.getname()
            if pil_fam:
                family = str(pil_fam).strip()
            if pil_sub and not subfamily:
                subfamily = str(pil_sub).strip()
        except Exception:
            pass

    has_override = bool(override_family and override_family.strip())
    if override_family and override_family.strip():
        family = override_family.strip()

    if not family:
        pretty_stem = re.sub(r"[_-]+", " ", path.stem).strip()
        family = pretty_stem or path.stem

    clean_sub = (subfamily or "Regular").strip() or "Regular"
    clean_full = (
        full_name.strip()
        if (not has_override and full_name and full_name.strip())
        else (family if clean_sub.lower() == "regular" else f"{family} {clean_sub}")
    )
    weight, is_italic = _infer_weight_and_italic(clean_sub, path.stem)
    return {
        "family": family,
        "subfamily": clean_sub,
        "full_name": clean_full,
        "weight": weight,
        "italic": is_italic,
    }


def validate_font_bytes(raw_bytes: bytes, extension: str) -> None:
    """Verify font file size and SFNT/WOFF magic signature before accepting upload/download."""
    ext = extension.lower()
    if ext not in SUPPORTED_FONT_EXTENSIONS:
        raise FontValidationError(
            f"Unsupported font extension {extension!r}. Supported formats: .ttf, .otf, .woff2, .woff."
        )
    if not raw_bytes or len(raw_bytes) < 64:
        raise FontValidationError("Font file is empty or truncated.")
    if len(raw_bytes) > MAX_FONT_FILE_BYTES:
        raise FontValidationError(
            f"Font file exceeds maximum allowed size ({MAX_FONT_FILE_BYTES // (1024 * 1024)} MB)."
        )

    header = raw_bytes[:4]
    if ext in (".ttf", ".otf"):
        if header not in _SFNT_MAGICS:
            raise FontValidationError(
                f"Invalid TrueType/OpenType font header ({header!r}) for '{ext}' file."
            )
    elif ext == ".woff" and header != b"wOFF":
        raise FontValidationError("Invalid WOFF font header.")
    elif ext == ".woff2" and header != b"wOF2":
        raise FontValidationError("Invalid WOFF2 font header.")


def sanitize_font_filename(filename: str) -> str:
    """Sanitize a font filename strictly against path traversal while preserving clean characters."""
    base_name = Path(filename.replace("\\", "/")).name.strip()
    if not base_name:
        raise FontValidationError("Font filename cannot be empty.")

    stem = Path(base_name).stem
    ext = Path(base_name).suffix.lower()
    if ext not in SUPPORTED_FONT_EXTENSIONS:
        raise FontValidationError(
            f"Unsupported font file format '{ext}'. Upload a .ttf, .otf, .woff2, or .woff file."
        )

    clean_stem = _SAFE_FILENAME_CHARS_RE.sub("_", stem).strip(" ._")
    if not clean_stem:
        clean_stem = "custom_font"
    return f"{clean_stem}{ext}"


def _build_font_url(filename: str) -> str:
    """Construct the `/api/fonts/file/{font_name}` URL for `@font-face` loading."""
    return f"/api/fonts/file/{quote(filename, safe='')}"


class FontManager:
    """Scans Windows system fonts, manages local custom/downloaded fonts, and downloads Google Fonts."""

    fonts_dir: Path
    downloaded_dir: Path
    custom_subdir: Path
    manifest_path: Path
    _lock: threading.RLock
    _system_cache_key: tuple[tuple[str, float, int], ...] | None
    _system_cache_entries: list[tuple[FontEntryDict, Path]]

    def __init__(self, fonts_dir: Path | None = None) -> None:
        self.fonts_dir = (fonts_dir if fonts_dir is not None else get_fonts_dir()).resolve()
        self.downloaded_dir = self.fonts_dir / "downloaded"
        self.custom_subdir = self.fonts_dir / "custom"
        self.manifest_path = self.downloaded_dir / "manifest.json"
        self._lock = threading.RLock()
        self._system_cache_key = None
        self._system_cache_entries = []

    def ensure_dirs(self) -> None:
        """Create the local persistent `fonts/` and `fonts/downloaded/` directories if absent."""
        self.fonts_dir.mkdir(parents=True, exist_ok=True)
        self.downloaded_dir.mkdir(parents=True, exist_ok=True)

    def _load_downloaded_manifest(self) -> dict[str, _DownloadedManifestEntry]:
        """Load `fonts/downloaded/manifest.json` mapping filename -> catalog metadata."""
        if not self.manifest_path.is_file():
            return {}
        try:
            parsed = _json_loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(parsed, Mapping):
            return {}
        result: dict[str, _DownloadedManifestEntry] = {}
        for k, v in cast(Mapping[object, object], parsed).items():
            if isinstance(v, Mapping):
                vmap = cast(Mapping[str, object], v)
                fam = str(vmap.get("family", "")).strip()
                raw_cid = vmap.get("catalog_id")
                cid = str(raw_cid).strip() if isinstance(raw_cid, str) and raw_cid.strip() else None
                src_url = str(vmap.get("source_url", "")).strip()
                if fam:
                    result[str(k)] = {
                        "family": fam,
                        "catalog_id": cid,
                        "source_url": src_url,
                    }
        return result

    def _save_downloaded_manifest(
        self,
        manifest: Mapping[str, _DownloadedManifestEntry],
    ) -> None:
        """Atomically persist `fonts/downloaded/manifest.json`."""
        self.ensure_dirs()
        tmp_path = self.manifest_path.with_suffix(".json.tmp")
        serialized = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        try:
            tmp_path.write_text(serialized, encoding="utf-8")
            os.replace(tmp_path, self.manifest_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    def _build_entry(
        self,
        path: Path,
        category: FontCategory,
        override_family: str | None = None,
    ) -> FontEntryDict | None:
        """Inspect a single font file on disk and construct a `FontEntryDict`."""
        ext = path.suffix.lower()
        fmt = FONT_EXT_TO_FORMAT.get(ext)
        if fmt is None:
            return None
        try:
            stat = path.stat()
        except OSError:
            return None
        if stat.st_size < 64:
            return None

        meta = extract_font_metadata(path, override_family=override_family)
        return {
            "id": f"{category}:{path.name}",
            "family": meta["family"],
            "subfamily": meta["subfamily"],
            "full_name": meta["full_name"],
            "filename": path.name,
            "category": category,
            "format": fmt,
            "weight": meta["weight"],
            "italic": meta["italic"],
            "size_bytes": int(stat.st_size),
            "url": _build_font_url(path.name),
        }

    def _scan_local_fonts(self) -> list[tuple[FontEntryDict, Path]]:
        """Scan local `fonts/` (custom) and `fonts/downloaded/` (downloaded) directories."""
        entries: list[tuple[FontEntryDict, Path]] = []
        seen_names: set[str] = set()

        # 1. Custom fonts: top-level `fonts/*.{ttf,otf,woff2,woff}` and `fonts/custom/*`
        for custom_dir in (self.fonts_dir, self.custom_subdir):
            if not custom_dir.is_dir():
                continue
            try:
                files = sorted(custom_dir.iterdir(), key=lambda p: p.name.lower())
            except OSError:
                continue
            for p in files:
                if not p.is_file() or p.suffix.lower() not in SUPPORTED_FONT_EXTENSIONS:
                    continue
                name_key = p.name.lower()
                if name_key in seen_names:
                    continue
                entry = self._build_entry(p, category="custom")
                if entry is not None:
                    seen_names.add(name_key)
                    entries.append((entry, p))

        # 2. Downloaded Google Fonts: `fonts/downloaded/*`
        manifest = self._load_downloaded_manifest()
        if self.downloaded_dir.is_dir():
            try:
                files = sorted(self.downloaded_dir.iterdir(), key=lambda p: p.name.lower())
            except OSError:
                files = []
            for p in files:
                if not p.is_file() or p.suffix.lower() not in SUPPORTED_FONT_EXTENSIONS:
                    continue
                name_key = p.name.lower()
                if name_key in seen_names:
                    continue
                man_entry = manifest.get(p.name)
                override_fam = man_entry["family"] if man_entry else None
                entry = self._build_entry(p, category="downloaded", override_family=override_fam)
                if entry is not None:
                    seen_names.add(name_key)
                    entries.append((entry, p))

        return entries

    def _scan_system_fonts(self, force_refresh: bool = False) -> list[tuple[FontEntryDict, Path]]:
        """Scan Windows system and user font directories with directory-mtime caching."""
        sys_dirs = get_windows_system_font_dirs()
        cache_key_items: list[tuple[str, float, int]] = []
        for d in sys_dirs:
            try:
                st = d.stat()
                cache_key_items.append((str(d).lower(), st.st_mtime, st.st_size))
            except OSError:
                continue
        cache_key = tuple(cache_key_items)

        with self._lock:
            if (
                not force_refresh
                and self._system_cache_key == cache_key
                and self._system_cache_entries
            ):
                return list(self._system_cache_entries)

        entries: list[tuple[FontEntryDict, Path]] = []
        seen_names: set[str] = set()
        # Scan user-installed Windows fonts first, then C:\Windows\Fonts
        ordered_dirs = list(reversed(sys_dirs)) if len(sys_dirs) > 1 else sys_dirs
        for font_dir in ordered_dirs:
            try:
                files = sorted(font_dir.iterdir(), key=lambda p: p.name.lower())
            except OSError:
                continue
            for p in files:
                if not p.is_file() or p.suffix.lower() not in SUPPORTED_FONT_EXTENSIONS:
                    continue
                name_key = p.name.lower()
                if name_key in seen_names:
                    continue
                entry = self._build_entry(p, category="system")
                if entry is not None:
                    seen_names.add(name_key)
                    entries.append((entry, p))

        with self._lock:
            self._system_cache_key = cache_key
            self._system_cache_entries = list(entries)

        return entries

    def _collect_all_installed(
        self,
        force_refresh: bool = False,
    ) -> list[tuple[FontEntryDict, Path]]:
        """Collect all installed fonts in priority order: `custom` -> `downloaded` -> `system`."""
        local_entries = self._scan_local_fonts()
        system_entries = self._scan_system_fonts(force_refresh=force_refresh)

        combined: list[tuple[FontEntryDict, Path]] = list(local_entries)
        seen_filenames: set[str] = {entry["filename"].lower() for entry, _ in local_entries}
        for entry, path in system_entries:
            if entry["filename"].lower() not in seen_filenames:
                seen_filenames.add(entry["filename"].lower())
                combined.append((entry, path))

        category_rank: dict[FontCategory, int] = {
            "custom": 0,
            "downloaded": 1,
            "system": 2,
        }
        combined.sort(
            key=lambda pair: (
                category_rank[pair[0]["category"]],
                pair[0]["family"].lower(),
                pair[0]["italic"],
                abs(pair[0]["weight"] - 400),
                pair[0]["filename"].lower(),
            )
        )
        return combined

    def _group_by_family(
        self,
        entries: Sequence[FontEntryDict],
    ) -> list[FontFamilyGroupDict]:
        """Group individual font variants into deduplicated family objects."""
        category_rank: dict[FontCategory, int] = {
            "custom": 0,
            "downloaded": 1,
            "system": 2,
        }
        grouped: dict[str, list[FontEntryDict]] = {}
        for item in entries:
            key = item["family"].lower()
            grouped.setdefault(key, []).append(item)

        groups: list[FontFamilyGroupDict] = []
        for items in grouped.values():
            # Pick best default variant (prefer Regular non-italic weight 400, prioritizing custom/downloaded)
            items_sorted = sorted(
                items,
                key=lambda x: (
                    category_rank[x["category"]],
                    x["italic"],
                    abs(x["weight"] - 400),
                    x["filename"].lower(),
                ),
            )
            primary = items_sorted[0]
            variants: list[FontVariantDict] = [
                {
                    "subfamily": v["subfamily"],
                    "filename": v["filename"],
                    "format": v["format"],
                    "weight": v["weight"],
                    "italic": v["italic"],
                    "url": v["url"],
                }
                for v in items_sorted
            ]
            groups.append(
                {
                    "family": primary["family"],
                    "category": primary["category"],
                    "default_file": primary["filename"],
                    "url": primary["url"],
                    "variants": variants,
                }
            )

        groups.sort(
            key=lambda g: (category_rank[g["category"]], g["family"].lower())
        )
        return groups

    def list_fonts(
        self,
        category: FontCategory | None = None,
        include_catalog: bool = True,
        force_refresh: bool = False,
    ) -> FontListResponseDict:
        """Enumerate system, custom, and downloaded fonts along with the curated Google Fonts catalog."""
        all_pairs = self._collect_all_installed(force_refresh=force_refresh)
        all_entries = [entry for entry, _ in all_pairs]

        system_count = sum(1 for e in all_entries if e["category"] == "system")
        custom_count = sum(1 for e in all_entries if e["category"] == "custom")
        downloaded_count = sum(1 for e in all_entries if e["category"] == "downloaded")

        filtered_entries = (
            [e for e in all_entries if e["category"] == category]
            if category is not None
            else all_entries
        )
        families = self._group_by_family(filtered_entries)

        # Map installed family name (lowercase) -> preferred FontEntryDict for catalog status
        installed_by_family: dict[str, FontEntryDict] = {}
        for item in all_entries:
            fam_key = item["family"].lower()
            if fam_key not in installed_by_family:
                installed_by_family[fam_key] = item

        catalog_list: list[CatalogFontDict] = []
        if include_catalog:
            for spec in CURATED_GOOGLE_FONTS:
                matched = installed_by_family.get(spec.family.lower())
                catalog_list.append(
                    {
                        "id": spec.id,
                        "family": spec.family,
                        "style_category": spec.style_category,
                        "scripts": list(spec.scripts),
                        "description": spec.description,
                        "installed": matched is not None,
                        "filename": matched["filename"] if matched is not None else None,
                        "url": matched["url"] if matched is not None else None,
                    }
                )

        return {
            "fonts_dir": str(self.fonts_dir),
            "system_font_dirs": [str(d) for d in get_windows_system_font_dirs()],
            "counts": {
                "total": len(filtered_entries),
                "families": len(families),
                "system": system_count,
                "custom": custom_count,
                "downloaded": downloaded_count,
                "catalog": len(CURATED_GOOGLE_FONTS),
            },
            "fonts": filtered_entries,
            "families": families,
            "catalog": catalog_list,
        }

    def resolve_font_file(self, font_name: str) -> tuple[Path, str]:
        """Resolve `font_name` (filename, stem, or family name) safely to `(file_path, media_type)`."""
        cleaned = font_name.strip()
        if (
            not cleaned
            or "/" in cleaned
            or "\\" in cleaned
            or ".." in cleaned
            or "\x00" in cleaned
        ):
            raise FontValidationError(f"Invalid font identifier: {font_name!r}")

        allowed_roots = [
            self.fonts_dir.resolve(),
            self.downloaded_dir.resolve(),
            self.custom_subdir.resolve(),
            *[d.resolve() for d in get_windows_system_font_dirs()],
        ]

        def _is_within_allowed(path: Path) -> bool:
            try:
                resolved = path.resolve()
            except OSError:
                return False
            return resolved.is_file() and any(
                resolved.parent == root for root in allowed_roots
            )

        all_pairs = self._collect_all_installed(force_refresh=False)
        lower_query = cleaned.lower()

        # 1. Exact filename match (e.g. `comic.ttf`, `CCWildWords.ttf`, `Bangers.ttf`)
        for entry, path in all_pairs:
            if entry["filename"].lower() == lower_query and _is_within_allowed(path):
                media_type = FONT_EXT_TO_MEDIA_TYPE.get(path.suffix.lower(), "application/octet-stream")
                return path.resolve(), media_type

        # 2. Stem match (e.g. `comic` -> `comic.ttf`, `CCWildWords` -> `CCWildWords.ttf`)
        for entry, path in all_pairs:
            if path.stem.lower() == lower_query and _is_within_allowed(path):
                media_type = FONT_EXT_TO_MEDIA_TYPE.get(path.suffix.lower(), "application/octet-stream")
                return path.resolve(), media_type

        # 3. Family name or full name match (e.g. `Comic Sans MS` -> Regular `comic.ttf`)
        for entry, path in all_pairs:
            if (
                entry["family"].lower() == lower_query
                or entry["full_name"].lower() == lower_query
            ) and _is_within_allowed(path):
                media_type = FONT_EXT_TO_MEDIA_TYPE.get(path.suffix.lower(), "application/octet-stream")
                return path.resolve(), media_type

        raise FileNotFoundError(f"Font '{cleaned}' not found in local or system font directories.")

    def save_uploaded_font(
        self,
        filename: str,
        raw_bytes: bytes,
        override_family: str | None = None,
        overwrite: bool = True,
    ) -> FontEntryDict:
        """Validate and atomically store an uploaded custom `.ttf`/`.otf`/`.woff2`/`.woff` font in `fonts/`."""
        safe_name = sanitize_font_filename(filename)
        ext = Path(safe_name).suffix.lower()
        validate_font_bytes(raw_bytes, ext)

        self.ensure_dirs()
        target_path = (self.fonts_dir / safe_name).resolve()
        if target_path.parent != self.fonts_dir.resolve():
            raise FontValidationError("Invalid target font path.")

        if target_path.exists() and not overwrite:
            raise FontValidationError(
                f"Custom font '{safe_name}' already exists. Set overwrite=true to replace it."
            )

        tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
        try:
            tmp_path.write_bytes(raw_bytes)
            # Extra FreeType sanity check for .ttf/.otf so corrupt binaries are rejected
            if ext in (".ttf", ".otf"):
                try:
                    _ = ImageFont.truetype(str(tmp_path), size=14)
                except Exception as exc:
                    raise FontValidationError(
                        f"Uploaded font '{safe_name}' could not be parsed as a valid font: {exc}"
                    ) from exc
            os.replace(tmp_path, target_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

        entry = self._build_entry(
            target_path,
            category="custom",
            override_family=override_family,
        )
        if entry is None:
            raise FontValidationError(f"Failed to index uploaded font '{safe_name}'.")
        return entry

    def _resolve_google_font_binary_url(
        self,
        family: str,
        weight: str | None = None,
        timeout: float = 15.0,
    ) -> str:
        """Query Google Fonts CSS2 API with a non-WOFF2 User-Agent to obtain a direct `.ttf`/`.otf` URL."""
        family_queries: list[str] = []
        if weight and weight.strip():
            family_queries.append(f"{family}:wght@{weight.strip()}")
        family_queries.extend([family, f"{family}:wght@400", f"{family}:wght@500", f"{family}:wght@700"])

        headers = {"User-Agent": _GOOGLE_FONTS_TTF_USER_AGENT}
        last_error = f"Font '{family}' not found on Google Fonts."

        for query in dict.fromkeys(family_queries):
            try:
                resp = requests.get(
                    GOOGLE_FONTS_CSS2_URL,
                    params={"family": query},
                    headers=headers,
                    timeout=timeout,
                )
            except requests.RequestException as exc:
                raise FontDownloadError(
                    f"Network error contacting Google Fonts for '{family}': {exc}"
                ) from exc

            if resp.status_code != 200:
                last_error = f"Google Fonts returned HTTP {resp.status_code} for '{query}'."
                continue

            urls = [m.group(1) for m in _URL_IN_CSS_RE.finditer(resp.text)]
            if not urls:
                last_error = f"Google Fonts CSS for '{query}' did not contain a downloadable font URL."
                continue

            # Prefer .ttf or .otf over .woff2/.woff
            for candidate in urls:
                lower_u = candidate.lower()
                if lower_u.endswith(".ttf") or lower_u.endswith(".otf"):
                    return candidate
            return urls[0]

        raise FontDownloadError(last_error)

    def download_catalog_fonts(
        self,
        requested_fonts: Sequence[str],
        weight: str | None = None,
        overwrite: bool = False,
        timeout: float = 30.0,
    ) -> FontDownloadResponseDict:
        """Download one or more Google Fonts into `fonts/downloaded/` for offline use."""
        self.ensure_dirs()
        results: list[FontDownloadItemResult] = []
        downloaded_count = 0
        skipped_count = 0
        failed_count = 0

        manifest = self._load_downloaded_manifest()
        manifest_changed = False

        for raw_req in requested_fonts:
            req_clean = raw_req.strip()
            if not req_clean:
                continue

            # Resolve against curated catalog by ID or family name, or accept raw Google Font family
            spec = _CATALOG_BY_ID.get(req_clean.lower()) or _CATALOG_BY_FAMILY_LOWER.get(
                req_clean.lower()
            )
            family_name = spec.family if spec is not None else req_clean
            catalog_id = spec.id if spec is not None else None

            safe_base = _SAFE_FILENAME_CHARS_RE.sub("_", family_name).strip(" ._") or "google_font"
            safe_base = re.sub(r"\s+", "_", safe_base)
            if weight and weight.strip() and weight.strip() != "400":
                safe_base = f"{safe_base}_{re.sub(r'[^0-9A-Za-z]+', '', weight.strip())}"

            # Check if already downloaded when overwrite=False
            existing_match: Path | None = None
            if not overwrite:
                for ext in (".ttf", ".otf", ".woff2", ".woff"):
                    cand = self.downloaded_dir / f"{safe_base}{ext}"
                    if cand.is_file() and cand.stat().st_size >= 64:
                        existing_match = cand
                        break

            if existing_match is not None:
                entry = self._build_entry(
                    existing_match,
                    category="downloaded",
                    override_family=family_name,
                )
                skipped_count += 1
                item_res: FontDownloadItemResult = {
                    "requested": req_clean,
                    "family": family_name,
                    "status": "skipped",
                    "filename": existing_match.name,
                    "url": _build_font_url(existing_match.name),
                    "size_bytes": int(existing_match.stat().st_size),
                }
                if entry is not None:
                    item_res["font"] = entry
                results.append(item_res)
                continue

            try:
                binary_url = self._resolve_google_font_binary_url(
                    family=family_name,
                    weight=weight,
                    timeout=min(15.0, timeout),
                )
                dl_resp = requests.get(binary_url, timeout=timeout)
                dl_resp.raise_for_status()
                font_bytes = dl_resp.content

                url_suffix = Path(binary_url.split("?", 1)[0]).suffix.lower()
                if url_suffix not in SUPPORTED_FONT_EXTENSIONS:
                    if font_bytes[:4] == b"OTTO":
                        url_suffix = ".otf"
                    elif font_bytes[:4] == b"wOF2":
                        url_suffix = ".woff2"
                    elif font_bytes[:4] == b"wOFF":
                        url_suffix = ".woff"
                    else:
                        url_suffix = ".ttf"

                validate_font_bytes(font_bytes, url_suffix)

                final_filename = f"{safe_base}{url_suffix}"
                final_path = self.downloaded_dir / final_filename
                tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
                try:
                    tmp_path.write_bytes(font_bytes)
                    os.replace(tmp_path, final_path)
                except Exception:
                    tmp_path.unlink(missing_ok=True)
                    raise

                manifest[final_filename] = {
                    "family": family_name,
                    "catalog_id": catalog_id,
                    "source_url": binary_url,
                }
                manifest_changed = True

                entry = self._build_entry(
                    final_path,
                    category="downloaded",
                    override_family=family_name,
                )
                downloaded_count += 1
                dl_item: FontDownloadItemResult = {
                    "requested": req_clean,
                    "family": family_name,
                    "status": "downloaded",
                    "filename": final_filename,
                    "url": _build_font_url(final_filename),
                    "size_bytes": len(font_bytes),
                }
                if entry is not None:
                    dl_item["font"] = entry
                results.append(dl_item)
            except (FontError, requests.RequestException, OSError) as exc:
                failed_count += 1
                results.append(
                    {
                        "requested": req_clean,
                        "family": family_name,
                        "status": "failed",
                        "filename": None,
                        "url": None,
                        "size_bytes": 0,
                        "error": str(exc),
                    }
                )

        if manifest_changed:
            self._save_downloaded_manifest(manifest)

        return {
            "downloaded_count": downloaded_count,
            "skipped_count": skipped_count,
            "failed_count": failed_count,
            "results": results,
        }
