"""Multi-Provider LLM Engine (Koharu-style) & Persistent Settings Store.

Supports 8 LLM translation providers out of the box:
1. `lmstudio`   — Local LM Studio (`http://localhost:1235/v1`, default model `"Gemma 4 E2B Uncensored HauhauCS Aggressive Q4 K P"`)
2. `ollama`     — Local Ollama (`http://localhost:11434/v1`, with `/api/tags` fallback)
3. `openai`     — OpenAI (`https://api.openai.com/v1`)
4. `gemini`     — Google Gemini OpenAI-compatible endpoint (`https://generativelanguage.googleapis.com/v1beta/openai`)
5. `anthropic`  — Anthropic Claude native Messages API (`https://api.anthropic.com/v1`)
6. `deepseek`   — DeepSeek (`https://api.deepseek.com/v1`)
7. `openrouter` — OpenRouter (`https://openrouter.ai/api/v1`)
8. `custom`     — Custom OpenAI-compatible endpoint (user-supplied Base URL + API Key)

Also manages:
- 16 supported target languages (`id`, `en`, `ja`, `ko`, `zh-CN`, `zh-TW`, `es`, `fr`, `de`, `pt`, `ru`, `vi`, `th`, `ms`, `ar`, `it`)
- Reasoning effort levels (`off`, `low`, `medium`, `high`)
- Glossary parsing (`source -> target`)
- Persistent configuration saved atomically to `%LOCALAPPDATA%/ComicTranslator/settings.json`
  when running as a PyInstaller frozen executable, or `backend/settings.json` during development.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Literal, NotRequired, TypedDict, cast

import requests

ProviderId = Literal[
    "lmstudio",
    "ollama",
    "openai",
    "gemini",
    "anthropic",
    "deepseek",
    "openrouter",
    "custom",
]

ProviderProtocol = Literal["openai", "anthropic"]

ReasoningEffort = Literal["off", "low", "medium", "high"]

ThemeOption = Literal["dark", "light", "system"]
AccentColorOption = Literal["amber", "teal", "crimson", "slate"]
UiDensityOption = Literal["compact", "comfortable"]
ExecutionModeOption = Literal["auto", "cpu"]
ReadingOrderOption = Literal["auto", "rtl", "ltr"]
TextAlignOption = Literal["left", "center", "right"]

DEFAULT_FONT_FALLBACK_CHAIN: tuple[str, ...] = (
    "CC Wild Words",
    "Anime Ace",
    "Comic Neue",
    "Noto Sans",
    "sans-serif",
)
DEFAULT_FONT_FAMILY_STACK = (
    '"CC Wild Words", "Anime Ace", "Comic Neue", "Noto Sans", sans-serif'
)

TargetLanguage = Literal[
    "id",
    "en",
    "ja",
    "ko",
    "zh-CN",
    "zh-TW",
    "es",
    "fr",
    "de",
    "pt",
    "ru",
    "vi",
    "th",
    "ms",
    "ar",
    "it",
]

DEFAULT_LM_STUDIO_BASE_URL = "http://localhost:1235/v1"
DEFAULT_LM_STUDIO_API_KEY = "lm-studio"
DEFAULT_LM_STUDIO_MODEL = "Gemma 4 E2B Uncensored HauhauCS Aggressive Q4 K P"

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"

SUPPORTED_PROVIDER_IDS: tuple[ProviderId, ...] = (
    "lmstudio",
    "ollama",
    "openai",
    "gemini",
    "anthropic",
    "deepseek",
    "openrouter",
    "custom",
)

SUPPORTED_REASONING_EFFORTS: tuple[ReasoningEffort, ...] = (
    "off",
    "low",
    "medium",
    "high",
)

SUPPORTED_TARGET_LANGUAGES: tuple[TargetLanguage, ...] = (
    "id",
    "en",
    "ja",
    "ko",
    "zh-CN",
    "zh-TW",
    "es",
    "fr",
    "de",
    "pt",
    "ru",
    "vi",
    "th",
    "ms",
    "ar",
    "it",
)

# Quantization / format tokens that local servers often strip from model IDs in /v1/models
_QUANT_TOKENS: frozenset[str] = frozenset(
    {
        "q2",
        "q3",
        "q4",
        "q5",
        "q6",
        "q8",
        "k",
        "p",
        "m",
        "s",
        "l",
        "xs",
        "xxs",
        "0",
        "1",
        "fp16",
        "bf16",
        "fp32",
        "gguf",
        "imat",
        "iq4",
        "iq3",
    }
)


class ProviderConnectionError(RuntimeError):
    """Raised when a configured LLM provider endpoint is unreachable or rejects credentials."""


class ProviderTranslationError(RuntimeError):
    """Raised when an LLM translation call fails unexpectedly."""


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    """Static definition and capabilities of a supported LLM provider."""

    id: ProviderId
    name: str
    default_base_url: str
    default_model: str
    default_api_key: str
    requires_api_key: bool
    requires_base_url: bool
    is_local: bool
    protocol: ProviderProtocol
    supports_json_schema: bool
    supports_vision: bool
    supports_top_k: bool
    supports_frequency_penalty: bool
    supports_reasoning_effort: bool
    extra_headers: tuple[tuple[str, str], ...] = ()
    fallback_models: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TargetLanguageSpec:
    """Metadata and aliases for a supported translation target language."""

    code: TargetLanguage
    name: str
    native_name: str
    rtl: bool
    aliases: tuple[str, ...]


class ProviderInfoDict(TypedDict):
    """Serializable provider preset metadata exposed via API."""

    id: ProviderId
    name: str
    default_base_url: str
    default_model: str
    requires_api_key: bool
    requires_base_url: bool
    is_local: bool
    protocol: ProviderProtocol
    supports_json_schema: bool
    supports_vision: bool
    supports_top_k: bool
    supports_frequency_penalty: bool
    supports_reasoning_effort: bool


class TargetLanguageInfoDict(TypedDict):
    """Serializable target language metadata exposed via API."""

    code: TargetLanguage
    name: str
    native_name: str
    rtl: bool


class ProviderProfileDict(TypedDict):
    """Saved per-provider connection profile inside `settings.json`."""

    base_url: str
    api_key: str
    model: str


class AppSettingsDict(TypedDict):
    """Complete persistent settings schema stored in `settings.json`."""

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
    # Tab 1: Appearance
    theme: ThemeOption
    accent_color: AccentColorOption
    ui_density: UiDensityOption
    # Tab 2: Pipeline
    execution_mode: ExecutionModeOption
    conf_threshold: float
    nms_threshold: float
    mask_threshold: float
    bbox_padding: int
    lama_tile_size: int
    reading_order: ReadingOrderOption
    auto_unload_vram: bool
    refine_mask: bool
    # Tab 5: Typesetting
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


class ProviderTestResult(TypedDict):
    """Structured result returned by `test_provider_connection` / `POST /api/providers/test`."""

    ok: bool
    connected: bool
    provider: ProviderId
    provider_name: str
    endpoint: str
    active_model: str | None
    resolved_model_id: str | None
    preferred_model: str
    models: list[str]
    available_models: list[str]
    vision_models: list[str]
    model_count: int
    latency_ms: int
    message: str
    error: NotRequired[str]


PROVIDER_PRESETS: dict[ProviderId, ProviderSpec] = {
    "lmstudio": ProviderSpec(
        id="lmstudio",
        name="LM Studio (Local)",
        default_base_url=DEFAULT_LM_STUDIO_BASE_URL,
        default_model=DEFAULT_LM_STUDIO_MODEL,
        default_api_key=DEFAULT_LM_STUDIO_API_KEY,
        requires_api_key=False,
        requires_base_url=True,
        is_local=True,
        protocol="openai",
        supports_json_schema=True,
        supports_vision=True,
        supports_top_k=True,
        supports_frequency_penalty=True,
        supports_reasoning_effort=True,
        fallback_models=(DEFAULT_LM_STUDIO_MODEL,),
    ),
    "ollama": ProviderSpec(
        id="ollama",
        name="Ollama (Local)",
        default_base_url=DEFAULT_OLLAMA_BASE_URL,
        default_model="qwen2.5:7b",
        default_api_key="ollama",
        requires_api_key=False,
        requires_base_url=True,
        is_local=True,
        protocol="openai",
        supports_json_schema=True,
        supports_vision=True,
        supports_top_k=True,
        supports_frequency_penalty=True,
        supports_reasoning_effort=True,
        fallback_models=("qwen2.5:7b", "gemma3:4b", "llama3.2-vision"),
    ),
    "openai": ProviderSpec(
        id="openai",
        name="OpenAI",
        default_base_url="https://api.openai.com/v1",
        default_model="gpt-4o-mini",
        default_api_key="",
        requires_api_key=True,
        requires_base_url=True,
        is_local=False,
        protocol="openai",
        supports_json_schema=True,
        supports_vision=True,
        supports_top_k=False,
        supports_frequency_penalty=True,
        supports_reasoning_effort=True,
        fallback_models=("gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "o4-mini"),
    ),
    "gemini": ProviderSpec(
        id="gemini",
        name="Google Gemini",
        default_base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        default_model="gemini-2.5-flash",
        default_api_key="",
        requires_api_key=True,
        requires_base_url=True,
        is_local=False,
        protocol="openai",
        supports_json_schema=True,
        supports_vision=True,
        supports_top_k=False,
        supports_frequency_penalty=False,
        supports_reasoning_effort=True,
        fallback_models=(
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
        ),
    ),
    "anthropic": ProviderSpec(
        id="anthropic",
        name="Anthropic (Claude)",
        default_base_url="https://api.anthropic.com/v1",
        default_model="claude-sonnet-4-20250514",
        default_api_key="",
        requires_api_key=True,
        requires_base_url=True,
        is_local=False,
        protocol="anthropic",
        supports_json_schema=False,
        supports_vision=True,
        supports_top_k=True,
        supports_frequency_penalty=False,
        supports_reasoning_effort=True,
        fallback_models=(
            "claude-sonnet-4-20250514",
            "claude-3-7-sonnet-latest",
            "claude-3-5-haiku-latest",
        ),
    ),
    "deepseek": ProviderSpec(
        id="deepseek",
        name="DeepSeek",
        default_base_url="https://api.deepseek.com/v1",
        default_model="deepseek-chat",
        default_api_key="",
        requires_api_key=True,
        requires_base_url=True,
        is_local=False,
        protocol="openai",
        supports_json_schema=False,
        supports_vision=False,
        supports_top_k=False,
        supports_frequency_penalty=True,
        supports_reasoning_effort=False,
        fallback_models=("deepseek-chat", "deepseek-reasoner"),
    ),
    "openrouter": ProviderSpec(
        id="openrouter",
        name="OpenRouter",
        default_base_url="https://openrouter.ai/api/v1",
        default_model="google/gemini-2.5-flash",
        default_api_key="",
        requires_api_key=True,
        requires_base_url=True,
        is_local=False,
        protocol="openai",
        supports_json_schema=True,
        supports_vision=True,
        supports_top_k=True,
        supports_frequency_penalty=True,
        supports_reasoning_effort=True,
        extra_headers=(
            ("HTTP-Referer", "https://github.com/comic-translator"),
            ("X-Title", "MangaFlow Comic Translator"),
        ),
        fallback_models=(
            "google/gemini-2.5-flash",
            "openai/gpt-4o-mini",
            "anthropic/claude-sonnet-4",
            "deepseek/deepseek-chat",
        ),
    ),
    "custom": ProviderSpec(
        id="custom",
        name="Custom (OpenAI-Compatible)",
        default_base_url="",
        default_model="",
        default_api_key="",
        requires_api_key=False,
        requires_base_url=True,
        is_local=False,
        protocol="openai",
        supports_json_schema=True,
        supports_vision=True,
        supports_top_k=True,
        supports_frequency_penalty=True,
        supports_reasoning_effort=True,
    ),
}

_PROVIDER_ALIASES: dict[str, ProviderId] = {
    "lmstudio": "lmstudio",
    "lm-studio": "lmstudio",
    "lm_studio": "lmstudio",
    "lm studio": "lmstudio",
    "ollama": "ollama",
    "openai": "openai",
    "gpt": "openai",
    "gemini": "gemini",
    "google": "gemini",
    "google-gemini": "gemini",
    "google_gemini": "gemini",
    "anthropic": "anthropic",
    "claude": "anthropic",
    "deepseek": "deepseek",
    "deep-seek": "deepseek",
    "openrouter": "openrouter",
    "open-router": "openrouter",
    "open_router": "openrouter",
    "custom": "custom",
    "openai-compatible": "custom",
    "openai_compatible": "custom",
}

TARGET_LANGUAGES: dict[TargetLanguage, TargetLanguageSpec] = {
    "id": TargetLanguageSpec(
        code="id",
        name="Indonesian",
        native_name="Bahasa Indonesia",
        rtl=False,
        aliases=("id", "ind", "indonesian", "bahasa", "bahasa indonesia", "indo"),
    ),
    "en": TargetLanguageSpec(
        code="en",
        name="English",
        native_name="English",
        rtl=False,
        aliases=("en", "en-us", "en-gb", "eng", "english"),
    ),
    "ja": TargetLanguageSpec(
        code="ja",
        name="Japanese",
        native_name="日本語",
        rtl=False,
        aliases=("ja", "jp", "jpn", "japanese", "日本語"),
    ),
    "ko": TargetLanguageSpec(
        code="ko",
        name="Korean",
        native_name="한국어",
        rtl=False,
        aliases=("ko", "kr", "kor", "korean", "한국어"),
    ),
    "zh-CN": TargetLanguageSpec(
        code="zh-CN",
        name="Chinese (Simplified)",
        native_name="简体中文",
        rtl=False,
        aliases=(
            "zh-cn",
            "zh_cn",
            "zh-hans",
            "zh_hans",
            "zh",
            "zho",
            "chi",
            "cn",
            "chinese",
            "chinese simplified",
            "simplified chinese",
            "chinese (simplified)",
            "简体中文",
            "中文",
        ),
    ),
    "zh-TW": TargetLanguageSpec(
        code="zh-TW",
        name="Chinese (Traditional)",
        native_name="繁體中文",
        rtl=False,
        aliases=(
            "zh-tw",
            "zh_tw",
            "zh-hant",
            "zh_hant",
            "zh-hk",
            "tw",
            "chinese traditional",
            "traditional chinese",
            "chinese (traditional)",
            "繁體中文",
        ),
    ),
    "es": TargetLanguageSpec(
        code="es",
        name="Spanish",
        native_name="Español",
        rtl=False,
        aliases=("es", "es-es", "es-mx", "es-419", "spa", "spanish", "español", "espanol"),
    ),
    "fr": TargetLanguageSpec(
        code="fr",
        name="French",
        native_name="Français",
        rtl=False,
        aliases=("fr", "fr-fr", "fra", "fre", "french", "français", "francais"),
    ),
    "de": TargetLanguageSpec(
        code="de",
        name="German",
        native_name="Deutsch",
        rtl=False,
        aliases=("de", "de-de", "deu", "ger", "german", "deutsch"),
    ),
    "pt": TargetLanguageSpec(
        code="pt",
        name="Portuguese",
        native_name="Português",
        rtl=False,
        aliases=("pt", "pt-br", "pt-pt", "por", "portuguese", "português", "portugues"),
    ),
    "ru": TargetLanguageSpec(
        code="ru",
        name="Russian",
        native_name="Русский",
        rtl=False,
        aliases=("ru", "rus", "russian", "русский"),
    ),
    "vi": TargetLanguageSpec(
        code="vi",
        name="Vietnamese",
        native_name="Tiếng Việt",
        rtl=False,
        aliases=("vi", "vn", "vie", "vietnamese", "tiếng việt", "tieng viet"),
    ),
    "th": TargetLanguageSpec(
        code="th",
        name="Thai",
        native_name="ไทย",
        rtl=False,
        aliases=("th", "tha", "thai", "ไทย", "ภาษาไทย"),
    ),
    "ms": TargetLanguageSpec(
        code="ms",
        name="Malay",
        native_name="Bahasa Melayu",
        rtl=False,
        aliases=("ms", "my", "msa", "may", "malay", "bahasa melayu", "melayu"),
    ),
    "ar": TargetLanguageSpec(
        code="ar",
        name="Arabic",
        native_name="العربية",
        rtl=True,
        aliases=("ar", "ara", "arabic", "العربية"),
    ),
    "it": TargetLanguageSpec(
        code="it",
        name="Italian",
        native_name="Italiano",
        rtl=False,
        aliases=("it", "ita", "italian", "italiano"),
    ),
}

_TARGET_LANGUAGE_ALIAS_MAP: dict[str, TargetLanguage] = {
    alias.lower(): spec.code
    for spec in TARGET_LANGUAGES.values()
    for alias in spec.aliases
}


def _json_loads(raw: str) -> object:
    """Parse a JSON string into a typed `object` (isolating stdlib `json.loads` `Any` return)."""
    res: object = json.loads(raw)  # pyright: ignore[reportAny]
    return res


def normalize_provider_id(provider: str | None) -> ProviderId:
    """Validate and normalize a provider identifier into a canonical `ProviderId`."""
    if provider is None or not str(provider).strip():
        return "lmstudio"
    cleaned = str(provider).strip().lower()
    resolved = _PROVIDER_ALIASES.get(cleaned)
    if resolved is None:
        raise ValueError(
            f"Unsupported provider {provider!r}. Must be one of {SUPPORTED_PROVIDER_IDS}."
        )
    return resolved


def get_provider_spec(provider: str | None) -> ProviderSpec:
    """Return the static `ProviderSpec` for `provider`."""
    return PROVIDER_PRESETS[normalize_provider_id(provider)]


def list_provider_presets() -> list[ProviderInfoDict]:
    """Return serializable metadata for all 8 supported LLM providers."""
    return [
        {
            "id": spec.id,
            "name": spec.name,
            "default_base_url": spec.default_base_url,
            "default_model": spec.default_model,
            "requires_api_key": spec.requires_api_key,
            "requires_base_url": spec.requires_base_url,
            "is_local": spec.is_local,
            "protocol": spec.protocol,
            "supports_json_schema": spec.supports_json_schema,
            "supports_vision": spec.supports_vision,
            "supports_top_k": spec.supports_top_k,
            "supports_frequency_penalty": spec.supports_frequency_penalty,
            "supports_reasoning_effort": spec.supports_reasoning_effort,
        }
        for spec in PROVIDER_PRESETS.values()
    ]


def normalize_target_language(target_lang: str) -> TargetLanguage:
    """Validate and normalize `target_lang` across all 16 supported target languages."""
    cleaned = target_lang.strip().lower()
    resolved = _TARGET_LANGUAGE_ALIAS_MAP.get(cleaned)
    if resolved is not None:
        return resolved
    raise ValueError(
        f"Unsupported target_lang {target_lang!r}. Must be one of {SUPPORTED_TARGET_LANGUAGES}."
    )


def list_target_languages() -> list[TargetLanguageInfoDict]:
    """Return serializable metadata for all supported translation target languages."""
    return [
        {
            "code": spec.code,
            "name": spec.name,
            "native_name": spec.native_name,
            "rtl": spec.rtl,
        }
        for spec in TARGET_LANGUAGES.values()
    ]


def normalize_reasoning_effort(effort: str | None) -> ReasoningEffort:
    """Validate and normalize `reasoning_effort` (`'off'`, `'low'`, `'medium'`, or `'high'`)."""
    if effort is None or not str(effort).strip():
        return "off"
    cleaned = str(effort).strip().lower()
    if cleaned in ("off", "none", "disabled", "false", "0"):
        return "off"
    if cleaned in ("low", "minimal"):
        return "low"
    if cleaned in ("medium", "med", "default"):
        return "medium"
    if cleaned in ("high", "max", "xhigh"):
        return "high"
    raise ValueError(
        f"Unsupported reasoning_effort {effort!r}. Must be one of {SUPPORTED_REASONING_EFFORTS}."
    )


def parse_glossary(raw_glossary: object) -> dict[str, str]:
    """Normalize a glossary from a dict, list of `{source, target}`, or `'source -> target'` lines."""
    parsed: dict[str, str] = {}
    if raw_glossary is None:
        return parsed

    if isinstance(raw_glossary, Mapping):
        for k, v in cast(Mapping[object, object], raw_glossary).items():
            src = str(k).strip()
            tgt = str(v).strip() if v is not None else ""
            if src and tgt:
                parsed[src] = tgt
        return parsed

    if isinstance(raw_glossary, str):
        text = raw_glossary.strip()
        if not text:
            return parsed
        # Support JSON-encoded glossary strings as well as Koharu-style `source -> target` lines
        if text.startswith("{") or text.startswith("["):
            try:
                loaded = _json_loads(text)
                return parse_glossary(loaded)
            except ValueError:
                pass
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            parts: list[str] | None = None
            for sep in ("->", "=>", "→", "=", ":"):
                if sep in line:
                    lhs, rhs = line.split(sep, 1)
                    parts = [lhs.strip(), rhs.strip()]
                    break
            if parts and parts[0] and parts[1]:
                parsed[parts[0]] = parts[1]
        return parsed

    if isinstance(raw_glossary, Sequence) and not isinstance(raw_glossary, (bytes, bytearray)):
        for entry in cast(Sequence[object], raw_glossary):
            if isinstance(entry, Mapping):
                entry_map = cast(Mapping[str, object], entry)
                src_obj = (
                    entry_map.get("source")
                    if entry_map.get("source") is not None
                    else entry_map.get("src")
                )
                tgt_obj = (
                    entry_map.get("target")
                    if entry_map.get("target") is not None
                    else entry_map.get("tgt")
                )
                src = str(src_obj).strip() if src_obj is not None else ""
                tgt = str(tgt_obj).strip() if tgt_obj is not None else ""
                if src and tgt:
                    parsed[src] = tgt
            elif isinstance(entry, str):
                parsed.update(parse_glossary(entry))
        return parsed

    return parsed


def normalize_provider_base_url(
    provider: ProviderId,
    base_url: str | None,
) -> str:
    """Normalize a provider's Base URL, applying preset defaults and `/v1` suffix when appropriate."""
    spec = PROVIDER_PRESETS[provider]
    cleaned = (base_url or "").strip().rstrip("/")
    if not cleaned:
        return spec.default_base_url.rstrip("/")

    if provider in ("lmstudio", "ollama"):
        lower = cleaned.lower()
        if not lower.endswith("/v1") and "/v1/" not in lower and "/api/" not in lower:
            cleaned = f"{cleaned}/v1"

    return cleaned


def resolve_provider_api_key(
    provider: ProviderId,
    api_key: str | None,
) -> str:
    """Resolve effective API key for `provider`, applying local placeholder keys when omitted."""
    spec = PROVIDER_PRESETS[provider]
    cleaned = (api_key or "").strip()
    if cleaned:
        return cleaned
    return spec.default_api_key


def _tokenize_model_name(name: str) -> list[str]:
    """Split a model name or ID into lowercase alphanumeric tokens."""
    return re.findall(r"[a-z0-9]+", name.lower())


def is_embedding_model(model_id: str) -> bool:
    """Return True if `model_id` appears to be an embedding, audio, or moderation model."""
    lower = model_id.lower()
    if "embed" in lower or lower.startswith("text-embedding"):
        return True
    non_chat_markers = (
        "whisper",
        "tts-",
        "dall-e",
        "text-moderation",
        "omni-moderation",
        "babbage-002",
        "davinci-002",
        "rerank",
    )
    return any(marker in lower for marker in non_chat_markers)


def is_likely_vision_model(model_id: str, provider: ProviderId) -> bool:
    """Heuristic check for whether `model_id` supports multimodal vision input."""
    lower = model_id.lower()
    if provider == "anthropic" and ("claude-3" in lower or "claude-4" in lower or "claude-sonnet" in lower or "claude-opus" in lower or "claude-haiku" in lower):
        return True
    if provider == "gemini" and "gemini" in lower and "embed" not in lower:
        return True
    vision_markers = (
        "vision",
        "vl",
        "vlm",
        "gemma-3",
        "gemma3",
        "gemma-4",
        "gemma4",
        "qwen2.5-vl",
        "qwen2-vl",
        "qwen3.5",
        "qwen3-vl",
        "pixtral",
        "ministral-3",
        "llava",
        "minicpm-v",
        "internvl",
        "gpt-4o",
        "gpt-4.1",
        "gpt-4-turbo",
        "gpt-5",
        "o1",
        "o3",
        "o4",
        "claude-3",
        "claude-4",
        "claude-sonnet",
        "claude-opus",
        "gemini-1.5",
        "gemini-2",
        "gemini-3",
    )
    return any(marker in lower for marker in vision_markers)


def score_model_match(preferred_name: str, candidate_id: str) -> float:
    """Compute a similarity score in `[0.0, 1.0]` between `preferred_name` and `candidate_id`."""
    pref_lower = preferred_name.lower().strip()
    cand_lower = candidate_id.lower().strip()
    if not pref_lower or not cand_lower:
        return 0.0
    if pref_lower == cand_lower:
        return 1.0

    pref_compact = re.sub(r"[^a-z0-9]", "", pref_lower)
    cand_compact = re.sub(r"[^a-z0-9]", "", cand_lower)
    if pref_compact and cand_compact and pref_compact == cand_compact:
        return 0.99

    pref_tokens = _tokenize_model_name(pref_lower)
    cand_tokens = _tokenize_model_name(cand_lower)
    if not pref_tokens or not cand_tokens:
        return 0.0

    pref_core = [t for t in pref_tokens if t not in _QUANT_TOKENS] or pref_tokens
    cand_core = [t for t in cand_tokens if t not in _QUANT_TOKENS] or cand_tokens

    pref_core_set = set(pref_core)
    cand_core_set = set(cand_core)
    common_core = pref_core_set & cand_core_set
    if not common_core:
        return 0.0

    union_core = pref_core_set | cand_core_set
    core_jaccard = len(common_core) / max(1, len(union_core))
    cand_coverage = len(common_core) / max(1, len(cand_core_set))
    pref_coverage = len(common_core) / max(1, len(pref_core_set))

    score = (core_jaccard * 0.5) + (cand_coverage * 0.3) + (pref_coverage * 0.2)
    if cand_compact and (cand_compact in pref_compact or pref_compact in cand_compact):
        score = min(0.98, score + 0.1)
    return score


def _fetch_lmstudio_v0_metadata(
    base_url: str,
    timeout: float = 3.0,
) -> tuple[str | None, set[str]]:
    """Query LM Studio's `/api/v0/models` for `(loaded_model_id, vlm_model_ids)`."""
    root_url = re.sub(r"/v1/?$", "", base_url.rstrip("/"))
    v0_url = f"{root_url}/api/v0/models"
    loaded_model: str | None = None
    vlm_models: set[str] = set()
    try:
        resp = requests.get(v0_url, timeout=timeout)
        if resp.status_code != 200:
            return None, vlm_models
        payload_obj = _json_loads(resp.text)
        if not isinstance(payload_obj, Mapping):
            return None, vlm_models
        payload_map = cast(Mapping[str, object], payload_obj)
        data_obj = payload_map.get("data")
        if not isinstance(data_obj, Sequence) or isinstance(data_obj, (str, bytes)):
            return None, vlm_models
        for entry_obj in data_obj:
            if not isinstance(entry_obj, Mapping):
                continue
            entry = cast(Mapping[str, object], entry_obj)
            model_id = str(entry.get("id", "")).strip()
            state = str(entry.get("state", "")).strip().lower()
            model_type = str(entry.get("type", "")).strip().lower()
            if not model_id or model_type == "embeddings" or is_embedding_model(model_id):
                continue
            if model_type == "vlm":
                vlm_models.add(model_id)
            if loaded_model is None and state == "loaded":
                loaded_model = model_id
    except (requests.RequestException, ValueError):
        return None, vlm_models
    return loaded_model, vlm_models


def _fetch_ollama_native_tags(base_url: str, timeout: float = 5.0) -> list[str]:
    """Query Ollama's native `GET /api/tags` endpoint as a fallback when `/v1/models` is unavailable."""
    root_url = re.sub(r"/v1/?$", "", base_url.rstrip("/"))
    tags_url = f"{root_url}/api/tags"
    resp = requests.get(tags_url, timeout=timeout)
    resp.raise_for_status()
    payload_obj = _json_loads(resp.text)
    if not isinstance(payload_obj, Mapping):
        return []
    payload_map = cast(Mapping[str, object], payload_obj)
    models_obj = payload_map.get("models")
    if not isinstance(models_obj, Sequence) or isinstance(models_obj, (str, bytes)):
        return []
    names: list[str] = []
    for entry_obj in models_obj:
        if isinstance(entry_obj, Mapping):
            entry = cast(Mapping[str, object], entry_obj)
            raw_name = entry.get("name") or entry.get("model")
            if raw_name is not None and str(raw_name).strip():
                names.append(str(raw_name).strip())
    return names


def _extract_http_error_detail(resp: requests.Response) -> str:
    """Extract a readable API error message from a non-2xx HTTP response."""
    try:
        parsed = _json_loads(resp.text)
        if isinstance(parsed, Mapping):
            pmap = cast(Mapping[str, object], parsed)
            err = pmap.get("error")
            if isinstance(err, Mapping):
                msg = cast(Mapping[str, object], err).get("message")
                if isinstance(msg, str) and msg.strip():
                    return msg.strip()
            elif isinstance(err, str) and err.strip():
                return err.strip()
            msg_top = pmap.get("message") or pmap.get("detail")
            if isinstance(msg_top, str) and msg_top.strip():
                return msg_top.strip()
    except ValueError:
        pass
    snippet = resp.text.strip().replace("\n", " ")[:240]
    return snippet or f"HTTP {resp.status_code}"


def list_provider_models(
    provider: str = "lmstudio",
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float = 8.0,
    include_embeddings: bool = True,
) -> list[str]:
    """Query the target provider's `/models` endpoint and return available model IDs.

    Raises `ProviderConnectionError` with a clear diagnostic message on network,
    configuration, or credential errors.
    """
    prov_id = normalize_provider_id(provider)
    spec = PROVIDER_PRESETS[prov_id]
    resolved_url = normalize_provider_base_url(prov_id, base_url)
    resolved_key = resolve_provider_api_key(prov_id, api_key)

    if spec.requires_base_url and not resolved_url:
        raise ProviderConnectionError(
            f"Base URL is required for {spec.name}."
        )
    if spec.requires_api_key and not resolved_key:
        raise ProviderConnectionError(
            f"API key is required for {spec.name}. Provide an API key to test or use this provider."
        )

    models_url = f"{resolved_url.rstrip('/')}/models"
    headers: dict[str, str] = {"Accept": "application/json"}
    for h_key, h_val in spec.extra_headers:
        headers[h_key] = h_val

    if spec.protocol == "anthropic":
        headers["x-api-key"] = resolved_key
        headers["anthropic-version"] = DEFAULT_ANTHROPIC_VERSION
    elif resolved_key:
        headers["Authorization"] = f"Bearer {resolved_key}"

    try:
        resp = requests.get(models_url, headers=headers, timeout=timeout)
    except requests.exceptions.ConnectionError as exc:
        if prov_id == "lmstudio":
            hint = "Ensure LM Studio is open and the Local Inference Server is running (default port 1235)."
        elif prov_id == "ollama":
            hint = "Ensure Ollama is running locally (default port 11434)."
        else:
            hint = f"Verify network connectivity and Base URL ({resolved_url})."
        raise ProviderConnectionError(
            f"Cannot connect to {spec.name} at {models_url}. {hint}"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise ProviderConnectionError(
            f"Timed out after {timeout:.1f}s while connecting to {spec.name} at {models_url}."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise ProviderConnectionError(
            f"HTTP error while querying {spec.name} models at {models_url}: {exc}"
        ) from exc

    if resp.status_code in (401, 403):
        detail = _extract_http_error_detail(resp)
        raise ProviderConnectionError(
            f"Authentication failed for {spec.name} (HTTP {resp.status_code}): {detail}"
        )

    if resp.status_code >= 400:
        if prov_id == "ollama":
            try:
                ollama_tags = _fetch_ollama_native_tags(resolved_url, timeout=timeout)
                return (
                    ollama_tags
                    if include_embeddings
                    else [m for m in ollama_tags if not is_embedding_model(m)]
                )
            except requests.RequestException:
                pass
        detail = _extract_http_error_detail(resp)
        raise ProviderConnectionError(
            f"{spec.name} returned HTTP {resp.status_code} at {models_url}: {detail}"
        )

    try:
        payload_obj = _json_loads(resp.text)
    except ValueError as exc:
        raise ProviderConnectionError(
            f"{spec.name} at {models_url} returned non-JSON payload: {resp.text[:200]!r}"
        ) from exc

    model_ids: list[str] = []
    if isinstance(payload_obj, Mapping):
        payload_map = cast(Mapping[str, object], payload_obj)
        data_obj = payload_map.get("data")
        if data_obj is None:
            data_obj = payload_map.get("models")
        if isinstance(data_obj, Sequence) and not isinstance(data_obj, (str, bytes)):
            for item_obj in data_obj:
                if isinstance(item_obj, Mapping):
                    item_map = cast(Mapping[str, object], item_obj)
                    raw_id = item_map.get("id") or item_map.get("name")
                    mid = str(raw_id).strip() if raw_id is not None else ""
                    if mid:
                        if prov_id == "gemini" and mid.startswith("models/"):
                            mid = mid[len("models/") :]
                        model_ids.append(mid)
                elif isinstance(item_obj, str) and item_obj.strip():
                    model_ids.append(item_obj.strip())
    elif isinstance(payload_obj, Sequence) and not isinstance(payload_obj, (str, bytes)):
        for item_obj in payload_obj:
            if isinstance(item_obj, str) and item_obj.strip():
                model_ids.append(item_obj.strip())
            elif isinstance(item_obj, Mapping):
                raw_id = cast(Mapping[str, object], item_obj).get("id")
                if raw_id is not None and str(raw_id).strip():
                    model_ids.append(str(raw_id).strip())
    else:
        raise ProviderConnectionError(
            f"Unexpected response format from {models_url}: {payload_obj!r}"
        )

    if not model_ids and prov_id == "ollama":
        try:
            model_ids = _fetch_ollama_native_tags(resolved_url, timeout=timeout)
        except requests.RequestException:
            pass

    # Preserve order while deduplicating
    deduped = list(dict.fromkeys(model_ids))
    if include_embeddings:
        return deduped
    filtered = [m for m in deduped if not is_embedding_model(m)]
    return filtered if filtered else deduped


def resolve_provider_model(
    provider: str = "lmstudio",
    preferred_model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float = 8.0,
    available_models: Sequence[str] | None = None,
) -> str:
    """Resolve the active model ID for `provider`.

    - For local providers (`lmstudio`, `ollama`), queries available models (or uses
      `available_models`), prefers exact/case-insensitive match, then fuzzy token match,
      then LM Studio's currently loaded model from `/api/v0/models`, then first non-embedding LLM.
    - For cloud/custom providers, returns `preferred_model` (or provider default) directly if
      `available_models` is not passed, or matches against `available_models` when provided.
    """
    prov_id = normalize_provider_id(provider)
    spec = PROVIDER_PRESETS[prov_id]
    resolved_url = normalize_provider_base_url(prov_id, base_url)
    pref_clean = (preferred_model or "").strip() or spec.default_model

    if prov_id == "gemini" and pref_clean.startswith("models/"):
        pref_clean = pref_clean[len("models/") :]

    # For remote/cloud providers, skip redundant /models calls during inference if model is set
    if not spec.is_local and available_models is None and pref_clean:
        return pref_clean

    all_models = (
        list(available_models)
        if available_models is not None
        else list_provider_models(
            provider=prov_id,
            base_url=resolved_url,
            api_key=api_key,
            timeout=timeout,
            include_embeddings=True,
        )
    )
    llm_models = [m for m in all_models if not is_embedding_model(m)]
    candidates = llm_models if llm_models else all_models

    if not candidates:
        if pref_clean and not spec.is_local:
            return pref_clean
        raise ProviderConnectionError(
            f"{spec.name} at {resolved_url} is reachable, but no models are installed or loaded."
        )

    # 1. Exact or case-insensitive match
    if pref_clean:
        for cand in candidates:
            if cand == pref_clean or cand.lower() == pref_clean.lower():
                return cand

        # 2. Fuzzy token match (e.g. "Gemma 4 E2B Uncensored HauhauCS Aggressive Q4 K P"
        #    -> "gemma-4-e2b-uncensored-hauhaucs-aggressive")
        scored: list[tuple[float, str]] = [
            (score_model_match(pref_clean, cand), cand) for cand in candidates
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        best_score, best_cand = scored[0]
        if best_score >= 0.50:
            return best_cand

    # 3. For LM Studio, prefer the currently loaded LLM from /api/v0/models
    if prov_id == "lmstudio":
        loaded_v0, _ = _fetch_lmstudio_v0_metadata(base_url=resolved_url, timeout=min(3.0, timeout))
        if loaded_v0 and loaded_v0 in candidates:
            return loaded_v0

    # 4. If a remote provider has a custom/explicit model string not listed in /models, trust it
    if pref_clean and not spec.is_local:
        return pref_clean

    return candidates[0]


def format_display_model_name(
    provider: ProviderId,
    resolved_model_id: str | None,
    preferred_model: str,
) -> str | None:
    """Format a user-friendly active model name while preserving canonical Gemma 4 E2B display name."""
    if resolved_model_id is None:
        return None
    if (
        provider == "lmstudio"
        and "gemma-4-e2b-uncensored-hauhaucs-aggressive" in resolved_model_id.lower()
        and (
            not preferred_model
            or "gemma 4 e2b" in preferred_model.lower()
            or "gemma-4-e2b" in preferred_model.lower()
        )
    ):
        return DEFAULT_LM_STUDIO_MODEL
    return resolved_model_id


def test_provider_connection(
    provider: str = "lmstudio",
    base_url: str | None = None,
    api_key: str | None = None,
    preferred_model: str | None = None,
    timeout: float = 10.0,
) -> ProviderTestResult:
    """Verify credentials & connectivity for `provider` and fetch its available models."""
    prov_id = normalize_provider_id(provider)
    spec = PROVIDER_PRESETS[prov_id]
    resolved_url = normalize_provider_base_url(prov_id, base_url)
    pref_model = (preferred_model or "").strip() or spec.default_model

    t0 = time.perf_counter()
    try:
        all_models = list_provider_models(
            provider=prov_id,
            base_url=resolved_url,
            api_key=api_key,
            timeout=timeout,
            include_embeddings=True,
        )
        chat_models = [m for m in all_models if not is_embedding_model(m)]
        listed_models = chat_models if chat_models else all_models

        vlm_from_server: set[str] = set()
        if prov_id == "lmstudio":
            _, vlm_from_server = _fetch_lmstudio_v0_metadata(
                base_url=resolved_url,
                timeout=min(3.0, timeout),
            )

        vision_models = [
            m
            for m in listed_models
            if m in vlm_from_server or is_likely_vision_model(m, prov_id)
        ]

        resolved_id: str | None = None
        if listed_models:
            resolved_id = resolve_provider_model(
                provider=prov_id,
                preferred_model=pref_model,
                base_url=resolved_url,
                api_key=api_key,
                timeout=timeout,
                available_models=all_models,
            )
        elif pref_model and not spec.is_local:
            resolved_id = pref_model

        display_model = format_display_model_name(prov_id, resolved_id, pref_model)
        latency_ms = int(round((time.perf_counter() - t0) * 1000.0))

        if not listed_models and spec.is_local:
            return {
                "ok": False,
                "connected": False,
                "provider": prov_id,
                "provider_name": spec.name,
                "endpoint": resolved_url,
                "active_model": None,
                "resolved_model_id": None,
                "preferred_model": pref_model,
                "models": [],
                "available_models": [],
                "vision_models": [],
                "model_count": 0,
                "latency_ms": latency_ms,
                "message": f"Connected to {spec.name}, but no models are installed or loaded.",
                "error": f"{spec.name} at {resolved_url} has no models available.",
            }

        return {
            "ok": True,
            "connected": True,
            "provider": prov_id,
            "provider_name": spec.name,
            "endpoint": resolved_url,
            "active_model": display_model,
            "resolved_model_id": resolved_id,
            "preferred_model": pref_model,
            "models": listed_models,
            "available_models": listed_models,
            "vision_models": vision_models,
            "model_count": len(listed_models),
            "latency_ms": latency_ms,
            "message": f"Connected to {spec.name}. Found {len(listed_models)} model(s).",
        }
    except ProviderConnectionError as exc:
        latency_ms = int(round((time.perf_counter() - t0) * 1000.0))
        return {
            "ok": False,
            "connected": False,
            "provider": prov_id,
            "provider_name": spec.name,
            "endpoint": resolved_url,
            "active_model": None,
            "resolved_model_id": None,
            "preferred_model": pref_model,
            "models": [],
            "available_models": [],
            "vision_models": [],
            "model_count": 0,
            "latency_ms": latency_ms,
            "message": str(exc),
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Persistent Settings Store
# ---------------------------------------------------------------------------


def is_frozen_executable() -> bool:
    """Return True when running inside a PyInstaller frozen executable (`comic-backend.exe`)."""
    return bool(getattr(sys, "frozen", False))


def get_settings_path() -> Path:
    """Resolve the persistent `settings.json` path.

    Priority:
    1. `MANGA_SETTINGS_PATH` environment variable (optional override).
    2. `%LOCALAPPDATA%/ComicTranslator/settings.json` when running as a frozen executable.
    3. `backend/settings.json` during development.
    """
    env_path = os.environ.get("MANGA_SETTINGS_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()

    if is_frozen_executable():
        local_app_data = os.environ.get("LOCALAPPDATA")
        base_dir = (
            Path(local_app_data).expanduser().resolve()
            if local_app_data
            else (Path.home() / "AppData" / "Local").resolve()
        )
        return base_dir / "ComicTranslator" / "settings.json"

    backend_dir = Path(__file__).resolve().parents[1]
    return backend_dir / "settings.json"


def build_default_provider_profiles() -> dict[str, ProviderProfileDict]:
    """Return default per-provider connection profiles for all 8 presets."""
    return {
        pid: {
            "base_url": spec.default_base_url,
            "api_key": spec.default_api_key,
            "model": spec.default_model,
        }
        for pid, spec in PROVIDER_PRESETS.items()
    }


def format_font_family_stack(chain: Sequence[str]) -> str:
    """Format an ordered font fallback chain list into a valid CSS `font-family` stack."""
    generics = {
        "sans-serif",
        "serif",
        "monospace",
        "cursive",
        "fantasy",
        "system-ui",
        "ui-sans-serif",
        "ui-serif",
        "ui-monospace",
    }
    formatted: list[str] = []
    seen: set[str] = set()
    for item in chain:
        cleaned = str(item).strip().strip('"').strip("'").strip()
        if not cleaned:
            continue
        lower = cleaned.lower()
        if lower in seen:
            continue
        seen.add(lower)
        if lower in generics:
            formatted.append(lower)
        elif " " in cleaned or "-" in cleaned:
            formatted.append(f'"{cleaned}"')
        else:
            formatted.append(cleaned)
    if not formatted:
        return DEFAULT_FONT_FAMILY_STACK
    if "sans-serif" not in seen and "serif" not in seen and "monospace" not in seen:
        formatted.append("sans-serif")
    return ", ".join(formatted)


def build_default_settings() -> AppSettingsDict:
    """Return a fresh copy of the default persistent application settings."""
    lm_spec = PROVIDER_PRESETS["lmstudio"]
    return {
        "provider": "lmstudio",
        "base_url": lm_spec.default_base_url,
        "api_key": lm_spec.default_api_key,
        "model": lm_spec.default_model,
        "timeout": 90.0,
        "source_lang": "ja",
        "target_lang": "id",
        "temperature": 0.2,
        "top_p": 0.95,
        "top_k": 40,
        "max_tokens": 2048,
        "frequency_penalty": 0.0,
        "reasoning_effort": "off",
        "strip_thinking": True,
        "use_vision": False,
        "vision_max_side": 1024,
        "custom_instructions": "",
        "glossary": {},
        "providers": build_default_provider_profiles(),
        # Tab 1: Appearance
        "theme": "dark",
        "accent_color": "amber",
        "ui_density": "compact",
        # Tab 2: Pipeline
        "execution_mode": "auto",
        "conf_threshold": 0.35,
        "nms_threshold": 0.45,
        "mask_threshold": 0.5,
        "bbox_padding": 6,
        "lama_tile_size": 512,
        "reading_order": "auto",
        "auto_unload_vram": True,
        "refine_mask": True,
        # Tab 5: Typesetting
        "font_family": DEFAULT_FONT_FAMILY_STACK,
        "font_fallback_chain": list(DEFAULT_FONT_FALLBACK_CHAIN),
        "auto_fit_font_size": True,
        "default_font_size": 18,
        "default_stroke_width": 4.0,
        "default_stroke_color": "#ffffff",
        "default_text_color": "#000000",
        "default_line_height": 1.18,
        "default_align": "center",
        "all_caps": False,
    }


def _clamp_float(val: object, low: float, high: float, default: float) -> float:
    if isinstance(val, bool):
        return default
    if isinstance(val, (int, float)):
        return max(low, min(high, float(val)))
    if isinstance(val, str):
        try:
            return max(low, min(high, float(val.strip())))
        except ValueError:
            return default
    return default


def _clamp_optional_float(
    val: object, low: float, high: float, default: float | None
) -> float | None:
    if val is None or val == "":
        return None
    if isinstance(val, bool):
        return default
    if isinstance(val, (int, float)):
        return max(low, min(high, float(val)))
    if isinstance(val, str):
        try:
            return max(low, min(high, float(val.strip())))
        except ValueError:
            return default
    return default


def _clamp_int(val: object, low: int, high: int, default: int) -> int:
    if isinstance(val, bool):
        return default
    if isinstance(val, (int, float)):
        return max(low, min(high, int(round(val))))
    if isinstance(val, str):
        try:
            return max(low, min(high, int(round(float(val.strip())))))
        except ValueError:
            return default
    return default


def _clamp_optional_int(
    val: object, low: int, high: int, default: int | None
) -> int | None:
    if val is None or val == "":
        return None
    if isinstance(val, bool):
        return default
    if isinstance(val, (int, float)):
        return max(low, min(high, int(round(val))))
    if isinstance(val, str):
        try:
            return max(low, min(high, int(round(float(val.strip())))))
        except ValueError:
            return default
    return default


def _coerce_bool(val: object, default: bool) -> bool:
    if isinstance(val, bool):
        return val
    if val is None:
        return default
    text = str(val).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    return default


def sanitize_settings_dict(
    raw: Mapping[str, object],
    base: AppSettingsDict | None = None,
) -> AppSettingsDict:
    """Merge and validate a raw settings mapping onto `base` (or defaults)."""
    current = (
        cast(AppSettingsDict, json.loads(json.dumps(base)))  # pyright: ignore[reportAny]
        if base is not None
        else build_default_settings()
    )

    # Ensure all 8 provider profiles exist
    profiles: dict[str, ProviderProfileDict] = build_default_provider_profiles()
    existing_profiles = current.get("providers", {})
    for pid, pdata in existing_profiles.items():
        if pid in PROVIDER_PRESETS:
            profiles[pid] = {
                "base_url": str(pdata.get("base_url", PROVIDER_PRESETS[cast(ProviderId, pid)].default_base_url)),
                "api_key": str(pdata.get("api_key", PROVIDER_PRESETS[cast(ProviderId, pid)].default_api_key)),
                "model": str(pdata.get("model", PROVIDER_PRESETS[cast(ProviderId, pid)].default_model)),
            }

    raw_profiles = raw.get("providers")
    if isinstance(raw_profiles, Mapping):
        for rpid, rpval in cast(Mapping[object, object], raw_profiles).items():
            try:
                norm_pid = normalize_provider_id(str(rpid))
            except ValueError:
                continue
            if isinstance(rpval, Mapping):
                rmap = cast(Mapping[str, object], rpval)
                cur_prof = profiles[norm_pid]
                profiles[norm_pid] = {
                    "base_url": (
                        str(rmap["base_url"]).strip()
                        if "base_url" in rmap and rmap["base_url"] is not None
                        else cur_prof["base_url"]
                    ),
                    "api_key": (
                        str(rmap["api_key"]).strip()
                        if "api_key" in rmap and rmap["api_key"] is not None
                        else cur_prof["api_key"]
                    ),
                    "model": (
                        str(rmap["model"]).strip()
                        if "model" in rmap and rmap["model"] is not None
                        else cur_prof["model"]
                    ),
                }

    prev_provider = current["provider"]
    if "provider" in raw and raw["provider"] is not None:
        active_provider = normalize_provider_id(str(raw["provider"]))
    else:
        active_provider = prev_provider

    provider_switched = active_provider != prev_provider
    active_profile = profiles[active_provider]

    if "base_url" in raw and raw["base_url"] is not None:
        base_url = normalize_provider_base_url(active_provider, str(raw["base_url"]))
    elif provider_switched:
        base_url = normalize_provider_base_url(active_provider, active_profile["base_url"])
    else:
        base_url = normalize_provider_base_url(active_provider, current["base_url"])

    if "api_key" in raw and raw["api_key"] is not None:
        api_key = str(raw["api_key"]).strip()
        if not api_key and PROVIDER_PRESETS[active_provider].default_api_key:
            api_key = PROVIDER_PRESETS[active_provider].default_api_key
    elif provider_switched:
        api_key = resolve_provider_api_key(active_provider, active_profile["api_key"])
    else:
        api_key = resolve_provider_api_key(active_provider, current["api_key"])

    if "model" in raw and raw["model"] is not None:
        model = str(raw["model"]).strip()
        if not model and not provider_switched:
            model = PROVIDER_PRESETS[active_provider].default_model
    elif provider_switched:
        model = active_profile["model"] or PROVIDER_PRESETS[active_provider].default_model
    else:
        model = current["model"] or PROVIDER_PRESETS[active_provider].default_model

    # Keep active provider's profile in sync with top-level connection fields
    profiles[active_provider] = {
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
    }

    source_lang = current["source_lang"]
    if "source_lang" in raw and raw["source_lang"] is not None:
        sl = str(raw["source_lang"]).strip().lower()
        if sl in ("ja", "jp", "japanese"):
            source_lang = "ja"
        elif sl in ("ko", "kr", "korean"):
            source_lang = "ko"
        elif sl in ("zh", "cn", "chinese"):
            source_lang = "zh"
        else:
            raise ValueError(f"Unsupported source_lang {raw['source_lang']!r}. Must be 'ja', 'ko', or 'zh'.")

    target_lang = current["target_lang"]
    if "target_lang" in raw and raw["target_lang"] is not None:
        target_lang = normalize_target_language(str(raw["target_lang"]))

    reasoning_effort = current["reasoning_effort"]
    if "reasoning_effort" in raw and raw["reasoning_effort"] is not None:
        reasoning_effort = normalize_reasoning_effort(str(raw["reasoning_effort"]))

    temperature = (
        _clamp_float(raw["temperature"], 0.0, 2.0, current["temperature"])
        if "temperature" in raw and raw["temperature"] is not None
        else current["temperature"]
    )
    top_p = (
        _clamp_optional_float(raw["top_p"], 0.0, 1.0, current["top_p"])
        if "top_p" in raw
        else current["top_p"]
    )
    top_k = (
        _clamp_optional_int(raw["top_k"], 0, 500, current["top_k"])
        if "top_k" in raw
        else current["top_k"]
    )
    max_tokens = (
        _clamp_int(raw["max_tokens"], 64, 32768, current["max_tokens"])
        if "max_tokens" in raw and raw["max_tokens"] is not None
        else current["max_tokens"]
    )
    frequency_penalty = (
        _clamp_optional_float(
            raw["frequency_penalty"], -2.0, 2.0, current["frequency_penalty"]
        )
        if "frequency_penalty" in raw
        else current["frequency_penalty"]
    )
    timeout = (
        _clamp_float(raw["timeout"], 5.0, 300.0, current["timeout"])
        if "timeout" in raw and raw["timeout"] is not None
        else current["timeout"]
    )
    use_vision = (
        _coerce_bool(raw["use_vision"], current["use_vision"])
        if "use_vision" in raw and raw["use_vision"] is not None
        else current["use_vision"]
    )
    vision_max_side = (
        _clamp_int(raw["vision_max_side"], 256, 2048, current["vision_max_side"])
        if "vision_max_side" in raw and raw["vision_max_side"] is not None
        else current["vision_max_side"]
    )
    custom_instructions = (
        str(raw["custom_instructions"]).strip()
        if "custom_instructions" in raw and raw["custom_instructions"] is not None
        else current["custom_instructions"]
    )
    glossary = (
        parse_glossary(raw["glossary"])
        if "glossary" in raw
        else dict(current["glossary"])
    )
    strip_thinking = (
        _coerce_bool(raw["strip_thinking"], current.get("strip_thinking", True))
        if "strip_thinking" in raw and raw["strip_thinking"] is not None
        else current.get("strip_thinking", True)
    )

    # Tab 1: Appearance
    theme: ThemeOption = current.get("theme", "dark")
    if "theme" in raw and raw["theme"] is not None:
        raw_theme = str(raw["theme"]).strip().lower()
        if raw_theme in ("dark", "light", "system"):
            theme = cast(ThemeOption, raw_theme)

    accent_color: AccentColorOption = current.get("accent_color", "amber")
    if "accent_color" in raw and raw["accent_color"] is not None:
        raw_accent = str(raw["accent_color"]).strip().lower()
        if raw_accent in ("amber", "teal", "crimson", "slate"):
            accent_color = cast(AccentColorOption, raw_accent)

    ui_density: UiDensityOption = current.get("ui_density", "compact")
    if "ui_density" in raw and raw["ui_density"] is not None:
        raw_density = str(raw["ui_density"]).strip().lower()
        if raw_density in ("compact", "comfortable"):
            ui_density = cast(UiDensityOption, raw_density)

    # Tab 2: Pipeline
    execution_mode: ExecutionModeOption = current.get("execution_mode", "auto")
    if "execution_mode" in raw and raw["execution_mode"] is not None:
        raw_exec = str(raw["execution_mode"]).strip().lower()
        if raw_exec in ("cpu", "cpu_only", "cpu only"):
            execution_mode = "cpu"
        else:
            execution_mode = "auto"
    os.environ["MANGA_FORCE_CPU"] = "1" if execution_mode == "cpu" else "0"

    conf_threshold = (
        _clamp_float(raw["conf_threshold"], 0.05, 0.95, current.get("conf_threshold", 0.35))
        if "conf_threshold" in raw and raw["conf_threshold"] is not None
        else current.get("conf_threshold", 0.35)
    )
    nms_threshold = (
        _clamp_float(raw["nms_threshold"], 0.05, 0.95, current.get("nms_threshold", 0.45))
        if "nms_threshold" in raw and raw["nms_threshold"] is not None
        else current.get("nms_threshold", 0.45)
    )
    mask_threshold = (
        _clamp_float(raw["mask_threshold"], 0.05, 0.95, current.get("mask_threshold", 0.5))
        if "mask_threshold" in raw and raw["mask_threshold"] is not None
        else current.get("mask_threshold", 0.5)
    )
    bbox_padding = (
        _clamp_int(raw["bbox_padding"], 0, 64, current.get("bbox_padding", 6))
        if "bbox_padding" in raw and raw["bbox_padding"] is not None
        else current.get("bbox_padding", 6)
    )
    lama_tile_size = (
        _clamp_int(raw["lama_tile_size"], 256, 1024, current.get("lama_tile_size", 512))
        if "lama_tile_size" in raw and raw["lama_tile_size"] is not None
        else current.get("lama_tile_size", 512)
    )

    reading_order: ReadingOrderOption = current.get("reading_order", "auto")
    if "reading_order" in raw and raw["reading_order"] is not None:
        raw_ro = str(raw["reading_order"]).strip().lower()
        if raw_ro in ("auto", "rtl", "ltr"):
            reading_order = cast(ReadingOrderOption, raw_ro)

    auto_unload_vram = (
        _coerce_bool(raw["auto_unload_vram"], current.get("auto_unload_vram", True))
        if "auto_unload_vram" in raw and raw["auto_unload_vram"] is not None
        else current.get("auto_unload_vram", True)
    )
    refine_mask = (
        _coerce_bool(raw["refine_mask"], current.get("refine_mask", True))
        if "refine_mask" in raw and raw["refine_mask"] is not None
        else current.get("refine_mask", True)
    )

    # Tab 5: Typesetting
    font_fallback_chain: list[str] = list(
        current.get("font_fallback_chain", list(DEFAULT_FONT_FALLBACK_CHAIN))
    )
    chain_updated = False
    if "font_fallback_chain" in raw and raw["font_fallback_chain"] is not None:
        raw_chain = raw["font_fallback_chain"]
        if isinstance(raw_chain, Sequence) and not isinstance(raw_chain, (str, bytes)):
            cleaned_chain: list[str] = []
            seen_chain: set[str] = set()
            for entry in cast(Sequence[object], raw_chain):
                entry_str = str(entry).strip().strip('"').strip("'").strip()
                if entry_str and entry_str.lower() not in seen_chain:
                    seen_chain.add(entry_str.lower())
                    cleaned_chain.append(entry_str)
            if cleaned_chain:
                font_fallback_chain = cleaned_chain
                chain_updated = True

    if "font_family" in raw and raw["font_family"] is not None and str(raw["font_family"]).strip():
        font_family = str(raw["font_family"]).strip()
        if not chain_updated:
            # Parse comma-separated font family string into fallback chain if chain wasn't explicitly provided
            parsed_chain = [
                part.strip().strip('"').strip("'").strip()
                for part in font_family.split(",")
                if part.strip().strip('"').strip("'").strip()
            ]
            if parsed_chain:
                font_fallback_chain = parsed_chain
    elif chain_updated:
        font_family = format_font_family_stack(font_fallback_chain)
    else:
        font_family = current.get("font_family", DEFAULT_FONT_FAMILY_STACK)

    auto_fit_font_size = (
        _coerce_bool(raw["auto_fit_font_size"], current.get("auto_fit_font_size", True))
        if "auto_fit_font_size" in raw and raw["auto_fit_font_size"] is not None
        else current.get("auto_fit_font_size", True)
    )
    default_font_size = (
        _clamp_int(raw["default_font_size"], 8, 96, current.get("default_font_size", 18))
        if "default_font_size" in raw and raw["default_font_size"] is not None
        else current.get("default_font_size", 18)
    )
    default_stroke_width = (
        _clamp_float(raw["default_stroke_width"], 0.0, 16.0, current.get("default_stroke_width", 4.0))
        if "default_stroke_width" in raw and raw["default_stroke_width"] is not None
        else current.get("default_stroke_width", 4.0)
    )
    default_stroke_color = (
        str(raw["default_stroke_color"]).strip()
        if "default_stroke_color" in raw
        and raw["default_stroke_color"] is not None
        and str(raw["default_stroke_color"]).strip()
        else current.get("default_stroke_color", "#ffffff")
    )
    default_text_color = (
        str(raw["default_text_color"]).strip()
        if "default_text_color" in raw
        and raw["default_text_color"] is not None
        and str(raw["default_text_color"]).strip()
        else current.get("default_text_color", "#000000")
    )
    default_line_height = (
        _clamp_float(raw["default_line_height"], 0.8, 2.5, current.get("default_line_height", 1.18))
        if "default_line_height" in raw and raw["default_line_height"] is not None
        else current.get("default_line_height", 1.18)
    )
    default_align: TextAlignOption = current.get("default_align", "center")
    if "default_align" in raw and raw["default_align"] is not None:
        raw_align = str(raw["default_align"]).strip().lower()
        if raw_align in ("left", "center", "right"):
            default_align = cast(TextAlignOption, raw_align)

    all_caps = (
        _coerce_bool(raw["all_caps"], current.get("all_caps", False))
        if "all_caps" in raw and raw["all_caps"] is not None
        else current.get("all_caps", False)
    )

    return {
        "provider": active_provider,
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
        "timeout": timeout,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_tokens,
        "frequency_penalty": frequency_penalty,
        "reasoning_effort": reasoning_effort,
        "strip_thinking": strip_thinking,
        "use_vision": use_vision,
        "vision_max_side": vision_max_side,
        "custom_instructions": custom_instructions,
        "glossary": glossary,
        "providers": profiles,
        "theme": theme,
        "accent_color": accent_color,
        "ui_density": ui_density,
        "execution_mode": execution_mode,
        "conf_threshold": conf_threshold,
        "nms_threshold": nms_threshold,
        "mask_threshold": mask_threshold,
        "bbox_padding": bbox_padding,
        "lama_tile_size": lama_tile_size,
        "reading_order": reading_order,
        "auto_unload_vram": auto_unload_vram,
        "refine_mask": refine_mask,
        "font_family": font_family,
        "font_fallback_chain": font_fallback_chain,
        "auto_fit_font_size": auto_fit_font_size,
        "default_font_size": default_font_size,
        "default_stroke_width": default_stroke_width,
        "default_stroke_color": default_stroke_color,
        "default_text_color": default_text_color,
        "default_line_height": default_line_height,
        "default_align": default_align,
        "all_caps": all_caps,
    }


class SettingsStore:
    """Thread-safe persistent JSON settings store with atomic writes."""

    path: Path
    _lock: threading.RLock
    _cache: AppSettingsDict | None

    def __init__(self, path: Path | None = None) -> None:
        self.path = path if path is not None else get_settings_path()
        self._lock = threading.RLock()
        self._cache = None

    def load(self, force_reload: bool = False) -> AppSettingsDict:
        """Load settings from disk, falling back cleanly to defaults if missing or unreadable."""
        with self._lock:
            if self._cache is not None and not force_reload:
                return cast(
                    AppSettingsDict,
                    json.loads(json.dumps(self._cache)),  # pyright: ignore[reportAny]
                )

            defaults = build_default_settings()
            if not self.path.is_file():
                self._cache = defaults
                return cast(
                    AppSettingsDict,
                    json.loads(json.dumps(defaults)),  # pyright: ignore[reportAny]
                )

            try:
                raw_text = self.path.read_text(encoding="utf-8")
                parsed_obj = _json_loads(raw_text)
                if isinstance(parsed_obj, Mapping):
                    loaded = sanitize_settings_dict(
                        cast(Mapping[str, object], parsed_obj),
                        base=defaults,
                    )
                else:
                    loaded = defaults
            except (OSError, ValueError):
                loaded = defaults

            self._cache = loaded
            return cast(
                AppSettingsDict,
                json.loads(json.dumps(loaded)),  # pyright: ignore[reportAny]
            )

    def save(self, settings: AppSettingsDict) -> AppSettingsDict:
        """Atomically persist `settings` to disk (`*.tmp` -> `os.replace`)."""
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
            serialized = json.dumps(settings, ensure_ascii=False, indent=2) + "\n"
            try:
                tmp_path.write_text(serialized, encoding="utf-8")
                os.replace(tmp_path, self.path)
            except Exception:
                tmp_path.unlink(missing_ok=True)
                raise

            self._cache = cast(
                AppSettingsDict,
                json.loads(json.dumps(settings)),  # pyright: ignore[reportAny]
            )
            return cast(
                AppSettingsDict,
                json.loads(json.dumps(settings)),  # pyright: ignore[reportAny]
            )

    def update(self, patch: Mapping[str, object]) -> AppSettingsDict:
        """Merge a partial or full settings update, validate, and persist atomically."""
        with self._lock:
            current = self.load()
            merged = sanitize_settings_dict(patch, base=current)
            return self.save(merged)
