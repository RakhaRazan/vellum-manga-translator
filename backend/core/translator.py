"""Multi-Provider Context-Aware Comic Dialogue Translator (Koharu-style).

Supports 8 providers via `backend.core.providers`:
- `lmstudio` (default `http://localhost:1235/v1`, default model `"Gemma 4 E2B Uncensored HauhauCS Aggressive Q4 K P"`)
- `ollama` (`http://localhost:11434/v1`)
- `openai` (`https://api.openai.com/v1`)
- `gemini` (`https://generativelanguage.googleapis.com/v1beta/openai`)
- `anthropic` (`https://api.anthropic.com/v1`, native Messages API)
- `deepseek` (`https://api.deepseek.com/v1`)
- `openrouter` (`https://openrouter.ai/api/v1`)
- `custom` (arbitrary OpenAI-compatible Base URL + API Key)

Advanced translation controls:
- Sampling: `temperature`, `top_p`, `top_k`, `max_tokens`, `frequency_penalty`.
- Reasoning: `reasoning_effort` (`off` / `low` / `medium` / `high`) + automatic stripping
  of `<think>...</think>` / `<thought>...</thought>` tags prior to JSON extraction.
- Vision (`use_vision`): attaches a resized JPEG comic page image in multimodal payloads
  for vision-capable models, with automatic text-only fallback if rejected.
- Custom Instruction Bar + Glossary (`source -> target`) injected into the system prompt.
- 16 target languages: `id`, `en`, `ja`, `ko`, `zh-CN`, `zh-TW`, `es`, `fr`, `de`, `pt`,
  `ru`, `vi`, `th`, `ms`, `ar`, `it`.
"""

import base64
from collections.abc import Mapping, Sequence
import io
import json
from pathlib import Path
import re
import sys
from typing import Literal, NotRequired, TypedDict, cast

import numpy as np
import openai
from openai import OpenAI, omit
from openai.types.chat import (
    ChatCompletionContentPartParam,
    ChatCompletionMessageParam,
)
from openai.types.shared_params import (
    ResponseFormatJSONObject,
    ResponseFormatJSONSchema,
)
from PIL import Image
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.detector import NDArrayU8, TextDirection  # noqa: E402
from backend.core.ocr import (  # noqa: E402
    OcrBubbleResult,
    SourceLanguage,
    normalize_language,
)
from backend.core.providers import (  # noqa: E402
    AppSettingsDict,
    DEFAULT_ANTHROPIC_VERSION,
    DEFAULT_LM_STUDIO_API_KEY,
    DEFAULT_LM_STUDIO_BASE_URL,
    DEFAULT_LM_STUDIO_MODEL,
    PROVIDER_PRESETS,
    ProviderConnectionError,
    ProviderId,
    ProviderSpec,
    ProviderTranslationError,
    ReasoningEffort,
    SUPPORTED_REASONING_EFFORTS,
    SUPPORTED_TARGET_LANGUAGES,
    TARGET_LANGUAGES,
    TargetLanguage,
    get_provider_spec,
    list_provider_models,
    normalize_provider_base_url,
    normalize_provider_id,
    normalize_reasoning_effort,
    normalize_target_language,
    parse_glossary,
    resolve_provider_api_key,
    resolve_provider_model,
)

# Backward-compatible exception aliases (SPEC.md v2 & existing routes)
LMStudioConnectionError = ProviderConnectionError
LMStudioTranslationError = ProviderTranslationError

OpenAIReasoningEffort = Literal["none", "minimal", "low", "medium", "high"]

type _JsonValue = (
    None
    | bool
    | int
    | float
    | str
    | Sequence[_JsonValue]
    | Mapping[str, _JsonValue]
)

_ANTHROPIC_THINKING_BUDGETS: dict[ReasoningEffort, int] = {
    "off": 0,
    "low": 1024,
    "medium": 4096,
    "high": 8192,
}

_DATA_URI_RE = re.compile(
    r"^data:(image/[a-zA-Z0-9.+-]+);base64,([A-Za-z0-9+/=\s]+)$",
    flags=re.IGNORECASE,
)


class DialogueBlockInput(TypedDict):
    """Normalized input dialogue block sent to the LLM for full-page translation."""

    id: str
    source_text: str


class TranslatedDialogueItem(TypedDict):
    """Translated dialogue item matching SPEC.md Section 5.2."""

    id: str
    source_text: str
    translated_text: str


class TranslatedBubbleResult(TypedDict):
    """Full OCR bubble metadata enriched with `translated_text` matching SPEC.md Section 5.1."""

    id: str
    order: int
    bbox: list[int]
    polygon: list[list[int]]
    source_text: str
    translated_text: str
    confidence: float
    text_direction: TextDirection


class PageTranslationResult(TypedDict):
    """Complete page translation response containing ID-mapped dict, dialogues, and runtime flags."""

    provider: ProviderId
    model: str
    source_language: SourceLanguage
    target_language: TargetLanguage
    vision_used: bool
    reasoning_effort: ReasoningEffort
    translations: dict[str, str]
    dialogues: list[TranslatedDialogueItem]


class LMStudioStatus(TypedDict):
    """Connectivity and active model status matching SPEC.md Section 5.5."""

    connected: bool
    endpoint: str
    active_model: str | None
    available_models: list[str]
    error: NotRequired[str]


# SPEC.md Section 5.2 Structured Translation JSON Schema
COMIC_TRANSLATION_JSON_SCHEMA: ResponseFormatJSONSchema = {
    "type": "json_schema",
    "json_schema": {
        "name": "comic_translation_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "dialogues": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "translated_text": {"type": "string"},
                        },
                        "required": ["id", "translated_text"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["dialogues"],
            "additionalProperties": False,
        },
    },
}

_JSON_OBJECT_FORMAT: ResponseFormatJSONObject = {"type": "json_object"}

# Tailored comic-book style rules for all 16 supported target languages
_TARGET_STYLE_RULES: dict[TargetLanguage, str] = {
    "id": (
        "Write natural, expressive spoken comic Indonesian (Bahasa Indonesia). "
        + "Keep character tone, emotion, and casual slang natural for speech bubbles "
        + "(use 'aku/kamu', 'kau', or casual 'gue/lu' when the speaker is rough/slangy; "
        + "never sound stiff or machine-translated)."
    ),
    "en": (
        "Write natural, idiomatic comic-book English. "
        + "Use punchy phrasing, natural contractions, and expressive slang that fits each character's emotion and tone."
    ),
    "ja": (
        "Write natural, expressive Japanese comic dialogue (日本語) matching each speaker's register, "
        + "pronouns, and sentence-ending particles naturally for manga bubbles."
    ),
    "ko": (
        "Write natural, expressive Korean comic dialogue (한국어) with appropriate speech levels "
        + "(반말/존댓말) matching character relationships and scene tension."
    ),
    "zh-CN": (
        "Write natural, idiomatic Simplified Chinese comic dialogue (简体中文) with lively pacing, "
        + "authentic colloquial expressions, and tone fitting each character."
    ),
    "zh-TW": (
        "Write natural, idiomatic Traditional Chinese comic dialogue (繁體中文, Taiwan/HK comic style) "
        + "with lively pacing and authentic spoken phrasing."
    ),
    "es": (
        "Write natural, expressive comic-book Spanish (Español) using colloquial phrasing, "
        + "appropriate register (tú/usted), and punchy dialogue suited for speech bubbles."
    ),
    "fr": (
        "Write natural, expressive comic-book French (Français) with authentic spoken tone, "
        + "appropriate tutoiement/vouvoiement, and crisp bubble-friendly phrasing."
    ),
    "de": (
        "Write natural, idiomatic comic-book German (Deutsch) with lively spoken phrasing, "
        + "natural du/Sie register, and punchy dialogue."
    ),
    "pt": (
        "Write natural, expressive comic-book Portuguese (Português) with lively colloquial phrasing "
        + "and authentic emotional nuance suitable for speech bubbles."
    ),
    "ru": (
        "Write natural, expressive comic-book Russian (Русский) with vivid colloquial tone, "
        + "appropriate ты/вы register, and punchy dialogue."
    ),
    "vi": (
        "Write natural, expressive comic-book Vietnamese (Tiếng Việt) using authentic pronouns "
        + "(tôi/cậu, tao/mày, anh/em, etc.) that accurately reflect character relationships and emotion."
    ),
    "th": (
        "Write natural, expressive comic-book Thai (ภาษาไทย) with fitting pronouns, particles "
        + "(ครับ/ค่ะ/นะ/เว้ย), and lively colloquial rhythm for comic bubbles."
    ),
    "ms": (
        "Write natural, expressive comic-book Malay (Bahasa Melayu) using natural spoken phrasing "
        + "(aku/kau, saya/awak) matching each character's tone and emotion."
    ),
    "ar": (
        "Write natural, expressive comic-book Arabic (العربية) with clear, vivid, and concise phrasing "
        + "suitable for comic speech bubbles while capturing each character's emotion."
    ),
    "it": (
        "Write natural, expressive comic-book Italian (Italiano) with idiomatic spoken tone, "
        + "appropriate tu/Lei register, and crisp bubble-friendly dialogue."
    ),
}


def _json_loads(raw: str) -> object:
    """Parse a JSON string into a typed `object` (isolating stdlib `json.loads` `Any` return)."""
    res: object = json.loads(raw)  # pyright: ignore[reportAny]
    return res


def normalize_source_language(source_lang: str) -> SourceLanguage:
    """Validate and normalize `source_lang` (`'ja'`, `'ko'`, or `'zh'`)."""
    return normalize_language(source_lang)


def list_lm_studio_models(
    base_url: str = DEFAULT_LM_STUDIO_BASE_URL,
    api_key: str = DEFAULT_LM_STUDIO_API_KEY,
    timeout: float = 5.0,
) -> list[str]:
    """Query `GET {base_url}/models` on LM Studio and return all available model IDs."""
    return list_provider_models(
        provider="lmstudio",
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        include_embeddings=True,
    )


def resolve_lm_studio_model(
    preferred_model: str = DEFAULT_LM_STUDIO_MODEL,
    base_url: str = DEFAULT_LM_STUDIO_BASE_URL,
    api_key: str = DEFAULT_LM_STUDIO_API_KEY,
    timeout: float = 5.0,
) -> str:
    """Resolve `preferred_model` against LM Studio `/v1/models` and `/api/v0/models`."""
    return resolve_provider_model(
        provider="lmstudio",
        preferred_model=preferred_model,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
    )


def encode_image_for_vision(
    image: NDArrayU8 | Image.Image | bytes | Path | str,
    max_side: int = 1024,
    jpeg_quality: int = 84,
) -> str:
    """Resize a comic page image to `<= max_side` px and encode as a compact JPEG data URI."""
    if isinstance(image, str):
        cleaned = image.strip()
        if _DATA_URI_RE.match(cleaned):
            return cleaned
        path_candidate = Path(cleaned)
        if path_candidate.is_file():
            pil_img = Image.open(path_candidate)
        else:
            raw_b64 = cleaned.split(",", 1)[1] if "," in cleaned else cleaned
            pil_img = Image.open(io.BytesIO(base64.b64decode(raw_b64)))
    elif isinstance(image, Path):
        pil_img = Image.open(image)
    elif isinstance(image, bytes):
        pil_img = Image.open(io.BytesIO(image))
    elif isinstance(image, Image.Image):
        pil_img = image
    else:
        arr = np.asarray(image, dtype=np.uint8)
        pil_img = Image.fromarray(arr)

    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    w, h = pil_img.size
    longest = max(w, h)
    clamped_max = max(256, min(2048, int(max_side)))
    if longest > clamped_max:
        scale = clamped_max / float(longest)
        new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def build_system_prompt(
    source_lang: SourceLanguage,
    target_lang: TargetLanguage,
    *,
    reasoning_effort: ReasoningEffort = "off",
    custom_instructions: str | None = None,
    glossary: Mapping[str, str] | None = None,
    has_page_image: bool = False,
) -> str:
    """Build a concise, dynamically tailored system prompt for comic page translation."""
    if source_lang == "ja":
        source_desc = "Japanese Manga (ja)"
        honorific_rule = (
            "Preserve Japanese honorifics naturally (e.g., -san, -senpai, -kun, -chan, -sama, -sensei "
            + "such as 'Tanaka-senpai', 'Sato-san') when appropriate for the target language."
        )
    elif source_lang == "ko":
        source_desc = "Korean Manhwa (ko)"
        honorific_rule = (
            "Preserve Korean honorifics and kinship/address terms naturally "
            + "(e.g., oppa, hyung, sunbae, noona, unnie, -nim, dongsaeng) when appropriate for the target language."
        )
    else:
        source_desc = "Chinese Manhua (zh)"
        honorific_rule = (
            "Preserve Chinese martial and familial honorifics accurately "
            + "(e.g., 哥哥 -> 'gege', 师兄 -> 'shixiong', 师姐 -> 'shijie', 师父 -> 'shifu', 师弟 -> 'shidi', 姐姐 -> 'jiejie') "
            + "when translating into non-Chinese languages."
        )

    tgt_spec = TARGET_LANGUAGES[target_lang]
    target_desc = f"{tgt_spec.name} ({tgt_spec.native_name}, code: {tgt_spec.code})"
    style_rule = _TARGET_STYLE_RULES[target_lang]

    thinking_directive = (
        "Do not think, analyze, or explain. "
        if reasoning_effort == "off"
        else "Keep any internal reasoning brief, then output strictly valid JSON. "
    )

    lines: list[str] = [
        f"You are an expert {source_desc} to {target_desc} comic translator. "
        + thinking_directive
        + f"Translate the full page of dialogue blocks from {source_desc} into {target_desc}.",
        "Rules:",
        "1. Read all blocks together as one continuous scene so conversational context, pronouns, and speaker tone remain consistent across bubbles.",
        f"2. {style_rule}",
        f"3. {honorific_rule}",
        '4. Output ONLY valid JSON in the exact format: {"dialogues": [{"id": "<block_id>", "translated_text": "<translation>"}]}. Keep every input block \'id\' unchanged.',
    ]

    if glossary:
        cleaned_pairs = [
            f"- {src_term.strip()} -> {tgt_term.strip()}"
            for src_term, tgt_term in glossary.items()
            if src_term.strip() and tgt_term.strip()
        ]
        if cleaned_pairs:
            lines.append(
                "Glossary (MUST follow strictly — always translate these terms as specified):\n"
                + "\n".join(cleaned_pairs)
            )

    if custom_instructions and custom_instructions.strip():
        lines.append(f"Custom Instructions (MUST follow): {custom_instructions.strip()}")

    if has_page_image:
        lines.append(
            "Visual Context: Use the attached comic page image to infer speaker gender, expressions, "
            + "action intensity, relationships, and scene mood so every bubble matches the artwork."
        )

    return "\n".join(lines)


def normalize_input_blocks(
    blocks: Sequence[Mapping[str, object]],
) -> list[DialogueBlockInput]:
    """Normalize an array of detected/OCR text blocks into `[{'id': ..., 'source_text': ...}]`."""
    normalized: list[DialogueBlockInput] = []
    for idx, raw_block in enumerate(blocks, start=1):
        raw_id = raw_block.get("id")
        block_id = (
            str(raw_id).strip()
            if raw_id is not None and str(raw_id).strip()
            else f"bubble_{idx:03d}"
        )

        text_val: object = raw_block.get("source_text")
        if text_val is None:
            text_val = raw_block.get("original_text")
        if text_val is None:
            text_val = raw_block.get("text")
        if text_val is None:
            text_val = ""

        cleaned_text = re.sub(r"\s+", " ", str(text_val)).strip()
        normalized.append(
            {
                "id": block_id,
                "source_text": cleaned_text,
            }
        )
    return normalized


def strip_thinking_tags(raw_content: str) -> str:
    """Remove `<think>...</think>`, `<thought>...</thought>`, and `<|thought|>...<|thought|>` blocks."""
    text = raw_content.strip()
    if not text:
        return ""

    # 1. Strip standard closed reasoning tags
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<thought>[\s\S]*?</thought>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<\|thought\|>[\s\S]*?<\|thought\|>", "", text, flags=re.IGNORECASE)

    # 2. Handle an unclosed <think> or <thought> tag (e.g. if a reasoning model omitted closing tag
    #    before emitting ```json or `{...}`)
    unclosed_match = re.search(r"<(?:think|thought|\|thought\|)>", text, flags=re.IGNORECASE)
    if unclosed_match:
        prefix = text[: unclosed_match.start()].strip()
        remainder = text[unclosed_match.end() :]
        fence_idx = remainder.find("```")
        brace_idx = remainder.find("{")
        candidates = [idx for idx in (fence_idx, brace_idx) if idx != -1]
        if candidates:
            after_thought = remainder[min(candidates) :].strip()
            text = f"{prefix}\n{after_thought}".strip() if prefix else after_thought
        elif prefix:
            text = prefix

    return text.strip()


def _strip_fences_and_noise(raw_content: str) -> str:
    """Strip reasoning tags, ```json markdown fences, outer prose, and trailing commas."""
    text = strip_thinking_tags(raw_content)
    if not text:
        return ""

    fence_match = re.search(
        r"```(?:json)?\s*([\s\S]*?)```",
        text,
        flags=re.IGNORECASE,
    )
    if fence_match:
        fenced_body = fence_match.group(1).strip()
        if fenced_body:
            text = fenced_body

    first_brace = text.find("{")
    first_bracket = text.find("[")
    start_idx = -1
    if first_brace != -1 and first_bracket != -1:
        start_idx = min(first_brace, first_bracket)
    elif first_brace != -1:
        start_idx = first_brace
    elif first_bracket != -1:
        start_idx = first_bracket

    if start_idx != -1:
        last_brace = text.rfind("}")
        last_bracket = text.rfind("]")
        end_idx = max(last_brace, last_bracket)
        if end_idx > start_idx:
            text = text[start_idx : end_idx + 1]

    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text.strip()


def _extract_mapping_from_parsed_json(parsed: object) -> dict[str, str]:
    """Convert any parsed JSON structure (SPEC schema, array of objects, or ID-keyed dict) into `{id: translated_text}`."""
    mapping: dict[str, str] = {}

    if isinstance(parsed, Mapping):
        parsed_map = cast(Mapping[str, object], parsed)

        for list_key in ("dialogues", "translations", "blocks", "items", "data", "results"):
            sub_val = parsed_map.get(list_key)
            if isinstance(sub_val, Sequence) and not isinstance(sub_val, (str, bytes)):
                sub_map = _extract_mapping_from_parsed_json(sub_val)
                if sub_map:
                    return sub_map
            elif isinstance(sub_val, Mapping):
                sub_map = _extract_mapping_from_parsed_json(
                    cast(Mapping[object, object], sub_val)
                )
                if sub_map:
                    return sub_map

        if "id" in parsed_map and any(
            k in parsed_map for k in ("translated_text", "translation", "text", "target_text")
        ):
            item_id = str(parsed_map.get("id", "")).strip()
            for t_key in ("translated_text", "translation", "text", "target_text"):
                t_val = parsed_map.get(t_key)
                if isinstance(t_val, str) and item_id:
                    mapping[item_id] = t_val.strip()
                    return mapping

        for k, v in parsed_map.items():
            key_str = str(k).strip()
            if not key_str:
                continue
            if isinstance(v, str):
                mapping[key_str] = v.strip()
            elif isinstance(v, Mapping):
                v_map = cast(Mapping[str, object], v)
                for t_key in ("translated_text", "translation", "text", "target_text"):
                    t_val = v_map.get(t_key)
                    if isinstance(t_val, str):
                        mapping[key_str] = t_val.strip()
                        break
        return mapping

    if isinstance(parsed, Sequence) and not isinstance(parsed, (str, bytes)):
        for entry_obj in parsed:
            if not isinstance(entry_obj, Mapping):
                continue
            entry = cast(Mapping[str, object], entry_obj)
            raw_id = entry.get("id")
            if raw_id is not None and str(raw_id).strip():
                item_id = str(raw_id).strip()
                for t_key in ("translated_text", "translation", "text", "target_text"):
                    t_val = entry.get(t_key)
                    if isinstance(t_val, str):
                        mapping[item_id] = t_val.strip()
                        break
            elif len(entry) == 1:
                for k, v in entry.items():
                    if isinstance(v, str):
                        mapping[str(k).strip()] = v.strip()
        return mapping

    return mapping


def _regex_salvage_translations(
    think_stripped_text: str,
    expected_ids: Sequence[str],
) -> dict[str, str]:
    """Regex fallback extractor if the LLM produced truncated or slightly malformed JSON."""
    salvaged: dict[str, str] = {}

    obj_pattern = re.compile(
        r'"id"\s*:\s*"([^"]+)"[\s\S]*?"(?:translated_text|translation|text|target_text)"\s*:\s*"((?:[^"\\]|\\.)*)"'
    )
    for match in obj_pattern.finditer(think_stripped_text):
        bid = match.group(1).strip()
        raw_val = match.group(2)
        try:
            unescaped = str(_json_loads(f'"{raw_val}"')).strip()
        except ValueError:
            unescaped = raw_val.replace('\\"', '"').replace("\\n", " ").strip()
        if bid:
            salvaged[bid] = unescaped

    for exp_id in expected_ids:
        if exp_id in salvaged:
            continue
        kv_pattern = re.compile(rf'"{re.escape(exp_id)}"\s*:\s*"((?:[^"\\]|\\.)*)"')
        kv_match = kv_pattern.search(think_stripped_text)
        if kv_match:
            raw_val = kv_match.group(1)
            try:
                unescaped = str(_json_loads(f'"{raw_val}"')).strip()
            except ValueError:
                unescaped = raw_val.replace('\\"', '"').replace("\\n", " ").strip()
            salvaged[exp_id] = unescaped

    return salvaged


def extract_and_clean_translations(
    raw_content: str,
    input_blocks: Sequence[DialogueBlockInput],
) -> dict[str, str]:
    """Extract and sanitize LLM JSON output into a strict `{block_id: translated_text}` dictionary.

    Automatically strips `<think>...</think>` blocks, ```json fences, trailing commas,
    and conversational prose, guaranteeing every input `block["id"]` is present in the output.
    """
    expected_ids = [b["id"] for b in input_blocks]
    think_stripped = strip_thinking_tags(raw_content)
    cleaned = _strip_fences_and_noise(think_stripped)

    extracted: dict[str, str] = {}
    if cleaned:
        try:
            parsed_obj = _json_loads(cleaned)
            extracted = _extract_mapping_from_parsed_json(parsed_obj)
        except ValueError:
            sanitized = re.sub(r"[\x00-\x1f]+", " ", cleaned)
            try:
                parsed_obj = _json_loads(sanitized)
                extracted = _extract_mapping_from_parsed_json(parsed_obj)
            except ValueError:
                extracted = {}

    if not extracted or len(extracted) < len(expected_ids):
        salvaged = _regex_salvage_translations(think_stripped, expected_ids)
        for k, v in salvaged.items():
            if k not in extracted or not extracted[k]:
                extracted[k] = v

    result: dict[str, str] = {}
    extracted_values_in_order = list(extracted.values())

    for idx, block in enumerate(input_blocks):
        bid = block["id"]
        src = block["source_text"]
        if not src:
            result[bid] = ""
            continue

        if bid in extracted and extracted[bid]:
            result[bid] = extracted[bid]
            continue

        matched_val: str | None = None
        for ext_k, ext_v in extracted.items():
            if ext_k.lower() == bid.lower() and ext_v:
                matched_val = ext_v
                break
        if matched_val is None:
            num_match = re.search(r"(\d+)$", bid)
            if num_match:
                num_str = num_match.group(1)
                num_int_str = str(int(num_str))
                for candidate_key in (num_str, num_int_str, f"bubble_{num_int_str}"):
                    if candidate_key in extracted and extracted[candidate_key]:
                        matched_val = extracted[candidate_key]
                        break

        if (
            matched_val is None
            and len(extracted_values_in_order) == len(input_blocks)
            and extracted_values_in_order[idx]
        ):
            matched_val = extracted_values_in_order[idx]

        result[bid] = matched_val if matched_val is not None else src

    return result


class ComicTranslator:
    """Multi-provider, vision-capable comic dialogue translator with advanced sampling & reasoning controls."""

    provider: ProviderId
    spec: ProviderSpec
    base_url: str
    api_key: str
    preferred_model: str
    temperature: float
    top_p: float | None
    top_k: int | None
    max_tokens: int
    frequency_penalty: float | None
    reasoning_effort: ReasoningEffort
    use_vision: bool
    vision_max_side: int
    custom_instructions: str
    glossary: dict[str, str]
    timeout: float
    client: OpenAI | None
    _resolved_model: str | None

    def __init__(
        self,
        provider: str = "lmstudio",
        base_url: str | None = None,
        api_key: str | None = None,
        preferred_model: str | None = None,
        temperature: float = 0.2,
        top_p: float | None = 0.95,
        top_k: int | None = 40,
        max_tokens: int = 2048,
        frequency_penalty: float | None = 0.0,
        reasoning_effort: str = "off",
        use_vision: bool = False,
        vision_max_side: int = 1024,
        custom_instructions: str = "",
        glossary: Mapping[str, str] | Sequence[Mapping[str, str]] | str | None = None,
        timeout: float = 90.0,
    ) -> None:
        self.provider = normalize_provider_id(provider)
        self.spec = get_provider_spec(self.provider)
        self.base_url = normalize_provider_base_url(self.provider, base_url)
        self.api_key = resolve_provider_api_key(self.provider, api_key)
        self.preferred_model = (preferred_model or "").strip() or self.spec.default_model
        self.temperature = max(0.0, min(2.0, float(temperature)))
        self.top_p = (
            max(0.0, min(1.0, float(top_p))) if top_p is not None else None
        )
        self.top_k = (
            max(0, min(500, int(top_k))) if top_k is not None else None
        )
        self.max_tokens = max(64, min(32768, int(max_tokens)))
        self.frequency_penalty = (
            max(-2.0, min(2.0, float(frequency_penalty)))
            if frequency_penalty is not None
            else None
        )
        self.reasoning_effort = normalize_reasoning_effort(reasoning_effort)
        self.use_vision = bool(use_vision)
        self.vision_max_side = max(256, min(2048, int(vision_max_side)))
        self.custom_instructions = (custom_instructions or "").strip()
        self.glossary = parse_glossary(glossary)
        self.timeout = max(5.0, min(300.0, float(timeout)))
        self._resolved_model = None

        if self.spec.protocol == "openai" and self.base_url:
            extra_headers = dict(self.spec.extra_headers) if self.spec.extra_headers else None
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key or "no-key-required",
                timeout=self.timeout,
                default_headers=extra_headers,
            )
        else:
            self.client = None

    @classmethod
    def from_settings(cls, settings: AppSettingsDict) -> "ComicTranslator":
        """Construct a `ComicTranslator` directly from a persisted `AppSettingsDict`."""
        return cls(
            provider=settings["provider"],
            base_url=settings["base_url"],
            api_key=settings["api_key"],
            preferred_model=settings["model"],
            temperature=settings["temperature"],
            top_p=settings["top_p"],
            top_k=settings["top_k"],
            max_tokens=settings["max_tokens"],
            frequency_penalty=settings["frequency_penalty"],
            reasoning_effort=settings["reasoning_effort"],
            use_vision=settings["use_vision"],
            vision_max_side=settings["vision_max_side"],
            custom_instructions=settings["custom_instructions"],
            glossary=settings["glossary"],
            timeout=settings["timeout"],
        )

    def resolve_model(self, force_refresh: bool = False) -> str:
        """Resolve the active model ID for the configured provider, caching the result."""
        if self._resolved_model is not None and not force_refresh:
            return self._resolved_model

        resolved = resolve_provider_model(
            provider=self.provider,
            preferred_model=self.preferred_model,
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=min(10.0, self.timeout),
        )
        self._resolved_model = resolved
        return resolved

    def get_status(self) -> LMStudioStatus:
        """Check connectivity and active model status for the configured provider."""
        try:
            models = list_provider_models(
                provider=self.provider,
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=5.0,
                include_embeddings=True,
            )
            active = resolve_provider_model(
                provider=self.provider,
                preferred_model=self.preferred_model,
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=5.0,
                available_models=models,
            )
            self._resolved_model = active
            return {
                "connected": True,
                "endpoint": self.base_url,
                "active_model": active,
                "available_models": models,
            }
        except ProviderConnectionError as exc:
            return {
                "connected": False,
                "endpoint": self.base_url,
                "active_model": None,
                "available_models": [],
                "error": str(exc),
            }

    def _map_openai_reasoning_effort(
        self,
        effort: ReasoningEffort,
    ) -> OpenAIReasoningEffort:
        if effort == "off":
            return "none"
        return effort

    def _call_anthropic_messages(
        self,
        model_id: str,
        system_prompt: str,
        user_payload: str,
        *,
        temperature: float,
        top_p: float | None,
        top_k: int | None,
        max_tokens: int,
        reasoning_effort: ReasoningEffort,
        page_image_data_uri: str | None = None,
    ) -> tuple[str, bool]:
        """Call Anthropic's native `POST /v1/messages` API with multimodal and thinking support."""
        if not self.api_key:
            raise ProviderConnectionError(
                "API key is required for Anthropic (Claude)."
            )

        messages_url = f"{self.base_url.rstrip('/')}/messages"
        headers: dict[str, str] = {
            "x-api-key": self.api_key,
            "anthropic-version": DEFAULT_ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

        text_content: list[dict[str, _JsonValue]] = [
            {"type": "text", "text": user_payload},
        ]
        vision_content: list[dict[str, _JsonValue]] | None = None
        if page_image_data_uri:
            match = _DATA_URI_RE.match(page_image_data_uri.strip())
            if match:
                media_type = match.group(1)
                b64_data = re.sub(r"\s+", "", match.group(2))
                vision_content = [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64_data,
                        },
                    },
                    {"type": "text", "text": user_payload},
                ]

        def _post_payload(
            content_blocks: list[dict[str, _JsonValue]],
            include_thinking: bool,
            include_optional_sampling: bool,
        ) -> requests.Response:
            budget = _ANTHROPIC_THINKING_BUDGETS[reasoning_effort] if include_thinking else 0
            eff_max_tokens = max(max_tokens, budget + 1024) if budget > 0 else max_tokens
            body: dict[str, _JsonValue] = {
                "model": model_id,
                "max_tokens": eff_max_tokens,
                "system": system_prompt,
                "messages": [{"role": "user", "content": content_blocks}],
            }
            if budget > 0:
                body["thinking"] = {"type": "enabled", "budget_tokens": budget}
            else:
                body["temperature"] = temperature
                if include_optional_sampling:
                    if top_k is not None and top_k > 0:
                        body["top_k"] = top_k
                    elif top_p is not None:
                        body["top_p"] = top_p

            try:
                return requests.post(
                    messages_url,
                    headers=headers,
                    json=body,
                    timeout=self.timeout,
                )
            except requests.exceptions.Timeout as exc:
                raise ProviderConnectionError(
                    f"Anthropic request to {messages_url} timed out after {self.timeout:.1f}s."
                ) from exc
            except requests.exceptions.ConnectionError as exc:
                raise ProviderConnectionError(
                    f"Failed to reach Anthropic API at {messages_url}."
                ) from exc
            except requests.exceptions.RequestException as exc:
                raise ProviderTranslationError(
                    f"Anthropic HTTP request error: {exc}"
                ) from exc

        def _run_with_fallbacks(
            content_blocks: list[dict[str, _JsonValue]],
        ) -> requests.Response:
            attempts: list[tuple[bool, bool]] = [
                (reasoning_effort != "off", True),
                (False, True),
                (False, False),
            ]
            # Deduplicate attempt configs while preserving order
            seen: set[tuple[bool, bool]] = set()
            last_resp: requests.Response | None = None
            for inc_think, inc_opt in attempts:
                if (inc_think, inc_opt) in seen:
                    continue
                seen.add((inc_think, inc_opt))
                resp = _post_payload(content_blocks, inc_think, inc_opt)
                last_resp = resp
                if resp.status_code != 400:
                    return resp
            assert last_resp is not None
            return last_resp

        vision_used = False
        if vision_content is not None:
            resp = _run_with_fallbacks(vision_content)
            if resp.status_code == 400:
                resp = _run_with_fallbacks(text_content)
            else:
                vision_used = resp.status_code < 400
        else:
            resp = _run_with_fallbacks(text_content)

        if resp.status_code in (401, 403):
            raise ProviderConnectionError(
                f"Anthropic authentication failed (HTTP {resp.status_code}): {resp.text[:240]}"
            )
        if resp.status_code >= 400:
            raise ProviderTranslationError(
                f"Anthropic returned HTTP {resp.status_code}: {resp.text[:300]}"
            )

        try:
            payload_obj = _json_loads(resp.text)
        except ValueError as exc:
            raise ProviderTranslationError(
                f"Anthropic returned non-JSON response: {resp.text[:200]!r}"
            ) from exc

        if not isinstance(payload_obj, Mapping):
            raise ProviderTranslationError("Anthropic returned unexpected response payload.")

        content_list = cast(Mapping[str, object], payload_obj).get("content")
        if not isinstance(content_list, Sequence) or isinstance(content_list, (str, bytes)):
            raise ProviderTranslationError("Anthropic returned empty content array.")

        text_parts: list[str] = []
        for part in content_list:
            if isinstance(part, Mapping):
                pmap = cast(Mapping[str, object], part)
                if pmap.get("type") == "text":
                    t = pmap.get("text")
                    if isinstance(t, str) and t.strip():
                        text_parts.append(t.strip())

        return "\n".join(text_parts).strip(), vision_used

    def _call_openai_chat_completion(
        self,
        model_id: str,
        system_prompt: str,
        user_payload: str,
        *,
        temperature: float,
        top_p: float | None,
        top_k: int | None,
        max_tokens: int,
        frequency_penalty: float | None,
        reasoning_effort: ReasoningEffort,
        page_image_data_uri: str | None = None,
    ) -> tuple[str, bool]:
        """Send an OpenAI-compatible chat completion with Structured JSON, sampling, reasoning, and vision."""
        if self.client is None:
            raise ProviderConnectionError(
                f"Provider '{self.provider}' is missing a valid Base URL."
            )
        if self.spec.requires_api_key and not self.api_key:
            raise ProviderConnectionError(
                f"API key is required for {self.spec.name}."
            )

        client = self.client
        text_messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ]

        vision_messages: list[ChatCompletionMessageParam] | None = None
        if page_image_data_uri:
            content_parts: list[ChatCompletionContentPartParam] = [
                {"type": "text", "text": user_payload},
                {"type": "image_url", "image_url": {"url": page_image_data_uri}},
            ]
            vision_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_parts},
            ]

        extra_body_payload: dict[str, object] | None = None
        if self.spec.supports_top_k and top_k is not None and top_k > 0:
            extra_body_payload = {"top_k": int(top_k)}

        openai_effort: OpenAIReasoningEffort | None = (
            self._map_openai_reasoning_effort(reasoning_effort)
            if self.spec.supports_reasoning_effort
            else None
        )
        effective_freq_penalty = (
            frequency_penalty
            if (self.spec.supports_frequency_penalty and frequency_penalty is not None)
            else None
        )

        def _try_messages(messages: list[ChatCompletionMessageParam]) -> object:
            # Attempt 1: Full structured schema + sampling + reasoning_effort + extra_body(top_k)
            try:
                return client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    response_format=(
                        COMIC_TRANSLATION_JSON_SCHEMA
                        if self.spec.supports_json_schema
                        else _JSON_OBJECT_FORMAT
                    ),
                    reasoning_effort=openai_effort if openai_effort is not None else omit,
                    temperature=temperature,
                    top_p=top_p if top_p is not None else omit,
                    frequency_penalty=(
                        effective_freq_penalty
                        if effective_freq_penalty is not None
                        else omit
                    ),
                    max_tokens=max_tokens,
                    extra_body=extra_body_payload,
                )
            except openai.BadRequestError:
                pass

            # Attempt 2: Keep sampling/json_object without extra_body/reasoning_effort/json_schema
            try:
                return client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    response_format=_JSON_OBJECT_FORMAT,
                    temperature=temperature,
                    top_p=top_p if top_p is not None else omit,
                    max_tokens=max_tokens,
                )
            except openai.BadRequestError:
                pass

            # Attempt 3: Plain chat completion with basic temperature + max_tokens
            try:
                return client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except openai.BadRequestError:
                # Attempt 4: Reasoning models (e.g. OpenAI o1/o3/o4) that require max_completion_tokens
                return client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    max_completion_tokens=max_tokens,
                )

        vision_used = False
        try:
            if vision_messages is not None:
                try:
                    response = _try_messages(vision_messages)
                    vision_used = True
                except openai.BadRequestError:
                    # Graceful fallback to text-only if the active model is not vision-enabled
                    response = _try_messages(text_messages)
                    vision_used = False
            else:
                response = _try_messages(text_messages)
                vision_used = False
        except openai.AuthenticationError as exc:
            raise ProviderConnectionError(
                f"Authentication failed for {self.spec.name} ({self.base_url}): {exc.message}"
            ) from exc
        except openai.APITimeoutError as exc:
            raise ProviderConnectionError(
                f"{self.spec.name} request to {self.base_url} timed out after {self.timeout:.1f}s."
            ) from exc
        except openai.APIConnectionError as exc:
            hint = (
                "Make sure LM Studio is running and listening on port 1235."
                if self.provider == "lmstudio"
                else (
                    "Make sure Ollama is running locally."
                    if self.provider == "ollama"
                    else f"Verify Base URL and network connectivity ({self.base_url})."
                )
            )
            raise ProviderConnectionError(
                f"Failed to reach {self.spec.name} at {self.base_url}. {hint}"
            ) from exc
        except openai.APIStatusError as exc:
            raise ProviderTranslationError(
                f"{self.spec.name} returned HTTP {exc.status_code}: {exc.message}"
            ) from exc

        choices: object = getattr(response, "choices", None)
        if not isinstance(choices, Sequence) or not choices:
            raise ProviderTranslationError(
                f"{self.spec.name} returned an empty choices array."
            )

        first_choice: object = choices[0]
        message_obj: object = getattr(first_choice, "message", None)
        content_obj: object = getattr(message_obj, "content", None)
        content_str = str(content_obj).strip() if isinstance(content_obj, str) else ""

        if not content_str:
            reasoning_obj: object = getattr(message_obj, "reasoning_content", None)
            if isinstance(reasoning_obj, str) and reasoning_obj.strip():
                content_str = reasoning_obj.strip()

        return content_str, vision_used

    def translate_blocks(
        self,
        blocks: Sequence[Mapping[str, object]],
        source_lang: str = "ja",
        target_lang: str = "id",
        custom_instructions: str | None = None,
        glossary: Mapping[str, str] | Sequence[Mapping[str, str]] | str | None = None,
        page_image_data_uri: str | None = None,
        *,
        page_image: NDArrayU8 | Image.Image | bytes | Path | None = None,
        use_vision: bool | None = None,
        vision_max_side: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        max_tokens: int | None = None,
        frequency_penalty: float | None = None,
        reasoning_effort: str | None = None,
    ) -> PageTranslationResult:
        """Translate a full page of comic dialogue blocks in one context-aware request."""
        src_lang = normalize_source_language(source_lang)
        tgt_lang = normalize_target_language(target_lang)
        normalized_blocks = normalize_input_blocks(blocks)
        model_id = self.resolve_model()

        eff_reasoning = (
            normalize_reasoning_effort(reasoning_effort)
            if reasoning_effort is not None
            else self.reasoning_effort
        )
        eff_temperature = (
            max(0.0, min(2.0, float(temperature)))
            if temperature is not None
            else self.temperature
        )
        eff_top_p = (
            max(0.0, min(1.0, float(top_p))) if top_p is not None else self.top_p
        )
        eff_top_k = (
            max(0, min(500, int(top_k))) if top_k is not None else self.top_k
        )
        eff_max_tokens = (
            max(64, min(32768, int(max_tokens)))
            if max_tokens is not None
            else self.max_tokens
        )
        eff_freq_penalty = (
            max(-2.0, min(2.0, float(frequency_penalty)))
            if frequency_penalty is not None
            else self.frequency_penalty
        )
        eff_use_vision = self.use_vision if use_vision is None else bool(use_vision)
        eff_vision_max_side = (
            max(256, min(2048, int(vision_max_side)))
            if vision_max_side is not None
            else self.vision_max_side
        )

        if not normalized_blocks:
            return {
                "provider": self.provider,
                "model": model_id,
                "source_language": src_lang,
                "target_language": tgt_lang,
                "vision_used": False,
                "reasoning_effort": eff_reasoning,
                "translations": {},
                "dialogues": [],
            }

        non_empty_blocks = [b for b in normalized_blocks if b["source_text"]]
        if not non_empty_blocks:
            empty_map = {b["id"]: "" for b in normalized_blocks}
            empty_items: list[TranslatedDialogueItem] = [
                {"id": b["id"], "source_text": "", "translated_text": ""}
                for b in normalized_blocks
            ]
            return {
                "provider": self.provider,
                "model": model_id,
                "source_language": src_lang,
                "target_language": tgt_lang,
                "vision_used": False,
                "reasoning_effort": eff_reasoning,
                "translations": empty_map,
                "dialogues": empty_items,
            }

        # Merge instance-level glossary with per-request glossary overrides
        merged_glossary: dict[str, str] = dict(self.glossary)
        if glossary is not None:
            merged_glossary.update(parse_glossary(glossary))

        eff_instructions = (
            custom_instructions.strip()
            if custom_instructions is not None
            else self.custom_instructions
        )

        # Prepare resized vision payload ONLY when `use_vision` is enabled
        resolved_image_uri: str | None = None
        if eff_use_vision:
            if page_image_data_uri and page_image_data_uri.strip():
                resolved_image_uri = encode_image_for_vision(
                    page_image_data_uri,
                    max_side=eff_vision_max_side,
                )
            elif page_image is not None:
                resolved_image_uri = encode_image_for_vision(
                    page_image,
                    max_side=eff_vision_max_side,
                )

        system_prompt = build_system_prompt(
            source_lang=src_lang,
            target_lang=tgt_lang,
            reasoning_effort=eff_reasoning,
            custom_instructions=eff_instructions,
            glossary=merged_glossary if merged_glossary else None,
            has_page_image=(resolved_image_uri is not None),
        )

        user_payload = json.dumps(
            {"dialogues": non_empty_blocks},
            ensure_ascii=False,
        )

        if self.spec.protocol == "anthropic":
            raw_output, vision_used = self._call_anthropic_messages(
                model_id=model_id,
                system_prompt=system_prompt,
                user_payload=user_payload,
                temperature=eff_temperature,
                top_p=eff_top_p,
                top_k=eff_top_k,
                max_tokens=eff_max_tokens,
                reasoning_effort=eff_reasoning,
                page_image_data_uri=resolved_image_uri,
            )
        else:
            raw_output, vision_used = self._call_openai_chat_completion(
                model_id=model_id,
                system_prompt=system_prompt,
                user_payload=user_payload,
                temperature=eff_temperature,
                top_p=eff_top_p,
                top_k=eff_top_k,
                max_tokens=eff_max_tokens,
                frequency_penalty=eff_freq_penalty,
                reasoning_effort=eff_reasoning,
                page_image_data_uri=resolved_image_uri,
            )

        translation_map = extract_and_clean_translations(
            raw_content=raw_output,
            input_blocks=normalized_blocks,
        )

        dialogues: list[TranslatedDialogueItem] = [
            {
                "id": b["id"],
                "source_text": b["source_text"],
                "translated_text": translation_map.get(b["id"], b["source_text"]),
            }
            for b in normalized_blocks
        ]

        return {
            "provider": self.provider,
            "model": model_id,
            "source_language": src_lang,
            "target_language": tgt_lang,
            "vision_used": vision_used,
            "reasoning_effort": eff_reasoning,
            "translations": translation_map,
            "dialogues": dialogues,
        }

    def translate_to_id_map(
        self,
        blocks: Sequence[Mapping[str, object]],
        source_lang: str = "ja",
        target_lang: str = "id",
    ) -> dict[str, str]:
        """Translate a full page of blocks and return a strict `{block_id: translated_text}` dict."""
        result = self.translate_blocks(
            blocks=blocks,
            source_lang=source_lang,
            target_lang=target_lang,
        )
        return result["translations"]

    def translate_ocr_bubbles(
        self,
        bubbles: Sequence[OcrBubbleResult],
        source_lang: str = "ja",
        target_lang: str = "id",
    ) -> list[TranslatedBubbleResult]:
        """Translate a sequence of `OcrBubbleResult` items and return enriched `TranslatedBubbleResult` objects."""
        block_inputs: list[dict[str, object]] = [
            {"id": b["id"], "source_text": b["source_text"]}
            for b in bubbles
        ]
        page_res = self.translate_blocks(
            blocks=block_inputs,
            source_lang=source_lang,
            target_lang=target_lang,
        )
        trans_map = page_res["translations"]

        enriched: list[TranslatedBubbleResult] = []
        for b in bubbles:
            enriched.append(
                {
                    "id": b["id"],
                    "order": b["order"],
                    "bbox": list(b["bbox"]),
                    "polygon": [list(pt) for pt in b["polygon"]],
                    "source_text": b["source_text"],
                    "translated_text": trans_map.get(b["id"], b["source_text"]),
                    "confidence": b["confidence"],
                    "text_direction": b["text_direction"],
                }
            )
        return enriched


__all__ = [
    "COMIC_TRANSLATION_JSON_SCHEMA",
    "ComicTranslator",
    "DEFAULT_LM_STUDIO_API_KEY",
    "DEFAULT_LM_STUDIO_BASE_URL",
    "DEFAULT_LM_STUDIO_MODEL",
    "DialogueBlockInput",
    "LMStudioConnectionError",
    "LMStudioStatus",
    "LMStudioTranslationError",
    "PROVIDER_PRESETS",
    "PageTranslationResult",
    "ProviderConnectionError",
    "ProviderId",
    "ProviderTranslationError",
    "ReasoningEffort",
    "SUPPORTED_REASONING_EFFORTS",
    "SUPPORTED_TARGET_LANGUAGES",
    "TargetLanguage",
    "TranslatedBubbleResult",
    "TranslatedDialogueItem",
    "build_system_prompt",
    "encode_image_for_vision",
    "extract_and_clean_translations",
    "list_lm_studio_models",
    "normalize_input_blocks",
    "normalize_reasoning_effort",
    "normalize_source_language",
    "normalize_target_language",
    "parse_glossary",
    "resolve_lm_studio_model",
    "strip_thinking_tags",
]


def _verify_translator() -> int:
    """Standalone CLI test verifying multi-provider LM Studio translation (Vision OFF & ON)."""
    print("=== Multi-Provider ComicTranslator Verification ===")
    translator = ComicTranslator(provider="lmstudio")

    status = translator.get_status()
    if not status["connected"]:
        print(
            f"[ERROR] {status.get('error', 'LM Studio is unreachable at http://localhost:1235/v1')}",
            file=sys.stderr,
        )
        return 1

    print(f"Provider       : {translator.provider}")
    print(f"Endpoint       : {status['endpoint']}")
    print(f"Preferred Model: {translator.preferred_model}")
    print(f"Resolved Model : {status['active_model']}")
    print(f"Available      : {len(status['available_models'])} models")

    sample_blocks: list[dict[str, object]] = [
        {"id": "bubble_001", "source_text": "田中先輩、何をしているんですか？ここは危険だぞ！"},
        {"id": "bubble_002", "source_text": "まさか…佐藤さんこそ、ここで会うとはな！"},
    ]

    # 1. Vision OFF
    res_off = translator.translate_blocks(
        blocks=sample_blocks,
        source_lang="ja",
        target_lang="id",
        use_vision=False,
        glossary={"先輩": "senpai"},
        custom_instructions="Keep dialogue punchy.",
    )
    print("\n--- Vision OFF (ja -> id) ---")
    print(json.dumps(res_off, ensure_ascii=False, indent=2))
    if res_off["vision_used"] is not False or not all(
        res_off["translations"].get(str(b["id"]), "").strip() for b in sample_blocks
    ):
        print("[ERROR] Vision OFF translation failed.", file=sys.stderr)
        return 1

    # 2. Vision ON with sample_ja.jpg if available
    sample_img = PROJECT_ROOT / "samples" / "sample_ja.jpg"
    if sample_img.is_file():
        res_on = translator.translate_blocks(
            blocks=sample_blocks,
            source_lang="ja",
            target_lang="en",
            page_image=sample_img,
            use_vision=True,
        )
        print("\n--- Vision ON (ja -> en) ---")
        print(json.dumps(res_on, ensure_ascii=False, indent=2))
        if not res_on["vision_used"]:
            print("[WARN] Vision ON fell back to text-only on the active model.")

    # 3. Verify <think> tag stripping & dirty JSON cleaner
    dirty_llm_sample = (
        "<think>Checking bubble_001: \"id\": \"bubble_001\", \"translated_text\": \"WRONG\"</think>\n"
        + "```json\n"
        + '{"dialogues": [{"id": "bubble_001", "translated_text": "Cleaned line 1",}, '
        + '{"id": "bubble_002", "translated_text": "Cleaned line 2"}]}\n'
        + "```"
    )
    cleaned_map = extract_and_clean_translations(
        raw_content=dirty_llm_sample,
        input_blocks=[
            {"id": "bubble_001", "source_text": "テスト1"},
            {"id": "bubble_002", "source_text": "テスト2"},
        ],
    )
    if (
        cleaned_map.get("bubble_001") != "Cleaned line 1"
        or cleaned_map.get("bubble_002") != "Cleaned line 2"
    ):
        print(f"[ERROR] JSON extractor / <think> stripper failed: {cleaned_map!r}", file=sys.stderr)
        return 1
    print("\n[OK] <think> tag stripping, glossary, and Vision OFF/ON verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_verify_translator())
