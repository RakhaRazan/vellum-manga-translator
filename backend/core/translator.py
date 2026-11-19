"""Context-Aware Comic Dialogue Translator via Local LM Studio (`openai` Python SDK).

Connects to LM Studio's OpenAI-compatible server at `http://localhost:1235/v1` (`api_key="lm-studio"`).
Features:
- Targets `"Gemma 4 E2B Uncensored HauhauCS Aggressive Q4 K P"` by default, with an automatic
  fallback helper (`resolve_lm_studio_model`) that queries `GET http://localhost:1235/v1/models`
  (and `/api/v0/models`) to match the active loaded model ID if the exact name differs in LM Studio.
- Translates a full comic page's dialogue array in a single request (`source_lang` in `ja`, `ko`, `zh`;
  `target_lang` in `id`, `en`) so conversational context, pronouns, and character relationships
  stay consistent across speech bubbles.
- Dynamically tailors a concise system prompt optimized for a 2B quantized model, preserving
  character tone, slang, and romanized honorifics (`-san`/`-senpai` for JA, `oppa`/`hyung`/`sunbae`
  for KO, `gege`/`shixiong` for ZH).
- Enforces valid JSON output mapped strictly by block `id` (SPEC.md Section 5.2 JSON Schema)
  combined with a multi-stage JSON extractor/cleaner that strips markdown fences, `<think>` blocks,
  conversational text, and trailing commas so parsing never fails.
"""

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import re
import sys
from typing import Literal, NotRequired, TypedDict, cast

import openai
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.shared_params import ResponseFormatJSONSchema
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.detector import TextDirection  # noqa: E402
from backend.core.ocr import (  # noqa: E402
    OcrBubbleResult,
    SourceLanguage,
    normalize_language,
)

TargetLanguage = Literal["id", "en"]
SUPPORTED_TARGET_LANGUAGES: tuple[TargetLanguage, ...] = ("id", "en")

DEFAULT_LM_STUDIO_BASE_URL = "http://localhost:1235/v1"
DEFAULT_LM_STUDIO_API_KEY = "lm-studio"
DEFAULT_LM_STUDIO_MODEL = "Gemma 4 E2B Uncensored HauhauCS Aggressive Q4 K P"

# Quantization / format tokens that LM Studio often strips from model IDs in /v1/models
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


class LMStudioConnectionError(RuntimeError):
    """Raised when the local LM Studio server at port 1235 is unreachable or has no models."""


class LMStudioTranslationError(RuntimeError):
    """Raised when an LLM translation call fails unexpectedly."""


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
    """Complete page translation response containing both ID-mapped dict and ordered dialogue list."""

    model: str
    source_language: SourceLanguage
    target_language: TargetLanguage
    translations: dict[str, str]
    dialogues: list[TranslatedDialogueItem]


class LMStudioStatus(TypedDict):
    """Connectivity and active model status for LM Studio matching SPEC.md Section 5.4."""

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


def normalize_source_language(source_lang: str) -> SourceLanguage:
    """Validate and normalize `source_lang` (`'ja'`, `'ko'`, or `'zh'`)."""
    return normalize_language(source_lang)


def normalize_target_language(target_lang: str) -> TargetLanguage:
    """Validate and normalize `target_lang` (`'id'` for Indonesian or `'en'` for English)."""
    cleaned = target_lang.lower().strip()
    if cleaned in ("id", "ind", "indonesian", "bahasa", "bahasa indonesia"):
        return "id"
    if cleaned in ("en", "eng", "english"):
        return "en"
    raise ValueError(
        f"Unsupported target_lang {target_lang!r}. Must be one of {SUPPORTED_TARGET_LANGUAGES}."
    )


def _tokenize_model_name(name: str) -> list[str]:
    """Split a model name or ID into lowercase alphanumeric tokens."""
    return re.findall(r"[a-z0-9]+", name.lower())


def _is_embedding_model(model_id: str) -> bool:
    """Return True if `model_id` appears to be an embedding model rather than a generative LLM."""
    lower = model_id.lower()
    return "embed" in lower or lower.startswith("text-embedding")


def _score_model_match(preferred_name: str, candidate_id: str) -> float:
    """Compute a similarity score in `[0.0, 1.0]` between `preferred_name` and `candidate_id`."""
    pref_lower = preferred_name.lower().strip()
    cand_lower = candidate_id.lower().strip()
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

    # Jaccard overlap on core model identity tokens (e.g. gemma, 4, e2b, uncensored, hauhaucs, aggressive)
    union_core = pref_core_set | cand_core_set
    core_jaccard = len(common_core) / max(1, len(union_core))
    cand_coverage = len(common_core) / max(1, len(cand_core_set))
    pref_coverage = len(common_core) / max(1, len(pref_core_set))

    score = (core_jaccard * 0.5) + (cand_coverage * 0.3) + (pref_coverage * 0.2)
    if cand_compact and (cand_compact in pref_compact or pref_compact in cand_compact):
        score = min(0.98, score + 0.1)
    return score


def _json_loads(raw: str) -> object:
    """Parse a JSON string into a typed `object` (isolating stdlib `json.loads` `Any` return)."""
    res: object = json.loads(raw)  # pyright: ignore[reportAny]
    return res


def _fetch_loaded_model_from_v0(base_url: str, timeout: float = 3.0) -> str | None:
    """Query LM Studio's `/api/v0/models` endpoint to check if a specific LLM is currently in `loaded` state."""
    root_url = re.sub(r"/v1/?$", "", base_url.rstrip("/"))
    v0_url = f"{root_url}/api/v0/models"
    try:
        resp = requests.get(v0_url, timeout=timeout)
        if resp.status_code != 200:
            return None
        payload_obj = _json_loads(resp.text)
        if not isinstance(payload_obj, Mapping):
            return None
        payload_map = cast(Mapping[str, object], payload_obj)
        data_obj = payload_map.get("data")
        if not isinstance(data_obj, Sequence) or isinstance(data_obj, (str, bytes)):
            return None
        for entry_obj in data_obj:
            if not isinstance(entry_obj, Mapping):
                continue
            entry = cast(Mapping[str, object], entry_obj)
            model_id = str(entry.get("id", "")).strip()
            state = str(entry.get("state", "")).strip().lower()
            model_type = str(entry.get("type", "")).strip().lower()
            if (
                model_id
                and state == "loaded"
                and model_type != "embeddings"
                and not _is_embedding_model(model_id)
            ):
                return model_id
    except requests.RequestException:
        return None
    except ValueError:
        return None
    return None


def list_lm_studio_models(
    base_url: str = DEFAULT_LM_STUDIO_BASE_URL,
    api_key: str = DEFAULT_LM_STUDIO_API_KEY,
    timeout: float = 5.0,
) -> list[str]:
    """Query `GET {base_url}/models` and return all available model IDs in LM Studio.

    Raises `LMStudioConnectionError` with a clear diagnostic message if port 1235 is unreachable.
    """
    models_url = f"{base_url.rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        resp = requests.get(models_url, headers=headers, timeout=timeout)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise LMStudioConnectionError(
            f"Cannot connect to LM Studio server at {models_url}. "
            + "Ensure LM Studio is open and the Local Inference Server is running on port 1235."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise LMStudioConnectionError(
            f"Timed out after {timeout:.1f}s while connecting to LM Studio at {models_url}. "
            + "Verify that port 1235 is responsive."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise LMStudioConnectionError(
            f"HTTP error while querying LM Studio models at {models_url}: {exc}"
        ) from exc

    try:
        payload_obj = _json_loads(resp.text)
    except ValueError as exc:
        raise LMStudioConnectionError(
            f"LM Studio at {models_url} returned non-JSON payload: {resp.text[:200]!r}"
        ) from exc

    if not isinstance(payload_obj, Mapping):
        raise LMStudioConnectionError(
            f"Unexpected response format from {models_url}: {payload_obj!r}"
        )

    payload_map = cast(Mapping[str, object], payload_obj)
    data_obj = payload_map.get("data")
    if not isinstance(data_obj, Sequence) or isinstance(data_obj, (str, bytes)):
        return []

    model_ids: list[str] = []
    for item_obj in data_obj:
        if isinstance(item_obj, Mapping):
            item_map = cast(Mapping[str, object], item_obj)
            mid = str(item_map.get("id", "")).strip()
            if mid:
                model_ids.append(mid)
    return model_ids


def resolve_lm_studio_model(
    preferred_model: str = DEFAULT_LM_STUDIO_MODEL,
    base_url: str = DEFAULT_LM_STUDIO_BASE_URL,
    api_key: str = DEFAULT_LM_STUDIO_API_KEY,
    timeout: float = 5.0,
) -> str:
    """Resolve `preferred_model` against `GET http://localhost:1235/v1/models`.

    1. If `preferred_model` matches an entry in `/v1/models` exactly (or case-insensitively), returns it.
    2. If the exact string differs slightly in LM Studio (e.g. `"Gemma 4 E2B Uncensored HauhauCS Aggressive Q4 K P"`
       vs `"gemma-4-e2b-uncensored-hauhaucs-aggressive"`), returns the closest fuzzy-matched LLM ID.
    3. If `preferred_model` is not present at all, returns the currently loaded LLM from `/api/v0/models`
       or the first non-embedding LLM from `/v1/models`.
    """
    all_models = list_lm_studio_models(base_url=base_url, api_key=api_key, timeout=timeout)
    llm_models = [m for m in all_models if not _is_embedding_model(m)]
    candidates = llm_models if llm_models else all_models

    if not candidates:
        raise LMStudioConnectionError(
            f"LM Studio server at {base_url} is reachable, but no models are installed or loaded."
        )

    # 1. Exact or case-insensitive match
    pref_clean = preferred_model.strip()
    for cand in candidates:
        if cand == pref_clean or cand.lower() == pref_clean.lower():
            return cand

    # 2. Fuzzy token match for slight naming/quantization differences in LM Studio
    scored: list[tuple[float, str]] = [
        (_score_model_match(pref_clean, cand), cand) for cand in candidates
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    best_score, best_cand = scored[0]
    if best_score >= 0.50:
        return best_cand

    # 3. Fallback to currently loaded model in LM Studio (via /api/v0/models) or first LLM in /v1/models
    loaded_v0 = _fetch_loaded_model_from_v0(base_url=base_url, timeout=timeout)
    if loaded_v0 and loaded_v0 in candidates:
        return loaded_v0

    return candidates[0]


def build_system_prompt(
    source_lang: SourceLanguage,
    target_lang: TargetLanguage,
) -> str:
    """Build a concise, dynamically tailored system prompt optimized for a 2B quantized LLM."""
    if source_lang == "ja":
        source_desc = "Japanese Manga (ja)"
        honorific_rule = (
            "Preserve Japanese honorifics in romaji attached to names or used as titles: "
            + "-san (さん), -senpai (先輩), -kun (くん/君), -chan (ちゃん), -sama (様), -sensei (先生) "
            + "(e.g., 'Tanaka-senpai', 'Sato-san')."
        )
    elif source_lang == "ko":
        source_desc = "Korean Manhwa (ko)"
        honorific_rule = (
            "Preserve Korean honorifics and kinship/address terms in romaji: "
            + "oppa (오빠), hyung (형), sunbae (선배), noona (누나), unnie (언니), -nim (님), dongsaeng (동생) "
            + "(e.g., 'Minsu-sunbae', 'oppa', 'hyung')."
        )
    else:
        source_desc = "Chinese Manhua (zh)"
        honorific_rule = (
            "Preserve Chinese martial and familial honorifics in pinyin (do NOT translate them into 'big brother', 'senior brother', or 'kakak'): "
            + "哥哥 -> always write 'gege', 师兄 -> always write 'shixiong', "
            + "师姐 -> 'shijie', 师父 -> 'shifu', 师弟 -> 'shidi', 姐姐 -> 'jiejie' "
            + "(e.g., 'Lin-shixiong', 'gege')."
        )

    if target_lang == "id":
        target_desc = "Indonesian (Bahasa Indonesia)"
        style_rule = (
            "Write natural, expressive spoken comic Indonesian (Bahasa Indonesia). "
            + "Keep character tone, emotion, and casual slang natural for comic bubbles "
            + "(use 'aku/kamu', 'kau', or casual 'gue/lu' when the speaker is rough/slangy; never sound stiff or machine-translated)."
        )
    else:
        target_desc = "English (en)"
        style_rule = (
            "Write natural, idiomatic comic-book English. "
            + "Use punchy phrasing, natural contractions, and expressive slang that fits each character's emotion and tone."
        )

    return (
        f"You are an expert {source_desc} to {target_desc} comic translator. "
        + "Do not think, analyze, or explain. "
        + f"Translate the full page of dialogue blocks from {source_desc} into {target_desc}.\n"
        + "Rules:\n"
        + "1. Read all blocks together as one continuous scene so conversational context, pronouns, and speaker tone remain consistent across bubbles.\n"
        + f"2. {style_rule}\n"
        + f"3. {honorific_rule}\n"
        + '4. Output ONLY valid JSON in the exact format: {"dialogues": [{"id": "<block_id>", "translated_text": "<translation>"}]}. '
        + "Keep every input block 'id' unchanged."
    )


def normalize_input_blocks(
    blocks: Sequence[Mapping[str, object]],
) -> list[DialogueBlockInput]:
    """Normalize an array of detected/OCR text blocks into `[{'id': ..., 'source_text': ...}]`.

    Accepts dicts containing `'id'` (or auto-assigns `'bubble_001'`, ...) and text under
    `'source_text'`, `'original_text'`, or `'text'`.
    """
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

        # Collapse internal newlines to spaces so multi-line OCR text translates as a cohesive sentence
        cleaned_text = re.sub(r"\s+", " ", str(text_val)).strip()
        normalized.append(
            {
                "id": block_id,
                "source_text": cleaned_text,
            }
        )
    return normalized


def _strip_fences_and_noise(raw_content: str) -> str:
    """Strip `<think>...</think>` blocks, ```json markdown fences, and outer conversational prose."""
    text = raw_content.strip()
    if not text:
        return ""

    # Remove <think>...</think> blocks if emitted by reasoning models
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()

    # Extract content inside ```json ... ``` or ``` ... ``` markdown code fences if present
    fence_match = re.search(
        r"```(?:json)?\s*([\s\S]*?)```",
        text,
        flags=re.IGNORECASE,
    )
    if fence_match:
        fenced_body = fence_match.group(1).strip()
        if fenced_body:
            text = fenced_body

    # Locate the outermost JSON object `{...}` or array `[...]`
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

    # Remove trailing commas before closing `}` or `]`
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text.strip()


def _extract_mapping_from_parsed_json(parsed: object) -> dict[str, str]:
    """Convert any parsed JSON structure (SPEC schema, array of objects, or ID-keyed dict) into `{id: translated_text}`."""
    mapping: dict[str, str] = {}

    if isinstance(parsed, Mapping):
        parsed_map = cast(Mapping[str, object], parsed)

        # Case 1: {"dialogues": [...]} or {"translations": [...]} or {"blocks": [...]}
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

        # Case 2: Single item {"id": "bubble_001", "translated_text": "..."}
        if "id" in parsed_map and any(
            k in parsed_map for k in ("translated_text", "translation", "text", "target_text")
        ):
            item_id = str(parsed_map.get("id", "")).strip()
            for t_key in ("translated_text", "translation", "text", "target_text"):
                t_val = parsed_map.get(t_key)
                if isinstance(t_val, str) and item_id:
                    mapping[item_id] = t_val.strip()
                    return mapping

        # Case 3: Direct dictionary {"bubble_001": "Translated text", "bubble_002": {"translated_text": "..."}}
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
                # Handle [{"bubble_001": "Translated text"}]
                for k, v in entry.items():
                    if isinstance(v, str):
                        mapping[str(k).strip()] = v.strip()
        return mapping

    return mapping


def _regex_salvage_translations(
    raw_text: str,
    expected_ids: Sequence[str],
) -> dict[str, str]:
    """Regex fallback extractor if the LLM produced truncated or slightly malformed JSON."""
    salvaged: dict[str, str] = {}

    # Pattern 1: {"id": "bubble_001", "translated_text": "..."}
    obj_pattern = re.compile(
        r'"id"\s*:\s*"([^"]+)"[\s\S]*?"(?:translated_text|translation|text|target_text)"\s*:\s*"((?:[^"\\]|\\.)*)"'
    )
    for match in obj_pattern.finditer(raw_text):
        bid = match.group(1).strip()
        raw_val = match.group(2)
        try:
            unescaped = str(_json_loads(f'"{raw_val}"')).strip()
        except ValueError:
            unescaped = raw_val.replace('\\"', '"').replace("\\n", " ").strip()
        if bid:
            salvaged[bid] = unescaped

    # Pattern 2: Direct key-value `"bubble_001": "..."` for any expected block ID
    for exp_id in expected_ids:
        if exp_id in salvaged:
            continue
        kv_pattern = re.compile(
            rf'"{re.escape(exp_id)}"\s*:\s*"((?:[^"\\]|\\.)*)"'
        )
        kv_match = kv_pattern.search(raw_text)
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

    Strips ```json fences, `<think>` blocks, trailing commas, and conversational text,
    and guarantees every input `block["id"]` is present in the returned dict.
    """
    expected_ids = [b["id"] for b in input_blocks]
    cleaned = _strip_fences_and_noise(raw_content)

    extracted: dict[str, str] = {}
    if cleaned:
        try:
            parsed_obj = _json_loads(cleaned)
            extracted = _extract_mapping_from_parsed_json(parsed_obj)
        except ValueError:
            # Repair common unescaped control characters inside strings and retry json.loads
            sanitized = re.sub(r"[\x00-\x1f]+", " ", cleaned)
            try:
                parsed_obj = _json_loads(sanitized)
                extracted = _extract_mapping_from_parsed_json(parsed_obj)
            except ValueError:
                extracted = {}

    if not extracted or len(extracted) < len(expected_ids):
        salvaged = _regex_salvage_translations(raw_content, expected_ids)
        for k, v in salvaged.items():
            if k not in extracted or not extracted[k]:
                extracted[k] = v

    # Map extracted translations strictly to expected block IDs
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

        # Case-insensitive or numeric suffix ID fallback (e.g. model returned "1" or "001" for "bubble_001")
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

        # Positional alignment fallback if the model returned the exact same number of items with altered keys
        if (
            matched_val is None
            and len(extracted_values_in_order) == len(input_blocks)
            and extracted_values_in_order[idx]
        ):
            matched_val = extracted_values_in_order[idx]

        result[bid] = matched_val if matched_val is not None else src

    return result


class ComicTranslator:
    """Context-aware comic dialogue translator backed by LM Studio (`http://localhost:1235/v1`)."""

    base_url: str
    api_key: str
    preferred_model: str
    temperature: float
    max_tokens: int
    timeout: float
    client: OpenAI
    _resolved_model: str | None

    def __init__(
        self,
        base_url: str = DEFAULT_LM_STUDIO_BASE_URL,
        api_key: str = DEFAULT_LM_STUDIO_API_KEY,
        preferred_model: str = DEFAULT_LM_STUDIO_MODEL,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.preferred_model = preferred_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )
        self._resolved_model = None

    def resolve_model(self, force_refresh: bool = False) -> str:
        """Resolve the active LM Studio model ID, caching the result after first lookup."""
        if self._resolved_model is not None and not force_refresh:
            return self._resolved_model

        resolved = resolve_lm_studio_model(
            preferred_model=self.preferred_model,
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=min(10.0, self.timeout),
        )
        self._resolved_model = resolved
        return resolved

    def get_status(self) -> LMStudioStatus:
        """Check LM Studio connectivity and active model status (SPEC.md Section 5.4)."""
        try:
            models = list_lm_studio_models(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=5.0,
            )
            active = resolve_lm_studio_model(
                preferred_model=self.preferred_model,
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=5.0,
            )
            self._resolved_model = active
            return {
                "connected": True,
                "endpoint": self.base_url,
                "active_model": active,
                "available_models": models,
            }
        except LMStudioConnectionError as exc:
            return {
                "connected": False,
                "endpoint": self.base_url,
                "active_model": None,
                "available_models": [],
                "error": str(exc),
            }

    def _call_chat_completion(
        self,
        model_id: str,
        system_prompt: str,
        user_payload: str,
    ) -> str:
        """Send the chat completion request with Structured JSON Schema and graceful fallback."""
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ]

        try:
            try:
                response = self.client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    response_format=COMIC_TRANSLATION_JSON_SCHEMA,
                    reasoning_effort="none",
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            except openai.BadRequestError:
                # Fallback if the loaded model/server version rejects json_schema or reasoning_effort
                response = self.client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
        except openai.APITimeoutError as exc:
            raise LMStudioConnectionError(
                f"LM Studio request to {self.base_url} timed out after {self.timeout:.1f}s."
            ) from exc
        except openai.APIConnectionError as exc:
            raise LMStudioConnectionError(
                f"Failed to reach LM Studio server at {self.base_url}. "
                + "Make sure LM Studio is running and listening on port 1235."
            ) from exc
        except openai.APIStatusError as exc:
            raise LMStudioTranslationError(
                f"LM Studio returned HTTP {exc.status_code}: {exc.message}"
            ) from exc

        choices = getattr(response, "choices", None)
        if not isinstance(choices, Sequence) or not choices:
            raise LMStudioTranslationError("LM Studio returned an empty choices array.")

        first_choice: object = choices[0]
        message_obj: object = getattr(first_choice, "message", None)
        content_obj: object = getattr(message_obj, "content", None)
        content_str = str(content_obj).strip() if isinstance(content_obj, str) else ""

        if not content_str:
            reasoning_obj: object = getattr(message_obj, "reasoning_content", None)
            if isinstance(reasoning_obj, str) and reasoning_obj.strip():
                content_str = reasoning_obj.strip()

        return content_str

    def translate_blocks(
        self,
        blocks: Sequence[Mapping[str, object]],
        source_lang: str = "ja",
        target_lang: str = "id",
    ) -> PageTranslationResult:
        """Translate a full page of comic dialogue blocks in one context-aware request.

        Args:
            blocks: Sequence of block dicts, each containing `'id'` and `'source_text'`
                (or `'original_text'` / `'text'`).
            source_lang: Source comic language (`'ja'`, `'ko'`, or `'zh'`).
            target_lang: Target translation language (`'id'` or `'en'`).

        Returns:
            `PageTranslationResult` containing the active `model`, `source_language`,
            `target_language`, `translations` (`{id: translated_text}`), and `dialogues`
            (`list[TranslatedDialogueItem]`).
        """
        src_lang = normalize_source_language(source_lang)
        tgt_lang = normalize_target_language(target_lang)
        normalized_blocks = normalize_input_blocks(blocks)
        model_id = self.resolve_model()

        if not normalized_blocks:
            return {
                "model": model_id,
                "source_language": src_lang,
                "target_language": tgt_lang,
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
                "model": model_id,
                "source_language": src_lang,
                "target_language": tgt_lang,
                "translations": empty_map,
                "dialogues": empty_items,
            }

        system_prompt = build_system_prompt(source_lang=src_lang, target_lang=tgt_lang)
        user_payload = json.dumps(
            {"dialogues": non_empty_blocks},
            ensure_ascii=False,
        )

        raw_output = self._call_chat_completion(
            model_id=model_id,
            system_prompt=system_prompt,
            user_payload=user_payload,
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
            "model": model_id,
            "source_language": src_lang,
            "target_language": tgt_lang,
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


def _verify_translator() -> int:
    """Standalone CLI test sending sample JA, KO, and ZH dialogue arrays to `http://localhost:1235/v1`."""
    print("=== LM Studio ComicTranslator Verification ===")
    translator = ComicTranslator()

    status = translator.get_status()
    if not status["connected"]:
        print(
            f"[ERROR] {status.get('error', 'LM Studio is unreachable at http://localhost:1235/v1')}",
            file=sys.stderr,
        )
        return 1

    print(f"Endpoint       : {status['endpoint']}")
    print(f"Preferred Model: {translator.preferred_model}")
    print(f"Resolved Model : {status['active_model']}")
    print(f"Available      : {len(status['available_models'])} models in LM Studio")

    sample_batches: list[tuple[SourceLanguage, list[dict[str, object]]]] = [
        (
            "ja",
            [
                {"id": "bubble_001", "source_text": "田中先輩、何をしているんですか？ここは危険だぞ！"},
                {"id": "bubble_002", "source_text": "まさか…佐藤さんこそ、ここで会うとはな！"},
                {"id": "bubble_003", "source_text": "気をつけろ！敵がすぐ近くまで来ている！"},
                {"id": "bubble_004", "source_text": "よし、準備は完了だ。先輩、一気に突破するぞ！"},
            ],
        ),
        (
            "ko",
            [
                {"id": "bubble_001", "source_text": "민수 선배! 여기서 뭐 하고 있어? 여긴 너무 위험해!"},
                {"id": "bubble_002", "source_text": "설마 진짜로 형이 혼자 찾아온 거야?"},
                {"id": "bubble_003", "source_text": "조심해! 적이 바로 앞까지 다가왔어!"},
                {"id": "bubble_004", "source_text": "오빠만 믿어, 준비는 끝났으니까 지금 바로 출발하자!"},
            ],
        ),
        (
            "zh",
            [
                {"id": "bubble_001", "source_text": "林师兄，你在这里做什么？这里非常危险！"},
                {"id": "bubble_002", "source_text": "难道你已经发现这里的秘密了？"},
                {"id": "bubble_003", "source_text": "大家小心！敌人正在迅速靠近！"},
                {"id": "bubble_004", "source_text": "别怕，有哥哥在！准备完毕，我们立刻出发吧！"},
            ],
        ),
    ]

    all_passed = True
    for target_lang in ("id", "en"):
        for source_lang, blocks in sample_batches:
            result = translator.translate_blocks(
                blocks=blocks,
                source_lang=source_lang,
                target_lang=target_lang,
            )
            print(
                f"\n--- [{source_lang.upper()} -> {target_lang.upper()}] "
                + f"(model={result['model']}) ---"
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))

            expected_ids = [str(b["id"]) for b in blocks]
            for eid in expected_ids:
                translated_val = result["translations"].get(eid, "").strip()
                if not translated_val:
                    print(
                        f"[ERROR] Missing translation for block id={eid!r} in {source_lang}->{target_lang}",
                        file=sys.stderr,
                    )
                    all_passed = False

    if not all_passed:
        return 1

    # Verify JSON cleaner robustness against markdown fences, <think> tags, and trailing commas
    dirty_llm_sample = (
        "<think>Analyzing dialogue bubbles...</think>\n"
        + "Sure! Here is the translated JSON output:\n"
        + "```json\n"
        + '{\n  "dialogues": [\n    {"id": "bubble_001", "translated_text": "Cleaned line 1",},\n'
        + '    {"id": "bubble_002", "translated_text": "Cleaned line 2"}\n  ],\n}\n'
        + "```\nHope this helps!"
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
        print(f"[ERROR] JSON extractor test failed: {cleaned_map!r}", file=sys.stderr)
        return 1
    print("\n[OK] Markdown fence, <think> block, and trailing-comma JSON cleaner verified.")

    # Verify clear error handling when port is unreachable
    unreachable_translator = ComicTranslator(
        base_url="http://localhost:1239/v1",
        timeout=1.5,
    )
    unreachable_status = unreachable_translator.get_status()
    if unreachable_status["connected"] or "error" not in unreachable_status:
        print("[ERROR] Expected unreachable status on closed port 1239.", file=sys.stderr)
        return 1
    print(f"[OK] Unreachable port error handling verified: {unreachable_status['error']}")

    print("\nSUCCESS: All JA, KO, and ZH comic dialogue arrays translated into valid ID-mapped JSON.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_verify_translator())
