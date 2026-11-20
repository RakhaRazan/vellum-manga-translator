"""End-to-end HTTP verification script for the FastAPI server (`backend/main.py` on port 8000).

Tests:
1. `GET /api/models/status`
2. `GET /api/lmstudio/status`
3. `GET /api/models/download` (SSE stream)
4. `POST /api/process-page` with `samples/sample_ja.jpg`, `samples/sample_ko.jpg`, and `samples/sample_zh.jpg`
   (`target_lang="id"`) verifying lightweight `/api/cache/images/{id}.png` URLs served via `FileResponse`
   (plus backward-compatible `include_base64=true` mode)
5. `POST /api/retranslate` re-translating the extracted JA blocks from `"ja"` to `"en"`

Usage:
    backend\\venv\\Scripts\\python.exe backend/test_api.py
"""

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import cast

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "http://127.0.0.1:8000"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _json_loads(raw: str) -> object:
    """Parse a JSON string into a typed `object`."""
    res: object = json.loads(raw)  # pyright: ignore[reportAny]
    return res


def _verify_cached_image_url(image_url: str) -> int:
    """Fetch `/api/cache/images/{id}.png` via HTTP and verify valid PNG `FileResponse` + CORS."""
    if not image_url.startswith("/api/cache/images/") or not image_url.endswith(".png"):
        raise AssertionError(
            f"Expected lightweight cache URL '/api/cache/images/{{id}}.png', got: {image_url[:80]!r}"
        )

    resp = requests.get(
        f"{BASE_URL}{image_url}",
        headers={"Origin": "http://127.0.0.1:5173"},
        timeout=15,
    )
    resp.raise_for_status()

    content_type = resp.headers.get("content-type", "")
    if "image/png" not in content_type.lower():
        raise AssertionError(
            f"Expected Content-Type 'image/png' from {image_url}, got {content_type!r}"
        )

    png_bytes = resp.content
    if not png_bytes.startswith(PNG_MAGIC) or len(png_bytes) < 1024:
        raise AssertionError(
            f"Cached image response from {image_url} is not a valid PNG ({len(png_bytes)} bytes)."
        )

    return len(png_bytes)


def run_api_verification() -> int:
    """Execute live HTTP tests against `http://127.0.0.1:8000`."""
    print(f"=== FastAPI Live Endpoint Verification ({BASE_URL}) ===")

    # 1. GET /api/models/status
    print("\n[1] Testing GET /api/models/status ...")
    resp_models = requests.get(f"{BASE_URL}/api/models/status", timeout=10)
    resp_models.raise_for_status()
    models_data = cast(Mapping[str, object], _json_loads(resp_models.text))
    print(json.dumps(models_data, ensure_ascii=False, indent=2))
    if not models_data.get("all_ready"):
        print("[ERROR] /api/models/status reported all_ready=False", file=sys.stderr)
        return 1

    # 2. GET /api/lmstudio/status
    print("\n[2] Testing GET /api/lmstudio/status ...")
    resp_lms = requests.get(f"{BASE_URL}/api/lmstudio/status", timeout=10)
    resp_lms.raise_for_status()
    lms_data = cast(Mapping[str, object], _json_loads(resp_lms.text))
    print(json.dumps(lms_data, ensure_ascii=False, indent=2))
    if not lms_data.get("connected"):
        print("[ERROR] /api/lmstudio/status reported connected=False", file=sys.stderr)
        return 1

    # 3. GET /api/models/download (SSE)
    print("\n[3] Testing GET /api/models/download (SSE) ...")
    with requests.get(f"{BASE_URL}/api/models/download", stream=True, timeout=15) as sse_resp:
        sse_resp.raise_for_status()
        sse_events: list[str] = []
        for raw_line in sse_resp.iter_lines(decode_unicode=True):
            if raw_line:
                line_str = str(raw_line)
                print(f"  SSE -> {line_str}")
                sse_events.append(line_str)
        if not sse_events:
            print("[ERROR] No SSE events received from /api/models/download", file=sys.stderr)
            return 1

    # 4. POST /api/process-page with JA / KO / ZH samples (verifying lightweight cached URL + FileResponse)
    sample_cases: list[tuple[str, Path]] = [
        ("ja", PROJECT_ROOT / "samples" / "sample_ja.jpg"),
        ("ko", PROJECT_ROOT / "samples" / "sample_ko.jpg"),
        ("zh", PROJECT_ROOT / "samples" / "sample_zh.jpg"),
    ]
    ja_blocks: list[Mapping[str, object]] = []

    for lang, sample_path in sample_cases:
        if not sample_path.is_file():
            print(f"[ERROR] Sample file not found: {sample_path}", file=sys.stderr)
            return 1

        print(f"\n[4.{lang.upper()}] Testing POST /api/process-page with {sample_path.name} ({lang} -> id) ...")
        with open(sample_path, "rb") as img_file:
            resp_proc = requests.post(
                f"{BASE_URL}/api/process-page",
                files={"image": (sample_path.name, img_file, "image/jpeg")},
                data={"source_lang": lang, "target_lang": "id"},
                timeout=120,
            )
        resp_proc.raise_for_status()
        proc_data = cast(Mapping[str, object], _json_loads(resp_proc.text))

        cleaned_image = str(proc_data.get("cleaned_image", ""))
        cleaned_image_url = str(proc_data.get("cleaned_image_url", ""))
        blocks_obj = proc_data.get("blocks")
        blocks_list: Sequence[Mapping[str, object]] = (
            cast(Sequence[Mapping[str, object]], blocks_obj)
            if isinstance(blocks_obj, Sequence) and not isinstance(blocks_obj, (str, bytes))
            else ()
        )

        try:
            png_byte_len = _verify_cached_image_url(cleaned_image)
        except AssertionError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1

        if cleaned_image_url != cleaned_image:
            print(
                f"[ERROR] cleaned_image_url ({cleaned_image_url!r}) != cleaned_image ({cleaned_image!r})",
                file=sys.stderr,
            )
            return 1

        summary_proc: dict[str, object] = {
            "page_id": proc_data.get("page_id"),
            "dimensions": proc_data.get("dimensions"),
            "source_lang": proc_data.get("source_lang"),
            "target_lang": proc_data.get("target_lang"),
            "model": proc_data.get("model"),
            "execution_provider": proc_data.get("execution_provider"),
            "tiles_inpainted": proc_data.get("tiles_inpainted"),
            "cleaned_image": cleaned_image,
            "cached_png_bytes": png_byte_len,
            "json_payload_bytes": len(resp_proc.content),
            "blocks": [
                {
                    "id": b.get("id"),
                    "bbox": b.get("bbox"),
                    "original_text": b.get("original_text"),
                    "translated_text": b.get("translated_text"),
                    "font_size": b.get("font_size"),
                    "color": b.get("color"),
                    "align": b.get("align"),
                }
                for b in blocks_list
            ],
        }
        print(json.dumps(summary_proc, ensure_ascii=False, indent=2))

        if (
            len(blocks_list) < 4
            or any(not str(b.get("original_text", "")).strip() for b in blocks_list)
            or any(not str(b.get("translated_text", "")).strip() for b in blocks_list)
        ):
            print(f"[ERROR] /api/process-page ({lang}) failed block validation.", file=sys.stderr)
            return 1

        if lang == "ja":
            ja_blocks = list(blocks_list)

    # 4b. Verify backward compatibility (`include_base64=true`) on sample_ko.jpg
    sample_ko_path = PROJECT_ROOT / "samples" / "sample_ko.jpg"
    print(f"\n[4.COMPAT] Testing POST /api/process-page backward-compat (include_base64=true) ...")
    with open(sample_ko_path, "rb") as img_file:
        resp_compat = requests.post(
            f"{BASE_URL}/api/process-page",
            files={"image": (sample_ko_path.name, img_file, "image/jpeg")},
            data={"source_lang": "ko", "target_lang": "id", "include_base64": "true"},
            timeout=120,
        )
    resp_compat.raise_for_status()
    compat_data = cast(Mapping[str, object], _json_loads(resp_compat.text))
    compat_b64 = str(compat_data.get("cleaned_image", ""))
    compat_url = str(compat_data.get("cleaned_image_url", ""))
    if not compat_b64.startswith("data:image/png;base64,") or not compat_url.startswith("/api/cache/images/"):
        print(
            "[ERROR] Backward-compatible include_base64=true response failed validation.",
            file=sys.stderr,
        )
        return 1
    print(
        f"  [OK] include_base64=true returned base64 ({len(compat_b64)} chars) + URL ({compat_url})"
    )

    # 5. POST /api/retranslate (switching target_lang from "id" to "en")
    print("\n[5] Testing POST /api/retranslate (ja -> en) ...")
    retranslate_payload: dict[str, object] = {
        "source_lang": "ja",
        "target_lang": "en",
        "blocks": ja_blocks,
    }
    resp_retrans = requests.post(
        f"{BASE_URL}/api/retranslate",
        data=json.dumps(retranslate_payload, ensure_ascii=False),
        headers={"Content-Type": "application/json"},
        timeout=60,
    )
    resp_retrans.raise_for_status()
    retrans_data = cast(Mapping[str, object], _json_loads(resp_retrans.text))
    retrans_blocks_obj = retrans_data.get("blocks")
    retrans_blocks: Sequence[Mapping[str, object]] = (
        cast(Sequence[Mapping[str, object]], retrans_blocks_obj)
        if isinstance(retrans_blocks_obj, Sequence)
        and not isinstance(retrans_blocks_obj, (str, bytes))
        else ()
    )
    print(
        json.dumps(
            {
                "model": retrans_data.get("model"),
                "source_lang": retrans_data.get("source_lang"),
                "target_lang": retrans_data.get("target_lang"),
                "blocks": [
                    {
                        "id": b.get("id"),
                        "bbox": b.get("bbox"),
                        "original_text": b.get("original_text"),
                        "translated_text": b.get("translated_text"),
                        "font_size": b.get("font_size"),
                        "color": b.get("color"),
                        "align": b.get("align"),
                    }
                    for b in retrans_blocks
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if len(retrans_blocks) != len(ja_blocks):
        print("[ERROR] /api/retranslate returned mismatched block count.", file=sys.stderr)
        return 1

    print(
        "\nSUCCESS: All FastAPI endpoints (/api/models/status, /api/lmstudio/status, "
        + "/api/models/download, /api/process-page [JA/KO/ZH + /api/cache/images FileResponse + include_base64], "
        + "/api/retranslate) verified!"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_api_verification())
