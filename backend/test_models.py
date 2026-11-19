"""CLI verification test for ModelManager (backend/core/model_manager.py).

Usage:
    backend\\venv\\Scripts\\python.exe backend/test_models.py
"""

import json
import socket
import sys
from pathlib import Path
from typing import Never

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.model_manager import ModelManager  # noqa: E402


def main() -> int:
    manager = ModelManager()
    print(f"=== ModelManager Verification (models_dir: {manager.models_dir}) ===")

    initial_status = manager.get_status()
    print(f"Initial all_ready: {initial_status['all_ready']}")
    print(f"Execution provider: {initial_status['execution_provider']}")

    if initial_status["all_ready"]:
        print(
            "[Offline Guard] All models already present. "
            + "Blocking socket connections to verify 100% offline mode..."
        )

        def _guard_connect(*_args: object, **_kwargs: object) -> Never:
            raise RuntimeError("Outbound network connection attempted during offline run!")

        socket.socket.connect = _guard_connect

    last_printed_bucket: dict[str, int] = {}

    for event in manager.download_missing_models():
        status = event["status"]
        if status in ("starting", "skipped", "completed"):
            print(
                f"[{status.upper()}] {event['message']} "
                + f"(overall: {event['overall_percent']:.1f}%)"
            )
        elif status == "downloading":
            file_key = event["file"] or ""
            pct = event["file_percent"]
            bucket = int(pct // 10) * 10
            if last_printed_bucket.get(file_key, -1) != bucket:
                last_printed_bucket[file_key] = bucket
                print(
                    f"  -> {event['message']} | Overall: {event['overall_percent']:.1f}%"
                )
        elif status == "file_completed":
            print(f"  [OK] {event['message']} | Overall: {event['overall_percent']:.1f}%")

    final_status = manager.get_status()
    print("\n=== Final Model Status ===")
    print(json.dumps(final_status, indent=2))

    if not final_status["all_ready"]:
        print("ERROR: Not all models are ready after download!", file=sys.stderr)
        return 1

    print("\nSUCCESS: All required model weights are verified in models/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
