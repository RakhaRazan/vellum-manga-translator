"""Pure-Python Native Desktop Host for MangaFlow AI using `pywebview`.

Starts the FastAPI/Uvicorn backend (serving both `/api/*` and the compiled React
static bundle at `/`) in a background daemon thread on `127.0.0.1`, waits until
`GET /api/models/status` responds, and opens a native desktop window via
`pywebview` (Microsoft Edge WebView2 on Windows).

Never spawns an external web browser (`webbrowser.open` is intentionally unused)
and exits cleanly via `os._exit(0)` on window close with zero orphan processes.
"""

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path
import socket
import sys
import threading
import time
from typing import TextIO
import urllib.error
import urllib.request


def _configure_frozen_stdio() -> TextIO | None:
    """Ensure `sys.stdout` and `sys.stderr` are valid writable streams in `--noconsole` builds.

    When packaged with PyInstaller `--noconsole`, Windows GUI executables start with
    `sys.stdout is None` and `sys.stderr is None`. Redirecting missing streams to a
    persistent log file under `%LOCALAPPDATA%\\ComicTranslator\\mangaflow.log` prevents
    third-party libraries (`uvicorn`, `loguru`, `tqdm`, `traceback`) from crashing on
    missing stdio handles while preserving startup diagnostics.
    """
    if sys.stdout is not None and sys.stderr is not None:
        return None

    stream: TextIO
    try:
        local_app_data = os.environ.get("LOCALAPPDATA")
        base_dir = (
            Path(local_app_data).expanduser().resolve()
            if local_app_data
            else (Path.home() / "AppData" / "Local").resolve()
        )
        log_dir = base_dir / "ComicTranslator"
        log_dir.mkdir(parents=True, exist_ok=True)
        stream = open(log_dir / "mangaflow.log", "w", encoding="utf-8", buffering=1)
    except OSError:
        stream = open(os.devnull, "w", encoding="utf-8")

    if sys.stdout is None:
        sys.stdout = stream
    if sys.stderr is None:
        sys.stderr = stream
    return stream


_FROZEN_LOG_STREAM = _configure_frozen_stdio()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if not getattr(sys, "frozen", False) and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn  # noqa: E402
import webview  # noqa: E402

from backend.main import (  # noqa: E402
    app as fastapi_app,
    mount_frontend_dist,
    pipeline_service,
)

HOST = "127.0.0.1"
DEFAULT_PORT = 8000
WINDOW_TITLE = "MangaFlow AI"
WINDOW_WIDTH = 1440
WINDOW_HEIGHT = 900
WINDOW_MIN_SIZE = (1024, 680)
WINDOW_BG_COLOR = "#111214"
READY_PATH = "/api/models/status"
READY_TIMEOUT_SECONDS = 30.0
READY_POLL_INTERVAL_SECONDS = 0.15


def _is_port_available(host: str, port: int) -> bool:
    """Return True if `(host, port)` has no active listener and can be bound exclusively."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.25)
        if probe.connect_ex((host, port)) == 0:
            return False

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def pick_available_port(host: str = HOST, preferred_port: int = DEFAULT_PORT) -> int:
    """Prefer `preferred_port` (8000), or auto-select an OS-assigned free loopback port."""
    if _is_port_available(host, preferred_port):
        return preferred_port

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _start_uvicorn_daemon(
    host: str,
    port: int,
) -> tuple[uvicorn.Server, threading.Thread, list[BaseException]]:
    """Launch FastAPI/Uvicorn in a background daemon thread without touching main-thread signals."""
    headless_io = (
        _FROZEN_LOG_STREAM is not None or sys.stdout is None or sys.stderr is None
    )
    config = uvicorn.Config(
        app=fastapi_app,
        host=host,
        port=port,
        log_level="warning" if headless_io else "info",
        access_log=False,
        use_colors=not headless_io,
        log_config=None if headless_io else uvicorn.config.LOGGING_CONFIG,
    )
    server = uvicorn.Server(config)
    # Signal handlers may only be installed on the main thread.
    server.install_signal_handlers = lambda: None  # type: ignore[method-assign]

    server_errors: list[BaseException] = []

    def _serve() -> None:
        try:
            server.run()
        except BaseException as exc:  # pragma: no cover - defensive crash capture
            server_errors.append(exc)

    thread = threading.Thread(
        target=_serve,
        name="mangaflow-fastapi-server",
        daemon=True,
    )
    thread.start()
    return server, thread, server_errors


def _wait_for_backend_ready(
    host: str,
    port: int,
    server_errors: list[BaseException],
    timeout_seconds: float = READY_TIMEOUT_SECONDS,
) -> None:
    """Poll `GET /api/models/status` on loopback until HTTP 200 is received."""
    url = f"http://{host}:{port}{READY_PATH}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout_seconds
    last_error: str | None = None

    while time.monotonic() < deadline:
        if server_errors:
            raise RuntimeError(
                f"Embedded FastAPI server crashed during startup: {server_errors[0]}"
            ) from server_errors[0]

        remaining = max(0.5, deadline - time.monotonic())
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with opener.open(req, timeout=min(10.0, remaining)) as resp:
                if resp.status == 200:
                    return
                last_error = f"HTTP {resp.status}"
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last_error = str(exc)

        time.sleep(READY_POLL_INTERVAL_SECONDS)

    raise TimeoutError(
        f"Timed out after {timeout_seconds:.1f}s waiting for {url} ({last_error or 'no response'})."
    )


def main() -> None:
    """Start the embedded FastAPI backend and launch the native `pywebview` window."""
    multiprocessing.freeze_support()

    dist_dir = mount_frontend_dist(fastapi_app)
    if dist_dir is None:
        print(
            "[ERROR] Compiled frontend bundle not found (`frontend/dist/index.html`).\n"
            "        Run `npm run build` inside `frontend/` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    port = pick_available_port(HOST, DEFAULT_PORT)
    app_url = f"http://{HOST}:{port}/"
    print(f"[MangaFlow AI] Serving static UI from: {dist_dir}", flush=True)
    print(f"[MangaFlow AI] Models directory     : {pipeline_service.model_manager.models_dir}", flush=True)
    print(f"[MangaFlow AI] Settings file        : {pipeline_service.settings_store.path}", flush=True)
    print(f"[MangaFlow AI] Fonts directory      : {pipeline_service.font_manager.fonts_dir}", flush=True)
    print(f"[MangaFlow AI] Starting embedded FastAPI backend on {app_url}", flush=True)

    server, server_thread, server_errors = _start_uvicorn_daemon(HOST, port)

    try:
        _wait_for_backend_ready(HOST, port, server_errors)
    except BaseException as exc:
        print(f"[ERROR] Backend failed to become ready: {exc}", file=sys.stderr, flush=True)
        server.should_exit = True
        server_thread.join(timeout=1.5)
        os._exit(1)

    print(f"[MangaFlow AI] Backend ready on {app_url}", flush=True)

    webview.settings["ALLOW_DOWNLOADS"] = True

    window = webview.create_window(
        title=WINDOW_TITLE,
        url=app_url,
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
        min_size=WINDOW_MIN_SIZE,
        resizable=True,
        background_color=WINDOW_BG_COLOR,
    )
    if window is None:
        print("[ERROR] Failed to create native pywebview window.", file=sys.stderr)
        server.should_exit = True
        server_thread.join(timeout=1.5)
        os._exit(1)

    def _on_window_closed() -> None:
        server.should_exit = True

    window.events.closed += _on_window_closed

    debug_mode = os.environ.get("MANGAFLOW_DEBUG", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    gui_backend = "edgechromium" if sys.platform == "win32" else None

    try:
        webview.start(gui=gui_backend, debug=debug_mode)
    finally:
        server.should_exit = True
        server_thread.join(timeout=1.5)
        try:
            pipeline_service.unload_all()
        except Exception:
            pass
        os._exit(0)


if __name__ == "__main__":
    main()
