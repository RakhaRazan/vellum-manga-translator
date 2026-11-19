# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec alias (`packaging/pyinstaller/backend.spec`) matching SPEC.md Section 2."""

from pathlib import Path

ROOT_SPEC = Path(SPECPATH).resolve().parent / "comic-backend.spec"
exec(compile(ROOT_SPEC.read_text(encoding="utf-8"), str(ROOT_SPEC), "exec"))
