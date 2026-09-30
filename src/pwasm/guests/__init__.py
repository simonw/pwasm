"""Run untrusted Python and JavaScript in interpreters compiled to
WebAssembly, executed by pwasm. See README.md in this directory for where
the .wasm files came from."""

from __future__ import annotations

import os
from pathlib import Path

from ..decoder import decode_module
from ..types import Module


def guest_path(name: str) -> Path:
    """Path of a bundled guest .wasm file (or one in $PWASM_GUESTS)."""
    directory = os.environ.get("PWASM_GUESTS")
    if directory and (Path(directory) / name).exists():
        return Path(directory) / name
    return Path(__file__).parent / name


def load_guest(name: str, wasm_path: str | os.PathLike | None = None) -> Module:
    return decode_module(Path(wasm_path or guest_path(name)).read_bytes())


from .micropython import MicroPython, PythonError  # noqa: E402
from .quickjs import JSError, QuickJS  # noqa: E402

__all__ = [
    "JSError",
    "MicroPython",
    "PythonError",
    "QuickJS",
    "guest_path",
    "load_guest",
]
