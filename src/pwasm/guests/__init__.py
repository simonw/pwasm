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


_modules: dict[tuple[str, float], Module] = {}


def load_guest(name: str, wasm_path: str | os.PathLike | None = None) -> Module:
    """Decode a guest module. Decoded modules are cached, so every instance
    of a guest shares the Python code compiled for its functions."""
    path = Path(wasm_path or guest_path(name)).resolve()
    key = (str(path), path.stat().st_mtime)
    module = _modules.get(key)
    if module is None:
        module = _modules[key] = decode_module(path.read_bytes())
    return module


from .micropython import MicroPython, PythonError  # noqa: E402
from .quickjs import JSError, QuickJS  # noqa: E402
from .mquickjs import MQuickJS  # noqa: E402

__all__ = [
    "JSError",
    "MicroPython",
    "MQuickJS",
    "PythonError",
    "QuickJS",
    "guest_path",
    "load_guest",
]
