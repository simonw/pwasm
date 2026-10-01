"""Test helpers: build modules from WebAssembly text using wasmtime's
wat2wasm (a dev-only dependency) and instantiate them with pwasm."""

import wasmtime

from pwasm import decode_module, instantiate


def wat2wasm(text: str) -> bytes:
    return bytes(wasmtime.wat2wasm(text))


def load(text: str, imports=None):
    """Compile WAT text and return a pwasm Instance."""
    return instantiate(decode_module(wat2wasm(text)), imports)
