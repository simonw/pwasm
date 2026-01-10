"""Pure Python WebAssembly Runtime.

A WebAssembly runtime implemented in pure Python with no dependencies.
"""

from .decoder import (
    decode_module,
    BinaryReader,
    decode_unsigned_leb128,
    decode_signed_leb128,
)
from .errors import WasmError, DecodeError, ValidationError, TrapError, LinkError
from .types import Module, FuncType, Function, Export, Import, Instruction
from .executor import instantiate, Instance

__version__ = "0.1.0"

__all__ = [
    # Main API
    "decode_module",
    "instantiate",
    "Instance",
    # Decoder internals (for testing)
    "BinaryReader",
    "decode_unsigned_leb128",
    "decode_signed_leb128",
    # Types
    "Module",
    "FuncType",
    "Function",
    "Export",
    "Import",
    "Instruction",
    # Errors
    "WasmError",
    "DecodeError",
    "ValidationError",
    "TrapError",
    "LinkError",
]


def hello() -> str:
    return "Hello from pwism!"
