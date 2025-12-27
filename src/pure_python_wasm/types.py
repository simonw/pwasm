"""WebAssembly type definitions."""

from dataclasses import dataclass
from typing import Any


# Value type constants
VALTYPE_I32 = "i32"
VALTYPE_I64 = "i64"
VALTYPE_F32 = "f32"
VALTYPE_F64 = "f64"
VALTYPE_FUNCREF = "funcref"
VALTYPE_EXTERNREF = "externref"

# Binary encoding of value types
VALTYPE_ENCODING = {
    0x7F: VALTYPE_I32,
    0x7E: VALTYPE_I64,
    0x7D: VALTYPE_F32,
    0x7C: VALTYPE_F64,
    0x70: VALTYPE_FUNCREF,
    0x6F: VALTYPE_EXTERNREF,
}

# Reverse mapping
VALTYPE_TO_BYTE = {v: k for k, v in VALTYPE_ENCODING.items()}

ValType = str  # One of the VALTYPE_* constants


@dataclass(frozen=True)
class FuncType:
    """WebAssembly function type (signature)."""

    params: tuple[ValType, ...]
    results: tuple[ValType, ...]

    def __repr__(self) -> str:
        params = ", ".join(self.params)
        results = ", ".join(self.results)
        return f"({params}) -> ({results})"


@dataclass(frozen=True)
class Limits:
    """Memory or table limits."""

    min: int
    max: int | None = None


@dataclass(frozen=True)
class MemoryType:
    """Memory type with limits."""

    limits: Limits


@dataclass(frozen=True)
class TableType:
    """Table type with element type and limits."""

    element_type: ValType
    limits: Limits


@dataclass(frozen=True)
class GlobalType:
    """Global type with value type and mutability."""

    valtype: ValType
    mutable: bool


# Export kinds
EXPORT_FUNC = "func"
EXPORT_TABLE = "table"
EXPORT_MEMORY = "memory"
EXPORT_GLOBAL = "global"

EXPORT_KIND_ENCODING = {
    0x00: EXPORT_FUNC,
    0x01: EXPORT_TABLE,
    0x02: EXPORT_MEMORY,
    0x03: EXPORT_GLOBAL,
}


@dataclass(frozen=True)
class Instruction:
    """A WebAssembly instruction."""

    opcode: str
    operand: Any = None  # Immediate value(s) if any

    def __repr__(self) -> str:
        if self.operand is not None:
            return f"{self.opcode} {self.operand}"
        return self.opcode


@dataclass
class Function:
    """A WebAssembly function (decoded from module)."""

    type_idx: int
    locals: tuple[ValType, ...]
    body: list[Instruction]
    jump_targets: dict[int, tuple[int | None, int]] = (
        None  # Cache: ip -> (else_ip, end_ip)
    )


@dataclass
class Export:
    """An export entry."""

    name: str
    kind: str  # One of EXPORT_* constants
    index: int


@dataclass
class Import:
    """An import entry."""

    module: str
    name: str
    kind: str
    desc: Any  # Type index for func, or limits/type for others


@dataclass
class Memory:
    """Memory declaration."""

    limits: Limits


@dataclass
class Table:
    """Table declaration."""

    element_type: ValType
    limits: Limits


@dataclass
class Global:
    """Global variable declaration."""

    type: GlobalType
    init: list[Instruction]


@dataclass
class Element:
    """Element segment for table initialization."""

    table_idx: int
    offset: list[Instruction]
    init: list[int]  # Function indices


@dataclass
class Data:
    """Data segment for memory initialization."""

    memory_idx: int
    offset: list[Instruction]
    init: bytes


@dataclass
class Module:
    """A decoded WebAssembly module."""

    types: list[FuncType]
    funcs: list[Function]
    tables: list[Table]
    mems: list[Memory]
    globals: list[Global]
    exports: list[Export]
    imports: list[Import]
    start: int | None
    elem: list[Element]
    data: list[Data]

    def __init__(self) -> None:
        self.types = []
        self.funcs = []
        self.tables = []
        self.mems = []
        self.globals = []
        self.exports = []
        self.imports = []
        self.start = None
        self.elem = []
        self.data = []
