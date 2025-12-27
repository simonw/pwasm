# Pure Python WebAssembly Runtime Specification

This document specifies the design and architecture of a pure Python WebAssembly (WASM) runtime with zero external dependencies.

## Overview

The runtime provides a complete implementation of the WebAssembly 1.0 specification (MVP) with selected post-MVP features, enabling Python applications to load and execute `.wasm` binary modules.

## Goals

1. **Pure Python**: No C extensions, Cython, or external dependencies
2. **Correctness**: Pass the official WebAssembly spec test suite
3. **Clarity**: Prioritize readable, well-documented code over raw performance
4. **Usability**: Provide a Pythonic API for embedding WASM in Python applications
5. **Interoperability**: Enable seamless function calls between Python and WASM

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Python Application                   │
├─────────────────────────────────────────────────────────┤
│                     Public API Layer                     │
│  - load_module(bytes) -> Module                          │
│  - instantiate(module, imports) -> Instance              │
│  - Instance.exports.func(*args) -> results               │
├─────────────────────────────────────────────────────────┤
│                    Runtime Components                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │   Decoder   │  │  Validator  │  │    Executor     │  │
│  │  (Binary)   │  │  (Types)    │  │  (Interpreter)  │  │
│  └─────────────┘  └─────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────┤
│                   Core Data Structures                   │
│  ┌───────┐ ┌────────┐ ┌───────┐ ┌──────┐ ┌─────────┐   │
│  │Module │ │Instance│ │Memory │ │Table │ │ Global  │   │
│  └───────┘ └────────┘ └───────┘ └──────┘ └─────────┘   │
│  ┌───────┐ ┌────────┐ ┌───────────────────────────┐    │
│  │ Stack │ │ Frame  │ │     Value Types           │    │
│  └───────┘ └────────┘ └───────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

## Component Specifications

### 1. Binary Decoder (`decoder.py`)

Parses WebAssembly binary format (`.wasm` files) into an abstract module representation.

#### Responsibilities:
- Read and validate the WASM magic number (`\0asm`) and version (1)
- Decode LEB128 variable-length integers (both signed and unsigned)
- Parse all section types in the correct order
- Decode instruction sequences into an AST

#### Section Types:
| ID | Name     | Description |
|----|----------|-------------|
| 0  | Custom   | Custom metadata (names, debug info) |
| 1  | Type     | Function type declarations |
| 2  | Import   | Imported functions, tables, memories, globals |
| 3  | Function | Function declarations (type indices) |
| 4  | Table    | Table declarations |
| 5  | Memory   | Linear memory declarations |
| 6  | Global   | Global variable declarations |
| 7  | Export   | Exported definitions |
| 8  | Start    | Start function index |
| 9  | Element  | Table element initializers |
| 10 | Code     | Function bodies |
| 11 | Data     | Memory data initializers |

#### Binary Encoding Details:
- **LEB128**: Variable-length integer encoding
  - Unsigned: 7 bits per byte, MSB indicates continuation
  - Signed: Same, but with sign extension on final byte
- **Vectors**: Length-prefixed sequences (`u32` count + elements)
- **Names**: UTF-8 encoded strings as byte vectors

### 2. Module Representation (`module.py`)

Represents a decoded but not yet instantiated WebAssembly module.

#### Data Structures:

```python
@dataclass
class FuncType:
    params: tuple[ValType, ...]   # Parameter types
    results: tuple[ValType, ...]  # Result types

@dataclass
class Function:
    type_idx: int
    locals: tuple[ValType, ...]
    body: list[Instruction]

@dataclass
class Table:
    element_type: RefType
    limits: Limits  # min, max?

@dataclass
class Memory:
    limits: Limits  # min pages, max pages?

@dataclass
class Global:
    type: GlobalType  # valtype + mutability
    init: list[Instruction]  # Constant expression

@dataclass
class Export:
    name: str
    kind: ExportKind  # func, table, memory, global
    index: int

@dataclass
class Import:
    module: str
    name: str
    desc: ImportDesc  # func type, table, memory, or global

@dataclass
class Module:
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
    customs: list[Custom]
```

### 3. Value Types (`types.py`)

WebAssembly's type system:

#### Numeric Types:
- `i32`: 32-bit integer (signed/unsigned interpretation depends on operation)
- `i64`: 64-bit integer
- `f32`: 32-bit IEEE 754 float
- `f64`: 64-bit IEEE 754 float

#### Reference Types (post-MVP):
- `funcref`: Function reference
- `externref`: External reference (opaque to WASM)

#### Value Representation in Python:
```python
class I32:
    """32-bit integer with proper wrapping semantics."""
    value: int  # Stored as signed, converted for unsigned ops

class I64:
    """64-bit integer with proper wrapping semantics."""
    value: int

class F32:
    """32-bit float with IEEE 754 semantics."""
    value: float  # Python float, truncated for storage

class F64:
    """64-bit float with IEEE 754 semantics."""
    value: float
```

### 4. Validator (`validator.py`)

Performs static validation of modules before instantiation.

#### Validation Phases:
1. **Structure validation**: Section ordering, limits checks
2. **Type validation**: All type indices in bounds
3. **Instruction validation**: Stack-based type checking
4. **Import/Export validation**: Name uniqueness, type compatibility

### 5. Instance (`instance.py`)

A module instance with concrete runtime state.

```python
@dataclass
class Instance:
    module: Module
    memories: list[MemoryInstance]
    tables: list[TableInstance]
    globals: list[GlobalInstance]
    funcs: list[FuncInstance]  # Including imports
    exports: ExportNamespace
```

### 6. Memory (`memory.py`)

Linear memory implementation (byte-addressable, little-endian).

```python
class MemoryInstance:
    data: bytearray
    min_pages: int
    max_pages: int | None

    PAGE_SIZE = 65536  # 64 KiB

    def load(self, addr: int, width: int) -> bytes: ...
    def store(self, addr: int, value: bytes) -> None: ...
    def grow(self, delta: int) -> int: ...  # Returns old size or -1
    def size(self) -> int: ...  # Returns size in pages
```

### 7. Table (`table.py`)

Table of function references (or externrefs).

```python
class TableInstance:
    elements: list[FuncRef | None]
    min_size: int
    max_size: int | None
```

### 8. Executor/Interpreter (`executor.py`)

Stack-based bytecode interpreter.

#### Execution State:
```python
@dataclass
class Frame:
    func: FuncInstance
    locals: list[Value]
    module: Instance

class Stack:
    values: list[Value]
    labels: list[Label]
    frames: list[Frame]
```

#### Instruction Categories:

**Control Flow:**
- `unreachable`, `nop`, `block`, `loop`, `if`, `else`, `end`
- `br`, `br_if`, `br_table`, `return`, `call`, `call_indirect`

**Parametric:**
- `drop`, `select`

**Variable Access:**
- `local.get`, `local.set`, `local.tee`
- `global.get`, `global.set`

**Memory Operations:**
- `i32.load`, `i64.load`, `f32.load`, `f64.load` (+ variants)
- `i32.store`, `i64.store`, `f32.store`, `f64.store` (+ variants)
- `memory.size`, `memory.grow`

**Numeric Operations:**
- Const: `i32.const`, `i64.const`, `f32.const`, `f64.const`
- Unary: `clz`, `ctz`, `popcnt`, `abs`, `neg`, `sqrt`, `ceil`, `floor`, `trunc`, `nearest`
- Binary: `add`, `sub`, `mul`, `div`, `rem`, `and`, `or`, `xor`, `shl`, `shr`, `rotl`, `rotr`
- Compare: `eqz`, `eq`, `ne`, `lt`, `gt`, `le`, `ge`
- Convert: Type conversions between i32/i64/f32/f64

### 9. Python Bridge (`bridge.py`)

Enables calling between Python and WASM.

#### Python → WASM:
```python
# Load and instantiate
module = load_module(wasm_bytes)
instance = instantiate(module)

# Call exported function
result = instance.exports.add(1, 2)  # Returns Python int
```

#### WASM → Python:
```python
# Define import functions
def py_print(x: int) -> None:
    print(x)

imports = {
    "env": {
        "print": py_print,
    }
}

instance = instantiate(module, imports)
```

#### Type Marshalling:
| WASM Type | Python Type |
|-----------|-------------|
| i32       | int (clamped to 32 bits) |
| i64       | int (clamped to 64 bits) |
| f32       | float (precision may differ) |
| f64       | float |
| funcref   | Callable |
| externref | Any |

## Error Handling

All errors are exceptions inheriting from `WasmError`:

```python
class WasmError(Exception): pass
class DecodeError(WasmError): pass      # Binary format errors
class ValidationError(WasmError): pass   # Type/structure errors
class TrapError(WasmError): pass         # Runtime traps
class LinkError(WasmError): pass         # Import resolution failures
```

### Trap Conditions:
- Integer divide by zero
- Integer overflow in signed division
- Invalid conversion to integer
- Unreachable instruction executed
- Out-of-bounds memory access
- Out-of-bounds table access
- Indirect call type mismatch
- Call stack exhaustion

## Public API

```python
# Main entry points
def load_module(source: bytes | BinaryIO | Path) -> Module:
    """Decode a WASM binary into a Module."""

def validate(module: Module) -> None:
    """Validate a module, raising ValidationError on failure."""

def instantiate(
    module: Module,
    imports: dict[str, dict[str, Any]] | None = None,
) -> Instance:
    """Create an instance from a module with optional imports."""

# Instance usage
instance = instantiate(module)
result = instance.exports.my_function(arg1, arg2)

# Memory access (if exported)
mem = instance.exports.memory
mem.read(addr, length) -> bytes
mem.write(addr, data: bytes)
```

## Supported WebAssembly Features

### MVP (1.0) - Full Support Target:
- [x] All value types (i32, i64, f32, f64)
- [x] All numeric operations
- [x] Control flow (block, loop, if, br, br_if, br_table, call, return)
- [x] Local and global variables
- [x] Linear memory (load, store, grow, size)
- [x] Tables and call_indirect
- [x] Imports and exports
- [x] Start function

### Post-MVP - Planned:
- [ ] Multi-value returns
- [ ] Reference types (funcref, externref)
- [ ] Bulk memory operations
- [ ] SIMD (partial, integer ops only)

### Not Planned:
- Threads and atomics (requires threading support)
- Exception handling proposal (complex, evolving)
- GC proposal (complex, evolving)
- Component model (higher-level, separate concern)

## Testing Strategy

1. **Unit tests**: Each component tested in isolation
2. **Integration tests**: End-to-end module loading and execution
3. **Spec tests**: Run official WebAssembly test suite
   - Located in `test/core/*.wast` of the spec repo
   - Requires a WAST parser for the test format

## File Structure

```
pure_python_wasm/
├── __init__.py          # Public API exports
├── types.py             # Value types and type definitions
├── decoder.py           # Binary format decoder
├── module.py            # Module representation
├── validator.py         # Static validation
├── instance.py          # Runtime instances
├── memory.py            # Linear memory implementation
├── table.py             # Function tables
├── executor.py          # Bytecode interpreter
├── bridge.py            # Python<->WASM interop
├── errors.py            # Exception classes
└── opcodes.py           # Instruction definitions

tests/
├── test_decoder.py
├── test_types.py
├── test_executor.py
├── test_memory.py
├── test_integration.py
└── test_spec/           # Spec test runner
```
