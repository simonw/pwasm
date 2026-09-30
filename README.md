# pwasm

[![PyPI](https://img.shields.io/pypi/v/pwasm.svg)](https://pypi.org/project/pwasm/)
[![Tests](https://github.com/simonw/pwasm/actions/workflows/test.yml/badge.svg)](https://github.com/simonw/pwasm/actions/workflows/test.yml)
[![Changelog](https://img.shields.io/github/v/release/simonw/pwasm?include_prereleases&label=changelog)](https://github.com/simonw/pwasm/releases)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/simonw/pwasm/blob/main/LICENSE)

A pure Python WebAssembly runtime.

> **Warning:** This is alpha software. It is significantly slower than WebAssembly runtimes with native extensions like [wasmtime-py](https://github.com/bytecodealliance/wasmtime-py).

## Overview

`pwasm` is a WebAssembly runtime written entirely in Python with zero external dependencies. It can load and execute `.wasm` binary modules without requiring any C extensions.

## Features

- **Pure Python** - No external dependencies or C extensions required
- **WebAssembly 2.0 core instructions** - everything except SIMD: integer and floating point arithmetic, conversions, sign extension, saturating truncation, multi-value blocks and functions, bulk memory and reference types
- **Passes the spec test suite** - the non-SIMD WebAssembly 2.0 core tests run in CI (validation-only tests are skipped)
- **Pythonic API** - Access exported functions directly as Python methods
- **Imports and linking** - Python functions, memories, globals and tables can be imported, including from other instances
- **Memories, tables and globals** accessible from Python

## Installation

```bash
pip install pwasm
```

Or with uv:

```bash
uv add pwasm
```

## Requirements

- Python 3.10+

## Usage

### Loading and Running a WebAssembly Module

```python
from pwasm import decode_module, instantiate

# Load a WASM module from bytes
with open("module.wasm", "rb") as f:
    wasm_bytes = f.read()

module = decode_module(wasm_bytes)
instance = instantiate(module)

# Call exported functions directly
result = instance.exports.add(2, 3)
print(result)  # 5
```

### Working with Multiple Functions

```python
from pwasm import decode_module, instantiate

module = decode_module(wasm_bytes)
instance = instantiate(module)

# Arithmetic operations
print(instance.exports.add(10, 20))       # 30
print(instance.exports.sub(50, 8))        # 42
print(instance.exports.mul(6, 7))         # 42
```

### Importing Python Functions

Functions a module imports can be supplied as Python callables, grouped by module name:

```python
from pwasm import decode_module, instantiate

def log(value):
    print("wasm says", value)

instance = instantiate(module, {"env": {"log": log}})
```

Host functions receive i32 and i64 arguments as signed Python integers and floats for f32 and f64. They can return `None`, a single value, or a tuple for functions with multiple results. Exceptions raised by a host function propagate out through the WebAssembly code to the Python caller.

An exported function from one instance can be imported by another:

```python
app = instantiate(app_module, {"lib": {"square": lib.exports.square}})
```

### Memories, Globals and Tables

Exported memories, globals and tables are available as objects:

```python
memory = instance.exports.memory
data = memory.read(ptr, 16)       # bytes
memory.write(ptr, b"hello")
memory.data                       # the underlying bytearray
memory.size                       # size in 64 KiB pages

counter = instance.exports.counter
counter.value                     # i32/i64 globals read as signed ints
counter.value = 5                 # mutable globals only

table = instance.exports.table
func = table.get(0)               # a function reference
func(1, 2)                        # function references can be called
```

Modules can also import memories, globals and tables, either created in Python or exported by another instance:

```python
from pwasm.runtime import GlobalInstance, MemoryInstance, TableInstance
from pwasm.types import GlobalType

memory = MemoryInstance(1, 10)  # min and max pages
instance = instantiate(module, {
    "env": {
        "memory": memory,
        "limit": 100,  # a plain number can supply an immutable global
        "counter": GlobalInstance(GlobalType("i32", mutable=True), 0),
        "table": TableInstance("funcref", 4),
    }
})
```

Host functions can call back into the instance that called them, and a Python exception raised inside a host function unwinds any WebAssembly frames in between.

### Error Handling

```python
from pwasm import decode_module, instantiate
from pwasm.errors import TrapError, DecodeError

# Handle runtime traps (e.g., division by zero)
try:
    instance.exports.div_s(10, 0)
except TrapError as e:
    print(f"Runtime trap: {e}")

# Handle malformed WASM
try:
    module = decode_module(b"invalid wasm")
except DecodeError as e:
    print(f"Decode error: {e}")
```

## Architecture

### Components

- **decoder.py** - Parses WebAssembly binary format with LEB128 decoding. Function bodies are decoded lazily, the first time they are needed
- **types.py** - WebAssembly type system (i32, i64, f32, f64, funcref, externref)
- **compiler.py** - Compiles each function, on its first call, into flat lists of internal opcodes and immediates
- **executor.py** - The interpreter loop, module instantiation and exports
- **runtime.py** - Memories, globals and function instances
- **numeric.py** - Numeric helpers (integer and floating point semantics)
- **errors.py** - Exception hierarchy (WasmError, DecodeError, ValidationError, TrapError, LinkError)

### Execution Model

Structured control flow is compiled away before execution: `block`, `loop` and `end` produce no code, and every branch becomes a jump to a precomputed instruction index. Operand stack heights are static in valid WebAssembly, so the compiler also knows exactly which values each branch needs to discard - the interpreter keeps no label stack.

Internally i32 and i64 values are stored as unsigned Python integers. Values are converted to signed integers when they are returned to Python code.

Each WebAssembly function call is a Python call of the interpreter, so exceptions raised by Python code propagate through WebAssembly frames.

## Development

```bash
# Clone and setup
git clone https://github.com/simonw/pwasm
cd pwasm

# Run tests
uv run pytest

# Format code
uv run black .
```

## License

Apache 2.0
