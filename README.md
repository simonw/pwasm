# pwism

A pure Python WebAssembly runtime. Zero dependencies.

## Installation

```bash
pip install pwism
```

## Usage

```python
from pwism import decode_module
from pwism.executor import instantiate

# Load a WASM module
with open("module.wasm", "rb") as f:
    module = decode_module(f.read())

# Instantiate and run
instance = instantiate(module)
result = instance.exports.add(2, 3)  # Call exported function
```

## Performance

pwism is compatible with both CPython and PyPy. For best performance, **use PyPy** which provides 5-9x speedup through its JIT compiler.

### Benchmark Results

| Benchmark | CPython 3.11 | PyPy 7.3 | Speedup |
|-----------|-------------|----------|---------|
| Simple Call | 680k ops/sec | 4.2M ops/sec | **6.2x** |
| Loop Execution | 22k ops/sec | 134k ops/sec | **6.2x** |
| Recursive Calls | 32k ops/sec | 187k ops/sec | **5.9x** |
| Heavy Arithmetic | 44k ops/sec | 219k ops/sec | **5.0x** |
| Nested Blocks | 88k ops/sec | 784k ops/sec | **8.9x** |

### Running with PyPy

```bash
# Install PyPy
# Ubuntu/Debian: apt install pypy3
# macOS: brew install pypy3

# Run your script with PyPy
pypy3 your_script.py
```

## Development

```bash
# Run tests
uv run pytest

# Run benchmarks
uv run python benchmarks/benchmark_wasm.py

# Format code
uv run black .
```

## License

Apache 2.0
