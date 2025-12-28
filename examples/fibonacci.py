#!/usr/bin/env python3
"""Run Fibonacci using pure WebAssembly (no JavaScript engine).

This demonstrates the pure-python-wasm interpreter executing a simple
Fibonacci function compiled directly to WebAssembly. This is MUCH faster
than running MicroQuickJS because:
1. No JavaScript parsing/compilation overhead
2. Direct WASM bytecode execution
3. Minimal WASM module (154 bytes vs 172KB for MicroQuickJS)

The Fibonacci function is implemented in WebAssembly Text (WAT) format
and compiled to WASM binary. Both recursive and iterative versions are
provided for comparison.
"""

import pure_python_wasm
from pathlib import Path
import time


def main():
    script_dir = Path(__file__).parent
    wasm_path = script_dir.parent / "tests" / "fibonacci.wasm"

    if not wasm_path.exists():
        print(f"Error: {wasm_path} not found")
        print("Compile with: wat2wasm tests/fibonacci.wat -o tests/fibonacci.wasm")
        return 1

    print("Loading Fibonacci WASM module...")
    with open(wasm_path, "rb") as f:
        wasm_bytes = f.read()

    module = pure_python_wasm.decode_module(wasm_bytes)
    print(f"  {len(wasm_bytes)} bytes, {len(module.funcs)} functions")

    instance = pure_python_wasm.instantiate(module, {})
    exports = {exp.name: exp.index for exp in instance.module.exports}

    print("\nFibonacci sequence using recursive implementation:")
    print("  n  | fib(n) | time")
    print("-----|--------|--------")

    for n in range(15):
        start = time.monotonic()
        result = pure_python_wasm.execute_function(instance, exports["fib"], [n])
        elapsed = time.monotonic() - start
        print(f"  {n:2} | {result:6} | {elapsed:.4f}s")

    print("\nFibonacci sequence using iterative implementation:")
    print("  n  | fib(n) | time")
    print("-----|--------|--------")

    for n in [10, 20, 30, 40, 50]:
        start = time.monotonic()
        result = pure_python_wasm.execute_function(instance, exports["fib_iter"], [n])
        elapsed = time.monotonic() - start
        print(f"  {n:2} | {result:10} | {elapsed:.6f}s")

    # Verify correctness
    expected = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
    print("\nVerifying correctness...")
    all_correct = True
    for n, exp in enumerate(expected):
        result = pure_python_wasm.execute_function(instance, exports["fib"], [n])
        if result != exp:
            print(f"  FAIL: fib({n}) = {result}, expected {exp}")
            all_correct = False

    if all_correct:
        print("  All values correct!")
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit(main())
