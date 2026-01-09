#!/usr/bin/env python3
"""Benchmark script for pure_python_wasm executor performance."""

import time
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pure_python_wasm import decode_module, instantiate


def make_simple_func(
    name: str, instr_bytes: bytes, num_params: int = 2, num_locals: int = 0
) -> bytes:
    """Helper to create a simple module with one exported function."""
    name_bytes = name.encode("utf-8")
    name_len = len(name_bytes)

    # Local declarations
    if num_locals > 0:
        local_decl = bytes([0x01, num_locals, 0x7F])  # num_locals of i32
        body_prefix = local_decl
    else:
        body_prefix = bytes([0x00])  # 0 locals

    body_size = len(body_prefix) + len(instr_bytes)

    # Type section
    if num_params == 0:
        type_section = bytes([0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7F])
    elif num_params == 1:
        type_section = bytes([0x01, 0x06, 0x01, 0x60, 0x01, 0x7F, 0x01, 0x7F])
    else:
        type_section = bytes([0x01, 0x07, 0x01, 0x60, 0x02, 0x7F, 0x7F, 0x01, 0x7F])

    func_section = bytes([0x03, 0x02, 0x01, 0x00])

    export_size = 1 + 1 + name_len + 1 + 1
    export_section = (
        bytes([0x07, export_size, 0x01, name_len]) + name_bytes + bytes([0x00, 0x00])
    )

    code_section = (
        bytes([0x0A, 2 + body_size, 0x01, body_size]) + body_prefix + instr_bytes
    )

    return (
        bytes([0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00])
        + type_section
        + func_section
        + export_section
        + code_section
    )


def benchmark_simple_arithmetic(num_ops=10000):
    """Benchmark simple arithmetic operations."""
    # Create a function that does: return a + b
    wasm = make_simple_func("add", bytes([0x20, 0x00, 0x20, 0x01, 0x6A, 0x0B]))
    module = decode_module(wasm)
    instance = instantiate(module)

    # Warm up
    instance.exports.add(1, 2)

    # Benchmark
    start = time.perf_counter()
    for i in range(num_ops):
        instance.exports.add(i, 1)
    elapsed = time.perf_counter() - start

    return {
        "name": "simple_add",
        "iterations": num_ops,
        "time_s": elapsed,
        "ops_per_sec": num_ops / elapsed if elapsed > 0 else 0,
    }


def benchmark_loop(iterations=1000):
    """Benchmark a counting loop inside WASM.

    (func (param $n i32) (result i32)
      (local $i i32)
      (local.set $i (i32.const 0))
      (block $exit
        (loop $loop
          (local.set $i (i32.add (local.get $i) (i32.const 1)))
          (br_if $loop (i32.lt_s (local.get $i) (local.get $n)))
        )
      )
      (local.get $i)
    )
    """
    instr = bytes(
        [
            # local.set 1, i32.const 0
            0x41,
            0x00,  # i32.const 0
            0x21,
            0x01,  # local.set 1 (i = 0)
            # block $exit
            0x02,
            0x40,  # block (void)
            # loop $loop
            0x03,
            0x40,  # loop (void)
            # i = i + 1
            0x20,
            0x01,  # local.get 1
            0x41,
            0x01,  # i32.const 1
            0x6A,  # i32.add
            0x21,
            0x01,  # local.set 1
            # br_if $loop if i < n
            0x20,
            0x01,  # local.get 1
            0x20,
            0x00,  # local.get 0 (n)
            0x48,  # i32.lt_s
            0x0D,
            0x00,  # br_if 0 (to loop)
            0x0B,  # end loop
            0x0B,  # end block
            # return i
            0x20,
            0x01,  # local.get 1
            0x0B,  # end func
        ]
    )
    wasm = make_simple_func("count", instr, num_params=1, num_locals=1)
    module = decode_module(wasm)
    instance = instantiate(module)

    # Warm up
    result = instance.exports.count(10)
    assert result == 10, f"Expected 10, got {result}"

    # Benchmark
    start = time.perf_counter()
    result = instance.exports.count(iterations)
    elapsed = time.perf_counter() - start

    return {
        "name": "loop",
        "iterations": iterations,
        "result": result,
        "time_s": elapsed,
        "wasm_ops_per_sec": (
            (iterations * 10) / elapsed if elapsed > 0 else 0
        ),  # ~10 ops per iteration
    }


def benchmark_nested_calls(num_calls=1000):
    """Benchmark function call overhead.

    Module with:
    - func 0: (param i32) -> i32, returns param + 1
    - func 1: calls func 0 (exported)
    """
    wasm = bytes(
        [
            0x00,
            0x61,
            0x73,
            0x6D,
            0x01,
            0x00,
            0x00,
            0x00,  # magic + version
            # Type section: (i32) -> i32
            0x01,
            0x06,
            0x01,
            0x60,
            0x01,
            0x7F,
            0x01,
            0x7F,
            # Function section: 2 functions of type 0
            0x03,
            0x03,
            0x02,
            0x00,
            0x00,
            # Export section: export func 1 as "f"
            0x07,
            0x05,
            0x01,
            0x01,
            0x66,
            0x00,
            0x01,  # "f"
            # Code section
            0x0A,
            0x10,
            0x02,
            # func 0: local.get 0, i32.const 1, i32.add, end
            0x07,
            0x00,
            0x20,
            0x00,
            0x41,
            0x01,
            0x6A,
            0x0B,
            # func 1: local.get 0, call 0, end
            0x06,
            0x00,
            0x20,
            0x00,
            0x10,
            0x00,
            0x0B,
        ]
    )
    module = decode_module(wasm)
    instance = instantiate(module)

    # Warm up
    assert instance.exports.f(5) == 6

    # Benchmark
    start = time.perf_counter()
    for i in range(num_calls):
        instance.exports.f(i)
    elapsed = time.perf_counter() - start

    return {
        "name": "nested_calls",
        "iterations": num_calls,
        "time_s": elapsed,
        "calls_per_sec": num_calls / elapsed if elapsed > 0 else 0,
    }


def benchmark_if_else(num_ops=5000):
    """Benchmark conditional branching.

    if param != 0: return 1 else return 0
    """
    instr = bytes(
        [
            0x20,
            0x00,  # local.get 0
            0x04,
            0x7F,  # if (result i32)
            0x41,
            0x01,  # i32.const 1
            0x05,  # else
            0x41,
            0x00,  # i32.const 0
            0x0B,  # end if
            0x0B,  # end func
        ]
    )
    wasm = make_simple_func("cond", instr, num_params=1)
    module = decode_module(wasm)
    instance = instantiate(module)

    # Warm up
    assert instance.exports.cond(1) == 1
    assert instance.exports.cond(0) == 0

    # Benchmark
    start = time.perf_counter()
    for i in range(num_ops):
        instance.exports.cond(i % 2)
    elapsed = time.perf_counter() - start

    return {
        "name": "if_else",
        "iterations": num_ops,
        "time_s": elapsed,
        "ops_per_sec": num_ops / elapsed if elapsed > 0 else 0,
    }


def benchmark_quickjs_init():
    """Benchmark QuickJS WASM initialization (sandbox_init)."""
    wasm_path = Path(__file__).parent / "demo" / "mquickjs.wasm"
    if not wasm_path.exists():
        return {
            "name": "quickjs_init",
            "skipped": True,
            "reason": "WASM file not found",
        }

    from qjs import QuickJSRuntime

    start = time.perf_counter()
    runtime = QuickJSRuntime(wasm_path)
    elapsed = time.perf_counter() - start

    return {
        "name": "quickjs_init",
        "time_s": elapsed,
    }


def run_benchmarks():
    """Run all benchmarks and print results."""
    print("=" * 60)
    print("Pure Python WASM Executor Benchmarks")
    print("=" * 60)
    print()

    benchmarks = [
        ("Simple Add (10k calls)", lambda: benchmark_simple_arithmetic(10000)),
        ("Loop (1k iterations)", lambda: benchmark_loop(1000)),
        ("Nested Calls (1k)", lambda: benchmark_nested_calls(1000)),
        ("If/Else (5k)", lambda: benchmark_if_else(5000)),
        ("QuickJS Init", benchmark_quickjs_init),
    ]

    results = []
    for name, bench_func in benchmarks:
        print(f"Running: {name}...")
        try:
            result = bench_func()
            results.append(result)

            if result.get("skipped"):
                print(f"  SKIPPED: {result.get('reason')}")
            else:
                print(f"  Time: {result['time_s']:.4f}s")
                for key in ["ops_per_sec", "calls_per_sec", "wasm_ops_per_sec"]:
                    if key in result:
                        print(f"  {key}: {result[key]:,.0f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback

            traceback.print_exc()
            results.append({"name": name, "error": str(e)})
        print()

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for r in results:
        if r.get("skipped") or r.get("error"):
            continue
        name = r["name"]
        time_s = r["time_s"]
        for key in ["ops_per_sec", "calls_per_sec", "wasm_ops_per_sec"]:
            if key in r:
                print(f"  {name}: {time_s:.4f}s ({r[key]:,.0f} {key})")
                break
        else:
            print(f"  {name}: {time_s:.4f}s")

    return results


if __name__ == "__main__":
    run_benchmarks()
