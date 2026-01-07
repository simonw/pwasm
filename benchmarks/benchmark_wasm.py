"""Benchmarks for the pure Python WebAssembly runtime.

This module provides benchmarks for:
1. Module decoding performance
2. Function execution performance (arithmetic, control flow, calls)
3. Specific hotspots like LEB128 decoding and instruction dispatch
"""

import time
import cProfile
import pstats
from io import StringIO

from pure_python_wasm import decode_module
from pure_python_wasm.executor import instantiate
from pure_python_wasm.decoder import (
    BinaryReader,
    decode_unsigned_leb128,
    decode_signed_leb128,
)


def encode_leb128(value: int) -> bytes:
    """Encode an unsigned integer as LEB128."""
    result = []
    while True:
        byte = value & 0x7F
        value >>= 7
        if value != 0:
            byte |= 0x80
        result.append(byte)
        if value == 0:
            break
    return bytes(result)


def make_simple_func(
    name: str, instr_bytes: bytes, num_params: int = 2, num_locals: int = 0
) -> bytes:
    """Helper to create a simple module with one exported function."""
    name_bytes = name.encode("utf-8")
    name_len = len(name_bytes)

    # Locals section
    if num_locals > 0:
        locals_section = (
            bytes([0x01]) + encode_leb128(num_locals) + bytes([0x7F])
        )  # i32 locals
    else:
        locals_section = bytes([0x00])  # no locals

    body_content = locals_section + instr_bytes
    body_size = len(body_content)

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

    body_size_enc = encode_leb128(body_size)
    code_body = body_size_enc + body_content
    code_section_content = bytes([0x01]) + code_body  # 1 function
    code_section = (
        bytes([0x0A]) + encode_leb128(len(code_section_content)) + code_section_content
    )

    return (
        bytes([0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00])
        + type_section
        + func_section
        + export_section
        + code_section
    )


# --- WASM modules for benchmarking ---

# Simple addition function
ADD_WASM = make_simple_func("add", bytes([0x20, 0x00, 0x20, 0x01, 0x6A, 0x0B]))

# Fibonacci-like iterative loop
# (func (param $n i32) (result i32) (local $a i32) (local $b i32) (local $i i32) (local $tmp i32)
#   i32.const 0  local.set 1  ;; a = 0
#   i32.const 1  local.set 2  ;; b = 1
#   i32.const 0  local.set 3  ;; i = 0
#   block
#     loop
#       local.get 3  local.get 0  i32.ge_s  br_if 1  ;; if i >= n, break
#       local.get 1  local.get 2  i32.add  local.set 4  ;; tmp = a + b
#       local.get 2  local.set 1  ;; a = b
#       local.get 4  local.set 2  ;; b = tmp
#       local.get 3  i32.const 1  i32.add  local.set 3  ;; i++
#       br 0
#     end
#   end
#   local.get 1
# )
FIB_WASM = make_simple_func(
    "fib",
    bytes(
        [
            0x41,
            0x00,
            0x21,
            0x01,  # i32.const 0, local.set 1 (a = 0)
            0x41,
            0x01,
            0x21,
            0x02,  # i32.const 1, local.set 2 (b = 1)
            0x41,
            0x00,
            0x21,
            0x03,  # i32.const 0, local.set 3 (i = 0)
            0x02,
            0x40,  # block
            0x03,
            0x40,  # loop
            0x20,
            0x03,
            0x20,
            0x00,
            0x4E,
            0x0D,
            0x01,  # local.get 3, local.get 0, i32.ge_s, br_if 1
            0x20,
            0x01,
            0x20,
            0x02,
            0x6A,
            0x21,
            0x04,  # local.get 1, local.get 2, i32.add, local.set 4
            0x20,
            0x02,
            0x21,
            0x01,  # local.get 2, local.set 1
            0x20,
            0x04,
            0x21,
            0x02,  # local.get 4, local.set 2
            0x20,
            0x03,
            0x41,
            0x01,
            0x6A,
            0x21,
            0x03,  # local.get 3, i32.const 1, i32.add, local.set 3
            0x0C,
            0x00,  # br 0
            0x0B,
            0x0B,  # end, end
            0x20,
            0x01,  # local.get 1
            0x0B,  # end
        ]
    ),
    num_params=1,
    num_locals=4,
)

# Recursive function call benchmark (factorial-like)
# This needs the function to call itself by index 0
RECURSIVE_CALL_WASM = make_simple_func(
    "fact",
    bytes(
        [
            # if (n <= 1) return 1 else return n * fact(n-1)
            0x20,
            0x00,  # local.get 0
            0x41,
            0x01,  # i32.const 1
            0x4C,  # i32.le_s
            0x04,
            0x7F,  # if (result i32)
            0x41,
            0x01,  # i32.const 1
            0x05,  # else
            0x20,
            0x00,  # local.get 0
            0x20,
            0x00,
            0x41,
            0x01,
            0x6B,  # local.get 0, i32.const 1, i32.sub
            0x10,
            0x00,  # call 0
            0x6C,  # i32.mul
            0x0B,  # end
            0x0B,  # end
        ]
    ),
    num_params=1,
)


# Heavy arithmetic - many operations in sequence
def make_heavy_arithmetic_wasm(num_ops: int = 50) -> bytes:
    """Create a function with many arithmetic operations."""
    # (func (param i32 i32) (result i32)
    #   local.get 0
    #   local.get 1
    #   i32.add
    #   local.get 1
    #   i32.mul
    #   local.get 0
    #   i32.sub
    #   ... repeated
    # )
    instrs = [0x20, 0x00]  # local.get 0
    for i in range(num_ops):
        instrs.extend([0x20, 0x01])  # local.get 1
        op = [0x6A, 0x6B, 0x6C, 0x71, 0x72, 0x73][i % 6]  # add, sub, mul, and, or, xor
        instrs.append(op)
    instrs.append(0x0B)  # end
    return make_simple_func("heavy", bytes(instrs))


HEAVY_ARITH_WASM = make_heavy_arithmetic_wasm(50)


# Benchmark: nested blocks and branches
def make_nested_blocks_wasm(depth: int = 10) -> bytes:
    """Create a function with nested blocks."""
    instrs = []
    # Open blocks
    for _ in range(depth):
        instrs.extend([0x02, 0x40])  # block (empty result)
    # Branch out
    instrs.extend([0x20, 0x00])  # local.get 0
    instrs.extend([0x0C, depth - 1])  # br to outermost
    # Close all blocks + return value
    for _ in range(depth):
        instrs.append(0x0B)  # end
    instrs.extend([0x20, 0x00])  # local.get 0
    instrs.append(0x0B)  # end func
    return make_simple_func("nested", bytes(instrs), num_params=1)


NESTED_BLOCKS_WASM = make_nested_blocks_wasm(20)


def benchmark_leb128_decode(iterations: int = 100000) -> float:
    """Benchmark LEB128 decoding performance."""
    # Create data with various LEB128 encodings
    test_data = (
        bytes(
            [
                0x00,  # 0
                0x7F,  # 127
                0x80,
                0x01,  # 128
                0xFF,
                0x7F,  # 16383
                0x80,
                0x80,
                0x04,  # 65536
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0x0F,  # max u32
            ]
        )
        * 1000
    )

    start = time.perf_counter()
    for _ in range(iterations // 1000):
        reader = BinaryReader(test_data)
        while not reader.eof():
            try:
                decode_unsigned_leb128(reader)
            except:
                break
    elapsed = time.perf_counter() - start
    return elapsed


def benchmark_decode_module(iterations: int = 1000) -> float:
    """Benchmark module decoding."""
    start = time.perf_counter()
    for _ in range(iterations):
        decode_module(FIB_WASM)
    elapsed = time.perf_counter() - start
    return elapsed


def benchmark_instantiate(iterations: int = 1000) -> float:
    """Benchmark module instantiation."""
    module = decode_module(FIB_WASM)
    start = time.perf_counter()
    for _ in range(iterations):
        instantiate(module)
    elapsed = time.perf_counter() - start
    return elapsed


def benchmark_simple_call(iterations: int = 10000) -> float:
    """Benchmark simple function execution (addition)."""
    module = decode_module(ADD_WASM)
    instance = instantiate(module)

    start = time.perf_counter()
    for i in range(iterations):
        instance.exports.add(i, i + 1)
    elapsed = time.perf_counter() - start
    return elapsed


def benchmark_loop_execution(iterations: int = 1000) -> float:
    """Benchmark loop-heavy execution (fibonacci)."""
    module = decode_module(FIB_WASM)
    instance = instantiate(module)

    start = time.perf_counter()
    for _ in range(iterations):
        instance.exports.fib(20)  # 20 iterations per call
    elapsed = time.perf_counter() - start
    return elapsed


def benchmark_recursive_calls(iterations: int = 1000) -> float:
    """Benchmark recursive function calls (factorial)."""
    module = decode_module(RECURSIVE_CALL_WASM)
    instance = instantiate(module)

    start = time.perf_counter()
    for _ in range(iterations):
        instance.exports.fact(10)  # 10 recursive calls per iteration
    elapsed = time.perf_counter() - start
    return elapsed


def benchmark_heavy_arithmetic(iterations: int = 1000) -> float:
    """Benchmark heavy arithmetic operations."""
    module = decode_module(HEAVY_ARITH_WASM)
    instance = instantiate(module)

    start = time.perf_counter()
    for i in range(iterations):
        instance.exports.heavy(i, i + 1)
    elapsed = time.perf_counter() - start
    return elapsed


def benchmark_nested_blocks(iterations: int = 10000) -> float:
    """Benchmark nested blocks and branching."""
    module = decode_module(NESTED_BLOCKS_WASM)
    instance = instantiate(module)

    start = time.perf_counter()
    for i in range(iterations):
        instance.exports.nested(i)
    elapsed = time.perf_counter() - start
    return elapsed


def run_all_benchmarks():
    """Run all benchmarks and print results."""
    print("=" * 60)
    print("Pure Python WASM Runtime Benchmarks")
    print("=" * 60)
    print()

    benchmarks = [
        ("LEB128 Decode (100k)", benchmark_leb128_decode, 100000),
        ("Module Decode (1k)", benchmark_decode_module, 1000),
        ("Instantiate (1k)", benchmark_instantiate, 1000),
        ("Simple Call (10k)", benchmark_simple_call, 10000),
        ("Loop Execution (1k x 20 iters)", benchmark_loop_execution, 1000),
        ("Recursive Calls (1k x 10 depth)", benchmark_recursive_calls, 1000),
        ("Heavy Arithmetic (1k x 100 ops)", benchmark_heavy_arithmetic, 1000),
        ("Nested Blocks (10k x 20 depth)", benchmark_nested_blocks, 10000),
    ]

    results = []
    for name, func, iters in benchmarks:
        elapsed = func(iters)
        ops_per_sec = iters / elapsed
        results.append((name, elapsed, ops_per_sec))
        print(f"{name:40} {elapsed:8.3f}s  ({ops_per_sec:,.0f} ops/sec)")

    print()
    return results


def profile_execution():
    """Profile execution to identify bottlenecks."""
    print("=" * 60)
    print("Profiling Loop Execution (fibonacci)")
    print("=" * 60)

    module = decode_module(FIB_WASM)
    instance = instantiate(module)

    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(500):
        instance.exports.fib(30)

    profiler.disable()

    s = StringIO()
    stats = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    stats.print_stats(30)
    print(s.getvalue())

    print("=" * 60)
    print("Profiling Heavy Arithmetic")
    print("=" * 60)

    module = decode_module(HEAVY_ARITH_WASM)
    instance = instantiate(module)

    profiler = cProfile.Profile()
    profiler.enable()

    for i in range(1000):
        instance.exports.heavy(i, i + 1)

    profiler.disable()

    s = StringIO()
    stats = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    stats.print_stats(30)
    print(s.getvalue())


if __name__ == "__main__":
    run_all_benchmarks()
    print()
    profile_execution()
