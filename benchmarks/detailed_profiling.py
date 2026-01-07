"""Detailed profiling to identify remaining optimization opportunities."""

import cProfile
import pstats
from io import StringIO
import time

from pure_python_wasm import decode_module
from pure_python_wasm.executor import instantiate

# Import the benchmark WASM modules
from benchmark_wasm import FIB_WASM, HEAVY_ARITH_WASM, RECURSIVE_CALL_WASM


def profile_with_line_stats():
    """Profile with detailed line-level statistics."""

    print("=" * 70)
    print("DETAILED PROFILING - Fibonacci (Loop-heavy)")
    print("=" * 70)

    module = decode_module(FIB_WASM)
    instance = instantiate(module)

    # Warm up
    for _ in range(10):
        instance.exports.fib(30)

    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(1000):
        instance.exports.fib(30)

    profiler.disable()

    s = StringIO()
    stats = pstats.Stats(profiler, stream=s).sort_stats("tottime")
    stats.print_stats(20)
    print(s.getvalue())

    # Print callers for execute_function
    print("\n--- Callers of execute_function ---")
    s = StringIO()
    stats = pstats.Stats(profiler, stream=s)
    stats.print_callers("execute_function")
    print(s.getvalue())

    print("\n" + "=" * 70)
    print("DETAILED PROFILING - Heavy Arithmetic")
    print("=" * 70)

    module = decode_module(HEAVY_ARITH_WASM)
    instance = instantiate(module)

    profiler = cProfile.Profile()
    profiler.enable()

    for i in range(2000):
        instance.exports.heavy(i, i + 1)

    profiler.disable()

    s = StringIO()
    stats = pstats.Stats(profiler, stream=s).sort_stats("tottime")
    stats.print_stats(20)
    print(s.getvalue())

    print("\n" + "=" * 70)
    print("DETAILED PROFILING - Recursive Calls")
    print("=" * 70)

    module = decode_module(RECURSIVE_CALL_WASM)
    instance = instantiate(module)

    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(2000):
        instance.exports.fact(12)  # 12! = 479001600

    profiler.disable()

    s = StringIO()
    stats = pstats.Stats(profiler, stream=s).sort_stats("tottime")
    stats.print_stats(20)
    print(s.getvalue())


def measure_overhead():
    """Measure specific overhead sources."""

    print("\n" + "=" * 70)
    print("OVERHEAD ANALYSIS")
    print("=" * 70)

    module = decode_module(FIB_WASM)
    instance = instantiate(module)

    # Measure wrapper overhead
    iterations = 10000

    # Direct execute_function call
    from pure_python_wasm.executor import execute_function

    start = time.perf_counter()
    for _ in range(iterations):
        execute_function(instance, 0, [30])
    direct_time = time.perf_counter() - start

    # Via wrapper
    start = time.perf_counter()
    for _ in range(iterations):
        instance.exports.fib(30)
    wrapper_time = time.perf_counter() - start

    print(f"Direct execute_function: {direct_time:.4f}s")
    print(f"Via exports wrapper:     {wrapper_time:.4f}s")
    print(
        f"Wrapper overhead:        {(wrapper_time - direct_time) / iterations * 1000000:.2f}µs per call"
    )

    # Measure list operations overhead
    print("\n--- List Operation Costs ---")

    stack = []
    iterations = 1000000

    start = time.perf_counter()
    for i in range(iterations):
        stack.append(i)
    append_time = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(iterations):
        stack.pop()
    pop_time = time.perf_counter() - start

    print(f"list.append: {append_time / iterations * 1000000:.3f}µs per op")
    print(f"list.pop:    {pop_time / iterations * 1000000:.3f}µs per op")

    # Measure string comparison overhead
    print("\n--- String Comparison Costs ---")

    opcodes = ["local.get", "local.set", "i32.add", "i32.const", "br_if", "end"]
    iterations = 1000000

    start = time.perf_counter()
    for _ in range(iterations):
        op = "i32.add"
        if op == "local.get":
            pass
        elif op == "local.set":
            pass
        elif op == "i32.const":
            pass
        elif op == "i32.add":
            pass
    string_compare_time = time.perf_counter() - start

    # With integer opcodes
    OP_LOCAL_GET = 0
    OP_LOCAL_SET = 1
    OP_I32_CONST = 2
    OP_I32_ADD = 3

    start = time.perf_counter()
    for _ in range(iterations):
        op = OP_I32_ADD
        if op == OP_LOCAL_GET:
            pass
        elif op == OP_LOCAL_SET:
            pass
        elif op == OP_I32_CONST:
            pass
        elif op == OP_I32_ADD:
            pass
    int_compare_time = time.perf_counter() - start

    print(f"String opcode comparison: {string_compare_time:.4f}s for {iterations} ops")
    print(f"Integer opcode comparison: {int_compare_time:.4f}s for {iterations} ops")
    print(
        f"String comparison overhead: {(string_compare_time - int_compare_time) / string_compare_time * 100:.1f}%"
    )

    # Measure attribute access overhead
    print("\n--- Attribute Access Costs ---")

    class Instr:
        def __init__(self, opcode, operand):
            self.opcode = opcode
            self.operand = operand

    class InstrSlots:
        __slots__ = ("opcode", "operand")

        def __init__(self, opcode, operand):
            self.opcode = opcode
            self.operand = operand

    instr = Instr("i32.add", 42)
    instr_slots = InstrSlots("i32.add", 42)
    instr_tuple = ("i32.add", 42)

    iterations = 1000000

    start = time.perf_counter()
    for _ in range(iterations):
        _ = instr.opcode
        _ = instr.operand
    class_time = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(iterations):
        _ = instr_slots.opcode
        _ = instr_slots.operand
    slots_time = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(iterations):
        _ = instr_tuple[0]
        _ = instr_tuple[1]
    tuple_time = time.perf_counter() - start

    print(f"Regular class attr:  {class_time:.4f}s")
    print(f"Slots class attr:    {slots_time:.4f}s")
    print(f"Tuple indexing:      {tuple_time:.4f}s")
    print(
        f"Potential speedup with tuples: {(class_time - tuple_time) / class_time * 100:.1f}%"
    )


def profile_instruction_frequency():
    """Profile which instructions are executed most frequently."""

    print("\n" + "=" * 70)
    print("INSTRUCTION FREQUENCY ANALYSIS")
    print("=" * 70)

    # Patch execute_function to count instructions
    from pure_python_wasm import executor

    instruction_counts = {}
    original_execute = executor.execute_function

    def counting_execute(instance, func_idx, args):
        func = instance.funcs[func_idx]
        body = func.body

        for instr in body:
            op = instr.opcode
            instruction_counts[op] = instruction_counts.get(op, 0) + 1

        return original_execute(instance, func_idx, args)

    # Run fibonacci
    module = decode_module(FIB_WASM)
    instance = instantiate(module)

    executor.execute_function = counting_execute

    for _ in range(100):
        instance.exports.fib(20)

    executor.execute_function = original_execute

    print("\nFibonacci instruction frequency (static count x 100 calls):")
    for op, count in sorted(instruction_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {op:20} {count:8}")


if __name__ == "__main__":
    profile_with_line_stats()
    measure_overhead()
    profile_instruction_frequency()
