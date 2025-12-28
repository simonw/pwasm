#!/usr/bin/env python3
"""Profile MicroQuickJS to find performance bottlenecks."""

import pure_python_wasm
from pathlib import Path
import time
import cProfile
import pstats
from io import StringIO


def main():
    wasm_path = Path(__file__).parent / "tests" / "mquickjs_standalone.wasm"

    if not wasm_path.exists():
        print(f"Error: {wasm_path} not found")
        return 1

    print("Loading MicroQuickJS WASM module...")
    with open(wasm_path, "rb") as f:
        wasm_bytes = f.read()

    module = pure_python_wasm.decode_module(wasm_bytes)

    # Set up minimal imports
    imports = {
        "env": {
            "setTempRet0": lambda x: None,
            "getTempRet0": lambda: 0,
            "_emscripten_throw_longjmp": lambda: None,
            "invoke_iii": lambda *a: 0,
            "invoke_iiii": lambda *a: 0,
            "invoke_iiiii": lambda *a: 0,
            "invoke_vi": lambda *a: None,
            "invoke_vii": lambda *a: None,
            "invoke_viii": lambda *a: None,
            "invoke_viiiii": lambda *a: None,
            "invoke_viiiiii": lambda *a: None,
        },
        "wasi_snapshot_preview1": {
            "args_sizes_get": lambda *a: 0,
            "args_get": lambda *a: 0,
            "proc_exit": lambda *a: None,
            "fd_close": lambda *a: 0,
            "fd_write": lambda *a: 0,
            "fd_seek": lambda *a: 0,
        },
    }

    instance = pure_python_wasm.instantiate(module, imports)
    exports = {exp.name: exp.index for exp in instance.module.exports}

    # Profile _start function to see real execution characteristics
    print("\nProfiling _start function (the real bottleneck)...")
    print("Running for a few seconds to collect stats...")

    profiler = cProfile.Profile()
    profiler.enable()

    start = time.time()
    max_time = 5  # Run for 5 seconds max

    # Use optimized execute_function_fast
    from pure_python_wasm.executor import (
        Label,
        get_jump_targets,
        execute_instruction,
        execute_function_fast,
        MASK_32,
        MASK_64,
        find_end,
    )

    func_idx = exports["_start"]
    func = instance.funcs[func_idx]
    func_type = instance.func_types[func.type_idx]

    # Set up execution state
    locals_list = []
    for valtype in func.locals:
        if valtype in ("i32", "i64"):
            locals_list.append(0)
        elif valtype in ("f32", "f64"):
            locals_list.append(0.0)
        else:
            locals_list.append(None)

    stack = []
    labels = [
        Label(arity=len(func_type.results), target=len(func.body) - 1, stack_height=0)
    ]
    body = func.body
    body_len = len(body)
    jump_targets = get_jump_targets(func, body)

    # Cache method references for speed
    stack_append = stack.append
    stack_pop = stack.pop
    labels_append = labels.append
    labels_pop = labels.pop
    _MASK_32 = MASK_32
    _MASK_64 = MASK_64

    instruction_count = 0
    ip = 0
    while ip < body_len:
        instr = body[ip]
        op = instr.opcode
        operand = instr.operand
        ip += 1

        # Inline most common ops for profiling accuracy
        if op == "br":
            depth = operand
            label_idx = len(labels) - 1 - depth
            label = labels[label_idx]
            for _ in range(depth + 1):
                labels_pop()
            if label.arity > 0:
                result_values = stack[-label.arity :]
                del stack[label.stack_height :]
                stack.extend(result_values)
            else:
                del stack[label.stack_height :]
            if label.is_loop:
                labels_append(label)
            ip = label.target + 1
            instruction_count += 1
            continue

        if op == "br_if":
            condition = stack_pop()
            if condition:
                depth = operand
                label_idx = len(labels) - 1 - depth
                label = labels[label_idx]
                for _ in range(depth + 1):
                    labels_pop()
                if label.arity > 0:
                    result_values = stack[-label.arity :]
                    del stack[label.stack_height :]
                    stack.extend(result_values)
                else:
                    del stack[label.stack_height :]
                if label.is_loop:
                    labels_append(label)
                ip = label.target + 1
            instruction_count += 1
            continue

        if op == "local.get":
            stack_append(locals_list[operand])
            instruction_count += 1
            continue

        if op == "local.set":
            locals_list[operand] = stack_pop()
            instruction_count += 1
            continue

        if op == "i32.const":
            val = operand & _MASK_32
            if val >= 0x80000000:
                val -= 0x100000000
            stack_append(val)
            instruction_count += 1
            continue

        if op == "i32.add":
            b, a = stack_pop(), stack_pop()
            val = (a + b) & _MASK_32
            if val >= 0x80000000:
                val -= 0x100000000
            stack_append(val)
            instruction_count += 1
            continue

        if op == "loop":
            blocktype = operand
            if blocktype == ():
                arity = 0
            elif isinstance(blocktype, tuple):
                arity = 1
            elif isinstance(blocktype, int):
                arity = len(instance.func_types[blocktype].params)
            else:
                arity = 0
            end_ip = (
                jump_targets[ip - 1][1]
                if ip - 1 in jump_targets
                else find_end(body, ip - 1, jump_targets)
            )
            labels_append(
                Label(
                    arity=arity,
                    target=ip - 1,
                    is_loop=True,
                    stack_height=len(stack) - arity,
                )
            )
            instruction_count += 1
            continue

        if op == "end":
            if labels:
                labels_pop()
            instruction_count += 1
            continue

        # Fall back for other ops
        result = execute_instruction(
            instr, stack, labels, locals_list, instance, body, ip, jump_targets
        )

        if result is not None:
            if result[0] == "branch":
                ip = result[1]
            elif result[0] == "return":
                break
            elif result[0] == "call":
                call_result = execute_function_fast(instance, result[1], result[2])
                if call_result is not None:
                    if isinstance(call_result, tuple):
                        stack.extend(call_result)
                    else:
                        stack_append(call_result)

        instruction_count += 1

        # Check timeout every 50000 instructions
        if instruction_count % 50000 == 0:
            elapsed = time.time() - start
            if elapsed > max_time:
                print(
                    f"  Stopped after {instruction_count:,} instructions in {elapsed:.2f}s"
                )
                print(f"  Rate: {instruction_count / elapsed:,.0f} instructions/second")
                break

    profiler.disable()
    elapsed = time.time() - start
    if instruction_count % 50000 != 0:  # Didn't hit timeout
        print(f"  Completed {instruction_count:,} instructions in {elapsed:.2f}s")

    profiler.disable()

    # Show profiling results
    print("\n" + "=" * 60)
    print("PROFILING RESULTS")
    print("=" * 60)

    s = StringIO()
    stats = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    stats.print_stats(30)
    print(s.getvalue())

    s = StringIO()
    stats = pstats.Stats(profiler, stream=s).sort_stats("tottime")
    stats.print_stats(20)
    print("\nBy total time:")
    print(s.getvalue())

    return 0


if __name__ == "__main__":
    exit(main())
