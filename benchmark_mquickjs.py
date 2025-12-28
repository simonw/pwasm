#!/usr/bin/env python3
"""Benchmark MicroQuickJS execution speed with global instruction counting."""

import pure_python_wasm
from pure_python_wasm.executor import (
    Label, get_jump_targets, execute_instruction, MASK_32, find_end
)
from pure_python_wasm.errors import TrapError
from pathlib import Path
import time


# Global counters for benchmarking
_instruction_count = 0
_start_time = 0
_max_time = 5.0
_check_interval = 50000
_last_report = 0
_should_stop = False
op_counts = {}


def execute_function_instrumented(instance, func_idx, args):
    """Instrumented function execution that counts all instructions globally."""
    global _instruction_count, _start_time, _last_report, _should_stop, _max_time, _check_interval, op_counts

    # Check if this is an imported function
    if func_idx in instance.imported_funcs:
        return instance.imported_funcs[func_idx](*args)

    func = instance.funcs[func_idx]
    if func is None:
        raise TrapError(f"Function {func_idx} is not available")
    func_type = instance.func_types[func.type_idx]

    # Set up locals
    locals_list = list(args)
    for valtype in func.locals:
        if valtype in ("i32", "i64"):
            locals_list.append(0)
        elif valtype in ("f32", "f64"):
            locals_list.append(0.0)
        else:
            locals_list.append(None)

    stack = []
    labels = []
    body = func.body
    body_len = len(body)

    labels.append(Label(arity=len(func_type.results), target=body_len - 1, stack_height=0))
    jump_targets = get_jump_targets(func, body)

    # Cache for speed
    stack_append = stack.append
    stack_pop = stack.pop
    labels_append = labels.append
    labels_pop = labels.pop
    _MASK_32 = MASK_32

    ip = 0
    while ip < body_len:
        if _should_stop:
            raise TimeoutError("Execution timeout")

        instr = body[ip]
        op = instr.opcode
        operand = instr.operand
        ip += 1
        _instruction_count += 1

        # Progress check
        if _instruction_count % _check_interval == 0:
            now = time.monotonic()
            elapsed = now - _start_time
            rate = _instruction_count / elapsed if elapsed > 0 else 0
            if now - _last_report >= 1.0:
                print(f"  Progress: {_instruction_count:,} instructions, {rate:,.0f}/sec", flush=True)
                _last_report = now
            if elapsed > _max_time:
                _should_stop = True
                raise TimeoutError(f"Stopped after {_instruction_count:,} instructions in {elapsed:.2f}s\n  Rate: {rate:,.0f}/sec")

        # Inline most common ops
        if op == "br":
            depth = operand
            label_idx = len(labels) - 1 - depth
            label = labels[label_idx]
            for _ in range(depth + 1):
                labels_pop()
            if label.arity > 0:
                result_values = stack[-label.arity:]
                del stack[label.stack_height:]
                stack.extend(result_values)
            else:
                del stack[label.stack_height:]
            if label.is_loop:
                labels_append(label)
            ip = label.target + 1
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
                    result_values = stack[-label.arity:]
                    del stack[label.stack_height:]
                    stack.extend(result_values)
                else:
                    del stack[label.stack_height:]
                if label.is_loop:
                    labels_append(label)
                ip = label.target + 1
            continue

        if op == "local.get":
            stack_append(locals_list[operand])
            continue

        if op == "local.set":
            locals_list[operand] = stack_pop()
            continue

        if op == "local.tee":
            locals_list[operand] = stack[-1]
            continue

        if op == "i32.const":
            val = operand & _MASK_32
            if val >= 0x80000000:
                val -= 0x100000000
            stack_append(val)
            continue

        if op == "i32.add":
            b, a = stack_pop(), stack_pop()
            val = (a + b) & _MASK_32
            if val >= 0x80000000:
                val -= 0x100000000
            stack_append(val)
            continue

        if op == "i32.sub":
            b, a = stack_pop(), stack_pop()
            val = (a - b) & _MASK_32
            if val >= 0x80000000:
                val -= 0x100000000
            stack_append(val)
            continue

        if op == "i32.eqz":
            stack_append(1 if stack_pop() == 0 else 0)
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
            end_ip = jump_targets[ip - 1][1] if ip - 1 in jump_targets else find_end(body, ip - 1, jump_targets)
            labels_append(Label(arity=arity, target=ip - 1, is_loop=True, stack_height=len(stack) - arity))
            continue

        if op == "block":
            blocktype = operand
            arity = 0 if blocktype == () else 1 if isinstance(blocktype, tuple) else 0
            end_ip = jump_targets[ip - 1][1] if ip - 1 in jump_targets else find_end(body, ip - 1, jump_targets)
            labels_append(Label(arity=arity, target=end_ip, stack_height=len(stack)))
            continue

        if op == "end":
            if labels:
                labels_pop()
            continue

        if op == "drop":
            stack_pop()
            continue

        if op == "nop":
            continue

        if op == "return":
            break

        # Track slow path ops
        op_counts[op] = op_counts.get(op, 0) + 1

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
                call_result = execute_function_instrumented(instance, result[1], result[2])
                if call_result is not None:
                    if isinstance(call_result, tuple):
                        stack.extend(call_result)
                    else:
                        stack_append(call_result)

    # Return results
    num_results = len(func_type.results)
    if num_results == 0:
        return None
    elif num_results == 1:
        return stack[-1] if stack else None
    else:
        return tuple(stack[-num_results:])


def main():
    global _instruction_count, _start_time, _last_report, _should_stop

    wasm_path = Path(__file__).parent / "tests" / "mquickjs_standalone.wasm"

    if not wasm_path.exists():
        print(f"Error: {wasm_path} not found")
        return 1

    print("Loading MicroQuickJS WASM module...")
    with open(wasm_path, "rb") as f:
        wasm_bytes = f.read()

    module = pure_python_wasm.decode_module(wasm_bytes)

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

    print("\nBenchmarking _start function execution...", flush=True)

    _instruction_count = 0
    _start_time = time.monotonic()
    _last_report = _start_time
    _should_stop = False

    try:
        execute_function_instrumented(instance, exports["_start"], [])
        elapsed = time.monotonic() - _start_time
        rate = _instruction_count / elapsed if elapsed > 0 else 0
        print(f"\n  Completed {_instruction_count:,} instructions in {elapsed:.2f}s")
        print(f"  Rate: {rate:,.0f} instructions/second")
    except TimeoutError as e:
        print(f"\n  {e}")
        if op_counts:
            print(f"\n  Slow path opcodes:")
            for op, cnt in sorted(op_counts.items(), key=lambda x: -x[1])[:15]:
                print(f"    {op}: {cnt:,}")
    except Exception as e:
        elapsed = time.monotonic() - _start_time
        rate = _instruction_count / elapsed if elapsed > 0 else 0
        print(f"\n  Error after {_instruction_count:,} instructions in {elapsed:.2f}s: {e}")
        print(f"  Rate: {rate:,.0f} instructions/second")

    return 0


if __name__ == "__main__":
    exit(main())
