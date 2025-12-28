#!/usr/bin/env python3
"""Benchmark NumPy executor vs standard executor."""

import pure_python_wasm
from pure_python_wasm.executor import (
    Label, get_jump_targets, MASK_32, find_end, execute_function_fast
)
from pure_python_wasm.executor_numpy import execute_function_numpy
from pure_python_wasm.errors import TrapError
from pathlib import Path
import time
import numpy as np


# Global counters
_instruction_count = 0
_start_time = 0
_max_time = 5.0
_check_interval = 50000
_should_stop = False


def execute_numpy_instrumented(instance, func_idx, args):
    """Instrumented NumPy executor for benchmarking."""
    global _instruction_count, _start_time, _should_stop

    # Check if imported
    if func_idx in instance.imported_funcs:
        return instance.imported_funcs[func_idx](*args)

    func = instance.funcs[func_idx]
    if func is None:
        raise TrapError(f"Function {func_idx} is not available")
    func_type = instance.func_types[func.type_idx]

    # Set up locals using numpy array
    n_locals = len(func_type.params) + len(func.locals)
    locals_arr = np.zeros(n_locals, dtype=np.int64)
    for i, arg in enumerate(args):
        locals_arr[i] = arg

    stack = np.zeros(1024, dtype=np.int64)
    sp = 0

    labels = [Label(arity=len(func_type.results), target=len(func.body) - 1, stack_height=0)]
    body = func.body
    body_len = len(body)
    jump_targets = get_jump_targets(func, body)

    ip = 0
    while ip < body_len:
        if _should_stop:
            raise TimeoutError("Timeout")

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
            print(f"  Progress: {_instruction_count:,} instructions, {rate:,.0f}/sec", flush=True)
            if elapsed > _max_time:
                _should_stop = True
                raise TimeoutError(f"Stopped after {_instruction_count:,} instructions")

        # Most common opcodes
        if op == "local.get":
            stack[sp] = locals_arr[operand]
            sp += 1
            continue

        if op == "local.set":
            sp -= 1
            locals_arr[operand] = stack[sp]
            continue

        if op == "local.tee":
            locals_arr[operand] = stack[sp - 1]
            continue

        if op == "i32.const":
            stack[sp] = np.int32(operand)
            sp += 1
            continue

        if op == "i32.add":
            sp -= 1
            b = np.int32(stack[sp])
            a = np.int32(stack[sp - 1])
            stack[sp - 1] = np.int32(a + b)
            continue

        if op == "i32.sub":
            sp -= 1
            b = np.int32(stack[sp])
            a = np.int32(stack[sp - 1])
            stack[sp - 1] = np.int32(a - b)
            continue

        if op == "i32.and":
            sp -= 1
            stack[sp - 1] = stack[sp - 1] & stack[sp]
            continue

        if op == "i32.or":
            sp -= 1
            stack[sp - 1] = stack[sp - 1] | stack[sp]
            continue

        if op == "i32.eqz":
            stack[sp - 1] = 1 if stack[sp - 1] == 0 else 0
            continue

        if op == "i32.eq":
            sp -= 1
            stack[sp - 1] = 1 if stack[sp - 1] == stack[sp] else 0
            continue

        if op == "i32.lt_u":
            sp -= 1
            a = np.uint32(stack[sp - 1])
            b = np.uint32(stack[sp])
            stack[sp - 1] = 1 if a < b else 0
            continue

        if op == "i32.gt_u":
            sp -= 1
            a = np.uint32(stack[sp - 1])
            b = np.uint32(stack[sp])
            stack[sp - 1] = 1 if a > b else 0
            continue

        if op == "i32.lt_s":
            sp -= 1
            a = np.int32(stack[sp - 1])
            b = np.int32(stack[sp])
            stack[sp - 1] = 1 if a < b else 0
            continue

        if op == "i32.gt_s":
            sp -= 1
            a = np.int32(stack[sp - 1])
            b = np.int32(stack[sp])
            stack[sp - 1] = 1 if a > b else 0
            continue

        if op == "i32.le_s":
            sp -= 1
            a = np.int32(stack[sp - 1])
            b = np.int32(stack[sp])
            stack[sp - 1] = 1 if a <= b else 0
            continue

        if op == "i32.le_u":
            sp -= 1
            a = np.uint32(stack[sp - 1])
            b = np.uint32(stack[sp])
            stack[sp - 1] = 1 if a <= b else 0
            continue

        if op == "i32.ge_s":
            sp -= 1
            a = np.int32(stack[sp - 1])
            b = np.int32(stack[sp])
            stack[sp - 1] = 1 if a >= b else 0
            continue

        if op == "i32.ge_u":
            sp -= 1
            a = np.uint32(stack[sp - 1])
            b = np.uint32(stack[sp])
            stack[sp - 1] = 1 if a >= b else 0
            continue

        if op == "i32.ne":
            sp -= 1
            stack[sp - 1] = 1 if stack[sp - 1] != stack[sp] else 0
            continue

        if op == "i32.shl":
            sp -= 1
            b = stack[sp] & 31
            a = np.uint32(stack[sp - 1])
            stack[sp - 1] = np.int32(a << b)
            continue

        if op == "i32.shr_u":
            sp -= 1
            b = stack[sp] & 31
            a = np.uint32(stack[sp - 1])
            stack[sp - 1] = np.int32(a >> b)
            continue

        if op == "i32.shr_s":
            sp -= 1
            b = stack[sp] & 31
            a = np.int32(stack[sp - 1])
            stack[sp - 1] = np.int32(a >> b)
            continue

        if op == "i32.mul":
            sp -= 1
            b = np.int32(stack[sp])
            a = np.int32(stack[sp - 1])
            stack[sp - 1] = np.int32(a * b)
            continue

        if op == "i32.xor":
            sp -= 1
            stack[sp - 1] = stack[sp - 1] ^ stack[sp]
            continue

        if op == "i32.load8_u":
            align, offset = operand
            addr = int(stack[sp - 1]) + offset
            stack[sp - 1] = instance.memories[0].data[addr]
            continue

        if op == "i32.load8_s":
            align, offset = operand
            addr = int(stack[sp - 1]) + offset
            val = instance.memories[0].data[addr]
            if val >= 128:
                val -= 256
            stack[sp - 1] = val
            continue

        if op == "i32.store8":
            align, offset = operand
            sp -= 2
            val = int(stack[sp + 1]) & 0xFF
            addr = int(stack[sp]) + offset
            instance.memories[0].data[addr] = val
            continue

        if op == "select":
            sp -= 1
            c = stack[sp]
            sp -= 1
            v2 = stack[sp]
            v1 = stack[sp - 1]
            stack[sp - 1] = v1 if c else v2
            continue

        if op == "i32.load":
            align, offset = operand
            addr = int(stack[sp - 1]) + offset
            mem = instance.memories[0].data
            val = int.from_bytes(mem[addr:addr + 4], "little", signed=True)
            stack[sp - 1] = val
            continue

        if op == "i32.store":
            align, offset = operand
            sp -= 2
            val = int(stack[sp + 1]) & MASK_32
            addr = int(stack[sp]) + offset
            mem = instance.memories[0].data
            mem[addr:addr + 4] = val.to_bytes(4, "little")
            continue

        if op == "global.get":
            stack[sp] = instance.globals[operand].value
            sp += 1
            continue

        if op == "global.set":
            sp -= 1
            instance.globals[operand].value = int(stack[sp])
            continue

        if op == "br":
            depth = operand
            label_idx = len(labels) - 1 - depth
            label = labels[label_idx]
            for _ in range(depth + 1):
                labels.pop()
            if label.arity > 0:
                result_sp = sp - label.arity
                for i in range(label.arity):
                    stack[label.stack_height + i] = stack[result_sp + i]
                sp = label.stack_height + label.arity
            else:
                sp = label.stack_height
            if label.is_loop:
                labels.append(label)
            ip = label.target + 1
            continue

        if op == "br_if":
            sp -= 1
            if stack[sp]:
                depth = operand
                label_idx = len(labels) - 1 - depth
                label = labels[label_idx]
                for _ in range(depth + 1):
                    labels.pop()
                if label.arity > 0:
                    result_sp = sp - label.arity
                    for i in range(label.arity):
                        stack[label.stack_height + i] = stack[result_sp + i]
                    sp = label.stack_height + label.arity
                else:
                    sp = label.stack_height
                if label.is_loop:
                    labels.append(label)
                ip = label.target + 1
            continue

        if op == "block":
            blocktype = operand
            arity = 0 if blocktype == () else 1 if isinstance(blocktype, tuple) else 0
            end_ip = jump_targets[ip - 1][1] if ip - 1 in jump_targets else find_end(body, ip - 1, jump_targets)
            labels.append(Label(arity=arity, target=end_ip, stack_height=sp))
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
            labels.append(Label(arity=arity, target=ip - 1, is_loop=True, stack_height=sp - arity))
            continue

        if op == "if":
            sp -= 1
            condition = stack[sp]
            blocktype = operand
            arity = 0 if blocktype == () else 1 if isinstance(blocktype, tuple) else 0
            if ip - 1 in jump_targets:
                else_ip, end_ip = jump_targets[ip - 1]
            else:
                from pure_python_wasm.executor import find_else_end
                else_ip, end_ip = find_else_end(body, ip - 1, jump_targets)
            if condition:
                labels.append(Label(arity=arity, target=end_ip, stack_height=sp))
            else:
                labels.append(Label(arity=arity, target=end_ip, stack_height=sp))
                if else_ip is not None:
                    ip = else_ip + 1
                else:
                    ip = end_ip
            continue

        if op == "else":
            if labels:
                label = labels[-1]
                ip = label.target
            continue

        if op == "end":
            if labels:
                labels.pop()
            continue

        if op == "drop":
            sp -= 1
            continue

        if op == "call":
            call_func_idx = operand
            called_func = instance.funcs[call_func_idx]
            if called_func is None:
                import_idx = call_func_idx
                imp = instance.module.imports[import_idx]
                call_func_type = instance.func_types[imp.desc]
            else:
                call_func_type = instance.func_types[called_func.type_idx]
            n_params = len(call_func_type.params)
            call_args = [int(stack[sp - n_params + i]) for i in range(n_params)]
            sp -= n_params
            call_result = execute_numpy_instrumented(instance, call_func_idx, call_args)
            if call_result is not None:
                if isinstance(call_result, tuple):
                    for r in call_result:
                        stack[sp] = r
                        sp += 1
                else:
                    stack[sp] = call_result
                    sp += 1
            continue

        if op == "nop":
            continue

        if op == "return":
            break

        if op == "i64.const":
            stack[sp] = operand
            sp += 1
            continue

        raise TrapError(f"Unimplemented: {op}")

    num_results = len(func_type.results)
    if num_results == 0:
        return None
    elif num_results == 1:
        return int(stack[sp - 1]) if sp > 0 else None
    else:
        return tuple(int(stack[sp - num_results + i]) for i in range(num_results))


def main():
    global _instruction_count, _start_time, _should_stop

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

    print("\nBenchmarking NumPy executor on _start function...", flush=True)

    _instruction_count = 0
    _start_time = time.monotonic()
    _should_stop = False

    try:
        execute_numpy_instrumented(instance, exports["_start"], [])
        elapsed = time.monotonic() - _start_time
        rate = _instruction_count / elapsed if elapsed > 0 else 0
        print(f"\n  Completed {_instruction_count:,} instructions in {elapsed:.2f}s")
        print(f"  Rate: {rate:,.0f} instructions/second")
    except TimeoutError as e:
        elapsed = time.monotonic() - _start_time
        rate = _instruction_count / elapsed if elapsed > 0 else 0
        print(f"\n  {e}")
        print(f"  Rate: {rate:,.0f} instructions/second")
    except Exception as e:
        elapsed = time.monotonic() - _start_time
        rate = _instruction_count / elapsed if elapsed > 0 else 0
        print(f"\n  Error after {_instruction_count:,} instructions in {elapsed:.2f}s: {e}")
        print(f"  Rate: {rate:,.0f} instructions/second")

    return 0


if __name__ == "__main__":
    exit(main())
