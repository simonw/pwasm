#!/usr/bin/env python3
"""Benchmark the optimized MicroQuickJS WASM build."""

import pure_python_wasm
from pure_python_wasm.executor import (
    Label,
    get_jump_targets,
    execute_instruction,
    MASK_32,
    find_end,
)
from pure_python_wasm.errors import TrapError
from pathlib import Path
import time


class LongjmpException(Exception):
    pass


class EmscriptenRuntime:
    def __init__(self):
        self.temp_ret0 = 0
        self.instance = None

    def set_instance(self, instance):
        self.instance = instance

    def _emscripten_throw_longjmp(self, *args):
        raise LongjmpException()

    def _make_invoke(self, signature):
        def invoke_func(func_idx, *args):
            try:
                if self.instance is None:
                    return 0
                table = self.instance.tables[0]
                if func_idx < 0 or func_idx >= len(table.elements):
                    return 0
                actual_func_idx = table.elements[func_idx]
                if actual_func_idx is None:
                    return 0
                result = execute_instrumented(
                    self.instance, actual_func_idx, list(args)
                )
                return result if result is not None else 0
            except LongjmpException:
                return 0
            except Exception:
                return 0

        return invoke_func


# Global counters
_instruction_count = 0
_start_time = 0
_max_time = 120.0  # 2 minutes - enough for simple JS
_check_interval = 1000000  # Less frequent reporting


def execute_instrumented(instance, func_idx, args):
    """Instrumented function execution with global counting."""
    global _instruction_count, _start_time

    if func_idx in instance.imported_funcs:
        return instance.imported_funcs[func_idx](*args)

    func = instance.funcs[func_idx]
    if func is None:
        raise TrapError(f"Function {func_idx} is not available")
    func_type = instance.func_types[func.type_idx]

    locals_list = list(args)
    for valtype in func.locals:
        locals_list.append(0 if valtype in ("i32", "i64") else 0.0)

    stack = []
    labels = [
        Label(arity=len(func_type.results), target=len(func.body) - 1, stack_height=0)
    ]
    body = func.body
    body_len = len(body)
    jump_targets = get_jump_targets(func, body)

    stack_append = stack.append
    stack_pop = stack.pop
    _MASK_32 = MASK_32

    ip = 0
    while ip < body_len:
        instr = body[ip]
        op = instr.opcode
        operand = instr.operand
        ip += 1
        _instruction_count += 1

        if _instruction_count % _check_interval == 0:
            now = time.monotonic()
            elapsed = now - _start_time
            rate = _instruction_count / elapsed if elapsed > 0 else 0
            print(f"  {_instruction_count:,} instructions, {rate:,.0f}/sec", flush=True)
            if elapsed > _max_time:
                raise TimeoutError(f"Stopped after {_instruction_count:,} instructions")

        # Inline common ops for speed
        if op == "local.get":
            stack_append(locals_list[operand])
            continue
        if op == "local.set":
            locals_list[operand] = stack_pop()
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
        if op == "end":
            if labels:
                labels.pop()
            continue
        if op == "nop":
            continue
        if op == "return":
            break

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
                call_result = execute_instrumented(instance, result[1], result[2])
                if call_result is not None:
                    if isinstance(call_result, tuple):
                        stack.extend(call_result)
                    else:
                        stack_append(call_result)

    num_results = len(func_type.results)
    if num_results == 0:
        return None
    elif num_results == 1:
        return stack[-1] if stack else None
    else:
        return tuple(stack[-num_results:])


def main():
    global _instruction_count, _start_time

    script_dir = Path(__file__).parent.parent
    wasm_path = script_dir / "tests" / "mquickjs_optimized.wasm"

    print("Loading optimized MicroQuickJS WASM module...")
    with open(wasm_path, "rb") as f:
        wasm_bytes = f.read()

    module = pure_python_wasm.decode_module(wasm_bytes)
    print(f"  {len(wasm_bytes):,} bytes, {len(module.funcs)} functions")

    runtime = EmscriptenRuntime()
    imports = {
        "a": {
            "a": lambda a, b, c, d: None,
            "b": lambda a, b, c, d: None,
            "c": runtime._emscripten_throw_longjmp,
            "d": runtime._make_invoke("iii"),
            "e": lambda *args: None,
            "f": lambda *args: None,
            "g": lambda *args: None,
            "h": runtime._make_invoke("iiii"),
            "i": runtime._make_invoke("iiiii"),
            "j": lambda a, b: 0,
            "k": lambda *args: None,
            "l": lambda a: 0,
            "m": lambda: None,
            "n": lambda: None,
            "o": lambda: 0.0,
        }
    }

    instance = pure_python_wasm.instantiate(module, imports)
    runtime.set_instance(instance)
    exports = {exp.name: exp.index for exp in instance.module.exports}

    # Call __wasm_call_ctors
    print("\nCalling __wasm_call_ctors...")
    execute_instrumented(instance, exports["q"], [])

    # Call _mquickjs_init
    print("Calling _mquickjs_init...")
    result = execute_instrumented(instance, exports["r"], [])
    print(f"  returned {result}")

    # Run simple JS
    print("\nRunning JavaScript: 42")
    js_code = b"42\x00"

    memory = instance.memories[0].data
    code_ptr = pure_python_wasm.execute_function(instance, exports["A"], [len(js_code)])
    memory[code_ptr : code_ptr + len(js_code)] = js_code
    print(f"  Code at {code_ptr}")

    _instruction_count = 0
    _start_time = time.monotonic()

    try:
        result = execute_instrumented(instance, exports["v"], [code_ptr])
        elapsed = time.monotonic() - _start_time
        rate = _instruction_count / elapsed if elapsed > 0 else 0
        print(f"\nCompleted in {elapsed:.2f}s")
        print(f"Instructions: {_instruction_count:,}")
        print(f"Rate: {rate:,.0f}/sec")
        print(f"Result: {result}")
    except TimeoutError as e:
        print(f"\n{e}")
    except LongjmpException:
        elapsed = time.monotonic() - _start_time
        rate = _instruction_count / elapsed if elapsed > 0 else 0
        print(f"\nLongjmp after {elapsed:.2f}s")
        print(f"Instructions: {_instruction_count:,}")
        print(f"Rate: {rate:,.0f}/sec")

    return 0


if __name__ == "__main__":
    exit(main())
