#!/usr/bin/env python3
"""Benchmark dictionary dispatch vs if-elif chains."""

import pure_python_wasm
from pure_python_wasm.executor import Label, get_jump_targets, find_end, MASK_32
from pure_python_wasm.errors import TrapError
from pathlib import Path
import time


# Global counters
_instruction_count = 0
_start_time = 0
_max_time = 5.0
_check_interval = 50000
_should_stop = False


def make_handler_dict():
    """Create a dictionary of opcode handlers."""

    def h_local_get(
        operand, stack, sp, locals_arr, labels, instance, body, ip, jump_targets
    ):
        stack[sp] = locals_arr[operand]
        return sp + 1, ip

    def h_local_set(
        operand, stack, sp, locals_arr, labels, instance, body, ip, jump_targets
    ):
        locals_arr[operand] = stack[sp - 1]
        return sp - 1, ip

    def h_local_tee(
        operand, stack, sp, locals_arr, labels, instance, body, ip, jump_targets
    ):
        locals_arr[operand] = stack[sp - 1]
        return sp, ip

    def h_i32_const(
        operand, stack, sp, locals_arr, labels, instance, body, ip, jump_targets
    ):
        val = operand & MASK_32
        if val >= 0x80000000:
            val -= 0x100000000
        stack[sp] = val
        return sp + 1, ip

    def h_i32_add(
        operand, stack, sp, locals_arr, labels, instance, body, ip, jump_targets
    ):
        b = stack[sp - 1]
        a = stack[sp - 2]
        val = (a + b) & MASK_32
        if val >= 0x80000000:
            val -= 0x100000000
        stack[sp - 2] = val
        return sp - 1, ip

    def h_i32_sub(
        operand, stack, sp, locals_arr, labels, instance, body, ip, jump_targets
    ):
        b = stack[sp - 1]
        a = stack[sp - 2]
        val = (a - b) & MASK_32
        if val >= 0x80000000:
            val -= 0x100000000
        stack[sp - 2] = val
        return sp - 1, ip

    def h_i32_mul(
        operand, stack, sp, locals_arr, labels, instance, body, ip, jump_targets
    ):
        b = stack[sp - 1]
        a = stack[sp - 2]
        val = (a * b) & MASK_32
        if val >= 0x80000000:
            val -= 0x100000000
        stack[sp - 2] = val
        return sp - 1, ip

    def h_i32_and(
        operand, stack, sp, locals_arr, labels, instance, body, ip, jump_targets
    ):
        stack[sp - 2] = stack[sp - 2] & stack[sp - 1]
        return sp - 1, ip

    def h_i32_or(
        operand, stack, sp, locals_arr, labels, instance, body, ip, jump_targets
    ):
        stack[sp - 2] = stack[sp - 2] | stack[sp - 1]
        return sp - 1, ip

    def h_i32_xor(
        operand, stack, sp, locals_arr, labels, instance, body, ip, jump_targets
    ):
        stack[sp - 2] = stack[sp - 2] ^ stack[sp - 1]
        return sp - 1, ip

    def h_i32_shl(
        operand, stack, sp, locals_arr, labels, instance, body, ip, jump_targets
    ):
        b = stack[sp - 1] & 31
        a = stack[sp - 2] & MASK_32
        val = (a << b) & MASK_32
        if val >= 0x80000000:
            val -= 0x100000000
        stack[sp - 2] = val
        return sp - 1, ip

    def h_i32_shr_u(
        operand, stack, sp, locals_arr, labels, instance, body, ip, jump_targets
    ):
        b = stack[sp - 1] & 31
        a = stack[sp - 2] & MASK_32
        val = a >> b
        if val >= 0x80000000:
            val -= 0x100000000
        stack[sp - 2] = val
        return sp - 1, ip

    def h_i32_eqz(
        operand, stack, sp, locals_arr, labels, instance, body, ip, jump_targets
    ):
        stack[sp - 1] = 1 if stack[sp - 1] == 0 else 0
        return sp, ip

    def h_i32_eq(
        operand, stack, sp, locals_arr, labels, instance, body, ip, jump_targets
    ):
        stack[sp - 2] = 1 if stack[sp - 2] == stack[sp - 1] else 0
        return sp - 1, ip

    def h_i32_ne(
        operand, stack, sp, locals_arr, labels, instance, body, ip, jump_targets
    ):
        stack[sp - 2] = 1 if stack[sp - 2] != stack[sp - 1] else 0
        return sp - 1, ip

    def h_i32_lt_u(
        operand, stack, sp, locals_arr, labels, instance, body, ip, jump_targets
    ):
        a = stack[sp - 2] & MASK_32
        b = stack[sp - 1] & MASK_32
        stack[sp - 2] = 1 if a < b else 0
        return sp - 1, ip

    def h_i32_gt_u(
        operand, stack, sp, locals_arr, labels, instance, body, ip, jump_targets
    ):
        a = stack[sp - 2] & MASK_32
        b = stack[sp - 1] & MASK_32
        stack[sp - 2] = 1 if a > b else 0
        return sp - 1, ip

    def h_i32_le_u(
        operand, stack, sp, locals_arr, labels, instance, body, ip, jump_targets
    ):
        a = stack[sp - 2] & MASK_32
        b = stack[sp - 1] & MASK_32
        stack[sp - 2] = 1 if a <= b else 0
        return sp - 1, ip

    def h_i32_ge_u(
        operand, stack, sp, locals_arr, labels, instance, body, ip, jump_targets
    ):
        a = stack[sp - 2] & MASK_32
        b = stack[sp - 1] & MASK_32
        stack[sp - 2] = 1 if a >= b else 0
        return sp - 1, ip

    def h_i32_lt_s(
        operand, stack, sp, locals_arr, labels, instance, body, ip, jump_targets
    ):
        a = stack[sp - 2]
        b = stack[sp - 1]
        stack[sp - 2] = 1 if a < b else 0
        return sp - 1, ip

    def h_i32_gt_s(
        operand, stack, sp, locals_arr, labels, instance, body, ip, jump_targets
    ):
        a = stack[sp - 2]
        b = stack[sp - 1]
        stack[sp - 2] = 1 if a > b else 0
        return sp - 1, ip

    def h_i32_load(
        operand, stack, sp, locals_arr, labels, instance, body, ip, jump_targets
    ):
        align, offset = operand
        addr = stack[sp - 1] + offset
        mem = instance.memories[0].data
        val = int.from_bytes(mem[addr : addr + 4], "little", signed=True)
        stack[sp - 1] = val
        return sp, ip

    def h_i32_load8_u(
        operand, stack, sp, locals_arr, labels, instance, body, ip, jump_targets
    ):
        align, offset = operand
        addr = stack[sp - 1] + offset
        stack[sp - 1] = instance.memories[0].data[addr]
        return sp, ip

    def h_i32_store(
        operand, stack, sp, locals_arr, labels, instance, body, ip, jump_targets
    ):
        align, offset = operand
        val = stack[sp - 1] & MASK_32
        addr = stack[sp - 2] + offset
        mem = instance.memories[0].data
        mem[addr : addr + 4] = val.to_bytes(4, "little")
        return sp - 2, ip

    def h_i32_store8(
        operand, stack, sp, locals_arr, labels, instance, body, ip, jump_targets
    ):
        align, offset = operand
        val = stack[sp - 1] & 0xFF
        addr = stack[sp - 2] + offset
        instance.memories[0].data[addr] = val
        return sp - 2, ip

    def h_global_get(
        operand, stack, sp, locals_arr, labels, instance, body, ip, jump_targets
    ):
        stack[sp] = instance.globals[operand].value
        return sp + 1, ip

    def h_global_set(
        operand, stack, sp, locals_arr, labels, instance, body, ip, jump_targets
    ):
        instance.globals[operand].value = stack[sp - 1]
        return sp - 1, ip

    def h_block(
        operand, stack, sp, locals_arr, labels, instance, body, ip, jump_targets
    ):
        arity = 0 if operand == () else 1 if isinstance(operand, tuple) else 0
        end_ip = (
            jump_targets[ip - 1][1]
            if ip - 1 in jump_targets
            else find_end(body, ip - 1, jump_targets)
        )
        labels.append(Label(arity=arity, target=end_ip, stack_height=sp))
        return sp, ip

    def h_loop(
        operand, stack, sp, locals_arr, labels, instance, body, ip, jump_targets
    ):
        if operand == ():
            arity = 0
        elif isinstance(operand, tuple):
            arity = 1
        elif isinstance(operand, int):
            arity = len(instance.func_types[operand].params)
        else:
            arity = 0
        labels.append(
            Label(arity=arity, target=ip - 1, is_loop=True, stack_height=sp - arity)
        )
        return sp, ip

    def h_if(operand, stack, sp, locals_arr, labels, instance, body, ip, jump_targets):
        condition = stack[sp - 1]
        sp -= 1
        arity = 0 if operand == () else 1 if isinstance(operand, tuple) else 0
        if ip - 1 in jump_targets:
            else_ip, end_ip = jump_targets[ip - 1]
        else:
            from pure_python_wasm.executor import find_else_end

            else_ip, end_ip = find_else_end(body, ip - 1, jump_targets)
        labels.append(Label(arity=arity, target=end_ip, stack_height=sp))
        if not condition:
            if else_ip is not None:
                ip = else_ip + 1
            else:
                ip = end_ip
        return sp, ip

    def h_else(
        operand, stack, sp, locals_arr, labels, instance, body, ip, jump_targets
    ):
        if labels:
            label = labels[-1]
            ip = label.target
        return sp, ip

    def h_end(operand, stack, sp, locals_arr, labels, instance, body, ip, jump_targets):
        if labels:
            labels.pop()
        return sp, ip

    def h_br(operand, stack, sp, locals_arr, labels, instance, body, ip, jump_targets):
        depth = operand
        label_idx = len(labels) - 1 - depth
        label = labels[label_idx]
        for _ in range(depth + 1):
            labels.pop()
        if label.arity > 0:
            for i in range(label.arity):
                stack[label.stack_height + i] = stack[sp - label.arity + i]
            sp = label.stack_height + label.arity
        else:
            sp = label.stack_height
        if label.is_loop:
            labels.append(label)
        return sp, label.target + 1

    def h_br_if(
        operand, stack, sp, locals_arr, labels, instance, body, ip, jump_targets
    ):
        sp -= 1
        if stack[sp]:
            depth = operand
            label_idx = len(labels) - 1 - depth
            label = labels[label_idx]
            for _ in range(depth + 1):
                labels.pop()
            if label.arity > 0:
                for i in range(label.arity):
                    stack[label.stack_height + i] = stack[sp - label.arity + i]
                sp = label.stack_height + label.arity
            else:
                sp = label.stack_height
            if label.is_loop:
                labels.append(label)
            ip = label.target + 1
        return sp, ip

    def h_drop(
        operand, stack, sp, locals_arr, labels, instance, body, ip, jump_targets
    ):
        return sp - 1, ip

    def h_select(
        operand, stack, sp, locals_arr, labels, instance, body, ip, jump_targets
    ):
        c = stack[sp - 1]
        v2 = stack[sp - 2]
        v1 = stack[sp - 3]
        stack[sp - 3] = v1 if c else v2
        return sp - 2, ip

    def h_nop(operand, stack, sp, locals_arr, labels, instance, body, ip, jump_targets):
        return sp, ip

    def h_i64_const(
        operand, stack, sp, locals_arr, labels, instance, body, ip, jump_targets
    ):
        stack[sp] = operand
        return sp + 1, ip

    return {
        "local.get": h_local_get,
        "local.set": h_local_set,
        "local.tee": h_local_tee,
        "i32.const": h_i32_const,
        "i32.add": h_i32_add,
        "i32.sub": h_i32_sub,
        "i32.mul": h_i32_mul,
        "i32.and": h_i32_and,
        "i32.or": h_i32_or,
        "i32.xor": h_i32_xor,
        "i32.shl": h_i32_shl,
        "i32.shr_u": h_i32_shr_u,
        "i32.eqz": h_i32_eqz,
        "i32.eq": h_i32_eq,
        "i32.ne": h_i32_ne,
        "i32.lt_u": h_i32_lt_u,
        "i32.gt_u": h_i32_gt_u,
        "i32.le_u": h_i32_le_u,
        "i32.ge_u": h_i32_ge_u,
        "i32.lt_s": h_i32_lt_s,
        "i32.gt_s": h_i32_gt_s,
        "i32.load": h_i32_load,
        "i32.load8_u": h_i32_load8_u,
        "i32.store": h_i32_store,
        "i32.store8": h_i32_store8,
        "global.get": h_global_get,
        "global.set": h_global_set,
        "block": h_block,
        "loop": h_loop,
        "if": h_if,
        "else": h_else,
        "end": h_end,
        "br": h_br,
        "br_if": h_br_if,
        "drop": h_drop,
        "select": h_select,
        "nop": h_nop,
        "i64.const": h_i64_const,
    }


HANDLERS = make_handler_dict()


def execute_dispatch(instance, func_idx, args):
    """Execute using dictionary dispatch."""
    global _instruction_count, _start_time, _should_stop

    if func_idx in instance.imported_funcs:
        return instance.imported_funcs[func_idx](*args)

    func = instance.funcs[func_idx]
    if func is None:
        raise TrapError(f"Function {func_idx} is not available")
    func_type = instance.func_types[func.type_idx]

    # Set up locals
    n_locals = len(func_type.params) + len(func.locals)
    locals_arr = [0] * n_locals
    for i, arg in enumerate(args):
        locals_arr[i] = arg

    stack = [0] * 1024
    sp = 0

    labels = [
        Label(arity=len(func_type.results), target=len(func.body) - 1, stack_height=0)
    ]
    body = func.body
    body_len = len(body)
    jump_targets = get_jump_targets(func, body)

    handlers = HANDLERS

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
            print(
                f"  Progress: {_instruction_count:,} instructions, {rate:,.0f}/sec",
                flush=True,
            )
            if elapsed > _max_time:
                _should_stop = True
                raise TimeoutError(f"Stopped after {_instruction_count:,} instructions")

        # Dictionary dispatch
        handler = handlers.get(op)
        if handler:
            sp, ip = handler(
                operand, stack, sp, locals_arr, labels, instance, body, ip, jump_targets
            )
        elif op == "call":
            call_func_idx = operand
            called_func = instance.funcs[call_func_idx]
            if called_func is None:
                imp = instance.module.imports[call_func_idx]
                call_func_type = instance.func_types[imp.desc]
            else:
                call_func_type = instance.func_types[called_func.type_idx]
            n_params = len(call_func_type.params)
            call_args = [stack[sp - n_params + i] for i in range(n_params)]
            sp -= n_params
            call_result = execute_dispatch(instance, call_func_idx, call_args)
            if call_result is not None:
                if isinstance(call_result, tuple):
                    for r in call_result:
                        stack[sp] = r
                        sp += 1
                else:
                    stack[sp] = call_result
                    sp += 1
        elif op == "return":
            break
        else:
            raise TrapError(f"Unimplemented: {op}")

    num_results = len(func_type.results)
    if num_results == 0:
        return None
    elif num_results == 1:
        return stack[sp - 1] if sp > 0 else None
    else:
        return tuple(stack[sp - num_results + i] for i in range(num_results))


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

    print("\nBenchmarking dictionary dispatch executor...", flush=True)

    _instruction_count = 0
    _start_time = time.monotonic()
    _should_stop = False

    try:
        execute_dispatch(instance, exports["_start"], [])
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
        print(f"\n  Error: {e}")
        print(f"  Rate: {rate:,.0f} instructions/second")

    return 0


if __name__ == "__main__":
    exit(main())
