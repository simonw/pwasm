"""NumPy-optimized WebAssembly executor.

Uses NumPy arrays for:
- Memory: numpy.ndarray instead of bytearray
- Stack: pre-allocated array with stack pointer
- Locals: numpy array with proper types
- Integer operations: native i32/i64 with proper overflow

This can provide significant speedup over pure Python lists.
"""

import numpy as np
from typing import Any
from .types import Function, FuncType
from .executor import (
    Instance, Label, get_jump_targets, find_end, find_else_end,
    MASK_32, MASK_64
)
from .errors import TrapError


def execute_function_numpy(instance: Instance, func_idx: int, args: list[Any]) -> Any:
    """Execute a function using NumPy-optimized operations."""
    # Check if this is an imported function
    if func_idx in instance.imported_funcs:
        return instance.imported_funcs[func_idx](*args)

    func = instance.funcs[func_idx]
    if func is None:
        raise TrapError(f"Function {func_idx} is not available")
    func_type = instance.func_types[func.type_idx]

    # Set up locals using numpy array
    n_locals = len(func_type.params) + len(func.locals)
    locals_arr = np.zeros(n_locals, dtype=np.int64)

    # Copy arguments
    for i, arg in enumerate(args):
        locals_arr[i] = arg

    # Use numpy array for stack (pre-allocated, fixed size)
    # Most WASM functions don't need huge stacks
    stack = np.zeros(1024, dtype=np.int64)
    sp = 0  # Stack pointer

    labels: list[Label] = []
    body = func.body
    body_len = len(body)

    labels.append(Label(arity=len(func_type.results), target=body_len - 1, stack_height=0))
    jump_targets = get_jump_targets(func, body)

    # Pre-compute memory view if available
    mem_view = None
    if instance.memories:
        mem_data = instance.memories[0].data
        mem_view = np.frombuffer(mem_data, dtype=np.uint8)

    ip = 0
    while ip < body_len:
        instr = body[ip]
        op = instr.opcode
        operand = instr.operand
        ip += 1

        # Most common opcodes first
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

        if op == "i32.mul":
            sp -= 1
            b = np.int32(stack[sp])
            a = np.int32(stack[sp - 1])
            stack[sp - 1] = np.int32(a * b)
            continue

        if op == "i32.and":
            sp -= 1
            stack[sp - 1] = stack[sp - 1] & stack[sp]
            continue

        if op == "i32.or":
            sp -= 1
            stack[sp - 1] = stack[sp - 1] | stack[sp]
            continue

        if op == "i32.xor":
            sp -= 1
            stack[sp - 1] = stack[sp - 1] ^ stack[sp]
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

        if op == "i32.eqz":
            stack[sp - 1] = 1 if stack[sp - 1] == 0 else 0
            continue

        if op == "i32.eq":
            sp -= 1
            stack[sp - 1] = 1 if stack[sp - 1] == stack[sp] else 0
            continue

        if op == "i32.ne":
            sp -= 1
            stack[sp - 1] = 1 if stack[sp - 1] != stack[sp] else 0
            continue

        if op == "i32.lt_s":
            sp -= 1
            a = np.int32(stack[sp - 1])
            b = np.int32(stack[sp])
            stack[sp - 1] = 1 if a < b else 0
            continue

        if op == "i32.lt_u":
            sp -= 1
            a = np.uint32(stack[sp - 1])
            b = np.uint32(stack[sp])
            stack[sp - 1] = 1 if a < b else 0
            continue

        if op == "i32.gt_s":
            sp -= 1
            a = np.int32(stack[sp - 1])
            b = np.int32(stack[sp])
            stack[sp - 1] = 1 if a > b else 0
            continue

        if op == "i32.gt_u":
            sp -= 1
            a = np.uint32(stack[sp - 1])
            b = np.uint32(stack[sp])
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

        if op == "i32.load":
            align, offset = operand
            sp -= 1
            addr = int(stack[sp]) + offset
            sp += 1
            mem = instance.memories[0].data
            val = int.from_bytes(mem[addr:addr + 4], "little", signed=True)
            stack[sp - 1] = val
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

        if op == "i32.store":
            align, offset = operand
            sp -= 2
            val = int(stack[sp + 1]) & MASK_32
            addr = int(stack[sp]) + offset
            mem = instance.memories[0].data
            mem[addr:addr + 4] = val.to_bytes(4, "little")
            continue

        if op == "i32.store8":
            align, offset = operand
            sp -= 2
            val = int(stack[sp + 1]) & 0xFF
            addr = int(stack[sp]) + offset
            instance.memories[0].data[addr] = val
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

        if op == "select":
            sp -= 1
            c = stack[sp]
            sp -= 1
            v2 = stack[sp]
            v1 = stack[sp - 1]
            stack[sp - 1] = v1 if c else v2
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
            call_result = execute_function_numpy(instance, call_func_idx, call_args)
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

        # i64 operations
        if op == "i64.const":
            stack[sp] = operand
            sp += 1
            continue

        if op == "i64.add":
            sp -= 1
            stack[sp - 1] = stack[sp - 1] + stack[sp]
            continue

        # Fallback - this shouldn't happen often
        raise TrapError(f"NumPy executor: Unimplemented instruction: {op}")

    # Return results
    num_results = len(func_type.results)
    if num_results == 0:
        return None
    elif num_results == 1:
        return int(stack[sp - 1]) if sp > 0 else None
    else:
        return tuple(int(stack[sp - num_results + i]) for i in range(num_results))
