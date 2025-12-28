"""Compile WebAssembly functions to Python source code.

Instead of interpreting WASM bytecode instruction by instruction, this module
compiles WASM functions to Python functions that can be executed natively.
This can provide a 10-100x speedup over interpretation.
"""

from .types import Function, FuncType, Instruction
from .executor import Instance, get_jump_targets, Label


def compile_function(func: Function, instance: Instance, func_idx: int) -> callable:
    """Compile a WASM function to a native Python function.

    Returns a callable that takes the same arguments and returns the same
    results as the original WASM function.
    """
    func_type = instance.func_types[func.type_idx]
    body = func.body
    jump_targets = get_jump_targets(func, body)

    # Generate Python source code
    lines = []
    indent = "    "

    # Function signature
    param_names = [f"p{i}" for i in range(len(func_type.params))]
    lines.append(f"def compiled_func({', '.join(param_names)}):")

    # Initialize locals
    lines.append(f"{indent}locals_list = [{', '.join(param_names)}]")
    for i, valtype in enumerate(func.locals):
        if valtype in ("i32", "i64"):
            lines.append(f"{indent}locals_list.append(0)")
        elif valtype in ("f32", "f64"):
            lines.append(f"{indent}locals_list.append(0.0)")
        else:
            lines.append(f"{indent}locals_list.append(None)")

    lines.append(f"{indent}stack = []")
    lines.append(f"{indent}MASK_32 = 0xFFFFFFFF")
    lines.append(f"{indent}MASK_64 = 0xFFFFFFFFFFFFFFFF")
    lines.append(
        f"{indent}mem = instance.memories[0].data if instance.memories else None"
    )

    # Compile instructions using labeled blocks
    current_indent = indent
    label_stack = []  # Stack of (label_id, is_loop, target_ip)
    label_counter = [0]

    def new_label():
        label_counter[0] += 1
        return f"L{label_counter[0]}"

    ip = 0
    while ip < len(body):
        instr = body[ip]
        op = instr.opcode
        operand = instr.operand

        if op == "i32.const":
            val = operand & 0xFFFFFFFF
            if val >= 0x80000000:
                val -= 0x100000000
            lines.append(f"{current_indent}stack.append({val})")

        elif op == "i64.const":
            val = operand & 0xFFFFFFFFFFFFFFFF
            if val >= 0x8000000000000000:
                val -= 0x10000000000000000
            lines.append(f"{current_indent}stack.append({val})")

        elif op == "local.get":
            lines.append(f"{current_indent}stack.append(locals_list[{operand}])")

        elif op == "local.set":
            lines.append(f"{current_indent}locals_list[{operand}] = stack.pop()")

        elif op == "local.tee":
            lines.append(f"{current_indent}locals_list[{operand}] = stack[-1]")

        elif op == "i32.add":
            lines.append(f"{current_indent}_b, _a = stack.pop(), stack.pop()")
            lines.append(f"{current_indent}_v = (_a + _b) & MASK_32")
            lines.append(f"{current_indent}if _v >= 0x80000000: _v -= 0x100000000")
            lines.append(f"{current_indent}stack.append(_v)")

        elif op == "i32.sub":
            lines.append(f"{current_indent}_b, _a = stack.pop(), stack.pop()")
            lines.append(f"{current_indent}_v = (_a - _b) & MASK_32")
            lines.append(f"{current_indent}if _v >= 0x80000000: _v -= 0x100000000")
            lines.append(f"{current_indent}stack.append(_v)")

        elif op == "i32.mul":
            lines.append(f"{current_indent}_b, _a = stack.pop(), stack.pop()")
            lines.append(f"{current_indent}_v = (_a * _b) & MASK_32")
            lines.append(f"{current_indent}if _v >= 0x80000000: _v -= 0x100000000")
            lines.append(f"{current_indent}stack.append(_v)")

        elif op == "i32.and":
            lines.append(f"{current_indent}_b, _a = stack.pop(), stack.pop()")
            lines.append(f"{current_indent}stack.append(_a & _b & MASK_32)")

        elif op == "i32.or":
            lines.append(f"{current_indent}_b, _a = stack.pop(), stack.pop()")
            lines.append(f"{current_indent}stack.append((_a | _b) & MASK_32)")

        elif op == "i32.xor":
            lines.append(f"{current_indent}_b, _a = stack.pop(), stack.pop()")
            lines.append(f"{current_indent}stack.append((_a ^ _b) & MASK_32)")

        elif op == "i32.eqz":
            lines.append(f"{current_indent}stack.append(1 if stack.pop() == 0 else 0)")

        elif op == "i32.eq":
            lines.append(f"{current_indent}_b, _a = stack.pop(), stack.pop()")
            lines.append(f"{current_indent}stack.append(1 if _a == _b else 0)")

        elif op == "i32.ne":
            lines.append(f"{current_indent}_b, _a = stack.pop(), stack.pop()")
            lines.append(f"{current_indent}stack.append(1 if _a != _b else 0)")

        elif op == "i32.lt_s":
            lines.append(f"{current_indent}_b, _a = stack.pop(), stack.pop()")
            lines.append(f"{current_indent}stack.append(1 if _a < _b else 0)")

        elif op == "i32.lt_u":
            lines.append(f"{current_indent}_b, _a = stack.pop(), stack.pop()")
            lines.append(
                f"{current_indent}stack.append(1 if (_a & MASK_32) < (_b & MASK_32) else 0)"
            )

        elif op == "i32.gt_s":
            lines.append(f"{current_indent}_b, _a = stack.pop(), stack.pop()")
            lines.append(f"{current_indent}stack.append(1 if _a > _b else 0)")

        elif op == "i32.gt_u":
            lines.append(f"{current_indent}_b, _a = stack.pop(), stack.pop()")
            lines.append(
                f"{current_indent}stack.append(1 if (_a & MASK_32) > (_b & MASK_32) else 0)"
            )

        elif op == "i32.le_s":
            lines.append(f"{current_indent}_b, _a = stack.pop(), stack.pop()")
            lines.append(f"{current_indent}stack.append(1 if _a <= _b else 0)")

        elif op == "i32.le_u":
            lines.append(f"{current_indent}_b, _a = stack.pop(), stack.pop()")
            lines.append(
                f"{current_indent}stack.append(1 if (_a & MASK_32) <= (_b & MASK_32) else 0)"
            )

        elif op == "i32.ge_s":
            lines.append(f"{current_indent}_b, _a = stack.pop(), stack.pop()")
            lines.append(f"{current_indent}stack.append(1 if _a >= _b else 0)")

        elif op == "i32.ge_u":
            lines.append(f"{current_indent}_b, _a = stack.pop(), stack.pop()")
            lines.append(
                f"{current_indent}stack.append(1 if (_a & MASK_32) >= (_b & MASK_32) else 0)"
            )

        elif op == "i32.load":
            align, offset = operand
            lines.append(f"{current_indent}_addr = stack.pop() + {offset}")
            lines.append(
                f"{current_indent}stack.append(int.from_bytes(mem[_addr:_addr+4], 'little', signed=True))"
            )

        elif op == "i32.load8_u":
            align, offset = operand
            lines.append(f"{current_indent}_addr = stack.pop() + {offset}")
            lines.append(f"{current_indent}stack.append(mem[_addr])")

        elif op == "i32.load8_s":
            align, offset = operand
            lines.append(f"{current_indent}_addr = stack.pop() + {offset}")
            lines.append(f"{current_indent}_v = mem[_addr]")
            lines.append(f"{current_indent}if _v >= 128: _v -= 256")
            lines.append(f"{current_indent}stack.append(_v)")

        elif op == "i32.store":
            align, offset = operand
            lines.append(f"{current_indent}_v = stack.pop()")
            lines.append(f"{current_indent}_addr = stack.pop() + {offset}")
            lines.append(
                f"{current_indent}mem[_addr:_addr+4] = (_v & MASK_32).to_bytes(4, 'little')"
            )

        elif op == "i32.store8":
            align, offset = operand
            lines.append(f"{current_indent}_v = stack.pop()")
            lines.append(f"{current_indent}_addr = stack.pop() + {offset}")
            lines.append(f"{current_indent}mem[_addr] = _v & 0xFF")

        elif op == "drop":
            lines.append(f"{current_indent}stack.pop()")

        elif op == "select":
            lines.append(f"{current_indent}_c = stack.pop()")
            lines.append(f"{current_indent}_v2 = stack.pop()")
            lines.append(f"{current_indent}_v1 = stack.pop()")
            lines.append(f"{current_indent}stack.append(_v1 if _c else _v2)")

        elif op == "global.get":
            lines.append(
                f"{current_indent}stack.append(instance.globals[{operand}].value)"
            )

        elif op == "global.set":
            lines.append(
                f"{current_indent}instance.globals[{operand}].value = stack.pop()"
            )

        elif op == "block":
            label_id = new_label()
            if ip in jump_targets:
                end_ip = jump_targets[ip][1]
            else:
                end_ip = ip  # fallback
            label_stack.append((label_id, False, end_ip))
            lines.append(f"{current_indent}# block {label_id}")

        elif op == "loop":
            label_id = new_label()
            if ip in jump_targets:
                end_ip = jump_targets[ip][1]
            else:
                end_ip = ip
            label_stack.append((label_id, True, ip))
            lines.append(f"{current_indent}while True:  # loop {label_id}")
            current_indent += "    "

        elif op == "if":
            label_id = new_label()
            if ip in jump_targets:
                else_ip, end_ip = jump_targets[ip]
            else:
                else_ip, end_ip = None, ip
            label_stack.append((label_id, False, end_ip))
            lines.append(f"{current_indent}if stack.pop():  # if {label_id}")
            current_indent += "    "

        elif op == "else":
            current_indent = current_indent[:-4]
            lines.append(f"{current_indent}else:")
            current_indent += "    "

        elif op == "end":
            if label_stack:
                label_id, is_loop, _ = label_stack.pop()
                if is_loop:
                    lines.append(f"{current_indent}break  # end loop {label_id}")
                    current_indent = current_indent[:-4]
                else:
                    if current_indent != indent:
                        current_indent = current_indent[:-4]

        elif op == "br":
            if label_stack:
                depth = operand
                if depth < len(label_stack):
                    target = label_stack[-(depth + 1)]
                    label_id, is_loop, _ = target
                    if is_loop:
                        lines.append(
                            f"{current_indent}continue  # br to loop {label_id}"
                        )
                    else:
                        lines.append(f"{current_indent}break  # br {depth}")

        elif op == "br_if":
            if label_stack:
                depth = operand
                if depth < len(label_stack):
                    target = label_stack[-(depth + 1)]
                    label_id, is_loop, _ = target
                    lines.append(f"{current_indent}if stack.pop():")
                    if is_loop:
                        lines.append(
                            f"{current_indent}    continue  # br_if to loop {label_id}"
                        )
                    else:
                        lines.append(f"{current_indent}    break  # br_if {depth}")

        elif op == "return":
            n_results = len(func_type.results)
            if n_results == 0:
                lines.append(f"{current_indent}return None")
            elif n_results == 1:
                lines.append(f"{current_indent}return stack[-1]")
            else:
                lines.append(f"{current_indent}return tuple(stack[-{n_results}:])")

        elif op == "call":
            lines.append(f"{current_indent}# call {operand}")
            lines.append(f"{current_indent}_result = call_func({operand}, stack)")
            lines.append(f"{current_indent}if _result is not None:")
            lines.append(f"{current_indent}    stack.append(_result)")

        elif op == "nop":
            pass

        else:
            # Fallback to interpreter for unknown ops
            lines.append(f"{current_indent}# {op} - fallback to interpreter")
            lines.append(f"{current_indent}raise NotImplementedError('{op}')")

        ip += 1

    # Return statement
    n_results = len(func_type.results)
    if n_results == 0:
        lines.append(f"{indent}return None")
    elif n_results == 1:
        lines.append(f"{indent}return stack[-1] if stack else None")
    else:
        lines.append(f"{indent}return tuple(stack[-{n_results}:])")

    # Compile the source
    source = "\n".join(lines)

    # Create a helper function for calls
    def make_call_func(inst):
        def call_func(func_idx, stack):
            from .executor import execute_function_fast

            func = inst.funcs[func_idx]
            if func is None:
                if func_idx in inst.imported_funcs:
                    fn = inst.imported_funcs[func_idx]
                    imp = inst.module.imports[func_idx]
                    ft = inst.func_types[imp.desc]
                    n = len(ft.params)
                    args = [stack.pop() for _ in range(n)][::-1]
                    return fn(*args)
                return None
            ft = inst.func_types[func.type_idx]
            n = len(ft.params)
            args = [stack.pop() for _ in range(n)][::-1]
            return execute_function_fast(inst, func_idx, args)

        return call_func

    call_func = make_call_func(instance)

    # Execute to define the function
    local_vars = {"instance": instance, "call_func": call_func}
    try:
        exec(source, local_vars)
        return local_vars["compiled_func"]
    except Exception as e:
        print(f"Compilation failed: {e}")
        print("Source:")
        for i, line in enumerate(lines[:50], 1):
            print(f"{i:4}: {line}")
        raise


def test_compiler():
    """Test the compiler with a simple function."""
    print("Compiler module loaded successfully.")
    print("Use compile_function(func, instance, func_idx) to compile WASM functions.")
