"""WebAssembly interpreter and module instantiation.

Functions are compiled on their first call (see compiler.py) into parallel
lists of internal opcodes and immediates, which `execute` interprets. Each
WebAssembly call is a Python call of `execute`, so Python exceptions raised by
host functions propagate through WebAssembly frames unchanged.
"""

from __future__ import annotations

import struct
from typing import Any, Callable

from .compiler import (
    BINOP,
    BR_IF,
    BR_IF_DROP,
    BR_TABLE,
    CALL,
    CALL0,
    CALL_HOST,
    CALLN,
    CONST,
    DROP,
    GLOBAL_GET,
    GLOBAL_SET,
    I32_ADD,
    I32_AND,
    I32_EQ,
    I32_EQZ,
    I32_GE_S,
    I32_GE_U,
    I32_GT_S,
    I32_GT_U,
    I32_LE_S,
    I32_LE_U,
    I32_LOAD,
    I32_LT_S,
    I32_LT_U,
    I32_MUL,
    I32_NE,
    I32_OR,
    I32_SHL,
    I32_SHR_U,
    I32_STORE,
    I32_SUB,
    I32_XOR,
    IF_FALSE,
    JMP,
    JMP_DROP,
    LOCAL_GET,
    LOCAL_SET,
    LOCAL_TEE,
    LOAD,
    LOAD_F32,
    LOAD_MASK,
    MEMORY_GROW,
    MEMORY_OPS,
    MEMORY_SIZE,
    STORE,
    STORE_F32,
    STORE_RAW,
    I32_LOAD8_U,
    I32_STORE8,
    RETURN,
    RETURN_IF,
    SELECT,
    UNOP,
    UNREACHABLE,
)
from .errors import LinkError, TrapError
from .numeric import MASK_32, MASK_64, F32NaN, f32_from_bits
from .runtime import (
    FROM_PYTHON,
    TO_PYTHON,
    GlobalInstance,
    HostFunction,
    MemoryInstance,
    WasmFunction,
)
from .types import Module, Instruction

# Kept for backwards compatibility with code importing these from here
__all__ = [
    "Instance",
    "instantiate",
    "execute",
    "MemoryInstance",
    "GlobalInstance",
    "WasmFunction",
    "HostFunction",
]

_unpack_u32 = struct.Struct("<I").unpack_from
_pack_u32 = struct.Struct("<I").pack_into
_unpack_f32 = struct.Struct("<f").unpack_from
_pack_f32 = struct.Struct("<f").pack_into

SIGN_32 = 0x80000000


def execute(func: WasmFunction, args: list) -> Any:
    """Run a WebAssembly function on internal values.

    `args` becomes the function's locals, so callers pass a list they own.
    Returns None, a single value, or a list of values."""
    code = func.code
    if code is None:
        code = func.compile()
    ops = code.ops
    imms = code.imms
    local = args
    if code.zeros:
        local += code.zeros
    stack: list = []
    push = stack.append
    pop = stack.pop
    mem = code.mem
    ip = 0
    try:
        while True:
            op = ops[ip]
            ip += 1
            if op == LOCAL_GET:
                push(local[imms[ip - 1]])
                continue
            if op == CONST:
                push(imms[ip - 1])
                continue
            if op == LOCAL_SET:
                local[imms[ip - 1]] = pop()
                continue
            if op == LOCAL_TEE:
                local[imms[ip - 1]] = stack[-1]
                continue
            if op == I32_ADD:
                b = pop()
                stack[-1] = (stack[-1] + b) & MASK_32
                continue
            if op == I32_LOAD:
                stack[-1] = _unpack_u32(mem, stack[-1] + imms[ip - 1])[0]
                continue
            if op == BR_IF:
                if pop():
                    ip = imms[ip - 1]
                continue
            if op == I32_STORE:
                v = pop()
                _pack_u32(mem, pop() + imms[ip - 1], v)
                continue
            if op == CALL:
                callee, n = imms[ip - 1]
                if n:
                    a = stack[-n:]
                    del stack[-n:]
                else:
                    a = []
                push(execute(callee, a))
                continue
            if op == JMP:
                ip = imms[ip - 1]
                continue
            if op < 20:
                if op == I32_AND:
                    b = pop()
                    stack[-1] &= b
                elif op == I32_EQZ:
                    stack[-1] = 1 if stack[-1] == 0 else 0
                elif op == I32_SHL:
                    b = pop()
                    stack[-1] = (stack[-1] << (b & 31)) & MASK_32
                elif op == I32_NE:
                    b = pop()
                    stack[-1] = 1 if stack[-1] != b else 0
                elif op == I32_SUB:
                    b = pop()
                    stack[-1] = (stack[-1] - b) & MASK_32
                elif op == I32_EQ:
                    b = pop()
                    stack[-1] = 1 if stack[-1] == b else 0
                elif op == GLOBAL_GET:
                    push(imms[ip - 1]._value)
                elif op == GLOBAL_SET:
                    imms[ip - 1]._value = pop()
                elif op == DROP:
                    pop()
                else:  # IF_FALSE
                    if not pop():
                        ip = imms[ip - 1]
                continue
            if op < 30:
                b = pop()
                a = stack[-1]
                if op == I32_LT_U:
                    stack[-1] = 1 if a < b else 0
                elif op == I32_LT_S:
                    stack[-1] = 1 if (a ^ SIGN_32) < (b ^ SIGN_32) else 0
                elif op == I32_GT_U:
                    stack[-1] = 1 if a > b else 0
                elif op == I32_GT_S:
                    stack[-1] = 1 if (a ^ SIGN_32) > (b ^ SIGN_32) else 0
                elif op == I32_LE_U:
                    stack[-1] = 1 if a <= b else 0
                elif op == I32_LE_S:
                    stack[-1] = 1 if (a ^ SIGN_32) <= (b ^ SIGN_32) else 0
                elif op == I32_GE_U:
                    stack[-1] = 1 if a >= b else 0
                elif op == I32_GE_S:
                    stack[-1] = 1 if (a ^ SIGN_32) >= (b ^ SIGN_32) else 0
                elif op == I32_OR:
                    stack[-1] = a | b
                else:  # I32_XOR
                    stack[-1] = a ^ b
                continue
            if op < 40:
                if op == I32_SHR_U:
                    b = pop()
                    stack[-1] >>= b & 31
                elif op == I32_MUL:
                    b = pop()
                    stack[-1] = (stack[-1] * b) & MASK_32
                elif op == SELECT:
                    c = pop()
                    b = pop()
                    if not c:
                        stack[-1] = b
                elif op == RETURN:
                    n = imms[ip - 1]
                    if n == 1:
                        return stack[-1]
                    if n == 0:
                        return None
                    return stack[-n:]
                elif op == JMP_DROP:
                    ip, lo, hi = imms[ip - 1]
                    del stack[lo:hi]
                elif op == BR_IF_DROP:
                    if pop():
                        ip, lo, hi = imms[ip - 1]
                        del stack[lo:hi]
                elif op == BR_TABLE:
                    table, default = imms[ip - 1]
                    i = pop()
                    entry = table[i] if i < len(table) else default
                    if entry is None:
                        ip = len(ops) - 1  # the final RETURN
                    else:
                        ip, lo, hi = entry
                        if hi > lo:
                            del stack[lo:hi]
                elif op == CALL0:
                    callee, n = imms[ip - 1]
                    if n:
                        a = stack[-n:]
                        del stack[-n:]
                    else:
                        a = []
                    execute(callee, a)
                elif op == CALLN:
                    callee, n = imms[ip - 1]
                    if n:
                        a = stack[-n:]
                        del stack[-n:]
                    else:
                        a = []
                    stack.extend(execute(callee, a))
                else:  # CALL_HOST
                    callee = imms[ip - 1]
                    n = callee.n_params
                    if n:
                        a = stack[-n:]
                        del stack[-n:]
                    else:
                        a = []
                    r = callee.call(a)
                    n = callee.n_results
                    if n == 1:
                        push(r)
                    elif n:
                        stack.extend(r)
                continue
            if op < 50:
                if op == LOAD:
                    unpack, off = imms[ip - 1]
                    stack[-1] = unpack(mem, stack[-1] + off)[0]
                elif op == I32_LOAD8_U:
                    stack[-1] = mem[stack[-1] + imms[ip - 1]]
                elif op == I32_STORE8:
                    v = pop()
                    mem[pop() + imms[ip - 1]] = v & 0xFF
                elif op == STORE_RAW:
                    pack, off = imms[ip - 1]
                    v = pop()
                    pack(mem, pop() + off, v)
                elif op == LOAD_MASK:
                    unpack, off, mask = imms[ip - 1]
                    stack[-1] = unpack(mem, stack[-1] + off)[0] & mask
                elif op == STORE:
                    pack, off, mask = imms[ip - 1]
                    v = pop()
                    pack(mem, pop() + off, v & mask)
                elif op == LOAD_F32:
                    a = stack[-1] + imms[ip - 1]
                    v = _unpack_f32(mem, a)[0]
                    if v != v:
                        v = f32_from_bits(_unpack_u32(mem, a)[0])
                    stack[-1] = v
                elif op == STORE_F32:
                    v = pop()
                    a = pop() + imms[ip - 1]
                    if type(v) is F32NaN:
                        _pack_u32(mem, a, v.bits)
                    else:
                        _pack_f32(mem, a, v)
                elif op == MEMORY_SIZE:
                    push(len(mem) >> 16)
                else:  # MEMORY_GROW
                    stack[-1] = imms[ip - 1].grow(stack[-1]) & MASK_32
                continue
            if op == BINOP:
                b = pop()
                stack[-1] = imms[ip - 1](stack[-1], b)
            elif op == UNOP:
                stack[-1] = imms[ip - 1](stack[-1])
            elif op == UNREACHABLE:
                raise TrapError("unreachable")
            elif op == RETURN_IF:
                if pop():
                    n = imms[ip - 1]
                    if n == 1:
                        return stack[-1]
                    if n == 0:
                        return None
                    return stack[-n:]
            else:
                raise TrapError(f"Unknown internal opcode {op}")
    except (struct.error, IndexError):
        if ops[ip - 1] in MEMORY_OPS:
            raise TrapError("out of bounds memory access") from None
        raise


def invoke(func: WasmFunction | HostFunction, args: list) -> Any:
    """Call any function object with a list of internal values."""
    if type(func) is WasmFunction:
        return execute(func, args)
    return func.call(args)


class ExportedFunction:
    """A callable for an exported function: converts Python arguments to
    internal values and results back (i32/i64 results are signed ints;
    multiple results come back as a tuple)."""

    __slots__ = ("func", "name", "_in", "_out")

    def __init__(self, func: WasmFunction | HostFunction, name: str = "") -> None:
        self.func = func
        self.name = name
        self._in = [FROM_PYTHON[t] for t in func.type.params]
        self._out = [TO_PYTHON[t] for t in func.type.results]

    def __call__(self, *args: Any) -> Any:
        if len(args) != len(self._in):
            raise TypeError(
                f"{self.name or 'function'}() takes {len(self._in)} arguments "
                f"({len(args)} given)"
            )
        values = [conv(v) for conv, v in zip(self._in, args)]
        try:
            result = invoke(self.func, values)
        except RecursionError:
            raise TrapError("call stack exhausted") from None
        out = self._out
        if not out:
            return None
        if len(out) == 1:
            return out[0](result)
        return tuple(conv(v) for conv, v in zip(out, result))

    @property
    def type(self):
        return self.func.type

    def __repr__(self) -> str:
        return f"<ExportedFunction {self.name} {self.func.type}>"


class ExportNamespace:
    """Namespace for accessing exports as attributes or items."""

    def __init__(self, instance: "Instance") -> None:
        object.__setattr__(self, "_instance", instance)
        object.__setattr__(self, "_exports", {})

    def _add(self, name: str, value: Any) -> None:
        # Make names that are not identifiers reachable as attributes too
        safe_name = name.replace("-", "_")
        if safe_name.isidentifier():
            object.__setattr__(self, safe_name, value)
        self._exports[name] = value

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._exports:
            return self._exports[name]
        raise AttributeError(f"No export named '{name}'")

    def __getitem__(self, name: str) -> Any:
        return self._exports[name]

    def __contains__(self, name: str) -> bool:
        return name in self._exports

    def __iter__(self):
        return iter(self._exports)


class Instance:
    """A WebAssembly module instance with runtime state."""

    def __init__(self, module: Module) -> None:
        self.module = module
        self.functions: list[WasmFunction | HostFunction] = []
        self.memories: list[MemoryInstance] = []
        self.globals: list[GlobalInstance] = []
        self.exports = ExportNamespace(self)

    @property
    def funcs(self) -> list:
        """Alias for `functions`."""
        return self.functions


def _eval_const(instance: Instance, expr: list[Instruction]) -> Any:
    """Evaluate a constant expression (global initialisers, offsets)."""
    stack: list = []
    for instr in expr:
        name = instr.opcode
        if name == "i32.const":
            stack.append(instr.operand & MASK_32)
        elif name == "i64.const":
            stack.append(instr.operand & MASK_64)
        elif name in ("f32.const", "f64.const"):
            stack.append(instr.operand)
        elif name == "global.get":
            stack.append(instance.globals[instr.operand]._value)
        elif name == "end":
            break
        else:
            raise LinkError(f"Unsupported constant expression: {name}")
    return stack[-1]


def _lookup_import(imports: dict, module_name: str, name: str) -> Any:
    try:
        return imports[module_name][name]
    except (KeyError, TypeError):
        raise LinkError(f"unknown import {module_name}.{name}") from None


def _import_function(value: Any, ftype: Any, imp: Any) -> WasmFunction | HostFunction:
    if isinstance(value, ExportedFunction):
        value = value.func
    if isinstance(value, (WasmFunction, HostFunction)):
        if value.type != ftype:
            raise LinkError(
                f"incompatible import type for {imp.module}.{imp.name}: "
                f"expected {ftype}, got {value.type}"
            )
        return value
    if callable(value):
        return HostFunction(ftype, value)
    raise LinkError(f"import {imp.module}.{imp.name} is not callable")


def instantiate(
    module: Module, imports: dict[str, dict[str, Any]] | None = None
) -> Instance:
    """Create an instance from a module.

    Args:
        module: The decoded module to instantiate
        imports: Optional import object mapping module -> name -> value

    Returns:
        An Instance ready for execution
    """
    imports = imports or {}
    instance = Instance(module)

    # Imports come first in each index space
    for imp in module.imports:
        value = _lookup_import(imports, imp.module, imp.name)
        if imp.kind == "func":
            instance.functions.append(
                _import_function(value, module.types[imp.desc], imp)
            )
        else:
            raise LinkError(f"Importing a {imp.kind} is not supported yet")

    n_imported = len(instance.functions)
    for index, func in enumerate(module.funcs):
        instance.functions.append(
            WasmFunction(
                module.types[func.type_idx], instance, func, n_imported + index
            )
        )

    for mem in module.mems:
        instance.memories.append(MemoryInstance(mem.limits.min, mem.limits.max))

    for glob in module.globals:
        instance.globals.append(
            GlobalInstance(glob.type, _eval_const(instance, glob.init))
        )

    for export in module.exports:
        if export.kind == "func":
            func = instance.functions[export.index]
            instance.exports._add(export.name, ExportedFunction(func, export.name))
        elif export.kind == "memory":
            instance.exports._add(export.name, instance.memories[export.index])
        elif export.kind == "global":
            instance.exports._add(export.name, instance.globals[export.index])

    for seg in module.data:
        if seg.memory_idx < 0:
            continue  # passive segment
        memory = instance.memories[seg.memory_idx]
        offset = _eval_const(instance, seg.offset)
        memory.write(offset, seg.init)

    if module.start is not None:
        invoke(instance.functions[module.start], [])

    return instance
