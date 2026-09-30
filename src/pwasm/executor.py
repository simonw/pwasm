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
    CALL_INDIRECT,
    CALLN,
    MISC,
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
    TICK,
    UNOP,
    UNREACHABLE,
)
from .codegen import is_cached
from .errors import LinkError, TrapError
from .numeric import MASK_32, MASK_64, F32NaN, f32_from_bits
from .runtime import (
    FROM_PYTHON,
    TO_PYTHON,
    GlobalInstance,
    HostFunction,
    Limits,
    MemoryInstance,
    TableInstance,
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
    Returns None, a single value, or a list (or tuple) of values."""
    pyfunc = func.pyfunc
    if pyfunc is not None:
        return pyfunc(*args)
    if func.countdown:
        func.countdown -= 1
        if not func.countdown and func.tier_up():
            return func.pyfunc(*args)
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
            elif op == CALL_INDIRECT:
                table, ftype = imms[ip - 1]
                i = pop()
                elements = table.elements
                if i >= len(elements):
                    raise TrapError("undefined element")
                f = elements[i]
                if f is None:
                    raise TrapError(f"uninitialized element {i}")
                if f.type is not ftype and f.type != ftype:
                    raise TrapError("indirect call type mismatch")
                n = f.n_params
                if n:
                    a = stack[-n:]
                    del stack[-n:]
                else:
                    a = []
                if type(f) is WasmFunction:
                    r = execute(f, a)
                else:
                    r = f.call(a)
                n = f.n_results
                if n == 1:
                    push(r)
                elif n:
                    stack.extend(r)
            elif op == MISC:
                fn, n_in, n_out = imms[ip - 1]
                if n_in:
                    a = stack[-n_in:]
                    del stack[-n_in:]
                    r = fn(*a)
                else:
                    r = fn()
                if n_out:
                    push(r)
            elif op == TICK:
                limits = imms[ip - 1]
                limits.countdown -= 1
                if limits.countdown < 0:
                    limits.refill()
            else:
                raise TrapError(f"Unknown internal opcode {op}")
    except (struct.error, IndexError):
        if ops[ip - 1] in MEMORY_OPS:
            raise TrapError("out of bounds memory access") from None
        raise


def invoke(func: WasmFunction | HostFunction, args: list) -> Any:
    """Call any function object with a list of internal values."""
    return func.entry(*args)


def call_with_python_values(func: WasmFunction | HostFunction, args: tuple) -> Any:
    """Call a function object with Python values (see ExportedFunction)."""
    return ExportedFunction(func)(*args)


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
        self.tables: list[TableInstance] = []
        self.globals: list[GlobalInstance] = []
        # Element and data segments; dropped segments become empty
        self.elements: list[list] = []
        self.datas: list[bytes] = []
        self.limits: Limits | None = None
        self.mode = DEFAULT_MODE
        # (function index, exception) for functions that could not be
        # compiled to Python and stay in the interpreter
        self.compile_errors: list = []
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
        elif name == "ref.null":
            stack.append(None)
        elif name == "ref.func":
            stack.append(instance.functions[instr.operand])
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


def _incompatible(imp: Any, detail: str) -> LinkError:
    return LinkError(f"incompatible import type for {imp.module}.{imp.name}: {detail}")


def _limits_match(size: int, maximum: int | None, limits: Any) -> bool:
    if size < limits.min:
        return False
    if limits.max is not None and (maximum is None or maximum > limits.max):
        return False
    return True


def _import_memory(value: Any, imp: Any) -> MemoryInstance:
    if not isinstance(value, MemoryInstance):
        raise _incompatible(imp, f"expected a memory, got {type(value).__name__}")
    if not _limits_match(value.size, value.max_pages, imp.desc):
        raise _incompatible(imp, "memory limits do not match")
    return value


def _import_table(value: Any, imp: Any) -> TableInstance:
    element_type, limits = imp.desc
    if not isinstance(value, TableInstance):
        raise _incompatible(imp, f"expected a table, got {type(value).__name__}")
    if value.element_type != element_type:
        raise _incompatible(imp, f"expected {element_type} table")
    if not _limits_match(value.size, value.max_size, limits):
        raise _incompatible(imp, "table limits do not match")
    return value


def _import_global(value: Any, imp: Any) -> GlobalInstance:
    gtype = imp.desc
    if isinstance(value, GlobalInstance):
        if value.type.valtype != gtype.valtype or value.type.mutable != gtype.mutable:
            raise _incompatible(imp, f"expected global {gtype}, got {value.type}")
        return value
    if gtype.mutable or not isinstance(value, (int, float)) or value is None:
        raise _incompatible(imp, f"expected a global, got {type(value).__name__}")
    # A plain Python number can supply an immutable global
    return GlobalInstance(gtype, FROM_PYTHON[gtype.valtype](value))


# How functions run:
# - "interpret": always in the interpreter
# - "compile": compiled to Python source on their first call
# - "auto": interpreted at first, compiled to Python once called
#   AUTO_THRESHOLD times (code that only runs once is not worth compiling)
MODES = ("interpret", "compile", "auto")
DEFAULT_MODE = "auto"
AUTO_THRESHOLD = 2


def instantiate(
    module: Module,
    imports: dict[str, dict[str, Any]] | None = None,
    *,
    limits: Limits | None = None,
    mode: str | None = None,
) -> Instance:
    """Create an instance from a module.

    Args:
        module: The decoded module to instantiate
        imports: Optional import object mapping module -> name -> value
        limits: Optional resource limits (fuel, deadline, memory cap)
        mode: "interpret", "compile" or "auto" (see MODES; default
            DEFAULT_MODE)

    Returns:
        An Instance ready for execution
    """
    imports = imports or {}
    mode = mode or DEFAULT_MODE
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    countdown = {"interpret": 0, "compile": 1, "auto": AUTO_THRESHOLD}[mode]
    instance = Instance(module)
    instance.limits = limits
    instance.mode = mode

    # Imports come first in each index space
    for imp in module.imports:
        value = _lookup_import(imports, imp.module, imp.name)
        if imp.kind == "func":
            instance.functions.append(
                _import_function(value, module.types[imp.desc], imp)
            )
        elif imp.kind == "memory":
            instance.memories.append(_import_memory(value, imp))
        elif imp.kind == "table":
            instance.tables.append(_import_table(value, imp))
        elif imp.kind == "global":
            instance.globals.append(_import_global(value, imp))
        else:
            raise LinkError(f"Unsupported import kind {imp.kind}")

    n_imported = len(instance.functions)
    for index, func in enumerate(module.funcs):
        wfunc = WasmFunction(
            module.types[func.type_idx], instance, func, n_imported + index
        )
        wfunc.countdown = countdown
        # Python code another instance already compiled is worth switching
        # to on the first call
        if mode == "auto" and is_cached(module, wfunc.index, limits is not None):
            wfunc.countdown = 1
        instance.functions.append(wfunc)

    for table in module.tables:
        instance.tables.append(
            TableInstance(table.element_type, table.limits.min, table.limits.max)
        )

    for mem in module.mems:
        memory = MemoryInstance(mem.limits.min, mem.limits.max)
        if limits is not None and limits.max_memory is not None:
            memory.limit_pages = limits.max_memory // MemoryInstance.PAGE_SIZE
            if mem.limits.min > memory.limit_pages:
                raise LinkError(
                    f"initial memory of {mem.limits.min} pages exceeds max_memory"
                )
        instance.memories.append(memory)

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
        elif export.kind == "table":
            instance.exports._add(export.name, instance.tables[export.index])

    functions = instance.functions
    for seg in module.elem:
        instance.elements.append(
            [
                (
                    functions[item]
                    if isinstance(item, int)
                    else _eval_const(instance, item)
                )
                for item in seg.init
            ]
        )
    instance.datas = [seg.init for seg in module.data]

    # Active segments are copied in order, then dropped. A trap leaves the
    # effects of earlier segments in place.
    for i, seg in enumerate(module.elem):
        if seg.mode == "active":
            refs = instance.elements[i]
            offset = _eval_const(instance, seg.offset)
            instance.tables[seg.table_idx].init(offset, refs, 0, len(refs))
        if seg.mode != "passive":
            instance.elements[i] = []

    for i, seg in enumerate(module.data):
        if seg.memory_idx < 0:
            continue  # passive segment
        memory = instance.memories[seg.memory_idx]
        offset = _eval_const(instance, seg.offset)
        memory.write(offset, seg.init)
        instance.datas[i] = b""

    if module.start is not None:
        invoke(instance.functions[module.start], [])

    return instance
