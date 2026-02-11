"""WebAssembly bytecode executor/interpreter.

Optimized version with:
- Pre-compiled parallel arrays (integer opcodes + operands)
- Bound method caching for stack operations
- Pre-computed control flow targets
"""

from dataclasses import dataclass, field
from typing import Any, Callable

from .types import Module, Function, FuncType, Instruction, Export, Memory, Global
from .errors import TrapError
from .opcodes import (
    OPCODE_NAMES,
    UNREACHABLE as OP_UNREACHABLE,
    NOP as OP_NOP,
    BLOCK as OP_BLOCK,
    LOOP as OP_LOOP,
    IF as OP_IF,
    ELSE as OP_ELSE,
    END as OP_END,
    BR as OP_BR,
    BR_IF as OP_BR_IF,
    BR_TABLE as OP_BR_TABLE,
    RETURN as OP_RETURN,
    CALL as OP_CALL,
    DROP as OP_DROP,
    SELECT as OP_SELECT,
    LOCAL_GET as OP_LOCAL_GET,
    LOCAL_SET as OP_LOCAL_SET,
    LOCAL_TEE as OP_LOCAL_TEE,
    GLOBAL_GET as OP_GLOBAL_GET,
    GLOBAL_SET as OP_GLOBAL_SET,
    I32_LOAD as OP_I32_LOAD,
    I32_CONST as OP_I32_CONST,
    I64_CONST as OP_I64_CONST,
    F32_CONST as OP_F32_CONST,
    F64_CONST as OP_F64_CONST,
    I32_EQZ as OP_I32_EQZ,
    I32_EQ as OP_I32_EQ,
    I32_NE as OP_I32_NE,
    I32_LT_S as OP_I32_LT_S,
    I32_LT_U as OP_I32_LT_U,
    I32_GT_S as OP_I32_GT_S,
    I32_GT_U as OP_I32_GT_U,
    I32_LE_S as OP_I32_LE_S,
    I32_LE_U as OP_I32_LE_U,
    I32_GE_S as OP_I32_GE_S,
    I32_GE_U as OP_I32_GE_U,
    I32_CLZ as OP_I32_CLZ,
    I32_CTZ as OP_I32_CTZ,
    I32_POPCNT as OP_I32_POPCNT,
    I32_ADD as OP_I32_ADD,
    I32_SUB as OP_I32_SUB,
    I32_MUL as OP_I32_MUL,
    I32_DIV_S as OP_I32_DIV_S,
    I32_DIV_U as OP_I32_DIV_U,
    I32_REM_S as OP_I32_REM_S,
    I32_REM_U as OP_I32_REM_U,
    I32_AND as OP_I32_AND,
    I32_OR as OP_I32_OR,
    I32_XOR as OP_I32_XOR,
    I32_SHL as OP_I32_SHL,
    I32_SHR_S as OP_I32_SHR_S,
    I32_SHR_U as OP_I32_SHR_U,
    I32_ROTL as OP_I32_ROTL,
    I32_ROTR as OP_I32_ROTR,
    I64_ADD as OP_I64_ADD,
    F32_ADD as OP_F32_ADD,
)

# Reverse mapping: string opcode name -> integer opcode
# Use first occurrence to prefer primary opcodes (e.g., SELECT over SELECT_T)
_NAME_TO_OPCODE: dict[str, int] = {}
for _k, _v in OPCODE_NAMES.items():
    if _v not in _NAME_TO_OPCODE:
        _NAME_TO_OPCODE[_v] = _k

# Super-instruction opcodes (>= 256, beyond WASM opcode range)
SUPER_COPY_LOCAL = 0x100  # local.get A, local.set B → operand=(A, B)
SUPER_GET_GET_ADD_SET = (
    0x101  # local.get A, local.get B, i32.add, local.set C → operand=(A, B, C)
)
SUPER_GET_CONST_ADD_SET = (
    0x102  # local.get A, i32.const V, i32.add, local.set B → operand=(A, V, B)
)


# Mask for 32-bit values - cached as module-level for faster access
_MASK_32 = 0xFFFFFFFF
_MASK_64 = 0xFFFFFFFFFFFFFFFF
_SIGN_32 = 0x80000000
_OVERFLOW_32 = 0x100000000
_SIGN_64 = 0x8000000000000000
_OVERFLOW_64 = 0x10000000000000000


def to_i32(value: int) -> int:
    """Convert to signed 32-bit integer."""
    value = value & _MASK_32
    if value >= _SIGN_32:
        value -= _OVERFLOW_32
    return value


def to_u32(value: int) -> int:
    """Convert to unsigned 32-bit integer."""
    return value & _MASK_32


def to_i64(value: int) -> int:
    """Convert to signed 64-bit integer."""
    value = value & _MASK_64
    if value >= _SIGN_64:
        value -= _OVERFLOW_64
    return value


def to_u64(value: int) -> int:
    """Convert to unsigned 64-bit integer."""
    return value & _MASK_64


@dataclass(slots=True)
class Label:
    """A control flow label for block/loop/if."""

    arity: int  # Number of values on stack when branching
    target: int  # Instruction index to jump to
    is_loop: bool = False  # True if this is a loop label


@dataclass(slots=True)
class Frame:
    """A call frame for a function invocation."""

    func_idx: int
    locals: list[Any]
    return_arity: int
    module: "Instance"


@dataclass(slots=True)
class MemoryInstance:
    """Runtime memory instance."""

    data: bytearray
    min_pages: int
    max_pages: int | None

    PAGE_SIZE = 65536

    def grow(self, delta: int) -> int:
        """Grow memory by delta pages. Returns old size or -1 on failure."""
        old_size = len(self.data) // self.PAGE_SIZE
        new_size = old_size + delta

        if self.max_pages is not None and new_size > self.max_pages:
            return -1

        # Check for reasonable upper limit (e.g., 4GB)
        if new_size > 65536:  # 4GB
            return -1

        self.data.extend(bytearray(delta * self.PAGE_SIZE))
        return old_size


@dataclass(slots=True)
class GlobalInstance:
    """Runtime global variable instance."""

    value: Any
    mutable: bool


class ExportNamespace:
    """Namespace for accessing exported functions."""

    def __init__(self, instance: "Instance") -> None:
        object.__setattr__(self, "_instance", instance)
        object.__setattr__(self, "_exports", {})

    def _add(self, name: str, value: Any) -> None:
        # Replace invalid Python identifiers
        safe_name = name.replace("-", "_")
        if safe_name.isidentifier():
            object.__setattr__(self, safe_name, value)
        self._exports[name] = value

    def __getattr__(self, name: str) -> Any:
        # Handle exports with names that needed sanitizing
        if name.startswith("_"):
            raise AttributeError(name)
        # Try original name first
        if name in self._exports:
            return self._exports[name]
        raise AttributeError(f"No export named '{name}'")

    def __getitem__(self, name: str) -> Any:
        return self._exports[name]


@dataclass
class Instance:
    """A WebAssembly module instance with runtime state."""

    module: Module
    funcs: list[Function]  # All functions (imports + module-defined)
    func_types: list[FuncType]
    memories: list[MemoryInstance]
    globals: list[GlobalInstance]
    exports: ExportNamespace = field(init=False)
    # Pre-computed control flow targets
    _control_flow_cache: dict = field(init=False, default_factory=dict)
    # Pre-compiled parallel arrays: func_idx -> (opcodes_list, operands_list)
    _compiled: dict = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.exports = ExportNamespace(self)
        self._control_flow_cache = {}
        self._compiled = {}
        self._setup_exports()
        self._precompute_control_flow()

    def _setup_exports(self) -> None:
        """Set up export accessors."""
        for export in self.module.exports:
            if export.kind == "func":
                func_idx = export.index
                # Create a callable wrapper
                wrapper = self._make_func_wrapper(func_idx)
                self.exports._add(export.name, wrapper)
            elif export.kind == "memory":
                self.exports._add(export.name, self.memories[export.index])
            elif export.kind == "global":
                self.exports._add(export.name, self.globals[export.index])

    def _make_func_wrapper(self, func_idx: int) -> Callable:
        """Create a Python callable that invokes a WASM function."""

        def wrapper(*args: Any) -> Any:
            return execute_function(self, func_idx, list(args))

        return wrapper

    def _precompute_control_flow(self) -> None:
        """Pre-compute control flow targets and compile to parallel arrays."""
        name_to_op = _NAME_TO_OPCODE

        for func_idx, func in enumerate(self.funcs):
            body = func.body
            body_len = len(body)
            cache = {}

            # Compile to parallel arrays
            ops = [0] * body_len
            operands_list = [None] * body_len

            for ip in range(body_len):
                instr = body[ip]
                # Compile to integer opcode
                ops[ip] = name_to_op.get(instr.opcode, -1)
                operands_list[ip] = instr.operand

                if instr.opcode in ("block", "loop"):
                    end_ip = _find_end_fast(body, ip + 1, body_len)
                    cache[ip] = end_ip
                elif instr.opcode == "if":
                    else_ip, end_ip = _find_else_end_fast(body, ip + 1, body_len)
                    cache[ip] = (else_ip, end_ip)

            self._control_flow_cache[func_idx] = cache

            # Fuse super-instructions
            ops, operands_list = _fuse_super_instructions(ops, operands_list, cache)

            self._compiled[func_idx] = (ops, operands_list)


def _fuse_super_instructions(
    ops: list[int], operands: list, cf_cache: dict
) -> tuple[list[int], list]:
    """Fuse common instruction patterns into super-instructions.

    Returns new (ops, operands) lists with fused instructions.
    Control flow targets in cf_cache use original IPs, so we build
    a mapping from old IP to new IP and adjust targets.
    """
    n = len(ops)
    # Positions that are control flow targets (cannot be fused into middle of super-instr)
    cf_targets: set[int] = set()
    for ip, target in cf_cache.items():
        cf_targets.add(ip)
        if isinstance(target, tuple):
            # if: (else_ip, end_ip)
            else_ip, end_ip = target
            if else_ip is not None:
                cf_targets.add(else_ip)
            cf_targets.add(end_ip)
        else:
            cf_targets.add(target)

    new_ops: list[int] = []
    new_operands: list = []
    ip_map: dict[int, int] = {}  # old_ip -> new_ip

    ip = 0
    while ip < n:
        ip_map[ip] = len(new_ops)

        # Try to match super-instruction patterns
        # Pattern: local.get A, local.get B, i32.add, local.set C (4 instructions)
        if (
            ip + 3 < n
            and ops[ip] == OP_LOCAL_GET
            and ops[ip + 1] == OP_LOCAL_GET
            and ops[ip + 2] == OP_I32_ADD
            and ops[ip + 3] == OP_LOCAL_SET
            and (ip + 1) not in cf_targets
            and (ip + 2) not in cf_targets
            and (ip + 3) not in cf_targets
        ):
            new_ops.append(SUPER_GET_GET_ADD_SET)
            new_operands.append((operands[ip], operands[ip + 1], operands[ip + 3]))
            # Map intermediate IPs
            ip_map[ip + 1] = len(new_ops) - 1
            ip_map[ip + 2] = len(new_ops) - 1
            ip_map[ip + 3] = len(new_ops) - 1
            ip += 4
            continue

        # Pattern: local.get A, i32.const V, i32.add, local.set B (4 instructions)
        if (
            ip + 3 < n
            and ops[ip] == OP_LOCAL_GET
            and ops[ip + 1] == OP_I32_CONST
            and ops[ip + 2] == OP_I32_ADD
            and ops[ip + 3] == OP_LOCAL_SET
            and (ip + 1) not in cf_targets
            and (ip + 2) not in cf_targets
            and (ip + 3) not in cf_targets
        ):
            new_ops.append(SUPER_GET_CONST_ADD_SET)
            new_operands.append((operands[ip], operands[ip + 1], operands[ip + 3]))
            ip_map[ip + 1] = len(new_ops) - 1
            ip_map[ip + 2] = len(new_ops) - 1
            ip_map[ip + 3] = len(new_ops) - 1
            ip += 4
            continue

        # Pattern: local.get A, local.set B (2 instructions)
        if (
            ip + 1 < n
            and ops[ip] == OP_LOCAL_GET
            and ops[ip + 1] == OP_LOCAL_SET
            and (ip + 1) not in cf_targets
        ):
            new_ops.append(SUPER_COPY_LOCAL)
            new_operands.append((operands[ip], operands[ip + 1]))
            ip_map[ip + 1] = len(new_ops) - 1
            ip += 2
            continue

        # No pattern matched, emit as-is
        new_ops.append(ops[ip])
        new_operands.append(operands[ip])
        ip += 1

    # Map remaining IPs (for body_len)
    ip_map[n] = len(new_ops)

    # Update control flow cache to use new IPs
    for old_ip in list(cf_cache.keys()):
        target = cf_cache[old_ip]
        new_key = ip_map.get(old_ip, old_ip)
        if isinstance(target, tuple):
            else_ip, end_ip = target
            new_else = ip_map.get(else_ip) if else_ip is not None else None
            new_end = ip_map.get(end_ip, end_ip)
            cf_cache[new_key] = (new_else, new_end)
        else:
            cf_cache[new_key] = ip_map.get(target, target)
        if new_key != old_ip:
            del cf_cache[old_ip]

    return new_ops, new_operands


def _find_end_fast(body: list[Instruction], start_ip: int, body_len: int) -> int:
    """Find the matching 'end' instruction for a block/loop starting at start_ip."""
    depth = 1
    ip = start_ip
    while ip < body_len:
        op = body[ip].opcode
        if op == "block" or op == "loop" or op == "if":
            depth += 1
        elif op == "end":
            depth -= 1
            if depth == 0:
                return ip
        ip += 1
    raise TrapError("No matching end found")


def _find_else_end_fast(
    body: list[Instruction], start_ip: int, body_len: int
) -> tuple[int | None, int]:
    """Find 'else' and 'end' for an if starting at start_ip."""
    depth = 1
    ip = start_ip
    else_ip = None
    while ip < body_len:
        op = body[ip].opcode
        if op == "block" or op == "loop" or op == "if":
            depth += 1
        elif op == "else" and depth == 1:
            else_ip = ip
        elif op == "end":
            depth -= 1
            if depth == 0:
                return else_ip, ip
        ip += 1
    raise TrapError("No matching end found for if")


def execute_function(instance: Instance, func_idx: int, args: list[Any]) -> Any:
    """Execute a function and return its result(s)."""
    func = instance.funcs[func_idx]
    func_type = instance.func_types[func.type_idx]

    # Set up locals: params + declared locals
    locals_list: list[Any] = list(args)

    # Add declared locals (initialized to zero)
    for valtype in func.locals:
        if valtype == "i32" or valtype == "i64":
            locals_list.append(0)
        elif valtype == "f32" or valtype == "f64":
            locals_list.append(0.0)
        else:
            locals_list.append(None)

    # Execute function body
    stack: list[Any] = []
    labels: list[Label] = []

    # Cache frequently accessed values
    cf_cache = instance._control_flow_cache.get(func_idx, {})
    result_count = len(func_type.results)

    # Get compiled parallel arrays (may be shorter than func.body due to fusion)
    ops, operands = instance._compiled[func_idx]
    body_len = len(ops)

    # Bound method caching for hot-path stack operations
    stack_append = stack.append
    stack_pop = stack.pop

    # Add implicit function-level label
    labels.append(Label(arity=result_count, target=body_len - 1))

    ip = 0  # Instruction pointer

    # Inline constants for faster access (local variables are fastest in CPython)
    mask_32 = _MASK_32
    sign_32 = _SIGN_32
    overflow_32 = _OVERFLOW_32

    while ip < body_len:
        op = ops[ip]
        ip += 1

        # Most common operations first for branch prediction

        # Super-instructions (fused common patterns) - checked first for hot loops
        if op == SUPER_GET_GET_ADD_SET:
            # local.get A, local.get B, i32.add, local.set C
            a_idx, b_idx, c_idx = operands[ip - 1]
            v = (locals_list[a_idx] + locals_list[b_idx]) & mask_32
            if v >= sign_32:
                v -= overflow_32
            locals_list[c_idx] = v
            continue

        if op == SUPER_COPY_LOCAL:
            # local.get A, local.set B
            locals_list[operands[ip - 1][1]] = locals_list[operands[ip - 1][0]]
            continue

        if op == SUPER_GET_CONST_ADD_SET:
            # local.get A, i32.const V, i32.add, local.set B
            a_idx, const_val, b_idx = operands[ip - 1]
            v = (locals_list[a_idx] + const_val) & mask_32
            if v >= sign_32:
                v -= overflow_32
            locals_list[b_idx] = v
            continue

        if op == OP_LOCAL_GET:
            stack_append(locals_list[operands[ip - 1]])
            continue

        if op == OP_LOCAL_SET:
            locals_list[operands[ip - 1]] = stack_pop()
            continue

        if op == OP_I32_CONST:
            v = operands[ip - 1] & mask_32
            if v >= sign_32:
                v -= overflow_32
            stack_append(v)
            continue

        if op == OP_I32_ADD:
            b = stack_pop()
            a = stack_pop()
            v = (a + b) & mask_32
            if v >= sign_32:
                v -= overflow_32
            stack_append(v)
            continue

        if op == OP_I32_SUB:
            b = stack_pop()
            a = stack_pop()
            v = (a - b) & mask_32
            if v >= sign_32:
                v -= overflow_32
            stack_append(v)
            continue

        if op == OP_I32_MUL:
            b = stack_pop()
            a = stack_pop()
            v = (a * b) & mask_32
            if v >= sign_32:
                v -= overflow_32
            stack_append(v)
            continue

        if op == OP_LOCAL_TEE:
            locals_list[operands[ip - 1]] = stack[-1]
            continue

        if op == OP_END:
            if labels:
                labels.pop()
            continue

        if op == OP_BR_IF:
            condition = stack_pop()
            if condition:
                depth = operands[ip - 1]
                label_idx = len(labels) - 1 - depth
                label = labels[label_idx]
                del labels[label_idx:]
                if label.is_loop:
                    labels.append(label)
                ip = label.target + 1
            continue

        if op == OP_BR:
            depth = operands[ip - 1]
            label_idx = len(labels) - 1 - depth
            label = labels[label_idx]
            del labels[label_idx:]
            if label.is_loop:
                labels.append(label)
            ip = label.target + 1
            continue

        if op == OP_BLOCK:
            blocktype = operands[ip - 1]
            arity = 0 if blocktype == () else 1 if isinstance(blocktype, tuple) else 0
            end_ip = cf_cache.get(ip - 1)
            if end_ip is None:
                end_ip = _find_end_fast(func.body, ip, body_len)
            labels.append(Label(arity=arity, target=end_ip))
            continue

        if op == OP_LOOP:
            labels.append(Label(arity=0, target=ip - 1, is_loop=True))
            continue

        if op == OP_IF:
            condition = stack_pop()
            blocktype = operands[ip - 1]
            arity = 0 if blocktype == () else 1 if isinstance(blocktype, tuple) else 0
            cached = cf_cache.get(ip - 1)
            if cached:
                else_ip, end_ip = cached
            else:
                else_ip, end_ip = _find_else_end_fast(func.body, ip, body_len)

            if condition:
                labels.append(Label(arity=arity, target=end_ip))
            else:
                labels.append(Label(arity=arity, target=end_ip))
                if else_ip is not None:
                    ip = else_ip + 1
                else:
                    ip = end_ip + 1
            continue

        if op == OP_ELSE:
            if labels:
                ip = labels[-1].target
            continue

        if op == OP_I32_GE_S:
            b = stack_pop()
            a = stack_pop()
            stack_append(1 if a >= b else 0)
            continue

        if op == OP_I32_LT_S:
            b = stack_pop()
            a = stack_pop()
            stack_append(1 if a < b else 0)
            continue

        if op == OP_I32_LE_S:
            b = stack_pop()
            a = stack_pop()
            stack_append(1 if a <= b else 0)
            continue

        if op == OP_I32_GT_S:
            b = stack_pop()
            a = stack_pop()
            stack_append(1 if a > b else 0)
            continue

        if op == OP_I32_EQ:
            b = stack_pop()
            a = stack_pop()
            stack_append(1 if a == b else 0)
            continue

        if op == OP_I32_NE:
            b = stack_pop()
            a = stack_pop()
            stack_append(1 if a != b else 0)
            continue

        if op == OP_I32_EQZ:
            stack[-1] = 1 if stack[-1] == 0 else 0
            continue

        if op == OP_CALL:
            callee_idx = operands[ip - 1]
            callee_func = instance.funcs[callee_idx]
            callee_type = instance.func_types[callee_func.type_idx]
            n_params = len(callee_type.params)
            if n_params > 0:
                call_args = stack[-n_params:]
                del stack[-n_params:]
            else:
                call_args = []
            call_result = execute_function(instance, callee_idx, call_args)
            if call_result is not None:
                if isinstance(call_result, tuple):
                    stack.extend(call_result)
                else:
                    stack_append(call_result)
            continue

        if op == OP_RETURN:
            break

        if op == OP_NOP:
            continue

        if op == OP_UNREACHABLE:
            raise TrapError("unreachable executed")

        if op == OP_DROP:
            stack_pop()
            continue

        if op == OP_SELECT:
            c = stack_pop()
            val2 = stack_pop()
            val1 = stack_pop()
            stack_append(val1 if c else val2)
            continue

        # Less common operations
        if op == OP_I64_CONST:
            v = operands[ip - 1] & _MASK_64
            if v >= _SIGN_64:
                v -= _OVERFLOW_64
            stack_append(v)
            continue

        if op == OP_I64_ADD:
            b = stack_pop()
            a = stack_pop()
            v = (a + b) & _MASK_64
            if v >= _SIGN_64:
                v -= _OVERFLOW_64
            stack_append(v)
            continue

        if op == OP_F32_CONST or op == OP_F64_CONST:
            stack_append(float(operands[ip - 1]))
            continue

        if op == OP_F32_ADD:
            b = stack_pop()
            a = stack_pop()
            stack_append(a + b)
            continue

        if op == OP_I32_LOAD:
            align, offset = operands[ip - 1]
            addr = stack_pop() + offset
            mem = instance.memories[0].data
            if addr < 0 or addr + 4 > len(mem):
                raise TrapError("out of bounds memory access")
            v = int.from_bytes(mem[addr : addr + 4], "little", signed=True)
            stack_append(v)
            continue

        if op == OP_GLOBAL_GET:
            stack_append(instance.globals[operands[ip - 1]].value)
            continue

        if op == OP_GLOBAL_SET:
            g = instance.globals[operands[ip - 1]]
            if not g.mutable:
                raise TrapError("Cannot set immutable global")
            g.value = stack_pop()
            continue

        if op == OP_I32_DIV_S:
            b = stack_pop()
            a = stack_pop()
            if b == 0:
                raise TrapError("integer divide by zero")
            if a == -0x80000000 and b == -1:
                raise TrapError("integer overflow")
            result = abs(a) // abs(b)
            if (a < 0) != (b < 0):
                result = -result
            v = result & mask_32
            if v >= sign_32:
                v -= overflow_32
            stack_append(v)
            continue

        if op == OP_I32_DIV_U:
            b = stack_pop()
            a = stack_pop()
            if b == 0:
                raise TrapError("integer divide by zero")
            stack_append((a & mask_32) // (b & mask_32))
            continue

        if op == OP_I32_REM_S:
            b = stack_pop()
            a = stack_pop()
            if b == 0:
                raise TrapError("integer divide by zero")
            result = abs(a) % abs(b)
            if a < 0:
                result = -result
            v = result & mask_32
            if v >= sign_32:
                v -= overflow_32
            stack_append(v)
            continue

        if op == OP_I32_REM_U:
            b = stack_pop()
            a = stack_pop()
            if b == 0:
                raise TrapError("integer divide by zero")
            stack_append((a & mask_32) % (b & mask_32))
            continue

        # Bitwise operations
        if op == OP_I32_AND:
            b = stack_pop()
            a = stack_pop()
            v = (a & mask_32) & (b & mask_32)
            if v >= sign_32:
                v -= overflow_32
            stack_append(v)
            continue

        if op == OP_I32_OR:
            b = stack_pop()
            a = stack_pop()
            v = (a & mask_32) | (b & mask_32)
            if v >= sign_32:
                v -= overflow_32
            stack_append(v)
            continue

        if op == OP_I32_XOR:
            b = stack_pop()
            a = stack_pop()
            v = (a & mask_32) ^ (b & mask_32)
            if v >= sign_32:
                v -= overflow_32
            stack_append(v)
            continue

        if op == OP_I32_SHL:
            b = stack_pop()
            a = stack_pop()
            v = ((a & mask_32) << ((b & mask_32) & 31)) & mask_32
            if v >= sign_32:
                v -= overflow_32
            stack_append(v)
            continue

        if op == OP_I32_SHR_S:
            b = stack_pop()
            a = stack_pop()
            shift = (b & mask_32) & 31
            v = (a >> shift) & mask_32
            if v >= sign_32:
                v -= overflow_32
            stack_append(v)
            continue

        if op == OP_I32_SHR_U:
            b = stack_pop()
            a = stack_pop()
            shift = (b & mask_32) & 31
            v = ((a & mask_32) >> shift) & mask_32
            if v >= sign_32:
                v -= overflow_32
            stack_append(v)
            continue

        if op == OP_I32_ROTL:
            b = stack_pop()
            a = stack_pop()
            shift = (b & mask_32) & 31
            ua = a & mask_32
            v = ((ua << shift) | (ua >> (32 - shift))) & mask_32
            if v >= sign_32:
                v -= overflow_32
            stack_append(v)
            continue

        if op == OP_I32_ROTR:
            b = stack_pop()
            a = stack_pop()
            shift = (b & mask_32) & 31
            ua = a & mask_32
            v = ((ua >> shift) | (ua << (32 - shift))) & mask_32
            if v >= sign_32:
                v -= overflow_32
            stack_append(v)
            continue

        # Unary operations
        if op == OP_I32_CLZ:
            a = stack_pop() & mask_32
            if a == 0:
                stack_append(32)
            else:
                stack_append(32 - a.bit_length())
            continue

        if op == OP_I32_CTZ:
            a = stack_pop() & mask_32
            if a == 0:
                stack_append(32)
            else:
                count = 0
                while (a & 1) == 0:
                    count += 1
                    a >>= 1
                stack_append(count)
            continue

        if op == OP_I32_POPCNT:
            a = stack_pop() & mask_32
            stack_append(bin(a).count("1"))
            continue

        # Unsigned comparisons
        if op == OP_I32_LT_U:
            b = stack_pop()
            a = stack_pop()
            stack_append(1 if (a & mask_32) < (b & mask_32) else 0)
            continue

        if op == OP_I32_GT_U:
            b = stack_pop()
            a = stack_pop()
            stack_append(1 if (a & mask_32) > (b & mask_32) else 0)
            continue

        if op == OP_I32_LE_U:
            b = stack_pop()
            a = stack_pop()
            stack_append(1 if (a & mask_32) <= (b & mask_32) else 0)
            continue

        if op == OP_I32_GE_U:
            b = stack_pop()
            a = stack_pop()
            stack_append(1 if (a & mask_32) >= (b & mask_32) else 0)
            continue

        if op == OP_BR_TABLE:
            label_indices, default = operands[ip - 1]
            idx = stack_pop()
            if 0 <= idx < len(label_indices):
                depth = label_indices[idx]
            else:
                depth = default
            label_idx = len(labels) - 1 - depth
            label = labels[label_idx]
            del labels[label_idx:]
            if label.is_loop:
                labels.append(label)
            ip = label.target + 1
            continue

        raise TrapError(f"Unimplemented instruction: {OPCODE_NAMES.get(op, hex(op))}")

    # Return results
    if result_count == 0:
        return None
    elif result_count == 1:
        return stack[-1] if stack else None
    else:
        return tuple(stack[-result_count:])


# Keep these for compatibility
def do_branch(stack: list[Any], labels: list[Label], depth: int) -> tuple:
    """Execute a branch to the given label depth."""
    label_idx = len(labels) - 1 - depth
    if label_idx < 0:
        raise TrapError(f"Invalid branch depth: {depth}")

    label = labels[label_idx]

    for _ in range(depth + 1):
        labels.pop()

    if label.is_loop:
        labels.append(label)

    return ("branch", label.target + 1 if not label.is_loop else label.target + 1)


def find_end(body: list[Instruction], start_ip: int) -> int:
    """Find the matching 'end' instruction for a block/loop starting at start_ip."""
    return _find_end_fast(body, start_ip, len(body))


def find_else_end(body: list[Instruction], start_ip: int) -> tuple[int | None, int]:
    """Find 'else' and 'end' for an if starting at start_ip."""
    return _find_else_end_fast(body, start_ip, len(body))


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

    # Collect function types
    func_types = list(module.types)

    # Collect all functions (imports first, then module-defined)
    funcs: list[Function] = []

    # Handle imported functions
    for imp in module.imports:
        if imp.kind == "func":
            pass

    # Add module-defined functions
    funcs.extend(module.funcs)

    # Initialize memories
    memories: list[MemoryInstance] = []
    for mem in module.mems:
        data = bytearray(mem.limits.min * MemoryInstance.PAGE_SIZE)
        memories.append(MemoryInstance(data, mem.limits.min, mem.limits.max))

    # Initialize data segments
    for data_seg in module.data:
        if data_seg.memory_idx >= 0 and memories:
            mem = memories[data_seg.memory_idx]
            offset = 0
            for instr in data_seg.offset:
                if instr.opcode == "i32.const":
                    offset = instr.operand
                    break
            mem.data[offset : offset + len(data_seg.init)] = data_seg.init

    # Initialize globals
    globals_list: list[GlobalInstance] = []
    for glob in module.globals:
        value: Any = 0
        for instr in glob.init:
            if instr.opcode in ("i32.const", "i64.const"):
                value = instr.operand
            elif instr.opcode in ("f32.const", "f64.const"):
                value = instr.operand
        globals_list.append(GlobalInstance(value, glob.type.mutable))

    instance = Instance(
        module=module,
        funcs=funcs,
        func_types=func_types,
        memories=memories,
        globals=globals_list,
    )

    # Run start function if present
    if module.start is not None:
        execute_function(instance, module.start, [])

    return instance
