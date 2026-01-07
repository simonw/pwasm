"""WebAssembly bytecode executor/interpreter.

Optimized version with:
- Dictionary-based instruction dispatch
- Pre-computed control flow targets
- Inlined integer conversion operations
"""

from dataclasses import dataclass, field
from typing import Any, Callable

from .types import Module, Function, FuncType, Instruction, Export, Memory, Global
from .errors import TrapError


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

    def __post_init__(self) -> None:
        self.exports = ExportNamespace(self)
        self._control_flow_cache = {}
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
        """Pre-compute control flow targets for all functions."""
        for func_idx, func in enumerate(self.funcs):
            body = func.body
            body_len = len(body)
            cache = {}

            for ip in range(body_len):
                instr = body[ip]
                if instr.opcode in ("block", "loop"):
                    end_ip = _find_end_fast(body, ip + 1, body_len)
                    cache[ip] = end_ip
                elif instr.opcode == "if":
                    else_ip, end_ip = _find_else_end_fast(body, ip + 1, body_len)
                    cache[ip] = (else_ip, end_ip)

            self._control_flow_cache[func_idx] = cache


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
    body = func.body
    body_len = len(body)
    cf_cache = instance._control_flow_cache.get(func_idx, {})
    result_count = len(func_type.results)

    # Add implicit function-level label
    labels.append(Label(arity=result_count, target=body_len - 1))

    ip = 0  # Instruction pointer

    # Inline constants for faster access
    mask_32 = _MASK_32
    sign_32 = _SIGN_32
    overflow_32 = _OVERFLOW_32

    while ip < body_len:
        instr = body[ip]
        op = instr.opcode
        ip += 1

        # Use a dispatch approach with early returns for common ops
        # Most common operations first for branch prediction

        if op == "local.get":
            stack.append(locals_list[instr.operand])
            continue

        if op == "local.set":
            locals_list[instr.operand] = stack.pop()
            continue

        if op == "i32.const":
            v = instr.operand & mask_32
            if v >= sign_32:
                v -= overflow_32
            stack.append(v)
            continue

        if op == "i32.add":
            b = stack.pop()
            a = stack.pop()
            v = (a + b) & mask_32
            if v >= sign_32:
                v -= overflow_32
            stack.append(v)
            continue

        if op == "i32.sub":
            b = stack.pop()
            a = stack.pop()
            v = (a - b) & mask_32
            if v >= sign_32:
                v -= overflow_32
            stack.append(v)
            continue

        if op == "i32.mul":
            b = stack.pop()
            a = stack.pop()
            v = (a * b) & mask_32
            if v >= sign_32:
                v -= overflow_32
            stack.append(v)
            continue

        if op == "local.tee":
            locals_list[instr.operand] = stack[-1]
            continue

        if op == "end":
            if labels:
                labels.pop()
            continue

        if op == "br_if":
            condition = stack.pop()
            if condition:
                depth = instr.operand
                label_idx = len(labels) - 1 - depth
                label = labels[label_idx]
                for _ in range(depth + 1):
                    labels.pop()
                if label.is_loop:
                    labels.append(label)
                ip = label.target + 1
            continue

        if op == "br":
            depth = instr.operand
            label_idx = len(labels) - 1 - depth
            label = labels[label_idx]
            for _ in range(depth + 1):
                labels.pop()
            if label.is_loop:
                labels.append(label)
            ip = label.target + 1
            continue

        if op == "block":
            blocktype = instr.operand
            arity = 0 if blocktype == () else 1 if isinstance(blocktype, tuple) else 0
            end_ip = cf_cache.get(ip - 1, _find_end_fast(body, ip, body_len))
            labels.append(Label(arity=arity, target=end_ip))
            continue

        if op == "loop":
            blocktype = instr.operand
            end_ip = cf_cache.get(ip - 1, _find_end_fast(body, ip, body_len))
            labels.append(Label(arity=0, target=ip - 1, is_loop=True))
            continue

        if op == "if":
            condition = stack.pop()
            blocktype = instr.operand
            arity = 0 if blocktype == () else 1 if isinstance(blocktype, tuple) else 0
            cached = cf_cache.get(ip - 1)
            if cached:
                else_ip, end_ip = cached
            else:
                else_ip, end_ip = _find_else_end_fast(body, ip, body_len)

            if condition:
                labels.append(Label(arity=arity, target=end_ip))
            else:
                labels.append(Label(arity=arity, target=end_ip))
                if else_ip is not None:
                    ip = else_ip + 1
                else:
                    ip = end_ip + 1
            continue

        if op == "else":
            if labels:
                label = labels[-1]
                ip = label.target
            continue

        if op == "i32.ge_s":
            b = stack.pop()
            a = stack.pop()
            stack.append(1 if a >= b else 0)
            continue

        if op == "i32.lt_s":
            b = stack.pop()
            a = stack.pop()
            stack.append(1 if a < b else 0)
            continue

        if op == "i32.le_s":
            b = stack.pop()
            a = stack.pop()
            stack.append(1 if a <= b else 0)
            continue

        if op == "i32.gt_s":
            b = stack.pop()
            a = stack.pop()
            stack.append(1 if a > b else 0)
            continue

        if op == "i32.eq":
            b = stack.pop()
            a = stack.pop()
            stack.append(1 if a == b else 0)
            continue

        if op == "i32.ne":
            b = stack.pop()
            a = stack.pop()
            stack.append(1 if a != b else 0)
            continue

        if op == "i32.eqz":
            stack.append(1 if stack.pop() == 0 else 0)
            continue

        if op == "call":
            callee_idx = instr.operand
            callee_func = instance.funcs[callee_idx]
            callee_type = instance.func_types[callee_func.type_idx]
            n_params = len(callee_type.params)
            call_args = [stack.pop() for _ in range(n_params)][::-1]
            call_result = execute_function(instance, callee_idx, call_args)
            if call_result is not None:
                if isinstance(call_result, tuple):
                    stack.extend(call_result)
                else:
                    stack.append(call_result)
            continue

        if op == "return":
            break

        if op == "nop":
            continue

        if op == "unreachable":
            raise TrapError("unreachable executed")

        if op == "drop":
            stack.pop()
            continue

        if op == "select":
            c = stack.pop()
            val2 = stack.pop()
            val1 = stack.pop()
            stack.append(val1 if c else val2)
            continue

        # Less common operations
        if op == "i64.const":
            v = instr.operand & _MASK_64
            if v >= _SIGN_64:
                v -= _OVERFLOW_64
            stack.append(v)
            continue

        if op == "f32.const" or op == "f64.const":
            stack.append(float(instr.operand))
            continue

        if op == "global.get":
            stack.append(instance.globals[instr.operand].value)
            continue

        if op == "global.set":
            g = instance.globals[instr.operand]
            if not g.mutable:
                raise TrapError("Cannot set immutable global")
            g.value = stack.pop()
            continue

        if op == "i32.div_s":
            b = stack.pop()
            a = stack.pop()
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
            stack.append(v)
            continue

        if op == "i32.div_u":
            b = stack.pop()
            a = stack.pop()
            if b == 0:
                raise TrapError("integer divide by zero")
            stack.append((a & mask_32) // (b & mask_32))
            continue

        if op == "i32.rem_s":
            b = stack.pop()
            a = stack.pop()
            if b == 0:
                raise TrapError("integer divide by zero")
            result = abs(a) % abs(b)
            if a < 0:
                result = -result
            v = result & mask_32
            if v >= sign_32:
                v -= overflow_32
            stack.append(v)
            continue

        if op == "i32.rem_u":
            b = stack.pop()
            a = stack.pop()
            if b == 0:
                raise TrapError("integer divide by zero")
            stack.append((a & mask_32) % (b & mask_32))
            continue

        # Bitwise operations
        if op == "i32.and":
            b = stack.pop()
            a = stack.pop()
            v = (a & mask_32) & (b & mask_32)
            if v >= sign_32:
                v -= overflow_32
            stack.append(v)
            continue

        if op == "i32.or":
            b = stack.pop()
            a = stack.pop()
            v = (a & mask_32) | (b & mask_32)
            if v >= sign_32:
                v -= overflow_32
            stack.append(v)
            continue

        if op == "i32.xor":
            b = stack.pop()
            a = stack.pop()
            v = (a & mask_32) ^ (b & mask_32)
            if v >= sign_32:
                v -= overflow_32
            stack.append(v)
            continue

        if op == "i32.shl":
            b = stack.pop()
            a = stack.pop()
            v = ((a & mask_32) << ((b & mask_32) & 31)) & mask_32
            if v >= sign_32:
                v -= overflow_32
            stack.append(v)
            continue

        if op == "i32.shr_s":
            b = stack.pop()
            a = stack.pop()
            shift = (b & mask_32) & 31
            v = (a >> shift) & mask_32
            if v >= sign_32:
                v -= overflow_32
            stack.append(v)
            continue

        if op == "i32.shr_u":
            b = stack.pop()
            a = stack.pop()
            shift = (b & mask_32) & 31
            v = ((a & mask_32) >> shift) & mask_32
            if v >= sign_32:
                v -= overflow_32
            stack.append(v)
            continue

        if op == "i32.rotl":
            b = stack.pop()
            a = stack.pop()
            shift = (b & mask_32) & 31
            ua = a & mask_32
            v = ((ua << shift) | (ua >> (32 - shift))) & mask_32
            if v >= sign_32:
                v -= overflow_32
            stack.append(v)
            continue

        if op == "i32.rotr":
            b = stack.pop()
            a = stack.pop()
            shift = (b & mask_32) & 31
            ua = a & mask_32
            v = ((ua >> shift) | (ua << (32 - shift))) & mask_32
            if v >= sign_32:
                v -= overflow_32
            stack.append(v)
            continue

        # Unary operations
        if op == "i32.clz":
            a = stack.pop() & mask_32
            if a == 0:
                stack.append(32)
            else:
                stack.append(32 - a.bit_length())
            continue

        if op == "i32.ctz":
            a = stack.pop() & mask_32
            if a == 0:
                stack.append(32)
            else:
                count = 0
                while (a & 1) == 0:
                    count += 1
                    a >>= 1
                stack.append(count)
            continue

        if op == "i32.popcnt":
            a = stack.pop() & mask_32
            stack.append(bin(a).count("1"))
            continue

        # Unsigned comparisons
        if op == "i32.lt_u":
            b = stack.pop()
            a = stack.pop()
            stack.append(1 if (a & mask_32) < (b & mask_32) else 0)
            continue

        if op == "i32.gt_u":
            b = stack.pop()
            a = stack.pop()
            stack.append(1 if (a & mask_32) > (b & mask_32) else 0)
            continue

        if op == "i32.le_u":
            b = stack.pop()
            a = stack.pop()
            stack.append(1 if (a & mask_32) <= (b & mask_32) else 0)
            continue

        if op == "i32.ge_u":
            b = stack.pop()
            a = stack.pop()
            stack.append(1 if (a & mask_32) >= (b & mask_32) else 0)
            continue

        if op == "br_table":
            label_indices, default = instr.operand
            idx = stack.pop()
            if 0 <= idx < len(label_indices):
                depth = label_indices[idx]
            else:
                depth = default
            label_idx = len(labels) - 1 - depth
            label = labels[label_idx]
            for _ in range(depth + 1):
                labels.pop()
            if label.is_loop:
                labels.append(label)
            ip = label.target + 1
            continue

        raise TrapError(f"Unimplemented instruction: {op}")

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
