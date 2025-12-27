"""WebAssembly bytecode executor/interpreter."""

from dataclasses import dataclass, field
from typing import Any, Callable

from .types import Module, Function, FuncType, Instruction, Export, Memory, Global
from .errors import TrapError


# Mask for 32-bit values
MASK_32 = 0xFFFFFFFF
MASK_64 = 0xFFFFFFFFFFFFFFFF


def to_i32(value: int) -> int:
    """Convert to signed 32-bit integer."""
    value = value & MASK_32
    if value >= 0x80000000:
        value -= 0x100000000
    return value


def to_u32(value: int) -> int:
    """Convert to unsigned 32-bit integer."""
    return value & MASK_32


def to_i64(value: int) -> int:
    """Convert to signed 64-bit integer."""
    value = value & MASK_64
    if value >= 0x8000000000000000:
        value -= 0x10000000000000000
    return value


def to_u64(value: int) -> int:
    """Convert to unsigned 64-bit integer."""
    return value & MASK_64


@dataclass
class Label:
    """A control flow label for block/loop/if."""

    arity: int  # Number of values on stack when branching
    target: int  # Instruction index to jump to
    is_loop: bool = False  # True if this is a loop label


@dataclass
class Frame:
    """A call frame for a function invocation."""

    func_idx: int
    locals: list[Any]
    return_arity: int
    module: "Instance"


@dataclass
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


@dataclass
class GlobalInstance:
    """Runtime global variable instance."""

    value: Any
    mutable: bool


class ExportNamespace:
    """Namespace for accessing exported functions."""

    def __init__(self, instance: "Instance") -> None:
        self._instance = instance
        self._exports: dict[str, Any] = {}

    def _add(self, name: str, value: Any) -> None:
        # Replace invalid Python identifiers
        safe_name = name.replace("-", "_")
        if safe_name.isidentifier():
            setattr(self, safe_name, value)
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

    def __post_init__(self) -> None:
        self.exports = ExportNamespace(self)
        self._setup_exports()

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


def execute_function(instance: Instance, func_idx: int, args: list[Any]) -> Any:
    """Execute a function and return its result(s)."""
    func = instance.funcs[func_idx]
    func_type = instance.func_types[func.type_idx]

    # Set up locals: params + declared locals
    locals_list: list[Any] = list(args)

    # Add declared locals (initialized to zero)
    for valtype in func.locals:
        if valtype in ("i32", "i64"):
            locals_list.append(0)
        elif valtype in ("f32", "f64"):
            locals_list.append(0.0)
        else:
            locals_list.append(None)

    # Execute function body
    stack: list[Any] = []
    labels: list[Label] = []

    # Add implicit function-level label
    labels.append(Label(arity=len(func_type.results), target=len(func.body) - 1))

    ip = 0  # Instruction pointer
    body = func.body

    while ip < len(body):
        instr = body[ip]
        ip += 1

        # Execute instruction
        result = execute_instruction(
            instr, stack, labels, locals_list, instance, body, ip
        )

        if result is not None:
            if result[0] == "branch":
                ip = result[1]
            elif result[0] == "return":
                break
            elif result[0] == "call":
                # Recursive call
                call_func_idx = result[1]
                call_args = result[2]
                call_result = execute_function(instance, call_func_idx, call_args)
                if call_result is not None:
                    if isinstance(call_result, tuple):
                        stack.extend(call_result)
                    else:
                        stack.append(call_result)

    # Return results
    if len(func_type.results) == 0:
        return None
    elif len(func_type.results) == 1:
        return stack[-1] if stack else None
    else:
        return tuple(stack[-len(func_type.results) :])


def execute_instruction(
    instr: Instruction,
    stack: list[Any],
    labels: list[Label],
    locals_list: list[Any],
    instance: Instance,
    body: list[Instruction],
    ip: int,
) -> tuple | None:
    """Execute a single instruction. Returns control flow action or None."""
    op = instr.opcode

    # Constants
    if op == "i32.const":
        stack.append(to_i32(instr.operand))
    elif op == "i64.const":
        stack.append(to_i64(instr.operand))
    elif op == "f32.const":
        stack.append(float(instr.operand))
    elif op == "f64.const":
        stack.append(float(instr.operand))

    # Local variables
    elif op == "local.get":
        stack.append(locals_list[instr.operand])
    elif op == "local.set":
        locals_list[instr.operand] = stack.pop()
    elif op == "local.tee":
        locals_list[instr.operand] = stack[-1]

    # Global variables
    elif op == "global.get":
        stack.append(instance.globals[instr.operand].value)
    elif op == "global.set":
        g = instance.globals[instr.operand]
        if not g.mutable:
            raise TrapError("Cannot set immutable global")
        g.value = stack.pop()

    # Parametric
    elif op == "drop":
        stack.pop()
    elif op == "select":
        c = stack.pop()
        val2 = stack.pop()
        val1 = stack.pop()
        stack.append(val1 if c else val2)

    # i32 arithmetic
    elif op == "i32.add":
        b, a = stack.pop(), stack.pop()
        stack.append(to_i32(a + b))
    elif op == "i32.sub":
        b, a = stack.pop(), stack.pop()
        stack.append(to_i32(a - b))
    elif op == "i32.mul":
        b, a = stack.pop(), stack.pop()
        stack.append(to_i32(a * b))
    elif op == "i32.div_s":
        b, a = stack.pop(), stack.pop()
        if b == 0:
            raise TrapError("integer divide by zero")
        if a == -0x80000000 and b == -1:
            raise TrapError("integer overflow")
        # Python division truncates toward negative infinity, we need toward zero
        result = abs(a) // abs(b)
        if (a < 0) != (b < 0):
            result = -result
        stack.append(to_i32(result))
    elif op == "i32.div_u":
        b, a = stack.pop(), stack.pop()
        if b == 0:
            raise TrapError("integer divide by zero")
        stack.append(to_u32(a) // to_u32(b))
    elif op == "i32.rem_s":
        b, a = stack.pop(), stack.pop()
        if b == 0:
            raise TrapError("integer divide by zero")
        # Python's % differs from C's for negative numbers
        result = abs(a) % abs(b)
        if a < 0:
            result = -result
        stack.append(to_i32(result))
    elif op == "i32.rem_u":
        b, a = stack.pop(), stack.pop()
        if b == 0:
            raise TrapError("integer divide by zero")
        stack.append(to_u32(a) % to_u32(b))

    # i32 bitwise
    elif op == "i32.and":
        b, a = stack.pop(), stack.pop()
        stack.append(to_i32(to_u32(a) & to_u32(b)))
    elif op == "i32.or":
        b, a = stack.pop(), stack.pop()
        stack.append(to_i32(to_u32(a) | to_u32(b)))
    elif op == "i32.xor":
        b, a = stack.pop(), stack.pop()
        stack.append(to_i32(to_u32(a) ^ to_u32(b)))
    elif op == "i32.shl":
        b, a = stack.pop(), stack.pop()
        stack.append(to_i32(to_u32(a) << (to_u32(b) & 31)))
    elif op == "i32.shr_s":
        b, a = stack.pop(), stack.pop()
        shift = to_u32(b) & 31
        stack.append(to_i32(a >> shift))
    elif op == "i32.shr_u":
        b, a = stack.pop(), stack.pop()
        shift = to_u32(b) & 31
        stack.append(to_i32(to_u32(a) >> shift))
    elif op == "i32.rotl":
        b, a = stack.pop(), stack.pop()
        shift = to_u32(b) & 31
        ua = to_u32(a)
        stack.append(to_i32((ua << shift) | (ua >> (32 - shift))))
    elif op == "i32.rotr":
        b, a = stack.pop(), stack.pop()
        shift = to_u32(b) & 31
        ua = to_u32(a)
        stack.append(to_i32((ua >> shift) | (ua << (32 - shift))))

    # i32 unary
    elif op == "i32.clz":
        a = to_u32(stack.pop())
        if a == 0:
            stack.append(32)
        else:
            stack.append(32 - a.bit_length())
    elif op == "i32.ctz":
        a = to_u32(stack.pop())
        if a == 0:
            stack.append(32)
        else:
            count = 0
            while (a & 1) == 0:
                count += 1
                a >>= 1
            stack.append(count)
    elif op == "i32.popcnt":
        a = to_u32(stack.pop())
        stack.append(bin(a).count("1"))

    # i32 comparison
    elif op == "i32.eqz":
        stack.append(1 if stack.pop() == 0 else 0)
    elif op == "i32.eq":
        b, a = stack.pop(), stack.pop()
        stack.append(1 if to_i32(a) == to_i32(b) else 0)
    elif op == "i32.ne":
        b, a = stack.pop(), stack.pop()
        stack.append(1 if to_i32(a) != to_i32(b) else 0)
    elif op == "i32.lt_s":
        b, a = stack.pop(), stack.pop()
        stack.append(1 if to_i32(a) < to_i32(b) else 0)
    elif op == "i32.lt_u":
        b, a = stack.pop(), stack.pop()
        stack.append(1 if to_u32(a) < to_u32(b) else 0)
    elif op == "i32.gt_s":
        b, a = stack.pop(), stack.pop()
        stack.append(1 if to_i32(a) > to_i32(b) else 0)
    elif op == "i32.gt_u":
        b, a = stack.pop(), stack.pop()
        stack.append(1 if to_u32(a) > to_u32(b) else 0)
    elif op == "i32.le_s":
        b, a = stack.pop(), stack.pop()
        stack.append(1 if to_i32(a) <= to_i32(b) else 0)
    elif op == "i32.le_u":
        b, a = stack.pop(), stack.pop()
        stack.append(1 if to_u32(a) <= to_u32(b) else 0)
    elif op == "i32.ge_s":
        b, a = stack.pop(), stack.pop()
        stack.append(1 if to_i32(a) >= to_i32(b) else 0)
    elif op == "i32.ge_u":
        b, a = stack.pop(), stack.pop()
        stack.append(1 if to_u32(a) >= to_u32(b) else 0)

    # i64 arithmetic
    elif op == "i64.add":
        b, a = stack.pop(), stack.pop()
        stack.append(to_i64(a + b))
    elif op == "i64.sub":
        b, a = stack.pop(), stack.pop()
        stack.append(to_i64(a - b))
    elif op == "i64.mul":
        b, a = stack.pop(), stack.pop()
        stack.append(to_i64(a * b))

    # i64 comparison
    elif op == "i64.eqz":
        stack.append(1 if stack.pop() == 0 else 0)
    elif op == "i64.eq":
        b, a = stack.pop(), stack.pop()
        stack.append(1 if to_i64(a) == to_i64(b) else 0)
    elif op == "i64.ne":
        b, a = stack.pop(), stack.pop()
        stack.append(1 if to_i64(a) != to_i64(b) else 0)
    elif op == "i64.lt_s":
        b, a = stack.pop(), stack.pop()
        stack.append(1 if to_i64(a) < to_i64(b) else 0)
    elif op == "i64.lt_u":
        b, a = stack.pop(), stack.pop()
        stack.append(1 if to_u64(a) < to_u64(b) else 0)
    elif op == "i64.gt_s":
        b, a = stack.pop(), stack.pop()
        stack.append(1 if to_i64(a) > to_i64(b) else 0)
    elif op == "i64.gt_u":
        b, a = stack.pop(), stack.pop()
        stack.append(1 if to_u64(a) > to_u64(b) else 0)
    elif op == "i64.le_s":
        b, a = stack.pop(), stack.pop()
        stack.append(1 if to_i64(a) <= to_i64(b) else 0)
    elif op == "i64.le_u":
        b, a = stack.pop(), stack.pop()
        stack.append(1 if to_u64(a) <= to_u64(b) else 0)
    elif op == "i64.ge_s":
        b, a = stack.pop(), stack.pop()
        stack.append(1 if to_i64(a) >= to_i64(b) else 0)
    elif op == "i64.ge_u":
        b, a = stack.pop(), stack.pop()
        stack.append(1 if to_u64(a) >= to_u64(b) else 0)

    # Memory load operations
    elif op == "i32.load":
        align, offset = instr.operand
        addr = stack.pop()
        ea = addr + offset
        mem = instance.memories[0].data
        if ea < 0 or ea + 4 > len(mem):
            raise TrapError("out of bounds memory access")
        val = int.from_bytes(mem[ea : ea + 4], "little", signed=True)
        stack.append(val)

    elif op == "i32.load8_u":
        align, offset = instr.operand
        addr = stack.pop()
        ea = addr + offset  # effective address
        if ea < 0 or ea >= len(instance.memories[0].data):
            raise TrapError("out of bounds memory access")
        stack.append(instance.memories[0].data[ea])

    elif op == "i32.load8_s":
        align, offset = instr.operand
        addr = stack.pop()
        ea = addr + offset
        if ea < 0 or ea >= len(instance.memories[0].data):
            raise TrapError("out of bounds memory access")
        val = instance.memories[0].data[ea]
        if val >= 128:
            val -= 256
        stack.append(val)

    # Memory store operations
    elif op == "i32.store":
        align, offset = instr.operand
        val = stack.pop()
        addr = stack.pop()
        ea = addr + offset
        mem = instance.memories[0].data
        if ea < 0 or ea + 4 > len(mem):
            raise TrapError("out of bounds memory access")
        mem[ea : ea + 4] = (val & MASK_32).to_bytes(4, "little")

    elif op == "i32.store8":
        align, offset = instr.operand
        val = stack.pop()
        addr = stack.pop()
        ea = addr + offset
        if ea < 0 or ea >= len(instance.memories[0].data):
            raise TrapError("out of bounds memory access")
        instance.memories[0].data[ea] = val & 0xFF

    # Memory size/grow
    elif op == "memory.size":
        mem = instance.memories[0]
        stack.append(len(mem.data) // mem.PAGE_SIZE)

    elif op == "memory.grow":
        delta = stack.pop()
        mem = instance.memories[0]
        result = mem.grow(delta)
        stack.append(result)

    # Control flow
    elif op == "nop":
        pass
    elif op == "unreachable":
        raise TrapError("unreachable executed")
    elif op == "return":
        return ("return",)
    elif op == "end":
        if labels:
            labels.pop()

    elif op == "block":
        # Find matching end
        blocktype = instr.operand
        arity = 0 if blocktype == () else 1 if isinstance(blocktype, tuple) else 0
        end_ip = find_end(body, ip)
        labels.append(Label(arity=arity, target=end_ip))

    elif op == "loop":
        # Loop label points to start of loop
        blocktype = instr.operand
        arity = 0  # Loop takes no values on branch (jumps to start)
        end_ip = find_end(body, ip)
        labels.append(Label(arity=arity, target=ip - 1, is_loop=True))

    elif op == "if":
        condition = stack.pop()
        blocktype = instr.operand
        arity = 0 if blocktype == () else 1 if isinstance(blocktype, tuple) else 0

        # Find else and end
        else_ip, end_ip = find_else_end(body, ip)

        if condition:
            # Execute then branch
            labels.append(Label(arity=arity, target=end_ip))
        else:
            # Skip to else or end
            labels.append(Label(arity=arity, target=end_ip))
            if else_ip is not None:
                # Branch past the else instruction to execute else body
                return ("branch", else_ip + 1)
            else:
                # No else branch, skip to end
                return ("branch", end_ip + 1)

    elif op == "else":
        # When we hit else during normal execution, skip to end
        if labels:
            label = labels[-1]
            return ("branch", label.target)

    elif op == "br":
        depth = instr.operand
        return do_branch(stack, labels, depth)

    elif op == "br_if":
        condition = stack.pop()
        if condition:
            depth = instr.operand
            return do_branch(stack, labels, depth)

    elif op == "br_table":
        label_indices, default = instr.operand
        idx = stack.pop()
        if 0 <= idx < len(label_indices):
            depth = label_indices[idx]
        else:
            depth = default
        return do_branch(stack, labels, depth)

    elif op == "call":
        func_idx = instr.operand
        func = instance.funcs[func_idx]
        func_type = instance.func_types[func.type_idx]
        n_params = len(func_type.params)
        args = [stack.pop() for _ in range(n_params)][::-1]
        return ("call", func_idx, args)

    else:
        raise TrapError(f"Unimplemented instruction: {op}")

    return None


def do_branch(stack: list[Any], labels: list[Label], depth: int) -> tuple:
    """Execute a branch to the given label depth."""
    # Count from innermost label
    label_idx = len(labels) - 1 - depth
    if label_idx < 0:
        raise TrapError(f"Invalid branch depth: {depth}")

    label = labels[label_idx]

    # Pop labels up to and including the target
    for _ in range(depth + 1):
        labels.pop()

    # For loops, we re-enter the loop (no stack adjustment needed for arity 0)
    # For blocks, we exit with arity values on stack
    if label.is_loop:
        # Re-add the loop label for the next iteration
        labels.append(label)

    return ("branch", label.target + 1 if not label.is_loop else label.target + 1)


def find_end(body: list[Instruction], start_ip: int) -> int:
    """Find the matching 'end' instruction for a block/loop starting at start_ip."""
    depth = 1
    ip = start_ip
    while ip < len(body):
        op = body[ip].opcode
        if op in ("block", "loop", "if"):
            depth += 1
        elif op == "end":
            depth -= 1
            if depth == 0:
                return ip
        ip += 1
    raise TrapError("No matching end found")


def find_else_end(body: list[Instruction], start_ip: int) -> tuple[int | None, int]:
    """Find 'else' and 'end' for an if starting at start_ip."""
    depth = 1
    ip = start_ip
    else_ip = None
    while ip < len(body):
        op = body[ip].opcode
        if op in ("block", "loop", "if"):
            depth += 1
        elif op == "else" and depth == 1:
            else_ip = ip
        elif op == "end":
            depth -= 1
            if depth == 0:
                return else_ip, ip
        ip += 1
    raise TrapError("No matching end found for if")


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
            # For now, we don't support imported functions in execution
            # We'd need to wrap Python callables
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
        if data_seg.memory_idx >= 0 and memories:  # Active segment
            mem = memories[data_seg.memory_idx]
            # Evaluate offset expression (simplified: assume i32.const)
            offset = 0
            for instr in data_seg.offset:
                if instr.opcode == "i32.const":
                    offset = instr.operand
                    break
            # Copy data
            mem.data[offset : offset + len(data_seg.init)] = data_seg.init

    # Initialize globals
    globals_list: list[GlobalInstance] = []
    for glob in module.globals:
        # Evaluate init expression (simplified: assume const)
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
