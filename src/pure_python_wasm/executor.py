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


class Label:
    """A control flow label for block/loop/if."""

    __slots__ = ("arity", "target", "is_loop", "stack_height")

    def __init__(
        self, arity: int, target: int, is_loop: bool = False, stack_height: int = 0
    ):
        self.arity = arity
        self.target = target
        self.is_loop = is_loop
        self.stack_height = stack_height


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


@dataclass
class TableInstance:
    """Runtime table instance.

    Elements can be:
    - None: null reference
    - int: function index (for funcref tables)
    - ('func', idx): function reference
    - ('extern', val): external reference
    """

    elements: list[Any]  # Reference values or None
    max: int | None = None
    elem_type: str = "funcref"  # 'funcref' or 'externref'


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
    funcs: list[Function | None]  # All functions (imports + module-defined)
    func_types: list[FuncType]
    memories: list[MemoryInstance]
    globals: list[GlobalInstance]
    tables: list[TableInstance] = field(default_factory=list)
    imported_funcs: dict[int, Callable] = field(default_factory=dict)  # idx -> callable
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
    """Execute a function and return its result(s).

    This uses the optimized fast executor by default.
    """
    # Use the optimized version
    return execute_function_fast(instance, func_idx, args)


def execute_function_original(
    instance: Instance, func_idx: int, args: list[Any]
) -> Any:
    """Execute a function using the original interpreter (for reference/debugging)."""
    # Check if this is an imported function
    if func_idx in instance.imported_funcs:
        imported_fn = instance.imported_funcs[func_idx]
        return imported_fn(*args)

    func = instance.funcs[func_idx]
    if func is None:
        raise TrapError(f"Function {func_idx} is not available (missing import?)")
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
    labels.append(
        Label(arity=len(func_type.results), target=len(func.body) - 1, stack_height=0)
    )

    ip = 0  # Instruction pointer
    body = func.body

    # Get cached jump targets (or compute them)
    jump_targets = get_jump_targets(func, body)

    while ip < len(body):
        instr = body[ip]
        ip += 1

        # Execute instruction
        result = execute_instruction(
            instr, stack, labels, locals_list, instance, body, ip, jump_targets
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


def execute_function_fast(instance: Instance, func_idx: int, args: list[Any]) -> Any:
    """Optimized function execution with inlined instruction dispatch."""
    # Check if this is an imported function
    if func_idx in instance.imported_funcs:
        return instance.imported_funcs[func_idx](*args)

    func = instance.funcs[func_idx]
    if func is None:
        raise TrapError(f"Function {func_idx} is not available (missing import?)")
    func_type = instance.func_types[func.type_idx]

    # Set up locals: params + declared locals
    locals_list: list[Any] = list(args)
    for valtype in func.locals:
        if valtype in ("i32", "i64"):
            locals_list.append(0)
        elif valtype in ("f32", "f64"):
            locals_list.append(0.0)
        else:
            locals_list.append(None)

    # Execute function body with inlined dispatch
    stack: list[Any] = []
    labels: list[Label] = []
    body = func.body
    body_len = len(body)

    # Add implicit function-level label
    labels.append(
        Label(arity=len(func_type.results), target=body_len - 1, stack_height=0)
    )

    # Get cached jump targets
    jump_targets = get_jump_targets(func, body)

    # Cache method references for speed
    stack_append = stack.append
    stack_pop = stack.pop
    labels_append = labels.append
    labels_pop = labels.pop

    # Constants for masking
    _MASK_32 = MASK_32
    _MASK_64 = MASK_64

    # Optimized main loop
    ip = 0
    while ip < body_len:
        instr = body[ip]
        op = instr.opcode
        operand = instr.operand
        ip += 1

        # Most common opcodes first (based on profiling)
        if op == "br":
            # Inline do_branch for performance
            depth = operand
            label_idx = len(labels) - 1 - depth
            label = labels[label_idx]
            for _ in range(depth + 1):
                labels_pop()
            if label.arity > 0:
                result_values = stack[-label.arity :]
                del stack[label.stack_height :]
                stack.extend(result_values)
            else:
                del stack[label.stack_height :]
            if label.is_loop:
                labels_append(label)
            ip = label.target + 1
            continue

        if op == "br_if":
            condition = stack_pop()
            if condition:
                depth = operand
                label_idx = len(labels) - 1 - depth
                label = labels[label_idx]
                for _ in range(depth + 1):
                    labels_pop()
                if label.arity > 0:
                    result_values = stack[-label.arity :]
                    del stack[label.stack_height :]
                    stack.extend(result_values)
                else:
                    del stack[label.stack_height :]
                if label.is_loop:
                    labels_append(label)
                ip = label.target + 1
            continue

        if op == "local.get":
            stack_append(locals_list[operand])
            continue

        if op == "local.set":
            locals_list[operand] = stack_pop()
            continue

        if op == "local.tee":
            locals_list[operand] = stack[-1]
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

        if op == "i32.sub":
            b, a = stack_pop(), stack_pop()
            val = (a - b) & _MASK_32
            if val >= 0x80000000:
                val -= 0x100000000
            stack_append(val)
            continue

        if op == "i32.and":
            b, a = stack_pop(), stack_pop()
            val = ((a & _MASK_32) & (b & _MASK_32)) & _MASK_32
            if val >= 0x80000000:
                val -= 0x100000000
            stack_append(val)
            continue

        if op == "i32.or":
            b, a = stack_pop(), stack_pop()
            val = ((a & _MASK_32) | (b & _MASK_32)) & _MASK_32
            if val >= 0x80000000:
                val -= 0x100000000
            stack_append(val)
            continue

        if op == "i32.eqz":
            stack_append(1 if stack_pop() == 0 else 0)
            continue

        if op == "i32.eq":
            b, a = stack_pop(), stack_pop()
            stack_append(1 if (a & _MASK_32) == (b & _MASK_32) else 0)
            continue

        if op == "i32.ne":
            b, a = stack_pop(), stack_pop()
            stack_append(1 if (a & _MASK_32) != (b & _MASK_32) else 0)
            continue

        if op == "i32.lt_u":
            b, a = stack_pop(), stack_pop()
            stack_append(1 if (a & _MASK_32) < (b & _MASK_32) else 0)
            continue

        if op == "i32.gt_u":
            b, a = stack_pop(), stack_pop()
            stack_append(1 if (a & _MASK_32) > (b & _MASK_32) else 0)
            continue

        if op == "i32.le_u":
            b, a = stack_pop(), stack_pop()
            stack_append(1 if (a & _MASK_32) <= (b & _MASK_32) else 0)
            continue

        if op == "i32.ge_u":
            b, a = stack_pop(), stack_pop()
            stack_append(1 if (a & _MASK_32) >= (b & _MASK_32) else 0)
            continue

        if op == "i32.lt_s":
            b, a = stack_pop(), stack_pop()
            # Convert to signed
            sa = (
                (a & _MASK_32) - 0x100000000
                if (a & _MASK_32) >= 0x80000000
                else (a & _MASK_32)
            )
            sb = (
                (b & _MASK_32) - 0x100000000
                if (b & _MASK_32) >= 0x80000000
                else (b & _MASK_32)
            )
            stack_append(1 if sa < sb else 0)
            continue

        if op == "i32.gt_s":
            b, a = stack_pop(), stack_pop()
            sa = (
                (a & _MASK_32) - 0x100000000
                if (a & _MASK_32) >= 0x80000000
                else (a & _MASK_32)
            )
            sb = (
                (b & _MASK_32) - 0x100000000
                if (b & _MASK_32) >= 0x80000000
                else (b & _MASK_32)
            )
            stack_append(1 if sa > sb else 0)
            continue

        if op == "i32.le_s":
            b, a = stack_pop(), stack_pop()
            sa = (
                (a & _MASK_32) - 0x100000000
                if (a & _MASK_32) >= 0x80000000
                else (a & _MASK_32)
            )
            sb = (
                (b & _MASK_32) - 0x100000000
                if (b & _MASK_32) >= 0x80000000
                else (b & _MASK_32)
            )
            stack_append(1 if sa <= sb else 0)
            continue

        if op == "i32.ge_s":
            b, a = stack_pop(), stack_pop()
            sa = (
                (a & _MASK_32) - 0x100000000
                if (a & _MASK_32) >= 0x80000000
                else (a & _MASK_32)
            )
            sb = (
                (b & _MASK_32) - 0x100000000
                if (b & _MASK_32) >= 0x80000000
                else (b & _MASK_32)
            )
            stack_append(1 if sa >= sb else 0)
            continue

        if op == "i32.mul":
            b, a = stack_pop(), stack_pop()
            val = (a * b) & _MASK_32
            if val >= 0x80000000:
                val -= 0x100000000
            stack_append(val)
            continue

        if op == "i32.xor":
            b, a = stack_pop(), stack_pop()
            val = ((a & _MASK_32) ^ (b & _MASK_32)) & _MASK_32
            if val >= 0x80000000:
                val -= 0x100000000
            stack_append(val)
            continue

        if op == "i32.shl":
            b, a = stack_pop(), stack_pop()
            val = ((a & _MASK_32) << ((b & _MASK_32) & 31)) & _MASK_32
            if val >= 0x80000000:
                val -= 0x100000000
            stack_append(val)
            continue

        if op == "i32.shr_u":
            b, a = stack_pop(), stack_pop()
            shift = (b & _MASK_32) & 31
            val = ((a & _MASK_32) >> shift) & _MASK_32
            if val >= 0x80000000:
                val -= 0x100000000
            stack_append(val)
            continue

        if op == "i32.shr_s":
            b, a = stack_pop(), stack_pop()
            shift = (b & _MASK_32) & 31
            sa = (
                (a & _MASK_32) - 0x100000000
                if (a & _MASK_32) >= 0x80000000
                else (a & _MASK_32)
            )
            val = (sa >> shift) & _MASK_32
            if val >= 0x80000000:
                val -= 0x100000000
            stack_append(val)
            continue

        if op == "i32.load":
            align, offset = operand
            addr = stack_pop()
            ea = addr + offset
            mem = instance.memories[0].data
            if ea < 0 or ea + 4 > len(mem):
                raise TrapError("out of bounds memory access")
            val = int.from_bytes(mem[ea : ea + 4], "little", signed=True)
            stack_append(val)
            continue

        if op == "i32.load8_u":
            align, offset = operand
            addr = stack_pop()
            ea = addr + offset
            mem = instance.memories[0].data
            if ea < 0 or ea >= len(mem):
                raise TrapError("out of bounds memory access")
            stack_append(mem[ea])
            continue

        if op == "i32.load8_s":
            align, offset = operand
            addr = stack_pop()
            ea = addr + offset
            mem = instance.memories[0].data
            if ea < 0 or ea >= len(mem):
                raise TrapError("out of bounds memory access")
            val = mem[ea]
            if val >= 128:
                val -= 256
            stack_append(val)
            continue

        if op == "i32.store":
            align, offset = operand
            val = stack_pop()
            addr = stack_pop()
            ea = addr + offset
            mem = instance.memories[0].data
            if ea < 0 or ea + 4 > len(mem):
                raise TrapError("out of bounds memory access")
            mem[ea : ea + 4] = (val & _MASK_32).to_bytes(4, "little")
            continue

        if op == "i32.store8":
            align, offset = operand
            val = stack_pop()
            addr = stack_pop()
            ea = addr + offset
            mem = instance.memories[0].data
            if ea < 0 or ea >= len(mem):
                raise TrapError("out of bounds memory access")
            mem[ea] = val & 0xFF
            continue

        if op == "global.get":
            stack_append(instance.globals[operand].value)
            continue

        if op == "global.set":
            instance.globals[operand].value = stack_pop()
            continue

        if op == "i64.const":
            val = operand & _MASK_64
            if val >= 0x8000000000000000:
                val -= 0x10000000000000000
            stack_append(val)
            continue

        if op == "i64.add":
            b, a = stack_pop(), stack_pop()
            val = (a + b) & _MASK_64
            if val >= 0x8000000000000000:
                val -= 0x10000000000000000
            stack_append(val)
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
            call_args = [stack_pop() for _ in range(n_params)][::-1]
            call_result = execute_function_fast(instance, call_func_idx, call_args)
            if call_result is not None:
                if isinstance(call_result, tuple):
                    stack.extend(call_result)
                else:
                    stack_append(call_result)
            continue

        if op == "if":
            condition = stack_pop()
            blocktype = operand
            arity = 0 if blocktype == () else 1 if isinstance(blocktype, tuple) else 0
            if ip - 1 in jump_targets:
                else_ip, end_ip = jump_targets[ip - 1]
            else:
                else_ip, end_ip = find_else_end(body, ip - 1, jump_targets)
            if condition:
                labels_append(
                    Label(arity=arity, target=end_ip, stack_height=len(stack))
                )
            else:
                labels_append(
                    Label(arity=arity, target=end_ip, stack_height=len(stack))
                )
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

        if op == "block":
            blocktype = operand
            arity = 0 if blocktype == () else 1 if isinstance(blocktype, tuple) else 0
            end_ip = (
                jump_targets[ip - 1][1]
                if ip - 1 in jump_targets
                else find_end(body, ip - 1, jump_targets)
            )
            labels_append(Label(arity=arity, target=end_ip, stack_height=len(stack)))
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
            end_ip = (
                jump_targets[ip - 1][1]
                if ip - 1 in jump_targets
                else find_end(body, ip - 1, jump_targets)
            )
            labels_append(
                Label(
                    arity=arity,
                    target=ip - 1,
                    is_loop=True,
                    stack_height=len(stack) - arity,
                )
            )
            continue

        if op == "end":
            if labels:
                labels_pop()
            continue

        if op == "drop":
            stack_pop()
            continue

        if op == "select":
            c = stack_pop()
            val2 = stack_pop()
            val1 = stack_pop()
            stack_append(val1 if c else val2)
            continue

        if op == "nop":
            continue

        if op == "return":
            break

        # Bulk memory operations (for optimized WASM builds)
        if op == "memory.copy":
            n = stack_pop()
            src = stack_pop()
            dst = stack_pop()
            mem = instance.memories[0].data
            if n > 0:
                mem[dst : dst + n] = bytes(mem[src : src + n])
            continue

        if op == "memory.fill":
            n = stack_pop()
            val = stack_pop() & 0xFF
            dst = stack_pop()
            mem = instance.memories[0].data
            if n > 0:
                mem[dst : dst + n] = bytes([val] * n)
            continue

        # Fall back to regular instruction dispatch for less common ops
        result = execute_instruction(
            instr, stack, labels, locals_list, instance, body, ip, jump_targets
        )

        if result is not None:
            if result[0] == "branch":
                ip = result[1]
            elif result[0] == "return":
                break
            elif result[0] == "call":
                call_result = execute_function_fast(instance, result[1], result[2])
                if call_result is not None:
                    if isinstance(call_result, tuple):
                        stack.extend(call_result)
                    else:
                        stack_append(call_result)

    # Return results
    num_results = len(func_type.results)
    if num_results == 0:
        return None
    elif num_results == 1:
        return stack[-1] if stack else None
    else:
        return tuple(stack[-num_results:])


def execute_instruction(
    instr: Instruction,
    stack: list[Any],
    labels: list[Label],
    locals_list: list[Any],
    instance: Instance,
    body: list[Instruction],
    ip: int,
    jump_targets: dict | None = None,
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

    # i64 bitwise operations
    elif op == "i64.and":
        b, a = stack.pop(), stack.pop()
        stack.append(to_i64((a & MASK_64) & (b & MASK_64)))
    elif op == "i64.or":
        b, a = stack.pop(), stack.pop()
        stack.append(to_i64((a & MASK_64) | (b & MASK_64)))
    elif op == "i64.xor":
        b, a = stack.pop(), stack.pop()
        stack.append(to_i64((a & MASK_64) ^ (b & MASK_64)))
    elif op == "i64.shl":
        b, a = stack.pop(), stack.pop()
        shift = b & 63  # Only use low 6 bits for shift amount
        stack.append(to_i64((a & MASK_64) << shift))
    elif op == "i64.shr_s":
        b, a = stack.pop(), stack.pop()
        shift = b & 63
        signed_a = to_i64(a)
        stack.append(signed_a >> shift)
    elif op == "i64.shr_u":
        b, a = stack.pop(), stack.pop()
        shift = b & 63
        stack.append((a & MASK_64) >> shift)
    elif op == "i64.rotl":
        b, a = stack.pop(), stack.pop()
        shift = b & 63
        ua = a & MASK_64
        result = ((ua << shift) | (ua >> (64 - shift))) & MASK_64
        stack.append(to_i64(result))
    elif op == "i64.clz":
        val = stack.pop() & MASK_64
        if val == 0:
            stack.append(64)
        else:
            count = 0
            while (val & (1 << 63)) == 0:
                count += 1
                val <<= 1
            stack.append(count)
    elif op == "i64.div_u":
        b, a = stack.pop(), stack.pop()
        ua, ub = a & MASK_64, b & MASK_64
        if ub == 0:
            raise TrapError("integer divide by zero")
        stack.append(ua // ub)
    elif op == "i64.rem_u":
        b, a = stack.pop(), stack.pop()
        ua, ub = a & MASK_64, b & MASK_64
        if ub == 0:
            raise TrapError("integer divide by zero")
        stack.append(ua % ub)

    # i64 extend operations
    elif op == "i64.extend_i32_s":
        val = stack.pop()
        # Sign extend 32-bit to 64-bit
        if val >= 0x80000000:
            val -= 0x100000000
        stack.append(val)
    elif op == "i64.extend_i32_u":
        val = stack.pop()
        # Zero extend 32-bit to 64-bit
        stack.append(val & MASK_32)

    # Type conversions
    elif op == "i32.wrap_i64":
        val = stack.pop()
        # Truncate to low 32 bits and interpret as signed i32
        result = val & MASK_32
        if result >= 0x80000000:
            result -= 0x100000000
        stack.append(result)

    elif op == "i32.trunc_f64_s":
        import math

        val = stack.pop()
        if math.isnan(val) or math.isinf(val):
            raise TrapError("invalid conversion to integer")
        if val >= 2147483648.0 or val < -2147483649.0:
            raise TrapError("integer overflow")
        stack.append(int(val))

    elif op == "i32.trunc_f64_u":
        import math

        val = stack.pop()
        if math.isnan(val) or math.isinf(val):
            raise TrapError("invalid conversion to integer")
        if val >= 4294967296.0 or val < -1.0:
            raise TrapError("integer overflow")
        stack.append(int(val) & MASK_32)

    elif op == "i32.trunc_f32_s":
        import math

        val = stack.pop()
        if math.isnan(val) or math.isinf(val):
            raise TrapError("invalid conversion to integer")
        if val >= 2147483648.0 or val < -2147483649.0:
            raise TrapError("integer overflow")
        stack.append(int(val))

    elif op == "f64.convert_i32_s":
        val = stack.pop()
        stack.append(float(to_i32(val)))

    elif op == "f64.convert_i32_u":
        val = stack.pop()
        stack.append(float(val & MASK_32))

    elif op == "f64.promote_f32":
        val = stack.pop()
        stack.append(float(val))

    elif op == "f32.demote_f64":
        import struct

        val = stack.pop()
        # Convert to f32 and back to handle precision
        (result,) = struct.unpack("<f", struct.pack("<f", val))
        stack.append(result)

    elif op == "i64.reinterpret_f64":
        import struct

        val = stack.pop()
        (result,) = struct.unpack("<q", struct.pack("<d", val))
        stack.append(result)

    elif op == "f64.reinterpret_i64":
        import struct

        val = stack.pop()
        (result,) = struct.unpack("<d", struct.pack("<q", val & MASK_64))
        stack.append(result)

    # f64 arithmetic
    elif op == "f64.add":
        b, a = stack.pop(), stack.pop()
        stack.append(a + b)

    elif op == "f64.sub":
        b, a = stack.pop(), stack.pop()
        stack.append(a - b)

    elif op == "f64.mul":
        b, a = stack.pop(), stack.pop()
        stack.append(a * b)

    elif op == "f64.div":
        b, a = stack.pop(), stack.pop()
        stack.append(a / b)

    elif op == "f64.abs":
        val = stack.pop()
        stack.append(abs(val))

    elif op == "f64.neg":
        val = stack.pop()
        stack.append(-val)

    elif op == "f64.copysign":
        import math

        b, a = stack.pop(), stack.pop()
        stack.append(math.copysign(a, b))

    # f64 comparison
    elif op == "f64.eq":
        b, a = stack.pop(), stack.pop()
        stack.append(1 if a == b else 0)

    elif op == "f64.ne":
        b, a = stack.pop(), stack.pop()
        stack.append(1 if a != b else 0)

    elif op == "f64.lt":
        b, a = stack.pop(), stack.pop()
        stack.append(1 if a < b else 0)

    elif op == "f64.gt":
        b, a = stack.pop(), stack.pop()
        stack.append(1 if a > b else 0)

    elif op == "f64.le":
        b, a = stack.pop(), stack.pop()
        stack.append(1 if a <= b else 0)

    elif op == "f64.ge":
        b, a = stack.pop(), stack.pop()
        stack.append(1 if a >= b else 0)

    # f32 operations
    elif op == "f32.abs":
        val = stack.pop()
        stack.append(abs(val))

    elif op == "f32.lt":
        b, a = stack.pop(), stack.pop()
        stack.append(1 if a < b else 0)

    elif op == "f32.ge":
        b, a = stack.pop(), stack.pop()
        stack.append(1 if a >= b else 0)

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

    elif op == "i32.load16_u":
        align, offset = instr.operand
        addr = stack.pop()
        ea = addr + offset
        mem = instance.memories[0].data
        if ea < 0 or ea + 2 > len(mem):
            raise TrapError("out of bounds memory access")
        stack.append(int.from_bytes(mem[ea : ea + 2], "little"))

    elif op == "i32.load16_s":
        align, offset = instr.operand
        addr = stack.pop()
        ea = addr + offset
        mem = instance.memories[0].data
        if ea < 0 or ea + 2 > len(mem):
            raise TrapError("out of bounds memory access")
        val = int.from_bytes(mem[ea : ea + 2], "little")
        if val >= 0x8000:
            val -= 0x10000
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

    elif op == "i64.load":
        align, offset = instr.operand
        addr = stack.pop()
        ea = addr + offset
        mem = instance.memories[0].data
        if ea < 0 or ea + 8 > len(mem):
            raise TrapError("out of bounds memory access")
        val = int.from_bytes(mem[ea : ea + 8], "little")
        # Convert to signed 64-bit
        if val >= 0x8000000000000000:
            val -= 0x10000000000000000
        stack.append(val)

    elif op == "i64.store":
        align, offset = instr.operand
        val = stack.pop()
        addr = stack.pop()
        ea = addr + offset
        mem = instance.memories[0].data
        if ea < 0 or ea + 8 > len(mem):
            raise TrapError("out of bounds memory access")
        mem[ea : ea + 8] = (val & MASK_64).to_bytes(8, "little")

    elif op == "i64.load8_s":
        align, offset = instr.operand
        addr = stack.pop()
        ea = addr + offset
        if ea < 0 or ea >= len(instance.memories[0].data):
            raise TrapError("out of bounds memory access")
        val = instance.memories[0].data[ea]
        if val >= 128:
            val -= 256
        stack.append(val)

    elif op == "i64.load8_u":
        align, offset = instr.operand
        addr = stack.pop()
        ea = addr + offset
        if ea < 0 or ea >= len(instance.memories[0].data):
            raise TrapError("out of bounds memory access")
        stack.append(instance.memories[0].data[ea])

    elif op == "i64.load16_s":
        align, offset = instr.operand
        addr = stack.pop()
        ea = addr + offset
        mem = instance.memories[0].data
        if ea < 0 or ea + 2 > len(mem):
            raise TrapError("out of bounds memory access")
        val = int.from_bytes(mem[ea : ea + 2], "little")
        if val >= 0x8000:
            val -= 0x10000
        stack.append(val)

    elif op == "i64.load16_u":
        align, offset = instr.operand
        addr = stack.pop()
        ea = addr + offset
        mem = instance.memories[0].data
        if ea < 0 or ea + 2 > len(mem):
            raise TrapError("out of bounds memory access")
        stack.append(int.from_bytes(mem[ea : ea + 2], "little"))

    elif op == "i64.load32_s":
        align, offset = instr.operand
        addr = stack.pop()
        ea = addr + offset
        mem = instance.memories[0].data
        if ea < 0 or ea + 4 > len(mem):
            raise TrapError("out of bounds memory access")
        val = int.from_bytes(mem[ea : ea + 4], "little")
        if val >= 0x80000000:
            val -= 0x100000000
        stack.append(val)

    elif op == "i64.load32_u":
        align, offset = instr.operand
        addr = stack.pop()
        ea = addr + offset
        mem = instance.memories[0].data
        if ea < 0 or ea + 4 > len(mem):
            raise TrapError("out of bounds memory access")
        stack.append(int.from_bytes(mem[ea : ea + 4], "little"))

    elif op == "i64.store8":
        align, offset = instr.operand
        val = stack.pop()
        addr = stack.pop()
        ea = addr + offset
        if ea < 0 or ea >= len(instance.memories[0].data):
            raise TrapError("out of bounds memory access")
        instance.memories[0].data[ea] = val & 0xFF

    elif op == "i64.store16":
        align, offset = instr.operand
        val = stack.pop()
        addr = stack.pop()
        ea = addr + offset
        mem = instance.memories[0].data
        if ea < 0 or ea + 2 > len(mem):
            raise TrapError("out of bounds memory access")
        mem[ea : ea + 2] = (val & 0xFFFF).to_bytes(2, "little")

    elif op == "i64.store32":
        align, offset = instr.operand
        val = stack.pop()
        addr = stack.pop()
        ea = addr + offset
        mem = instance.memories[0].data
        if ea < 0 or ea + 4 > len(mem):
            raise TrapError("out of bounds memory access")
        mem[ea : ea + 4] = (val & 0xFFFFFFFF).to_bytes(4, "little")

    elif op == "i32.store16":
        align, offset = instr.operand
        val = stack.pop()
        addr = stack.pop()
        ea = addr + offset
        mem = instance.memories[0].data
        if ea < 0 or ea + 2 > len(mem):
            raise TrapError("out of bounds memory access")
        mem[ea : ea + 2] = (val & 0xFFFF).to_bytes(2, "little")

    # Float load operations
    elif op == "f32.load":
        import struct

        align, offset = instr.operand
        addr = stack.pop()
        ea = addr + offset
        mem = instance.memories[0].data
        if ea < 0 or ea + 4 > len(mem):
            raise TrapError("out of bounds memory access")
        (val,) = struct.unpack("<f", bytes(mem[ea : ea + 4]))
        stack.append(val)

    elif op == "f64.load":
        import struct

        align, offset = instr.operand
        addr = stack.pop()
        ea = addr + offset
        mem = instance.memories[0].data
        if ea < 0 or ea + 8 > len(mem):
            raise TrapError("out of bounds memory access")
        (val,) = struct.unpack("<d", bytes(mem[ea : ea + 8]))
        stack.append(val)

    # Float store operations
    elif op == "f32.store":
        import struct

        align, offset = instr.operand
        val = stack.pop()
        addr = stack.pop()
        ea = addr + offset
        mem = instance.memories[0].data
        if ea < 0 or ea + 4 > len(mem):
            raise TrapError("out of bounds memory access")
        mem[ea : ea + 4] = struct.pack("<f", val)

    elif op == "f64.store":
        import struct

        align, offset = instr.operand
        val = stack.pop()
        addr = stack.pop()
        ea = addr + offset
        mem = instance.memories[0].data
        if ea < 0 or ea + 8 > len(mem):
            raise TrapError("out of bounds memory access")
        mem[ea : ea + 8] = struct.pack("<d", val)

    # Memory size/grow
    elif op == "memory.size":
        mem = instance.memories[0]
        stack.append(len(mem.data) // mem.PAGE_SIZE)

    elif op == "memory.grow":
        delta = stack.pop()
        mem = instance.memories[0]
        result = mem.grow(delta)
        stack.append(result)

    # Table operations
    elif op == "table.get":
        table_idx = instr.operand
        idx = stack.pop()
        if table_idx >= len(instance.tables):
            raise TrapError("undefined table")
        table = instance.tables[table_idx]
        if idx < 0 or idx >= len(table.elements):
            raise TrapError("out of bounds table access")
        stack.append(table.elements[idx])

    elif op == "table.set":
        table_idx = instr.operand
        val = stack.pop()
        idx = stack.pop()
        if table_idx >= len(instance.tables):
            raise TrapError("undefined table")
        table = instance.tables[table_idx]
        if idx < 0 or idx >= len(table.elements):
            raise TrapError("out of bounds table access")
        table.elements[idx] = val

    elif op == "table.size":
        table_idx = instr.operand
        if table_idx >= len(instance.tables):
            raise TrapError("undefined table")
        table = instance.tables[table_idx]
        stack.append(len(table.elements))

    elif op == "table.grow":
        table_idx = instr.operand
        n = stack.pop()  # Number of elements to grow
        init = stack.pop()  # Initial value for new elements
        if table_idx >= len(instance.tables):
            raise TrapError("undefined table")
        table = instance.tables[table_idx]
        old_size = len(table.elements)
        # Check if growth would exceed maximum
        if table.max is not None and old_size + n > table.max:
            stack.append(-1)  # Return -1 on failure
        else:
            table.elements.extend([init] * n)
            stack.append(old_size)  # Return old size on success

    elif op == "table.fill":
        table_idx = instr.operand
        n = stack.pop()  # Number of elements
        val = stack.pop()  # Value to fill
        i = stack.pop()  # Start index
        if table_idx >= len(instance.tables):
            raise TrapError("undefined table")
        table = instance.tables[table_idx]
        if i < 0 or i + n > len(table.elements):
            raise TrapError("out of bounds table access")
        for j in range(n):
            table.elements[i + j] = val

    # Reference operations
    elif op == "ref.null":
        # Push null reference of specified type
        stack.append(None)

    elif op == "ref.is_null":
        val = stack.pop()
        stack.append(1 if val is None else 0)

    elif op == "ref.func":
        func_idx = instr.operand
        # Push a function reference
        stack.append(("func", func_idx))

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
        # ip is already incremented, so use ip-1 for cache lookup (the actual block position)
        end_ip = find_end(body, ip - 1, jump_targets)
        labels.append(Label(arity=arity, target=end_ip, stack_height=len(stack)))

    elif op == "loop":
        # Loop label points to start of loop
        blocktype = instr.operand
        # For loops, arity is the number of PARAMETERS (values needed when branching back)
        if blocktype == ():
            arity = 0
        elif isinstance(blocktype, tuple):
            arity = 1  # Single value type like ('i32',)
        elif isinstance(blocktype, int):
            # Type index - get parameters count from function type
            func_type = instance.func_types[blocktype]
            arity = len(func_type.params)
        else:
            arity = 0
        # ip is already incremented, so use ip-1 for cache lookup (the actual loop position)
        end_ip = find_end(body, ip - 1, jump_targets)
        # For loops, stack_height is BEFORE the parameters (they're consumed on entry)
        labels.append(
            Label(
                arity=arity,
                target=ip - 1,
                is_loop=True,
                stack_height=len(stack) - arity,
            )
        )

    elif op == "if":
        condition = stack.pop()
        blocktype = instr.operand
        arity = 0 if blocktype == () else 1 if isinstance(blocktype, tuple) else 0

        # Find else and end
        # ip is already incremented, so use ip-1 for cache lookup (the actual if position)
        else_ip, end_ip = find_else_end(body, ip - 1, jump_targets)

        if condition:
            # Execute then branch
            labels.append(Label(arity=arity, target=end_ip, stack_height=len(stack)))
        else:
            # Skip to else or end
            labels.append(Label(arity=arity, target=end_ip, stack_height=len(stack)))
            if else_ip is not None:
                # Branch past the else instruction to execute else body
                return ("branch", else_ip + 1)
            else:
                # No else branch, skip to end instruction to pop the label
                return ("branch", end_ip)

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
        if func is None:
            # Imported function - get type from import descriptor
            import_idx = func_idx  # imports come first
            imp = instance.module.imports[import_idx]
            func_type = instance.func_types[imp.desc]
        else:
            func_type = instance.func_types[func.type_idx]
        n_params = len(func_type.params)
        args = [stack.pop() for _ in range(n_params)][::-1]
        return ("call", func_idx, args)

    elif op == "call_indirect":
        type_idx, table_idx = instr.operand
        # Pop the table index from stack
        idx = stack.pop()
        # Get function index from table
        if table_idx >= len(instance.tables):
            raise TrapError("undefined table")
        table = instance.tables[table_idx]
        if idx < 0 or idx >= len(table.elements):
            raise TrapError("undefined element")
        func_idx = table.elements[idx]
        if func_idx is None:
            raise TrapError("uninitialized element")
        # Validate function type matches (compare actual types, not indices)
        func = instance.funcs[func_idx]
        if func is None:
            raise TrapError("indirect call to imported function not supported")
        expected_type = instance.func_types[type_idx]
        actual_type = instance.func_types[func.type_idx]
        if (
            expected_type.params != actual_type.params
            or expected_type.results != actual_type.results
        ):
            raise TrapError("indirect call type mismatch")
        func_type = expected_type
        n_params = len(func_type.params)
        args = [stack.pop() for _ in range(n_params)][::-1]
        return ("call", func_idx, args)

    # Bulk memory operations
    elif op == "memory.copy":
        n = stack.pop()  # number of bytes
        src = stack.pop()  # source address
        dst = stack.pop()  # destination address
        mem = instance.memories[0].data
        if n > 0:
            if src < 0 or src + n > len(mem) or dst < 0 or dst + n > len(mem):
                raise TrapError("out of bounds memory access")
            # Handle overlapping regions correctly
            mem[dst : dst + n] = bytes(mem[src : src + n])

    elif op == "memory.fill":
        n = stack.pop()  # number of bytes
        val = stack.pop() & 0xFF  # byte value
        dst = stack.pop()  # destination address
        mem = instance.memories[0].data
        if n > 0:
            if dst < 0 or dst + n > len(mem):
                raise TrapError("out of bounds memory access")
            mem[dst : dst + n] = bytes([val] * n)

    elif op == "memory.init":
        n = stack.pop()  # number of bytes
        src = stack.pop()  # source offset in data segment
        dst = stack.pop()  # destination address in memory
        data_idx, mem_idx = instr.operand
        mem = instance.memories[mem_idx].data
        data_seg = instance.module.datas[data_idx]
        if n > 0:
            if src < 0 or src + n > len(data_seg.init) or dst < 0 or dst + n > len(mem):
                raise TrapError("out of bounds memory access")
            mem[dst : dst + n] = data_seg.init[src : src + n]

    elif op == "data.drop":
        # Mark data segment as dropped (no-op in our implementation)
        pass

    # Saturating truncation operations
    elif op == "i32.trunc_sat_f32_s":
        val = stack.pop()
        if math.isnan(val):
            stack.append(0)
        elif val >= 2147483647.0:
            stack.append(2147483647)
        elif val <= -2147483648.0:
            stack.append(-2147483648)
        else:
            stack.append(int(val))

    elif op == "i32.trunc_sat_f32_u":
        val = stack.pop()
        if math.isnan(val) or val < 0:
            stack.append(0)
        elif val >= 4294967295.0:
            stack.append(4294967295)
        else:
            stack.append(int(val))

    elif op == "i32.trunc_sat_f64_s":
        val = stack.pop()
        if math.isnan(val):
            stack.append(0)
        elif val >= 2147483647.0:
            stack.append(2147483647)
        elif val <= -2147483648.0:
            stack.append(-2147483648)
        else:
            stack.append(int(val))

    elif op == "i32.trunc_sat_f64_u":
        val = stack.pop()
        if math.isnan(val) or val < 0:
            stack.append(0)
        elif val >= 4294967295.0:
            stack.append(4294967295)
        else:
            stack.append(int(val))

    elif op == "i64.trunc_sat_f32_s":
        val = stack.pop()
        if math.isnan(val):
            stack.append(0)
        elif val >= 9223372036854775807.0:
            stack.append(9223372036854775807)
        elif val <= -9223372036854775808.0:
            stack.append(-9223372036854775808)
        else:
            stack.append(int(val))

    elif op == "i64.trunc_sat_f32_u":
        val = stack.pop()
        if math.isnan(val) or val < 0:
            stack.append(0)
        elif val >= 18446744073709551615.0:
            stack.append(18446744073709551615)
        else:
            stack.append(int(val))

    elif op == "i64.trunc_sat_f64_s":
        val = stack.pop()
        if math.isnan(val):
            stack.append(0)
        elif val >= 9223372036854775807.0:
            stack.append(9223372036854775807)
        elif val <= -9223372036854775808.0:
            stack.append(-9223372036854775808)
        else:
            stack.append(int(val))

    elif op == "i64.trunc_sat_f64_u":
        val = stack.pop()
        if math.isnan(val) or val < 0:
            stack.append(0)
        elif val >= 18446744073709551615.0:
            stack.append(18446744073709551615)
        else:
            stack.append(int(val))

    else:
        raise TrapError(f"Unimplemented instruction: {op}")

    return None


def do_branch(stack: list[Any], labels: list[Label], depth: int) -> tuple:
    """Execute a branch to the given label depth."""
    # Count from innermost label
    label_idx = len(labels) - 1 - depth
    if label_idx < 0:
        raise TrapError(f"Invalid branch depth: {depth} (have {len(labels)} labels)")

    label = labels[label_idx]

    # Pop labels up to and including the target
    for _ in range(depth + 1):
        labels.pop()

    # Restore stack: keep only arity values on top, discard down to stack_height
    if label.arity > 0:
        # Save the result values
        result_values = stack[-label.arity :]
        # Restore stack to the height when block was entered
        del stack[label.stack_height :]
        # Push back the result values
        stack.extend(result_values)
    else:
        # No result values, just restore stack height
        del stack[label.stack_height :]

    # For loops, we re-enter the loop
    if label.is_loop:
        # Re-add the loop label for the next iteration
        labels.append(label)

    return ("branch", label.target + 1)


def precompute_jump_targets(
    body: list[Instruction],
) -> dict[int, tuple[int | None, int]]:
    """Precompute jump targets for all block/loop/if instructions.

    Returns a dict mapping instruction index to (else_ip, end_ip).
    For block/loop, else_ip is None. For if, else_ip may be set.
    """
    targets: dict[int, tuple[int | None, int]] = {}
    # Stack of (start_ip, is_if, else_ip)
    stack: list[tuple[int, bool, int | None]] = []
    body_len = len(body)

    for ip in range(body_len):
        op = body[ip].opcode
        if op in ("block", "loop"):
            stack.append((ip, False, None))
        elif op == "if":
            stack.append((ip, True, None))
        elif op == "else":
            # Record else position for the innermost if
            if stack and stack[-1][1]:  # is_if
                start_ip, _, _ = stack[-1]
                stack[-1] = (start_ip, True, ip)
        elif op == "end":
            if stack:
                start_ip, is_if, else_ip = stack.pop()
                targets[start_ip] = (else_ip, ip)

    return targets


def get_jump_targets(
    func: Function, body: list[Instruction]
) -> dict[int, tuple[int | None, int]]:
    """Get jump targets for a function, computing and caching if needed."""
    if func.jump_targets is None:
        func.jump_targets = precompute_jump_targets(body)
    return func.jump_targets


def find_end(
    body: list[Instruction], start_ip: int, jump_targets: dict | None = None
) -> int:
    """Find the matching 'end' instruction for a block/loop starting at start_ip."""
    if jump_targets is not None and start_ip in jump_targets:
        return jump_targets[start_ip][1]
    # Fallback to scanning
    depth = 1
    ip = start_ip
    body_len = len(body)
    while ip < body_len:
        op = body[ip].opcode
        if op in ("block", "loop", "if"):
            depth += 1
        elif op == "end":
            depth -= 1
            if depth == 0:
                return ip
        ip += 1
    raise TrapError("No matching end found")


def find_else_end(
    body: list[Instruction], start_ip: int, jump_targets: dict | None = None
) -> tuple[int | None, int]:
    """Find 'else' and 'end' for an if starting at start_ip."""
    if jump_targets is not None and start_ip in jump_targets:
        return jump_targets[start_ip]
    # Fallback to scanning
    depth = 1
    ip = start_ip
    else_ip = None
    body_len = len(body)
    while ip < body_len:
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
    funcs: list[Function | None] = []
    imported_funcs: dict[int, Callable] = {}

    # Handle imported functions
    func_idx = 0
    for imp in module.imports:
        if imp.kind == "func":
            # Look up the imported function
            mod_imports = imports.get(imp.module, {})
            func_callable = mod_imports.get(imp.name)
            if func_callable is not None:
                imported_funcs[func_idx] = func_callable
            # Add placeholder to maintain correct indices
            funcs.append(None)
            func_idx += 1

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

    # Initialize tables
    tables: list[TableInstance] = []
    for table in module.tables:
        elements: list[Any] = [None] * table.limits.min
        elem_type = (
            table.element_type if isinstance(table.element_type, str) else "funcref"
        )
        tables.append(TableInstance(elements, table.limits.max, elem_type))

    # Initialize element segments
    for elem in module.elem:
        if elem.table_idx >= 0 and tables:  # Active segment
            table = tables[elem.table_idx]
            # Evaluate offset expression (simplified: assume i32.const)
            offset = 0
            for instr in elem.offset:
                if instr.opcode == "i32.const":
                    offset = instr.operand
                    break
            # Copy function indices
            for i, func_idx in enumerate(elem.init):
                table.elements[offset + i] = func_idx

    instance = Instance(
        module=module,
        funcs=funcs,
        func_types=func_types,
        memories=memories,
        globals=globals_list,
        tables=tables,
        imported_funcs=imported_funcs,
    )

    # Run start function if present
    if module.start is not None:
        execute_function(instance, module.start, [])

    return instance
