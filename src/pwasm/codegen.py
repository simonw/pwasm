"""Compile WebAssembly functions to Python source code.

The interpreter (executor.py) dispatches on every instruction. This module
instead translates a whole WebAssembly function into a Python function, so
CPython (or PyPy) runs it directly:

- The operand stack disappears. Pure instructions that cannot trap are
  folded into Python expressions over local variables (l0, l1, ...) and
  temporaries (t1, t2, ...). Instructions that can trap or have side effects
  (loads, stores, calls, division, ...) become statements, in program order.
- i32/i64 values are unsigned ints, like in the interpreter, but masking to
  32 or 64 bits is delayed until a value is stored, compared, used as an
  address or otherwise needs to be exact: +, -, *, &, |, ^ and << all give
  the right answer modulo 2**32 on unmasked inputs.
- Structured control flow becomes Python `while True:` loops and `if`
  statements ("structured" code). CPython limits nesting to 20 loops and
  100 levels of indentation, and C switch statements compile to very deeply
  nested blocks, so functions that would exceed those limits become a state
  machine over basic blocks instead: `while True:` around a binary search
  on a `pc` variable.

Generated code runs in a per-instance namespace (see instance_namespace)
holding the memory, globals, other functions (f0, f1, ...) and helpers, so
the compiled code objects can be shared by every instance of a module.
"""

from __future__ import annotations

import bisect
import linecache
import math
import struct
import sys
from types import FunctionType
from typing import Any

from . import numeric as num
from .errors import TrapError, WasmError
from .numeric import F32NaN, MASK_32, MASK_64

MASKS = {"i32": "0xFFFFFFFF", "i64": "0xFFFFFFFFFFFFFFFF"}
SIGNS = {"i32": 0x80000000, "i64": 0x8000000000000000}
BITS = {"i32": 32, "i64": 64}
ZERO_LITERAL = {
    "i32": "0",
    "i64": "0",
    "f32": "0.0",
    "f64": "0.0",
    "funcref": "None",
    "externref": "None",
}

# Structured code is used when it fits comfortably in CPython's limits
MAX_PY_LOOPS = 18
MAX_INDENT = 80
# Longer expressions are assigned to a temporary
MAX_EXPR = 300

# For testing: translate every function as a state machine, or lower every
# chain of blocks (see Translator.open_chain) even when not needed
FORCE_STATE_MACHINE = False
FORCE_CHAINS = False

_EMPTY: frozenset = frozenset()


class V:
    """A value on the symbolic operand stack: a Python expression."""

    __slots__ = (
        "expr",
        "type",
        "masked",
        "deps",
        "gread",
        "simple",
        "stable",
        "boolean",
        "const",
        "wrap",
    )

    def __init__(
        self,
        expr: str,
        type: str,
        masked: bool = True,
        deps: frozenset = _EMPTY,
        gread: bool = False,
        simple: bool = False,
        stable: bool = False,
        boolean: bool = False,
        const: Any = None,
        wrap: tuple[str, int] | None = None,
    ) -> None:
        self.expr = expr
        self.type = type
        # value already reduced to 32/64 bits (always true for non-integers)
        self.masked = masked
        # local variables the expression reads
        self.deps = deps
        # the expression reads globals
        self.gread = gread
        # usable as an operand without parentheses
        self.simple = simple
        # its value cannot change: a constant or a temporary
        self.stable = stable
        # a Python bool (a comparison)
        self.boolean = boolean
        # the integer value of a constant
        self.const = const
        # (x, d): the expression is x + d, for a masked simple x and a small
        # constant d, so masking can be a test instead of an `&`
        self.wrap = wrap


def op(v: V) -> str:
    """The expression as an operand of a larger expression."""
    return v.expr if v.simple else f"({v.expr})"


def masked(v: V) -> str:
    """The expression, reduced to its type's width if necessary."""
    if v.masked:
        return v.expr
    if v.wrap is not None:
        # faster than masking with & in CPython
        x, d = v.wrap
        full = 1 << BITS[v.type]
        if d > 0:
            return f"{x} + {d} if {x} < {full - d} else {x} - {full - d}"
        return f"{x} - {-d} if {x} >= {-d} else {x} + {full + d}"
    return f"{op(v)} & {MASKS[v.type]}"


def masked_op(v: V) -> str:
    """masked(), usable as an operand."""
    if v.masked:
        return op(v)
    if v.wrap is not None:
        return f"({masked(v)})"
    return f"({op(v)} & {MASKS[v.type]})"


class Label:
    """Compile-time state for a block, loop, if or the function body."""

    __slots__ = (
        "kind",
        "id",
        "param_types",
        "result_types",
        "height",
        "targeted",
        "pyloop",
        "crossed",
        "flag_target",
        "branched",
        "vars",
        "params",
        "entry",
        "else_seen",
        "then_reach",
        "arm_start",
        "start_pc",
        "end_pc",
        "else_pc",
        "chain",
        "seg",
    )

    def __init__(
        self, kind: str, id: int, param_types: tuple, result_types: tuple, height: int
    ):
        self.kind = kind
        self.id = id
        self.param_types = param_types
        self.result_types = result_types
        self.height = height  # stack height below the params
        self.targeted = False  # some branch targets this label
        self.pyloop = False  # emitted as a Python `while True:`
        self.crossed = False  # a branch passed through this Python loop
        self.flag_target = False  # a branch reaches this label via br_
        self.branched = False  # a branch to the end of this label was emitted
        self.vars: list[str] = []  # variables holding the results
        self.params: list[str] = []  # loop: variables holding the params
        self.entry: list[V] = []  # if: the stack when it was entered
        self.else_seen = False
        self.then_reach = False
        self.arm_start = 0
        self.start_pc = 0
        self.end_pc: int | None = None
        self.else_pc = 0
        self.chain: Chain | None = None  # set for the blocks of a chain
        self.seg: int | None = None  # chain segment starting at our end

    @property
    def branch_types(self) -> tuple:
        return self.param_types if self.kind == "loop" else self.result_types


class Chain:
    """A chain of consecutive blocks translated as one Python loop.

    `block block block ... end A end B end C` has code segments: the code
    inside the innermost block (0), then after each `end` (1, 2, ...).
    Branches only go forward, to a later segment, so the chain becomes
    `while True:` around a binary search on a segment variable, with a last
    "exit" leaf that leaves the loop. This is how C switch statements
    compile, and it keeps them from nesting deeper than Python allows.

    The search tree is weighted (by how often each leaf is expected to be
    entered) so that the busiest segments take the fewest tests, but it is
    never more than two levels deeper than a balanced tree.
    """

    __slots__ = (
        "id",
        "var",
        "n",
        "leaf",
        "base",
        "cont_id",
        "cont_used",
        "holder",
        "paths",
    )

    def __init__(self, id: int, weights: list[int]) -> None:
        self.id = id
        self.var = f"s{id}"
        self.n = len(weights)  # leaves: the segments plus the exit leaf
        self.leaf = -1
        self.base = 0  # indentation inside the while loop
        self.cont_id = 0  # br_ value meaning "continue the chain loop"
        self.cont_used = False
        self.holder: Label | None = None  # the outermost block
        self.paths: list = [None] * self.n
        prefix = [0]
        for w in weights:
            prefix.append(prefix[-1] + w)
        self._split(0, self.n, tree_depth(self.n), prefix, [])

    def _split(self, lo: int, hi: int, budget: int, prefix: list, path: list):
        if hi - lo == 1:
            self.paths[lo] = path
            return
        # the split nearest half the weight, leaving each side few enough
        # leaves for the remaining depth
        half = 1 << (budget - 1)
        first, last = max(lo + 1, hi - half), min(hi - 1, lo + half)
        target = (prefix[lo] + prefix[hi]) / 2
        m = bisect.bisect_left(prefix, target, first, last + 1)
        candidates = [c for c in (m - 1, m) if first <= c <= last]
        mid = min(
            candidates, key=lambda c: (abs(prefix[c] - target), abs(2 * c - lo - hi))
        )
        self._split(lo, mid, budget - 1, prefix, path + [(mid, 0)])
        self._split(mid, hi, budget - 1, prefix, path + [(mid, 1)])

    def path(self, i: int) -> list[tuple[int, int]]:
        """(mid, 0 for `if var < mid` / 1 for `else`) down to leaf i."""
        return self.paths[i]


def tree_depth(n: int) -> int:
    """Maximum depth of the dispatch tree of a chain with n leaves."""
    return min(n - 1, (n - 1).bit_length() + 2)


def _function_types(module: Any) -> list:
    types = getattr(module, "_function_types", None)
    if types is None:
        types = [module.types[i.desc] for i in module.imports if i.kind == "func"]
        types += [module.types[f.type_idx] for f in module.funcs]
        module._function_types = types
    return types


def _global_types(module: Any) -> list:
    types = getattr(module, "_global_types", None)
    if types is None:
        types = [i.desc.valtype for i in module.imports if i.kind == "global"]
        types += [g.type.valtype for g in module.globals]
        module._global_types = types
    return types


def _table_types(module: Any) -> list:
    types = getattr(module, "_table_types", None)
    if types is None:
        types = [i.desc[0] for i in module.imports if i.kind == "table"]
        types += [t.element_type for t in module.tables]
        module._table_types = types
    return types


def _float_literal(value: float, type: str) -> str:
    if type == "f32" and isinstance(value, F32NaN):
        return f"_f32b({value.bits:#x})"
    if value != value:
        bits = struct.unpack("<Q", struct.pack("<d", value))[0]
        return f"_f64b({bits:#x})"
    if value == math.inf:
        return "_INF"
    if value == -math.inf:
        return "_NINF"
    return repr(float(value))


# name -> template for loads; {A} is the address
LOADS = {
    "i32.load": ("i32", "_u32(mem, {A})[0]"),
    "i32.load8_u": ("i32", "mem[{A}]"),
    "i32.load8_s": ("i32", "_s8(mem, {A})[0] & 0xFFFFFFFF"),
    "i32.load16_u": ("i32", "_u16(mem, {A})[0]"),
    "i32.load16_s": ("i32", "_s16(mem, {A})[0] & 0xFFFFFFFF"),
    "i64.load": ("i64", "_u64(mem, {A})[0]"),
    "i64.load8_u": ("i64", "mem[{A}]"),
    "i64.load8_s": ("i64", "_s8(mem, {A})[0] & 0xFFFFFFFFFFFFFFFF"),
    "i64.load16_u": ("i64", "_u16(mem, {A})[0]"),
    "i64.load16_s": ("i64", "_s16(mem, {A})[0] & 0xFFFFFFFFFFFFFFFF"),
    "i64.load32_u": ("i64", "_u32(mem, {A})[0]"),
    "i64.load32_s": ("i64", "_s32(mem, {A})[0] & 0xFFFFFFFFFFFFFFFF"),
    "f32.load": ("f32", "_ldf32(mem, {A})"),
    "f64.load": ("f64", "_ud(mem, {A})[0]"),
}

# Loads that read aligned addresses from a memoryview of the memory (see
# MemoryInstance.views): name -> (view, size in bytes). Faster than struct.
VIEW_LOADS = {
    "i32.load": ("_M32", 4),
    "i32.load16_u": ("_M16", 2),
    "i64.load": ("_M64", 8),
    "i64.load16_u": ("_M16", 2),
    "i64.load32_u": ("_M32", 4),
    "f64.load": ("_MD", 8),
}

# memoryviews use the machine's byte order
USE_VIEWS = sys.byteorder == "little"

# name -> (how to format the value, template); {A} address, {V} value
STORES = {
    "i32.store": ("masked", "_p32(mem, {A}, {V})"),
    "i32.store8": ("operand", "mem[{A}] = {V} & 0xFF"),
    "i32.store16": ("operand", "_p16(mem, {A}, {V} & 0xFFFF)"),
    "i64.store": ("masked", "_p64(mem, {A}, {V})"),
    "i64.store8": ("operand", "mem[{A}] = {V} & 0xFF"),
    "i64.store16": ("operand", "_p16(mem, {A}, {V} & 0xFFFF)"),
    "i64.store32": ("operand", "_p32(mem, {A}, {V} & 0xFFFFFFFF)"),
    "f32.store": ("plain", "_stf32(mem, {A}, {V})"),
    "f64.store": ("plain", "_pd(mem, {A}, {V})"),
}

# Pure unary operations implemented by calling a function from numeric.py
# (or a namespace helper): name -> (result type, function name)
UNARY_CALLS = {
    "i32.clz": ("i32", "i32_clz"),
    "i32.ctz": ("i32", "i32_ctz"),
    "i32.popcnt": ("i32", "i32_popcnt"),
    "i64.clz": ("i64", "i64_clz"),
    "i64.ctz": ("i64", "i64_ctz"),
    "i64.popcnt": ("i64", "i32_popcnt"),
    "f32.abs": ("f32", "f32_abs"),
    "f32.neg": ("f32", "f32_neg"),
    "f32.sqrt": ("f32", "f32_sqrt"),
    "f32.ceil": ("f32", "f64_ceil"),
    "f32.floor": ("f32", "f64_floor"),
    "f32.trunc": ("f32", "f64_trunc"),
    "f32.nearest": ("f32", "f64_nearest"),
    "f64.abs": ("f64", "_fabs"),
    "f64.sqrt": ("f64", "f64_sqrt"),
    "f64.ceil": ("f64", "f64_ceil"),
    "f64.floor": ("f64", "f64_floor"),
    "f64.trunc": ("f64", "f64_trunc"),
    "f64.nearest": ("f64", "f64_nearest"),
    "i32.trunc_sat_f32_s": ("i32", "i32_trunc_sat_s"),
    "i32.trunc_sat_f32_u": ("i32", "i32_trunc_sat_u"),
    "i32.trunc_sat_f64_s": ("i32", "i32_trunc_sat_s"),
    "i32.trunc_sat_f64_u": ("i32", "i32_trunc_sat_u"),
    "i64.trunc_sat_f32_s": ("i64", "i64_trunc_sat_s"),
    "i64.trunc_sat_f32_u": ("i64", "i64_trunc_sat_u"),
    "i64.trunc_sat_f64_s": ("i64", "i64_trunc_sat_s"),
    "i64.trunc_sat_f64_u": ("i64", "i64_trunc_sat_u"),
    "f32.convert_i32_s": ("f32", "f32_convert_i32_s"),
    "f32.convert_i32_u": ("f32", "f32_convert_i32_u"),
    "f32.convert_i64_s": ("f32", "f32_convert_i64_s"),
    "f32.convert_i64_u": ("f32", "f32_convert_i64_u"),
    "f64.convert_i32_s": ("f64", "f64_convert_i32_s"),
    "f64.convert_i32_u": ("f64", "f64_convert_u"),
    "f64.convert_i64_s": ("f64", "f64_convert_i64_s"),
    "f64.convert_i64_u": ("f64", "f64_convert_u"),
    "f32.demote_f64": ("f32", "f32_demote_f64"),
    "f64.promote_f32": ("f64", "f64_promote_f32"),
    "i32.reinterpret_f32": ("i32", "f32_to_bits"),
    "f32.reinterpret_i32": ("f32", "f32_from_bits"),
    "i64.reinterpret_f64": ("i64", "i64_reinterpret_f64"),
    "f64.reinterpret_i64": ("f64", "f64_reinterpret_i64"),
}

# Unary operations that can trap: statements
TRAPPING_UNARY = {
    "i32.trunc_f32_s": ("i32", "i32_trunc_s"),
    "i32.trunc_f32_u": ("i32", "i32_trunc_u"),
    "i32.trunc_f64_s": ("i32", "i32_trunc_s"),
    "i32.trunc_f64_u": ("i32", "i32_trunc_u"),
    "i64.trunc_f32_s": ("i64", "i64_trunc_s"),
    "i64.trunc_f32_u": ("i64", "i64_trunc_u"),
    "i64.trunc_f64_s": ("i64", "i64_trunc_s"),
    "i64.trunc_f64_u": ("i64", "i64_trunc_u"),
}

# Pure binary float operations via function calls
FLOAT_BINARY_CALLS = {
    "f32.div": "f32_div",
    "f32.min": "f32_min",
    "f32.max": "f32_max",
    "f32.copysign": "f32_copysign",
    "f64.div": "f64_div",
    "f64.min": "f64_min",
    "f64.max": "f64_max",
    "f64.copysign": "_copysign",
}

COMPARE = {"eq": "==", "ne": "!=", "lt": "<", "gt": ">", "le": "<=", "ge": ">="}


class Translator:
    """Translates one function body to the source of a Python function."""

    def __init__(
        self, module: Any, index: int, func: Any, ftype: Any, has_limits: bool
    ):
        self.module = module
        self.types = module.types
        self.function_types = _function_types(module)
        self.global_types = _global_types(module)
        self.table_types = _table_types(module)
        self.index = index
        self.func = func
        self.ftype = ftype
        self.has_limits = has_limits
        self.local_types = list(ftype.params) + list(func.locals)
        self.stack: list[V] = []
        self.lines: list[tuple[int, str]] = []
        self.indent = 0
        self.ntmp = 0
        self.nlabel = 0
        self.dead = False
        self.labels: list[Label] = []
        self.uses_flag = False
        # state machine
        self.npc = 0
        self.blocks: list[tuple[int, list]] = []
        self.cur_pc = 0

    # --- output ---

    def emit(self, text: str) -> None:
        self.lines.append((self.indent, text))

    def emit_lines(self, lines: list[str]) -> None:
        for line in lines:
            self.emit(line)

    def new_tmp(self) -> str:
        self.ntmp += 1
        return f"t{self.ntmp}"

    # --- the symbolic stack ---

    def pop(self) -> V:
        return self.stack.pop()

    def popn(self, n: int) -> list[V]:
        if not n:
            return []
        values = self.stack[-n:]
        del self.stack[-n:]
        return values

    def to_temp(self, v: V) -> V:
        if v.stable:
            return v
        t = self.new_tmp()
        self.emit(f"{t} = {v.expr}")
        return V(t, v.type, v.masked, simple=True, stable=True, boolean=v.boolean)

    def push(self, v: V) -> None:
        if len(v.expr) > MAX_EXPR:
            v = self.to_temp(v)
        self.stack.append(v)

    def push_expr(
        self,
        expr: str,
        type: str,
        operands: tuple,
        masked: bool = True,
        boolean: bool = False,
        wrap: tuple[str, int] | None = None,
    ) -> None:
        deps = _EMPTY
        gread = False
        for o in operands:
            deps = deps | o.deps
            gread = gread or o.gread
        self.push(V(expr, type, masked, deps, gread, boolean=boolean, wrap=wrap))

    def push_stmt(self, expr: str, type: str, masked: bool = True) -> None:
        """Evaluate expr now (it may trap or have side effects)."""
        t = self.new_tmp()
        self.emit(f"{t} = {expr}")
        self.stack.append(V(t, type, masked, simple=True, stable=True))

    def settle(self) -> None:
        """Assign every value that could change to a temporary: needed
        before control flow, since assignments emitted inside a branch would
        not happen on other paths."""
        stack = self.stack
        for i, v in enumerate(stack):
            if not v.stable:
                stack[i] = self.to_temp(v)

    def settle_local(self, n: int) -> None:
        stack = self.stack
        for i, v in enumerate(stack):
            if n in v.deps:
                stack[i] = self.to_temp(v)

    def settle_globals(self) -> None:
        stack = self.stack
        for i, v in enumerate(stack):
            if v.gread:
                stack[i] = self.to_temp(v)

    def cond(self, v: V) -> str:
        return v.expr if v.boolean else masked(v)

    # --- translation ---

    def signature(self, blocktype: Any) -> tuple[tuple, tuple]:
        if blocktype == ():
            return (), ()
        if isinstance(blocktype, tuple):
            return (), blocktype
        t = self.types[blocktype]
        return tuple(t.params), tuple(t.results)

    def prepass(self) -> tuple[set, bool, dict]:
        """Which constructs are branch targets, whether the function fits in
        structured Python, and which chains of blocks to lower (start ip ->
        number of blocks) to make it fit."""
        body = self.func.body
        open_: list[int] = []
        targeted: set = set()
        for ip, ins in enumerate(body):
            name = ins.opcode
            if name in ("block", "loop", "if"):
                open_.append(ip)
            elif name == "end":
                if open_:
                    open_.pop()
            elif name in ("br", "br_if"):
                if ins.operand < len(open_):
                    targeted.add(open_[-1 - ins.operand])
            elif name == "br_table":
                depths, default = ins.operand
                for d in list(depths) + [default]:
                    if d < len(open_):
                        targeted.add(open_[-1 - d])
        if FORCE_STATE_MACHINE:
            return targeted, False, {}
        if not FORCE_CHAINS and self.fits(targeted, {}):
            return targeted, True, {}
        chains = self.find_chains()
        if self.fits(targeted, chains):
            return targeted, True, chains
        return targeted, False, {}

    def find_chains(self) -> dict[int, int]:
        body = self.func.body

        def chainable(ip: int) -> bool:
            ins = body[ip]
            return ins.opcode == "block" and not self.signature(ins.operand)[0]

        chains = {}
        ip = 0
        while ip < len(body):
            if chainable(ip):
                end = ip
                while end < len(body) and chainable(end):
                    end += 1
                if end - ip >= 2:
                    chains[ip] = end - ip
                ip = end
            else:
                ip += 1
        return chains

    def find_loop_chains(self) -> dict[int, int]:
        """Loops with a chain directly in their body (loop ip -> chain start
        ip). The loop and its first such chain share one Python loop, where
        the loop's start is the first leaf of the chain's dispatch tree."""
        found = {}
        open_: list[tuple[str, int]] = []
        for ip, ins in enumerate(self.func.body):
            if ip in self.chains and open_:
                kind, parent = open_[-1]
                if kind == "loop" and parent not in found:
                    found[parent] = ip
            if ins.opcode in ("block", "loop", "if"):
                open_.append((ins.opcode, ip))
            elif ins.opcode == "end" and open_:
                open_.pop()
        return found

    def fits(self, targeted: set, chains: dict[int, int]) -> bool:
        """Whether structured code stays within CPython's nesting limits."""
        members = {}
        for start, k in chains.items():
            for ip in range(start, start + k):
                members[ip] = (start, k)
        nesting: list[tuple[int, int]] = []
        loops = indent = max_loops = max_indent = 0
        for ip, ins in enumerate(self.func.body):
            name = ins.opcode
            if name in ("block", "loop", "if"):
                if ip in members:
                    start, k = members[ip]
                    depth = tree_depth(k + 1)
                    cost = (1, 1 + depth) if ip == start else (0, 0)
                else:
                    py = name == "loop" or ip in targeted
                    cost = (int(py), int(py) + (1 if name == "if" else 0))
                nesting.append(cost)
                loops += cost[0]
                indent += cost[1]
                max_loops = max(max_loops, loops)
                max_indent = max(max_indent, indent)
            elif name == "end" and nesting:
                py, width = nesting.pop()
                loops -= py
                indent -= width
        return max_loops <= MAX_PY_LOOPS and max_indent <= MAX_INDENT

    def translate(self) -> str:
        targeted, self.structured, self.chains = self.prepass()
        self.loop_chains = self.find_loop_chains() if self.chains else {}
        self.merged: dict[int, Chain] = {}  # chain start -> chain of its loop
        chain_rest = set()
        for start, k in self.chains.items():
            chain_rest.update(range(start + 1, start + k))
        func_label = Label("func", 0, (), tuple(self.ftype.results), 0)
        self.labels.append(func_label)
        body = self.func.body
        skip = 0
        for ip, ins in enumerate(body):
            name = ins.opcode
            arg = ins.operand
            if self.dead:
                if name in ("block", "loop", "if"):
                    skip += 1
                    continue
                if skip:
                    if name == "end":
                        skip -= 1
                    continue
                if name not in ("end", "else"):
                    continue
            if ip in chain_rest:
                continue  # opened with the first block of its chain
            handler = self.HANDLERS.get(name)
            if handler is not None:
                handler(self, name, arg)
            elif ip in self.chains:
                self.open_chain(ip, self.chains[ip])
            elif ip in self.loop_chains:
                self.open_loop_chain(arg, self.loop_chains[ip])
            elif name in ("block", "loop", "if"):
                self.open(name, arg, ip in targeted)
            elif name == "end":
                if self.close():
                    break
            else:
                raise WasmError(f"cannot compile instruction {name}")
        return self.assemble()

    # --- instructions ---

    def i_const(self, name: str, arg: Any) -> None:
        type = name[:3]
        if type in MASKS:
            value = arg & (MASK_32 if type == "i32" else MASK_64)
            self.stack.append(
                V(str(value), type, simple=True, stable=True, const=value)
            )
        else:
            literal = _float_literal(arg, type)
            simple = not literal.startswith("-")
            self.stack.append(V(literal, type, simple=simple, stable=True))

    def i_local_get(self, name: str, n: int) -> None:
        self.stack.append(
            V(f"l{n}", self.local_types[n], deps=frozenset((n,)), simple=True)
        )

    def i_local_set(self, name: str, n: int) -> None:
        v = self.pop()
        self.settle_local(n)
        value = masked(v)
        if value != f"l{n}":
            self.emit(f"l{n} = {value}")
        if name == "local.tee":
            self.i_local_get(name, n)

    def i_global_get(self, name: str, n: int) -> None:
        self.stack.append(
            V(f"g{n}._value", self.global_types[n], gread=True, simple=True)
        )

    def i_global_set(self, name: str, n: int) -> None:
        v = self.pop()
        self.settle_globals()
        self.emit(f"g{n}._value = {masked(v)}")

    def i_drop(self, name: str, arg: Any) -> None:
        self.pop()

    def i_select(self, name: str, arg: Any) -> None:
        c = self.pop()
        b = self.pop()
        a = self.pop()
        test = self.cond(c)
        if not c.simple:
            test = f"({test})"
        self.push_expr(
            f"{op(a)} if {test} else {op(b)}",
            a.type,
            (a, b, c),
            masked=a.masked and b.masked,
            boolean=a.boolean and b.boolean,
        )

    def i_int_binary(self, name: str, arg: Any) -> None:
        t, operation = name.split(".")
        b = self.pop()
        a = self.pop()
        mask = MASKS[t]
        sign = SIGNS[t]
        shift_mask = BITS[t] - 1
        if operation in ("add", "sub", "mul"):
            symbol = {"add": "+", "sub": "-", "mul": "*"}[operation]
            if a.const is not None and b.const is not None:
                value = eval(f"{a.const} {symbol} {b.const}") & (
                    MASK_32 if t == "i32" else MASK_64
                )
                self.stack.append(
                    V(str(value), t, simple=True, stable=True, const=value)
                )
                return
            if operation != "mul" and self.add_constant(operation, a, b):
                return
            self.push_expr(f"{op(a)} {symbol} {op(b)}", t, (a, b), masked=False)
        elif operation == "and":
            self.push_expr(f"{op(a)} & {op(b)}", t, (a, b), masked=a.masked or b.masked)
        elif operation in ("or", "xor"):
            symbol = "|" if operation == "or" else "^"
            self.push_expr(
                f"{op(a)} {symbol} {op(b)}", t, (a, b), masked=a.masked and b.masked
            )
        elif operation in ("shl", "shr_u", "shr_s"):
            amount = (
                str(b.const & shift_mask)
                if b.const is not None
                else f"{op(b)} & {shift_mask}"
            )
            if operation == "shl":
                self.push_expr(f"{op(a)} << ({amount})", t, (a, b), masked=False)
            elif operation == "shr_u":
                self.push_expr(f"{masked_op(a)} >> ({amount})", t, (a, b))
            else:
                self.push_expr(
                    f"(({masked_op(a)} ^ {sign}) - {sign}) >> ({amount})",
                    t,
                    (a, b),
                    masked=False,
                )
        elif operation in ("rotl", "rotr"):
            self.push_expr(f"{t}_{operation}({masked(a)}, {masked(b)})", t, (a, b))
        elif operation in ("eq", "ne"):
            symbol = COMPARE[operation]
            self.push_expr(
                f"{masked_op(a)} {symbol} {masked_op(b)}", "i32", (a, b), boolean=True
            )
        elif operation[:2] in ("lt", "gt", "le", "ge"):
            symbol = COMPARE[operation[:2]]
            if operation.endswith("_u"):
                self.push_expr(
                    f"{masked_op(a)} {symbol} {masked_op(b)}",
                    "i32",
                    (a, b),
                    boolean=True,
                )
            else:
                left = (
                    str(a.const ^ sign)
                    if a.const is not None
                    else f"({masked_op(a)} ^ {sign})"
                )
                right = (
                    str(b.const ^ sign)
                    if b.const is not None
                    else f"({masked_op(b)} ^ {sign})"
                )
                self.push_expr(f"{left} {symbol} {right}", "i32", (a, b), boolean=True)
        elif operation in ("div_s", "div_u", "rem_s", "rem_u"):
            full = MASK_32 if t == "i32" else MASK_64
            if b.const:  # cannot divide by zero
                if operation == "div_u":
                    self.push_expr(f"{masked_op(a)} // {b.const}", t, (a,))
                    return
                if operation == "rem_u":
                    self.push_expr(f"{masked_op(a)} % {b.const}", t, (a,))
                    return
                if b.const != full:  # and cannot overflow (x / -1)
                    self.push_expr(f"{t}_{operation}({masked(a)}, {b.const})", t, (a,))
                    return
            self.push_stmt(f"{t}_{operation}({masked(a)}, {masked(b)})", t)
        else:
            raise WasmError(f"cannot compile instruction {name}")

    def add_constant(self, operation: str, a: V, b: V) -> bool:
        """x + c or x - c, for a masked simple x, as x + d for a small
        (positive or negative) d. Returns False for other operands."""
        if b.const is not None:
            x, c = a, b.const
        elif operation == "add" and a.const is not None:
            x, c = b, a.const
        else:
            return False
        # x is used twice: it should be a variable
        if not (x.masked and x.expr.isidentifier()):
            return False
        full = 1 << BITS[x.type]
        d = (c if operation == "add" else -c) % full
        if d >= full // 2:
            d -= full
        if d == 0:
            self.stack.append(x)
        else:
            expr = f"{x.expr} + {d}" if d > 0 else f"{x.expr} - {-d}"
            self.push_expr(expr, x.type, (x,), masked=False, wrap=(x.expr, d))
        return True

    def i_int_unary(self, name: str, arg: Any) -> None:
        a = self.pop()
        t, operation = name.split(".")
        if operation == "eqz":
            if a.boolean:
                self.push_expr(f"not {op(a)}", "i32", (a,), boolean=True)
            else:
                self.push_expr(f"{masked_op(a)} == 0", "i32", (a,), boolean=True)
        elif operation in ("extend8_s", "extend16_s", "extend32_s"):
            bits = int(operation[6:-2])
            low = (1 << bits) - 1
            top = 1 << (bits - 1)
            self.push_expr(
                f"(({op(a)} & {low:#x}) ^ {top:#x}) - {top:#x}", t, (a,), masked=False
            )
        elif name == "i32.wrap_i64":
            self.stack.append(
                V(a.expr, "i32", False, a.deps, a.gread, a.simple, a.stable)
            )
        elif name == "i64.extend_i32_u":
            self.push_expr(masked(a), "i64", (a,))
        elif name == "i64.extend_i32_s":
            self.push_expr(
                f"({masked_op(a)} ^ 0x80000000) - 0x80000000", "i64", (a,), masked=False
            )
        else:
            raise WasmError(f"cannot compile instruction {name}")

    def i_unary_call(self, name: str, arg: Any) -> None:
        a = self.pop()
        type, function = UNARY_CALLS[name]
        self.push_expr(f"{function}({masked(a)})", type, (a,))

    def i_trapping_unary(self, name: str, arg: Any) -> None:
        a = self.pop()
        type, function = TRAPPING_UNARY[name]
        self.push_stmt(f"{function}({masked(a)})", type)

    def i_float_binary(self, name: str, arg: Any) -> None:
        b = self.pop()
        a = self.pop()
        t, operation = name.split(".")
        if name in FLOAT_BINARY_CALLS:
            self.push_expr(f"{FLOAT_BINARY_CALLS[name]}({a.expr}, {b.expr})", t, (a, b))
        elif operation in COMPARE:
            self.push_expr(
                f"{op(a)} {COMPARE[operation]} {op(b)}", "i32", (a, b), boolean=True
            )
        else:
            symbol = {"add": "+", "sub": "-", "mul": "*"}[operation]
            expr = f"{op(a)} {symbol} {op(b)}"
            if t == "f32":
                expr = f"_f32r({expr})"
            self.push_expr(expr, t, (a, b))

    def i_f64_neg(self, name: str, arg: Any) -> None:
        a = self.pop()
        self.push_expr(f"-{op(a)}", "f64", (a,))

    def i_ref_null(self, name: str, arg: Any) -> None:
        self.stack.append(V("None", arg, simple=True, stable=True))

    def i_ref_is_null(self, name: str, arg: Any) -> None:
        a = self.pop()
        self.push_expr(f"{op(a)} is None", "i32", (a,), boolean=True)

    def i_ref_func(self, name: str, n: int) -> None:
        self.stack.append(V(f"F{n}", "funcref", simple=True, stable=True))

    def address(self, addr: V, offset: int) -> str:
        if offset:
            return f"{masked_op(addr)} + {offset}"
        return masked(addr)

    def i_load(self, name: str, arg: Any) -> None:
        addr = self.pop()
        type, template = LOADS[name]
        offset = arg[1]
        if name not in VIEW_LOADS or not USE_VIEWS:
            self.push_stmt(template.format(A=self.address(addr, offset)), type)
            return
        view, size = VIEW_LOADS[name]
        shift = size.bit_length() - 1
        if addr.const is not None:
            a = addr.const + offset
            if a % size:
                self.push_stmt(template.format(A=a), type)
            else:
                self.push_stmt(f"{view}[{a >> shift}]", type)
            return
        if not (addr.simple and addr.masked):
            addr = self.to_temp(V(masked(addr), "i32", simple=False))
        base = addr.expr
        if offset % size:
            # the address is used three times: compute it once
            addr = self.to_temp(V(f"{base} + {offset}", "i32", simple=False))
            base, offset = addr.expr, 0
        index = f"{base} >> {shift}"
        if offset:
            index = f"({index}) + {offset >> shift}"
        fallback = template.format(A=self.address(addr, offset))
        self.push_stmt(
            f"{view}[{index}] if not {base} & {size - 1} else {fallback}", type
        )

    def i_store(self, name: str, arg: Any) -> None:
        v = self.pop()
        addr = self.pop()
        style, template = STORES[name]
        value = {"masked": masked(v), "operand": op(v), "plain": v.expr}[style]
        self.emit(template.format(A=self.address(addr, arg[1]), V=value))

    def i_memory_size(self, name: str, arg: Any) -> None:
        self.push_stmt("len(mem) >> 16", "i32")

    def i_memory_grow(self, name: str, arg: Any) -> None:
        v = self.pop()
        self.push_stmt(f"_M0.grow({masked(v)}) & 0xFFFFFFFF", "i32")

    def i_bulk(self, name: str, arg: Any) -> None:
        if name == "data.drop":
            self.emit(f"_ddrop({arg})")
            return
        if name == "elem.drop":
            self.emit(f"_edrop({arg})")
            return
        c, b, a = self.pop(), self.pop(), self.pop()
        args = f"{masked(a)}, {masked(b)}, {masked(c)}"
        if name == "memory.fill":
            self.emit(f"_mfill({args})")
        elif name == "memory.copy":
            self.emit(f"_mcopy({args})")
        elif name == "memory.init":
            self.emit(f"_minit({arg}, {args})")
        elif name == "table.copy":
            self.emit(f"_tcopy({arg[0]}, {arg[1]}, {args})")
        elif name == "table.init":
            self.emit(f"_tinit({arg[0]}, {arg[1]}, {args})")

    def i_table(self, name: str, n: int) -> None:
        if name == "table.get":
            i = self.pop()
            self.push_stmt(f"_tget({n}, {masked(i)})", self.table_types[n])
        elif name == "table.set":
            v = self.pop()
            i = self.pop()
            self.emit(f"_tset({n}, {masked(i)}, {v.expr})")
        elif name == "table.size":
            self.push_stmt(f"_tsize({n})", "i32")
        elif name == "table.grow":
            count = self.pop()
            init = self.pop()
            self.push_stmt(f"_tgrow({n}, {init.expr}, {masked(count)})", "i32")
        elif name == "table.fill":
            count = self.pop()
            v = self.pop()
            i = self.pop()
            self.emit(f"_tfill({n}, {masked(i)}, {v.expr}, {masked(count)})")

    def call_results(self, expr: str, results: tuple) -> None:
        if not results:
            self.emit(expr)
        elif len(results) == 1:
            self.push_stmt(expr, results[0])
        else:
            temps = [self.new_tmp() for _ in results]
            self.emit(f"{', '.join(temps)} = {expr}")
            for t, type in zip(temps, results):
                self.stack.append(V(t, type, simple=True, stable=True))

    def i_call(self, name: str, n: int) -> None:
        ftype = self.function_types[n]
        args = self.popn(len(ftype.params))
        # the callee can change globals (but not our locals)
        self.settle_globals()
        self.call_results(f"f{n}({', '.join(masked(a) for a in args)})", ftype.results)

    def i_call_indirect(self, name: str, arg: Any) -> None:
        type_index, table = arg
        ftype = self.types[type_index]
        i = self.pop()
        args = self.popn(len(ftype.params))
        self.settle_globals()
        target = f"_ci(T{table}, TY{type_index}, {masked(i)})"
        self.call_results(
            f"{target}({', '.join(masked(a) for a in args)})", ftype.results
        )

    def i_nop(self, name: str, arg: Any) -> None:
        pass

    def i_unreachable(self, name: str, arg: Any) -> None:
        self.emit('raise _Trap("unreachable")')
        self.dead = True

    def i_return(self, name: str, arg: Any) -> None:
        self.emit(self.return_stmt())
        self.dead = True

    def i_br(self, name: str, depth: int) -> None:
        self.emit_lines(self.branch(depth))
        self.dead = True

    def i_br_if(self, name: str, depth: int) -> None:
        c = self.pop()
        lines = self.branch(depth)
        self.emit(f"if {self.cond(c)}:")
        self.indent += 1
        self.emit_lines(lines)
        self.indent -= 1

    def i_br_table(self, name: str, arg: Any) -> None:
        depths, default = arg
        index = self.to_temp(V(masked(self.pop()), "i32", simple=False)).expr
        targets = [self.labels[-1 - d] for d in list(depths) + [default]]
        if not self.structured and all(
            t.kind != "func" and not t.branch_types for t in targets
        ):
            pcs = [self.target_pc(t) for t in targets]
            if depths:
                table = "(" + ", ".join(map(str, pcs[:-1])) + ",)"
                self.emit(
                    f"pc = {table}[{index}] if {index} < {len(depths)} else {pcs[-1]}"
                )
            else:
                self.emit(f"pc = {pcs[-1]}")
            self.emit("continue")
            self.dead = True
            return
        if self.structured and self.chain_table(index, depths, targets):
            self.dead = True
            return
        groups: dict[int, list[int]] = {}
        for i, d in enumerate(depths):
            if d != default:
                groups.setdefault(d, []).append(i)
        first = True
        for d, indexes in groups.items():
            if len(indexes) == 1:
                test = f"{index} == {indexes[0]}"
            else:
                test = f"{index} in {tuple(indexes)!r}"
            self.emit(f"{'if' if first else 'elif'} {test}:")
            self.indent += 1
            self.emit_lines(self.branch(d))
            self.indent -= 1
            first = False
        if first:
            self.emit_lines(self.branch(default))
        else:
            self.emit("else:")
            self.indent += 1
            self.emit_lines(self.branch(default))
            self.indent -= 1
        self.dead = True

    def chain_table(self, index: str, depths: Any, targets: list) -> bool:
        """br_table into the blocks of a chain: look the segment up in a
        tuple (after testing for any targets outside the chain). Returns
        False if the targets are not mostly in one chain."""
        if len(depths) < 2 or any(t.branch_types for t in targets):
            return False
        counts: dict[Chain, int] = {}
        for t in targets[:-1]:
            if t.chain is not None:
                counts[t.chain] = counts.get(t.chain, 0) + 1
        if not counts:
            return False
        chain = max(counts, key=counts.__getitem__)
        outside: dict[int, list[int]] = {}
        segs = []
        for i, (d, t) in enumerate(zip(depths, targets)):
            if t.chain is chain:
                segs.append(self.chain_seg(t))
            else:
                outside.setdefault(d, []).append(i)
                segs.append(0)  # never used
        if counts[chain] < 2 or len(outside) > counts[chain]:
            return False
        first = True
        for d, indexes in outside.items():
            if len(indexes) == 1:
                test = f"{index} == {indexes[0]}"
            else:
                test = f"{index} in {tuple(indexes)!r}"
            self.emit(f"{'if' if first else 'elif'} {test}:")
            self.indent += 1
            self.emit_lines(self.branch(d))
            self.indent -= 1
            first = False
        table = "(" + ", ".join(map(str, segs)) + ",)"
        n = len(depths)
        default = targets[-1]
        if default.chain is chain:
            if not first:
                self.emit("else:")
                self.indent += 1
            dseg = self.chain_seg(default)
            self.emit(f"{chain.var} = {table}[{index}] if {index} < {n} else {dseg}")
            self.emit_lines(self.chain_jump(chain))
            if not first:
                self.indent -= 1
            return True
        self.emit(f"{'if' if first else 'elif'} {index} < {n}:")
        self.indent += 1
        self.emit(f"{chain.var} = {table}[{index}]")
        self.emit_lines(self.chain_jump(chain))
        self.indent -= 1
        self.emit("else:")
        self.indent += 1
        self.emit_lines(self.branch(len(self.labels) - 1 - self.labels.index(default)))
        self.indent -= 1
        return True

    def chain_seg(self, target: Label) -> int:
        """The dispatch leaf a branch to a block of a chain goes to."""
        if target.seg is not None:
            return target.seg
        target.branched = True
        return target.chain.n - 1  # the exit leaf

    # --- control flow ---

    def return_stmt(self) -> str:
        n = len(self.ftype.results)
        if n == 0:
            return "return"
        values = self.stack[-n:]
        return "return " + ", ".join(masked(v) for v in values)

    def new_label(self, kind: str, blocktype: Any) -> Label:
        params, results = self.signature(blocktype)
        self.nlabel += 1
        return Label(kind, self.nlabel, params, results, len(self.stack) - len(params))

    def new_pc(self) -> int:
        self.npc += 1
        return self.npc

    def target_pc(self, label: Label) -> int:
        if label.kind == "loop":
            return label.start_pc
        if label.end_pc is None:
            label.end_pc = self.new_pc()
        label.branched = True
        return label.end_pc

    def assign(self, names: list[str], values: list[V]) -> list[str]:
        if not names:
            return []
        return [f"{', '.join(names)} = {', '.join(masked(v) for v in values)}"]

    def branch(self, depth: int) -> list[str]:
        """Lines that branch to the label `depth` levels out."""
        position = len(self.labels) - 1 - depth
        target = self.labels[position]
        if target.kind == "func":
            return [self.return_stmt()]
        types = target.branch_types
        values = self.stack[len(self.stack) - len(types) :] if types else []
        names = target.params if target.kind == "loop" else target.vars
        lines = self.assign(names, values)
        if target.seg is not None:
            # to a later segment of a chain
            chain = target.chain
            return lines + [f"{chain.var} = {target.seg}"] + self.chain_jump(chain)
        if target.kind != "loop":
            target.branched = True
        if not self.structured:
            return lines + [f"pc = {self.target_pc(target)}", "continue"]
        inner = [label for label in self.labels[position + 1 :] if label.pyloop]
        if not inner:
            return lines + ["continue" if target.kind == "loop" else "break"]
        self.uses_flag = True
        target.flag_target = True
        for label in inner:
            label.crossed = True
        return lines + [f"br_ = {target.id}", "break"]

    def tick(self) -> None:
        if self.has_limits:
            self.emit("_L.countdown -= 1")
            self.emit("if _L.countdown < 0:")
            self.emit("    _L.refill()")

    def start_block(self, pc: int) -> None:
        """State machine: start a new basic block."""
        if not self.dead:
            self.emit(f"pc = {pc}")
            self.emit("continue")
        self.blocks.append((self.cur_pc, self.lines))
        self.lines = []
        self.indent = 0
        self.cur_pc = pc
        self.dead = False

    def chain_weights(self, start: int, k: int, merged: bool) -> list[int]:
        """How often each leaf of a chain's dispatch tree is likely to be
        entered, estimated from the number of branches to it (plus one for
        falling into a segment). The first segment is entered every time."""
        body = self.func.body
        weights = [0] * (k + 1)
        member = k - 1  # the innermost open block of the chain
        depth = 0  # constructs open inside the current segment

        def targets(d: int) -> int:
            # the leaf of a branch to the block at depth d, or 0 (none)
            rel = d - depth
            return k - (member - rel) if 0 <= rel <= member else 0

        ip = start + k
        while member >= 0:
            ins = body[ip]
            name = ins.opcode
            if name in ("block", "loop", "if"):
                depth += 1
            elif name == "end":
                if depth:
                    depth -= 1
                else:
                    if member or merged:
                        weights[k - member] += 1  # falling through
                    member -= 1
            elif name in ("br", "br_if"):
                leaf = targets(ins.operand)
                if leaf < k or merged:  # else a plain break out of the chain
                    weights[leaf] += 1
            elif name == "br_table":
                depths, default = ins.operand
                for leaf in set(map(targets, set(depths) | {default})):
                    weights[leaf] += 1
            ip += 1
        weights[0] = sum(weights) + 1
        return weights

    def new_chain(self, start: int, merged: bool) -> Chain:
        self.nlabel += 1
        weights = self.chain_weights(start, self.chains[start], merged)
        chain = Chain(self.nlabel, weights)
        self.nlabel += 1
        chain.cont_id = self.nlabel
        return chain

    def open_chain(self, start: int, k: int) -> None:
        self.settle()
        chain = self.merged.get(start)
        merged = chain is not None
        if chain is None:
            chain = self.new_chain(start, False)
        height = len(self.stack)
        labels = []
        for j in range(k):  # outermost first
            self.nlabel += 1
            results = self.signature(self.func.body[start + j].operand)[1]
            label = Label("block", self.nlabel, (), results, height)
            label.vars = [f"r{label.id}_{i}" for i in range(len(results))]
            label.chain = chain
            # the end of the outermost block is the exit leaf; when the
            # chain shares its loop, the code after it goes there
            label.seg = k - j if j or merged else None
            labels.append(label)
        self.labels.extend(labels)
        if merged:
            return
        labels[0].pyloop = True
        chain.holder = labels[0]
        self.emit(f"{chain.var} = 0")
        self.emit("while True:")
        self.indent += 1
        chain.base = self.indent
        self.enter_leaf(chain, 0)

    def open_loop_chain(self, blocktype: Any, start: int) -> None:
        """A loop that shares its Python loop with a chain in its body."""
        self.settle()
        label = self.new_label("loop", blocktype)
        label.targeted = True
        self.loop_params(label)
        chain = self.merged[start] = self.new_chain(start, True)
        label.chain = chain
        label.seg = 0
        label.pyloop = True
        chain.holder = label
        self.emit(f"{chain.var} = 0")
        self.emit("while True:")
        self.indent += 1
        chain.base = self.indent
        self.labels.append(label)
        self.enter_leaf(chain, 0)
        self.tick()

    def enter_leaf(self, chain: Chain, i: int) -> None:
        """Emit the if/else lines leading to leaf i of the dispatch tree."""
        new = chain.path(i)
        level = 0
        if chain.leaf >= 0:
            old = chain.path(chain.leaf)
            while old[level] == new[level]:
                level += 1
            self.indent = chain.base + level
            self.emit("else:")
            level += 1
        for depth in range(level, len(new)):
            self.indent = chain.base + depth
            self.emit(f"if {chain.var} < {new[depth][0]}:")
        self.indent = chain.base + len(new)
        chain.leaf = i

    def close_chain_block(self, label: Label) -> None:
        chain = label.chain
        fall = not self.dead
        if fall:
            self.emit_lines(self.assign(label.vars, self.results_of(label)))
        self.stack = self.stack[: label.height] + [
            V(name, type, simple=True, stable=True)
            for name, type in zip(label.vars, label.result_types)
        ]
        if label.seg is not None:
            # the end of a segment: fall through into the next one
            if fall:
                self.emit(f"{chain.var} = {label.seg}")
                self.emit("continue")
            self.enter_leaf(chain, label.seg)
            self.dead = False
            return
        # the end of the chain
        if fall:
            self.emit("break")
        self.enter_leaf(chain, chain.n - 1)
        self.emit("break")
        self.indent = chain.base - 1
        if label.crossed:
            self.flag_check()
        self.dead = not (fall or label.branched)

    def chain_jump(self, chain: Chain) -> list[str]:
        """Lines that restart the chain loop (after setting its variable)."""
        position = self.labels.index(chain.holder)
        inner = [label for label in self.labels[position + 1 :] if label.pyloop]
        if not inner:
            return ["continue"]
        self.uses_flag = True
        chain.cont_used = True
        for label in inner:
            label.crossed = True
        return [f"br_ = {chain.cont_id}", "break"]

    def open(self, name: str, blocktype: Any, targeted: bool) -> None:
        cond = self.pop() if name == "if" else None
        self.settle()
        label = self.new_label(name, blocktype)
        label.targeted = targeted
        n_results = len(label.result_types)
        if name == "loop":
            self.loop_params(label)
            if self.structured:
                label.pyloop = True
                self.emit("while True:")
                self.indent += 1
            else:
                label.start_pc = self.new_pc()
                self.start_block(label.start_pc)
            self.tick()
        elif name == "block":
            if n_results and targeted:
                label.vars = [f"r{label.id}_{k}" for k in range(n_results)]
            if self.structured and targeted:
                label.pyloop = True
                self.emit("while True:")
                self.indent += 1
        else:  # if
            if n_results:
                label.vars = [f"r{label.id}_{k}" for k in range(n_results)]
            label.entry = list(self.stack)
            test = self.cond(cond)
            if self.structured:
                if targeted:
                    label.pyloop = True
                    self.emit("while True:")
                    self.indent += 1
                self.emit(f"if {test}:")
                self.indent += 1
                label.arm_start = len(self.lines)
            else:
                label.else_pc = self.new_pc()
                self.emit(f"if not ({test}):")
                self.emit(f"    pc = {label.else_pc}")
                self.emit("    continue")
        self.labels.append(label)

    def loop_params(self, label: Label) -> None:
        """Loop parameters live in variables (branches to the loop set them)."""
        if label.param_types:
            label.params = [f"p{label.id}_{k}" for k in range(len(label.param_types))]
            values = self.popn(len(label.param_types))
            self.emit_lines(self.assign(label.params, values))
            for p, type in zip(label.params, label.param_types):
                self.stack.append(V(p, type, simple=True, stable=True))

    def results_of(self, label: Label) -> list[V]:
        n = len(label.result_types)
        return self.stack[len(self.stack) - n :] if n else []

    def do_else(self) -> None:
        label = self.labels[-1]
        fall = not self.dead
        if fall:
            self.emit_lines(self.assign(label.vars, self.results_of(label)))
        label.then_reach = fall
        label.else_seen = True
        if self.structured:
            if len(self.lines) == label.arm_start:
                self.emit("pass")
            self.indent -= 1
            self.emit("else:")
            self.indent += 1
            label.arm_start = len(self.lines)
        else:
            if fall:
                self.emit(f"pc = {self.target_pc(label)}")
                self.emit("continue")
                self.dead = True
            self.start_block(label.else_pc)
        self.stack = list(label.entry)
        self.dead = False

    def close(self) -> bool:
        """Handle `end`. Returns True at the end of the function."""
        label = self.labels.pop()
        if label.chain is not None and label.kind == "block":
            self.close_chain_block(label)
            return False
        fall = not self.dead
        if label.kind == "func":
            if fall:
                self.emit(self.return_stmt())
            return True
        if label.kind == "loop":
            # results only arrive by falling through, so they stay symbolic
            if self.structured:
                if fall:
                    self.emit("break")
                if label.chain is not None:
                    self.indent = label.chain.base - 1
                else:
                    self.indent -= 1
                if label.crossed:
                    self.flag_check()
            self.dead = not fall
            return False
        if fall:
            self.emit_lines(self.assign(label.vars, self.results_of(label)))
        reach = fall
        if label.kind == "if":
            entry_values = (
                label.entry[len(label.entry) - len(label.param_types) :]
                if label.param_types
                else []
            )
            if self.structured:
                if len(self.lines) == label.arm_start:
                    self.emit("pass")
                if not label.else_seen and label.vars:
                    # no else: the params pass through as the results
                    self.indent -= 1
                    self.emit("else:")
                    self.indent += 1
                    self.emit_lines(self.assign(label.vars, entry_values))
                self.indent -= 1
            else:
                if not label.else_seen:
                    if fall:
                        self.emit(f"pc = {self.target_pc(label)}")
                        self.emit("continue")
                        self.dead = True
                    self.start_block(label.else_pc)
                    self.emit_lines(self.assign(label.vars, entry_values))
                    self.dead = False
                    fall = True
                if fall and not self.dead:
                    self.emit(f"pc = {self.target_pc(label)}")
                    self.emit("continue")
                    self.dead = True
                if label.end_pc is not None:
                    self.start_block(label.end_pc)
            reach = reach or label.then_reach or not label.else_seen
        elif not self.structured and label.end_pc is not None:
            self.start_block(label.end_pc)
        if self.structured and label.pyloop:
            # leave the `while True:` if the end of the construct is reached
            # by falling through (from either arm of an if)
            if reach:
                self.emit("break")
            self.indent -= 1
            if label.crossed:
                self.flag_check()
        reach = reach or label.branched
        base = self.stack[: label.height]
        if label.vars:
            self.stack = base + [
                V(name, type, simple=True, stable=True)
                for name, type in zip(label.vars, label.result_types)
            ]
        elif fall:
            self.stack = base + self.results_of(label)
        else:
            self.stack = base + [
                V("None", t, simple=True, stable=True) for t in label.result_types
            ]
        self.dead = not reach
        return False

    def flag_check(self) -> None:
        """After a Python loop that a branch passed through: keep leaving
        loops until the branch reaches its target."""
        parent = next((label for label in reversed(self.labels) if label.pyloop), None)
        self.emit("if br_:")
        self.indent += 1
        if parent is not None and parent.flag_target:
            self.emit(f"if br_ == {parent.id}:")
            self.emit("    br_ = 0")
            self.emit("    continue" if parent.kind == "loop" else "    break")
        if parent is not None and parent.chain is not None and parent.chain.cont_used:
            # a branch to a later segment of the chain
            self.emit(f"if br_ == {parent.chain.cont_id}:")
            self.emit("    br_ = 0")
            self.emit("    continue")
        self.emit("break")
        self.indent -= 1

    # --- output ---

    def assemble(self) -> str:
        name = f"f{self.index}"
        n_params = len(self.ftype.params)
        out = [f"def {name}({', '.join(f'l{i}' for i in range(n_params))}):"]
        by_zero: dict[str, list[str]] = {}
        for i, type in enumerate(self.local_types[n_params:], n_params):
            by_zero.setdefault(ZERO_LITERAL[type], []).append(f"l{i}")
        for zero, names in by_zero.items():
            for start in range(0, len(names), 50):
                out.append(f"    {' = '.join(names[start:start + 50])} = {zero}")
        if self.uses_flag:
            out.append("    br_ = 0")
        if self.has_limits:
            out += [
                "    _L.countdown -= 1",
                "    if _L.countdown < 0:",
                "        _L.refill()",
            ]
        if self.structured:
            out.append("    try:")
            body = [("    " * (indent + 2)) + text for indent, text in self.lines]
            out += body or ["        pass"]
        else:
            self.blocks.append((self.cur_pc, self.lines))
            out.append("    pc = 0")
            out.append("    try:")
            out.append("        while True:")
            _dispatch(sorted(self.blocks), 3, out)
        out += [
            "    except (_StructError, IndexError) as e:",
            "        if e.__traceback__.tb_next is None:",
            '            raise _Trap("out of bounds memory access") from None',
            "        raise",
        ]
        return "\n".join(out) + "\n"

    HANDLERS: dict[str, Any] = {}


def _dispatch(blocks: list, indent: int, out: list[str]) -> None:
    """Binary search on pc over basic blocks (sorted by pc)."""
    pad = "    " * indent

    def body(lines: list, level: int) -> None:
        if not lines:
            out.append("    " * level + "pass")
        for i, text in lines:
            out.append("    " * (level + i) + text)

    if len(blocks) == 1:
        body(blocks[0][1], indent)
    elif len(blocks) <= 4:
        for k, (pc, lines) in enumerate(blocks):
            if k == 0:
                out.append(f"{pad}if pc == {pc}:")
            elif k < len(blocks) - 1:
                out.append(f"{pad}elif pc == {pc}:")
            else:
                out.append(f"{pad}else:")
            body(lines, indent + 1)
    else:
        middle = len(blocks) // 2
        out.append(f"{pad}if pc < {blocks[middle][0]}:")
        _dispatch(blocks[:middle], indent + 1, out)
        out.append(f"{pad}else:")
        _dispatch(blocks[middle:], indent + 1, out)


def _register_handlers() -> None:
    h = Translator.HANDLERS
    for t in ("i32", "i64"):
        h[f"{t}.const"] = Translator.i_const
        for operation in (
            "add sub mul and or xor shl shr_u shr_s rotl rotr eq ne lt_s lt_u "
            "gt_s gt_u le_s le_u ge_s ge_u div_s div_u rem_s rem_u"
        ).split():
            h[f"{t}.{operation}"] = Translator.i_int_binary
        for operation in ("eqz", "extend8_s", "extend16_s"):
            h[f"{t}.{operation}"] = Translator.i_int_unary
    h["i64.extend32_s"] = Translator.i_int_unary
    h["i32.wrap_i64"] = Translator.i_int_unary
    h["i64.extend_i32_u"] = Translator.i_int_unary
    h["i64.extend_i32_s"] = Translator.i_int_unary
    for t in ("f32", "f64"):
        h[f"{t}.const"] = Translator.i_const
        for operation in ("add", "sub", "mul", "div", "min", "max", "copysign"):
            h[f"{t}.{operation}"] = Translator.i_float_binary
        for operation in COMPARE:
            h[f"{t}.{operation}"] = Translator.i_float_binary
    h["f64.neg"] = Translator.i_f64_neg
    for name in UNARY_CALLS:
        h[name] = Translator.i_unary_call
    for name in TRAPPING_UNARY:
        h[name] = Translator.i_trapping_unary
    for name in LOADS:
        h[name] = Translator.i_load
    for name in STORES:
        h[name] = Translator.i_store
    h["local.get"] = Translator.i_local_get
    h["local.set"] = Translator.i_local_set
    h["local.tee"] = Translator.i_local_set
    h["global.get"] = Translator.i_global_get
    h["global.set"] = Translator.i_global_set
    h["drop"] = Translator.i_drop
    h["select"] = Translator.i_select
    h["select_t"] = Translator.i_select
    h["memory.size"] = Translator.i_memory_size
    h["memory.grow"] = Translator.i_memory_grow
    for name in (
        "memory.fill",
        "memory.copy",
        "memory.init",
        "data.drop",
        "table.copy",
        "table.init",
        "elem.drop",
    ):
        h[name] = Translator.i_bulk
    for name in ("table.get", "table.set", "table.size", "table.grow", "table.fill"):
        h[name] = Translator.i_table
    h["ref.null"] = Translator.i_ref_null
    h["ref.is_null"] = Translator.i_ref_is_null
    h["ref.func"] = Translator.i_ref_func
    h["call"] = Translator.i_call
    h["call_indirect"] = Translator.i_call_indirect
    h["nop"] = Translator.i_nop
    h["unreachable"] = Translator.i_unreachable
    h["return"] = Translator.i_return
    h["br"] = Translator.i_br
    h["br_if"] = Translator.i_br_if
    h["br_table"] = Translator.i_br_table
    h["else"] = lambda self, name, arg: self.do_else()


_register_handlers()


# --- runtime support ---


def _ldf32(mem: bytearray, address: int) -> float:
    try:
        value = _unpack_f32(mem, address)[0]
    except struct.error:
        raise TrapError("out of bounds memory access") from None
    if value != value:
        return num.f32_from_bits(_unpack_u32(mem, address)[0])
    return value


def _stf32(mem: bytearray, address: int, value: float) -> None:
    try:
        if type(value) is F32NaN:
            _pack_u32(mem, address, value.bits)
        else:
            _pack_f32(mem, address, value)
    except struct.error:
        raise TrapError("out of bounds memory access") from None


def _call_indirect(table: Any, ftype: Any, i: int) -> Any:
    elements = table.elements
    if i >= len(elements):
        raise TrapError("undefined element")
    f = elements[i]
    if f is None:
        raise TrapError(f"uninitialized element {i}")
    if f.type is not ftype and f.type != ftype:
        raise TrapError("indirect call type mismatch")
    return f.entry


_unpack_f32 = struct.Struct("<f").unpack_from
_pack_f32 = struct.Struct("<f").pack_into
_unpack_u32 = struct.Struct("<I").unpack_from
_pack_u32 = struct.Struct("<I").pack_into

_BASE_NAMESPACE: dict[str, Any] = {
    name: getattr(num, name)
    for name in dir(num)
    if name.startswith(("i32_", "i64_", "f32_", "f64_"))
}
_BASE_NAMESPACE.update(
    {
        "_f32r": num.f32_round,
        "_f32b": num.f32_from_bits,
        "_f64b": num.f64_reinterpret_i64,
        "_fabs": math.fabs,
        "_copysign": math.copysign,
        "_INF": math.inf,
        "_NINF": -math.inf,
        "_Trap": TrapError,
        "_StructError": struct.error,
        "_u16": struct.Struct("<H").unpack_from,
        "_u32": _unpack_u32,
        "_u64": struct.Struct("<Q").unpack_from,
        "_s8": struct.Struct("<b").unpack_from,
        "_s16": struct.Struct("<h").unpack_from,
        "_s32": struct.Struct("<i").unpack_from,
        "_ud": struct.Struct("<d").unpack_from,
        "_p16": struct.Struct("<H").pack_into,
        "_p32": _pack_u32,
        "_p64": struct.Struct("<Q").pack_into,
        "_pd": struct.Struct("<d").pack_into,
        "_ldf32": _ldf32,
        "_stf32": _stf32,
        "_ci": _call_indirect,
        "__builtins__": __builtins__,
    }
)


def _instance_helpers(instance: Any) -> dict[str, Any]:
    memories = instance.memories
    tables = instance.tables
    mem = memories[0].data if memories else None

    def mfill(d: int, value: int, n: int) -> None:
        if d + n > len(mem):
            raise TrapError("out of bounds memory access")
        mem[d : d + n] = bytes((value & 0xFF,)) * n

    def mcopy(d: int, s: int, n: int) -> None:
        if s + n > len(mem) or d + n > len(mem):
            raise TrapError("out of bounds memory access")
        mem[d : d + n] = mem[s : s + n]

    def minit(segment: int, d: int, s: int, n: int) -> None:
        data = instance.datas[segment]
        if s + n > len(data) or d + n > len(mem):
            raise TrapError("out of bounds memory access")
        mem[d : d + n] = data[s : s + n]

    def ddrop(segment: int) -> None:
        instance.datas[segment] = b""

    def tcopy(dst: int, src: int, d: int, s: int, n: int) -> None:
        dst_elements = tables[dst].elements
        src_elements = tables[src].elements
        if s + n > len(src_elements) or d + n > len(dst_elements):
            raise TrapError("out of bounds table access")
        dst_elements[d : d + n] = src_elements[s : s + n]

    def tinit(segment: int, table: int, d: int, s: int, n: int) -> None:
        tables[table].init(d, instance.elements[segment], s, n)

    def edrop(segment: int) -> None:
        instance.elements[segment] = []

    return {
        "_mfill": mfill,
        "_mcopy": mcopy,
        "_minit": minit,
        "_ddrop": ddrop,
        "_tget": lambda t, i: tables[t].get(i),
        "_tset": lambda t, i, v: tables[t].set(i, v),
        "_tsize": lambda t: len(tables[t].elements),
        "_tgrow": lambda t, init, n: tables[t].grow(n, init) & MASK_32,
        "_tfill": lambda t, i, v, n: tables[t].fill(i, v, n),
        "_tcopy": tcopy,
        "_tinit": tinit,
        "_edrop": edrop,
    }


def instance_namespace(instance: Any) -> dict[str, Any]:
    """The globals dictionary generated code for this instance runs in."""
    namespace = getattr(instance, "_namespace", None)
    if namespace is not None:
        return namespace
    namespace = dict(_BASE_NAMESPACE)
    memory = instance.memories[0] if instance.memories else None
    namespace["mem"] = memory.data if memory is not None else None
    namespace["_M0"] = memory
    if memory is not None and USE_VIEWS:
        memory.views(namespace)
    namespace["_L"] = instance.limits
    for j, f in enumerate(instance.functions):
        namespace[f"f{j}"] = f.entry
        namespace[f"F{j}"] = f
        if hasattr(f, "_refs"):
            f._refs.append((namespace, f"f{j}"))
    for j, g in enumerate(instance.globals):
        namespace[f"g{j}"] = g
    for k, t in enumerate(instance.module.types):
        namespace[f"TY{k}"] = t
    for j, t in enumerate(instance.tables):
        namespace[f"T{j}"] = t
    namespace.update(_instance_helpers(instance))
    instance._namespace = namespace
    return namespace


def python_source(wfunc: Any) -> str:
    """The Python source code pwasm generates for a WebAssembly function."""
    translator = Translator(
        wfunc.instance.module,
        wfunc.index,
        wfunc.func,
        wfunc.type,
        wfunc.instance.limits is not None,
    )
    return translator.translate()


def _cache_key(index: int, has_limits: bool) -> tuple:
    return (index, has_limits, FORCE_STATE_MACHINE, FORCE_CHAINS)


def is_cached(module: Any, index: int, has_limits: bool) -> bool:
    """Whether Python code for this function has already been compiled."""
    cache = getattr(module, "_python_code", None)
    return bool(cache) and _cache_key(index, has_limits) in cache


def compile_to_python(wfunc: Any) -> FunctionType:
    """Compile a WasmFunction to a Python function (code objects are cached
    on the module and shared between its instances)."""
    instance = wfunc.instance
    module = instance.module
    cache = getattr(module, "_python_code", None)
    if cache is None:
        cache = module._python_code = {}
    key = _cache_key(wfunc.index, instance.limits is not None)
    code = cache.get(key)
    if code is None:
        source = python_source(wfunc)
        filename = f"<pwasm {id(module):x} f{wfunc.index}>"
        linecache.cache[filename] = (
            len(source),
            None,
            source.splitlines(True),
            filename,
        )
        scratch: dict[str, Any] = {}
        exec(compile(source, filename, "exec"), scratch)
        code = scratch[f"f{wfunc.index}"].__code__
        cache[key] = code
    return FunctionType(code, instance_namespace(instance), f"f{wfunc.index}")
