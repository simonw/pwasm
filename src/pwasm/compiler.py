"""Compile WebAssembly function bodies into flat code for the executor.

Structured control flow is compiled away: `block`, `loop` and `end` emit
nothing, and every branch becomes a jump to a precomputed instruction index.
Because operand stack heights are static in valid WebAssembly, the compiler
also knows exactly which values a branch has to discard, so the executor
never keeps a label stack.

Compiled code is two parallel lists: `ops` (internal opcodes, below) and
`imms` (their immediates).
"""

from __future__ import annotations

import operator
import struct
from typing import Any

from . import numeric as num
from .errors import WasmError
from .numeric import MASK_32, MASK_64
from .runtime import ZERO, HostFunction

# Internal opcodes. The executor tests the first group one by one (they are
# the most frequent instructions in compiled C), then dispatches on ranges of
# ten, so keep related rare opcodes together.
(
    LOCAL_GET,
    CONST,
    LOCAL_SET,
    LOCAL_TEE,
    I32_ADD,
    I32_LOAD,
    BR_IF,
    I32_STORE,
    CALL,
    JMP,
    # 10
    I32_AND,
    I32_EQZ,
    I32_SHL,
    I32_NE,
    I32_SUB,
    I32_EQ,
    GLOBAL_GET,
    GLOBAL_SET,
    DROP,
    IF_FALSE,
    # 20
    I32_LT_U,
    I32_LT_S,
    I32_GT_U,
    I32_GT_S,
    I32_LE_U,
    I32_LE_S,
    I32_GE_U,
    I32_GE_S,
    I32_OR,
    I32_XOR,
    # 30
    I32_SHR_U,
    I32_MUL,
    SELECT,
    RETURN,
    JMP_DROP,
    BR_IF_DROP,
    BR_TABLE,
    CALL0,
    CALLN,
    CALL_HOST,
    # 40: memory
    LOAD,
    LOAD_MASK,
    STORE,
    STORE_RAW,
    I32_LOAD8_U,
    I32_STORE8,
    LOAD_F32,
    STORE_F32,
    MEMORY_SIZE,
    MEMORY_GROW,
    # 50
    BINOP,
    UNOP,
    UNREACHABLE,
    RETURN_IF,
) = range(54)

# Opcodes whose failures (struct.error / IndexError) mean an out of bounds
# memory access rather than a bug.
MEMORY_OPS = frozenset(
    {
        I32_LOAD,
        I32_STORE,
        LOAD,
        LOAD_MASK,
        STORE,
        STORE_RAW,
        I32_LOAD8_U,
        I32_STORE8,
        LOAD_F32,
        STORE_F32,
    }
)

INLINE_BINOPS = {
    "i32.add": I32_ADD,
    "i32.sub": I32_SUB,
    "i32.mul": I32_MUL,
    "i32.and": I32_AND,
    "i32.or": I32_OR,
    "i32.xor": I32_XOR,
    "i32.shl": I32_SHL,
    "i32.shr_u": I32_SHR_U,
    "i32.eq": I32_EQ,
    "i32.ne": I32_NE,
    "i32.lt_u": I32_LT_U,
    "i32.lt_s": I32_LT_S,
    "i32.gt_u": I32_GT_U,
    "i32.gt_s": I32_GT_S,
    "i32.le_u": I32_LE_U,
    "i32.le_s": I32_LE_S,
    "i32.ge_u": I32_GE_U,
    "i32.ge_s": I32_GE_S,
}

INLINE_UNOPS = {
    "i32.eqz": I32_EQZ,
}

# Binary and unary operations executed by calling a Python function.
BINOP_FUNCS: dict[str, Any] = {
    "i32.div_s": num.i32_div_s,
    "i32.div_u": num.i32_div_u,
    "i32.rem_s": num.i32_rem_s,
    "i32.rem_u": num.i32_rem_u,
    "i32.shr_s": num.i32_shr_s,
    "i32.rotl": num.i32_rotl,
    "i32.rotr": num.i32_rotr,
    "i64.add": num.i64_add,
    "i64.sub": num.i64_sub,
    "i64.mul": num.i64_mul,
    "i64.div_s": num.i64_div_s,
    "i64.div_u": num.i64_div_u,
    "i64.rem_s": num.i64_rem_s,
    "i64.rem_u": num.i64_rem_u,
    "i64.and": operator.and_,
    "i64.or": operator.or_,
    "i64.xor": operator.xor,
    "i64.shl": num.i64_shl,
    "i64.shr_s": num.i64_shr_s,
    "i64.shr_u": num.i64_shr_u,
    "i64.rotl": num.i64_rotl,
    "i64.rotr": num.i64_rotr,
    "i64.eq": operator.eq,
    "i64.ne": operator.ne,
    "i64.lt_u": operator.lt,
    "i64.gt_u": operator.gt,
    "i64.le_u": operator.le,
    "i64.ge_u": operator.ge,
    "i64.lt_s": num.i64_lt_s,
    "i64.gt_s": num.i64_gt_s,
    "i64.le_s": num.i64_le_s,
    "i64.ge_s": num.i64_ge_s,
    "f32.add": num.f32_add,
}

UNOP_FUNCS: dict[str, Any] = {
    "i32.clz": num.i32_clz,
    "i32.ctz": num.i32_ctz,
    "i32.popcnt": num.i32_popcnt,
    "i32.extend8_s": num.i32_extend8_s,
    "i32.extend16_s": num.i32_extend16_s,
    "i32.wrap_i64": num.i32_wrap_i64,
    "i64.extend_i32_s": num.i64_extend_i32_s,
    "i64.clz": num.i64_clz,
    "i64.ctz": num.i64_ctz,
    "i64.popcnt": num.i32_popcnt,
    "i64.eqz": num.i64_eqz,
    "i64.extend8_s": num.i64_extend8_s,
    "i64.extend16_s": num.i64_extend16_s,
    "i64.extend32_s": num.i64_extend32_s,
}

# Instructions that leave the (internal representation of the) value alone
IDENTITY_UNOPS = {"i64.extend_i32_u"}


def _unpacker(fmt: str):
    return struct.Struct(fmt).unpack_from


def _packer(fmt: str):
    return struct.Struct(fmt).pack_into


# name -> (opcode, function building the immediate from the offset)
LOADS: dict[str, tuple[int, Any]] = {
    "i32.load": (I32_LOAD, lambda off: off),
    "i32.load8_u": (I32_LOAD8_U, lambda off: off),
    "i64.load8_u": (I32_LOAD8_U, lambda off: off),
    "i32.load8_s": (LOAD_MASK, lambda off: (_unpacker("<b"), off, MASK_32)),
    "i32.load16_s": (LOAD_MASK, lambda off: (_unpacker("<h"), off, MASK_32)),
    "i32.load16_u": (LOAD, lambda off: (_unpacker("<H"), off)),
    "i64.load": (LOAD, lambda off: (_unpacker("<Q"), off)),
    "i64.load8_s": (LOAD_MASK, lambda off: (_unpacker("<b"), off, MASK_64)),
    "i64.load16_s": (LOAD_MASK, lambda off: (_unpacker("<h"), off, MASK_64)),
    "i64.load16_u": (LOAD, lambda off: (_unpacker("<H"), off)),
    "i64.load32_s": (LOAD_MASK, lambda off: (_unpacker("<i"), off, MASK_64)),
    "i64.load32_u": (LOAD, lambda off: (_unpacker("<I"), off)),
    "f32.load": (LOAD_F32, lambda off: off),
    "f64.load": (LOAD, lambda off: (_unpacker("<d"), off)),
}

STORES: dict[str, tuple[int, Any]] = {
    "i32.store": (I32_STORE, lambda off: off),
    "i32.store8": (I32_STORE8, lambda off: off),
    "i64.store8": (I32_STORE8, lambda off: off),
    "i32.store16": (STORE, lambda off: (_packer("<H"), off, 0xFFFF)),
    "i64.store16": (STORE, lambda off: (_packer("<H"), off, 0xFFFF)),
    "i64.store32": (STORE, lambda off: (_packer("<I"), off, MASK_32)),
    "i64.store": (STORE_RAW, lambda off: (_packer("<Q"), off)),
    "f32.store": (STORE_F32, lambda off: off),
    "f64.store": (STORE_RAW, lambda off: (_packer("<d"), off)),
}


class Code:
    """Compiled code for one function."""

    __slots__ = ("ops", "imms", "zeros", "mem")

    def __init__(self, ops: list, imms: list, zeros: list, mem: Any) -> None:
        self.ops = ops
        self.imms = imms
        self.zeros = zeros
        self.mem = mem


class _Label:
    """Compile-time control frame for block/loop/if and the function body."""

    __slots__ = (
        "kind",
        "height",
        "n_params",
        "n_results",
        "start",
        "fixups",
        "if_jump",
    )

    def __init__(
        self, kind: str, height: int, n_params: int, n_results: int, start: int
    ):
        self.kind = kind
        self.height = height  # operand stack height below the block's params
        self.n_params = n_params
        self.n_results = n_results
        self.start = start  # loop: branch target
        self.fixups: list = []  # forward branches to patch at `end`
        self.if_jump: int | None = None  # index of the IF_FALSE to patch

    @property
    def arity(self) -> int:
        return self.n_params if self.kind == "loop" else self.n_results


def compile_function(wfunc: Any) -> Code:
    instance = wfunc.instance
    func = wfunc.func
    types = instance.module.types
    ops: list[int] = []
    imms: list[Any] = []

    def emit(op: int, imm: Any = None) -> None:
        ops.append(op)
        imms.append(imm)

    def block_signature(blocktype: Any) -> tuple[int, int]:
        if blocktype == ():
            return 0, 0
        if isinstance(blocktype, tuple):
            return 0, len(blocktype)
        t = types[blocktype]
        return len(t.params), len(t.results)

    labels = [_Label("func", 0, 0, wfunc.n_results, 0)]
    height = 0
    dead = False  # current code is unreachable (after br, return, ...)
    skip = 0  # nesting depth of blocks inside dead code

    def branch_target(label: _Label, entry: list | None = None) -> list:
        """[target, drop_lo, drop_hi] for a branch from the current height.
        Forward targets are patched when the label's `end` is reached."""
        lo = label.height
        hi = height - label.arity
        target = label.start if label.kind == "loop" else None
        entry = [target, lo, max(hi, lo)]
        if target is None:
            label.fixups.append(entry)
        return entry

    def emit_branch(label: _Label, conditional: bool) -> None:
        if label.kind == "func":
            emit(RETURN_IF if conditional else RETURN, label.n_results)
            return
        target, lo, hi = entry = branch_target(label)
        if hi > lo:
            emit(BR_IF_DROP if conditional else JMP_DROP, entry)
        else:
            if target is None:
                label.fixups[-1] = len(ops)  # patch the imm itself
            emit(BR_IF if conditional else JMP, target)

    for instr in func.body:
        name = instr.opcode
        arg = instr.operand

        if dead:
            if name in ("block", "loop", "if"):
                skip += 1
                continue
            if skip:
                if name == "end":
                    skip -= 1
                continue
            if name not in ("end", "else"):
                continue

        if name == "local.get":
            emit(LOCAL_GET, arg)
            height += 1
        elif name == "i32.const":
            emit(CONST, arg & MASK_32)
            height += 1
        elif name == "local.set":
            emit(LOCAL_SET, arg)
            height -= 1
        elif name == "local.tee":
            emit(LOCAL_TEE, arg)
        elif name in INLINE_BINOPS:
            emit(INLINE_BINOPS[name])
            height -= 1
        elif name in LOADS:
            op, make_imm = LOADS[name]
            emit(op, make_imm(arg[1]))
        elif name in STORES:
            op, make_imm = STORES[name]
            emit(op, make_imm(arg[1]))
            height -= 2
        elif name == "br_if":
            height -= 1
            emit_branch(labels[-1 - arg], True)
        elif name == "br":
            emit_branch(labels[-1 - arg], False)
            dead = True
        elif name == "call":
            callee = instance.functions[arg]
            n_params = callee.n_params
            n_results = callee.n_results
            if isinstance(callee, HostFunction):
                emit(CALL_HOST, callee)
            elif n_results == 1:
                emit(CALL, (callee, n_params))
            elif n_results == 0:
                emit(CALL0, (callee, n_params))
            else:
                emit(CALLN, (callee, n_params))
            height += n_results - n_params
        elif name == "block":
            n_params, n_results = block_signature(arg)
            labels.append(_Label("block", height - n_params, n_params, n_results, 0))
        elif name == "loop":
            n_params, n_results = block_signature(arg)
            labels.append(
                _Label("loop", height - n_params, n_params, n_results, len(ops))
            )
        elif name == "if":
            height -= 1
            n_params, n_results = block_signature(arg)
            label = _Label("if", height - n_params, n_params, n_results, 0)
            label.if_jump = len(ops)
            emit(IF_FALSE, None)
            labels.append(label)
        elif name == "else":
            label = labels[-1]
            if not dead:
                emit(JMP, None)
                label.fixups.append(len(ops) - 1)
            imms[label.if_jump] = len(ops)
            label.if_jump = None
            height = label.height + label.n_params
            dead = False
        elif name == "end":
            label = labels.pop()
            height = label.height + label.n_results
            dead = False
            if label.kind == "func":
                end = len(ops)
                emit(RETURN, label.n_results)
            else:
                end = len(ops)
            if label.if_jump is not None:
                imms[label.if_jump] = end
            for fixup in label.fixups:
                if isinstance(fixup, int):
                    imms[fixup] = end
                else:
                    fixup[0] = end
            if label.kind == "func":
                break
        elif name == "i64.const":
            emit(CONST, arg & MASK_64)
            height += 1
        elif name in ("f32.const", "f64.const"):
            emit(CONST, arg if name == "f32.const" else float(arg))
            height += 1
        elif name in INLINE_UNOPS:
            emit(INLINE_UNOPS[name])
        elif name in BINOP_FUNCS:
            emit(BINOP, BINOP_FUNCS[name])
            height -= 1
        elif name in UNOP_FUNCS:
            emit(UNOP, UNOP_FUNCS[name])
        elif name in IDENTITY_UNOPS:
            pass
        elif name == "global.get":
            emit(GLOBAL_GET, instance.globals[arg])
            height += 1
        elif name == "global.set":
            emit(GLOBAL_SET, instance.globals[arg])
            height -= 1
        elif name == "drop":
            emit(DROP)
            height -= 1
        elif name in ("select", "select_t"):
            emit(SELECT)
            height -= 2
        elif name == "br_table":
            height -= 1
            depths, default = arg
            entries = []
            for depth in list(depths) + [default]:
                label = labels[-1 - depth]
                if label.kind == "func":
                    entries.append(None)
                else:
                    entries.append(branch_target(label))
            emit(BR_TABLE, (entries[:-1], entries[-1]))
            dead = True
        elif name == "return":
            emit(RETURN, wfunc.n_results)
            dead = True
        elif name == "unreachable":
            emit(UNREACHABLE)
            dead = True
        elif name == "memory.size":
            emit(MEMORY_SIZE)
            height += 1
        elif name == "memory.grow":
            emit(MEMORY_GROW, instance.memories[0])
        elif name == "nop":
            pass
        else:
            raise WasmError(f"Unsupported instruction: {name}")

    zeros = [ZERO[t] for t in func.locals]
    memories = instance.memories
    mem = memories[0].data if memories else None
    return Code(ops, imms, zeros, mem)
