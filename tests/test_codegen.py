"""Compiling WebAssembly functions to Python source code."""

import struct
import traceback

import pytest

from pwasm import TrapError, decode_module, instantiate
from pwasm.codegen import python_source
from pwasm.runtime import Limits
from wat import wat2wasm

ADD = """(module (func (export "add") (param i32 i32) (result i32)
  (i32.add (local.get 0) (local.get 1))))"""

COUNT = """(module (func (export "count") (param i32) (result i32) (local i32)
  (block $done
    (loop $l
      (br_if $done (i32.ge_u (local.get 1) (local.get 0)))
      (local.set 1 (i32.add (local.get 1) (i32.const 1)))
      (br $l)))
  (local.get 1)))"""


def load(text, mode="compile", imports=None, **kwargs):
    return instantiate(decode_module(wat2wasm(text)), imports, mode=mode, **kwargs)


def switch_module(n):
    """A function whose br_table switch nests n + 1 blocks deep."""
    body = "(br_table " + " ".join(f"$b{k}" for k in range(n + 1)) + " (local.get 0))"
    for k in range(n + 1):
        body = f"(block $b{k} {body})"
        if k < n:
            body += f" (return (i32.const {k * 10}))"
    return (
        f'(module (func (export "f") (param i32) (result i32) {body} (i32.const 999)))'
    )


def nested_loops_module(n):
    """n nested loops; the innermost counts up to the parameter."""
    body = (
        "(local.set 1 (i32.add (local.get 1) (i32.const 1)))"
        f"(br_if $l{n - 1} (i32.lt_u (local.get 1) (local.get 0)))"
    )
    for k in reversed(range(n)):
        body = f"(loop $l{k} {body})"
    return f'(module (func (export "f") (param i32) (result i32) (local i32) {body} (local.get 1)))'


def test_compile_mode_compiles_on_first_call():
    inst = load(ADD)
    func = inst.functions[0]
    assert func.pyfunc is None
    assert inst.exports.add(2, 3) == 5
    assert func.pyfunc is not None
    assert inst.exports.add(-1, 1) == 0


def test_interpret_mode_never_compiles():
    inst = load(ADD, mode="interpret")
    for _ in range(20):
        assert inst.exports.add(2, 3) == 5
    assert inst.functions[0].pyfunc is None


def test_auto_mode_compiles_functions_once_they_are_called_again():
    inst = load(ADD, mode="auto")
    inst.exports.add(1, 1)
    assert inst.functions[0].pyfunc is None
    for _ in range(10):
        inst.exports.add(1, 1)
    assert inst.functions[0].pyfunc is not None


def test_expressions_are_folded():
    inst = load(ADD)
    source = python_source(inst.functions[0])
    assert "return (l0 + l1) & 0xFFFFFFFF" in source


def test_shallow_functions_use_structured_control_flow():
    inst = load(COUNT)
    assert inst.exports.count(7) == 7
    source = python_source(inst.functions[0])
    assert "while True:" in source
    assert "pc = " not in source


@pytest.mark.parametrize("n", [3, 40, 300])
def test_deep_switches(n):
    inst = load(switch_module(n))
    for k in range(n):
        assert inst.exports.f(k) == k * 10
    assert inst.exports.f(n) == 999
    assert inst.exports.f(10_000) == 999
    source = python_source(inst.functions[0])
    # deep switches stay structured: the chain of blocks becomes one loop
    # that dispatches on a segment variable
    assert "pc = " not in source
    assert ("s1 = (" in source) == (n > 3)


CHAIN = """(module (func (export "f") (param i32) (result i32) (local i32)
  (block $b1
    (block $b2
      (block $b3
        (block $b4
          (br_table $b4 $b3 $b2 $b1 (local.get 0)))
        ;; case 0: add 1, then fall through into case 1
        (local.set 1 (i32.add (local.get 1) (i32.const 1))))
      ;; case 1: a loop that jumps to case 2, or out of the switch
      (loop $l
        (local.set 1 (i32.add (local.get 1) (i32.const 10)))
        (br_if $b2 (i32.eq (local.get 1) (i32.const 21)))
        (br_if $b1 (i32.gt_u (local.get 1) (i32.const 25)))
        (br $l)))
    ;; case 2
    (local.set 1 (i32.add (local.get 1) (i32.const 100))))
  (local.get 1)))"""


@pytest.mark.parametrize("force", [False, True])
def test_block_chains(monkeypatch, force):
    import pwasm.codegen

    monkeypatch.setattr(pwasm.codegen, "FORCE_CHAINS", force)
    inst = load(CHAIN)
    assert [inst.exports.f(i) for i in range(5)] == [121, 30, 100, 0, 0]
    assert ("s1 = " in python_source(inst.functions[0])) == force


def test_chain_dispatch_trees_favour_heavy_segments():
    from pwasm.codegen import Chain

    weights = [100, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 60, 1]
    chain = Chain(1, weights)
    n = len(weights)
    depths = [len(chain.path(i)) for i in range(n)]
    assert depths[0] == 1
    assert depths[16] <= 3
    # never much deeper than a balanced tree
    assert max(depths) <= (n - 1).bit_length() + 2
    # each path of tests narrows down to its leaf
    for i in range(n):
        lo, hi = 0, n
        for mid, side in chain.path(i):
            assert lo < mid < hi
            lo, hi = (lo, mid) if side == 0 else (mid, hi)
        assert (lo, hi) == (i, i + 1)


def test_chains_test_for_the_first_segment_first():
    inst = load(switch_module(40))
    inst.exports.f(1)
    lines = python_source(inst.functions[0]).splitlines()
    start = lines.index("        s1 = 0")
    assert lines[start + 1 : start + 3] == [
        "        while True:",
        "            if s1 < 1:",
    ]


TYPED_CHAIN = """(module (func (export "f") (param i32) (result i32) (local i32)
  (block $b1
    (block $b2 (result i32)
      (block $b3
        (block $b4
          (br_table $b4 $b3 $b1 (local.get 0)))
        ;; case 0
        (local.set 1 (i32.const 5)))
      ;; case 1, and case 0 falls through: $b2 gets a value either way
      (drop (br_if $b2 (i32.const 7) (i32.eqz (local.get 1))))
      (i32.add (local.get 1) (i32.const 1000)))
    (local.set 1 (i32.mul (i32.const 2))))
  (local.get 1)))"""


@pytest.mark.parametrize("force", [False, True])
def test_block_chains_with_results(monkeypatch, force):
    import pwasm.codegen

    monkeypatch.setattr(pwasm.codegen, "FORCE_CHAINS", force)
    inst = load(TYPED_CHAIN)
    assert [inst.exports.f(i) for i in range(4)] == [2010, 14, 0, 0]
    # one chain, so the br_table is a lookup
    assert ("s1 = (" in python_source(inst.functions[0])) == force


OUTER_TARGETS = """(module (func (export "f") (param i32) (result i32) (local i32)
  (block $out
    (loop $top
      (local.set 1 (i32.add (local.get 1) (i32.const 1)))
      (block $b1
        (block $b2
          (block $b3
            ;; 0 -> $b3, 1 -> $b2, 2 -> $out, 3 -> $top, default -> $b1
            (br_table $b3 $b2 $out $top $b1
              (i32.sub (local.get 0) (local.get 1))))
          (local.set 1 (i32.add (local.get 1) (i32.const 10))))
        (local.set 1 (i32.add (local.get 1) (i32.const 100))))
      (local.set 1 (i32.add (local.get 1) (i32.const 1000)))))
  (local.get 1)))"""


@pytest.mark.parametrize("force", [False, True])
def test_br_table_into_a_chain_and_out_of_it(monkeypatch, force):
    import pwasm.codegen

    monkeypatch.setattr(pwasm.codegen, "FORCE_CHAINS", force)
    inst = load(OUTER_TARGETS)
    results = [inst.exports.f(i) for i in range(-1, 7)]
    assert results == [1001, 1001, 1111, 1101, 1, 2, 1001, 1001]
    assert ("s3 = (" in python_source(inst.functions[0])) == force


INTERPRETER_LOOP = """(module (func (export "f") (param i32) (result i32) (local i32 i32)
  (loop $main
    (local.set 1 (i32.add (local.get 1) (i32.const 1)))
    (block $done
      (block $c2
        (block $c1
          (block $c0
            (br_table $c0 $c1 $c2 $done (i32.rem_u (local.get 1) (i32.const 4))))
          ;; case 0: go round again
          (local.set 2 (i32.add (local.get 2) (i32.const 1)))
          (br $main))
        ;; case 1: falls into case 2
        (local.set 2 (i32.add (local.get 2) (i32.const 10)))
        SPIN)
      ;; case 2
      (local.set 2 (i32.add (local.get 2) (i32.const 100)))
      (br_if $main (i32.lt_u (local.get 1) (local.get 0))))
    (local.set 2 (i32.add (local.get 2) (i32.const 1000)))
    (br_if $main (i32.lt_u (local.get 1) (local.get 0))))
  (local.get 2)))"""

# an inner loop in case 1 that sometimes branches straight back to $main
SPIN = """(loop $spin
          (local.set 2 (i32.add (local.get 2) (i32.const 10)))
          (br_if $main (i32.eq (i32.and (local.get 2) (i32.const 7)) (i32.const 3)))
          (br_if $spin (i32.lt_u (i32.and (local.get 2) (i32.const 0xF0)) (i32.const 0x40))))"""


@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("spin", [False, True])
def test_loops_around_chains(monkeypatch, force, spin):
    import pwasm.codegen

    text = INTERPRETER_LOOP.replace("SPIN", SPIN if spin else "")
    expected = [load(text, mode="interpret").exports.f(n) for n in range(12)]
    monkeypatch.setattr(pwasm.codegen, "FORCE_CHAINS", force)
    inst = load(text)
    assert [inst.exports.f(n) for n in range(12)] == expected
    if not spin:
        # the loop and the chain share one Python loop: going round the
        # loop is just another jump in the chain
        assert ("br_" in python_source(inst.functions[0])) == (not force)


VIEWS = """(module (memory 1)
  (func (export "u32") (param i32) (result i32) (i32.load (local.get 0)))
  (func (export "u32_4") (param i32) (result i32) (i32.load offset=4 (local.get 0)))
  (func (export "u32_5") (param i32) (result i32) (i32.load offset=5 (local.get 0)))
  (func (export "u16") (param i32) (result i32) (i32.load16_u offset=2 (local.get 0)))
  (func (export "u64") (param i32) (result i64) (i64.load offset=8 (local.get 0)))
  (func (export "u64_32") (param i32) (result i64) (i64.load32_u (local.get 0)))
  (func (export "f64") (param i32) (result f64) (f64.load offset=8 (local.get 0)))
  (func (export "const") (result i32) (i32.load (i32.const 8)))
  (func (export "grow") (param i32) (result i32) (memory.grow (local.get 0))))"""

# export -> (struct format of the result as pwasm returns it, offset)
VIEW_LOADS = {
    "u32": ("<i", 0),
    "u32_4": ("<i", 4),
    "u32_5": ("<i", 5),
    "u16": ("<H", 2),
    "u64": ("<q", 8),
    "u64_32": ("<I", 0),
    "f64": ("<d", 8),
}


def test_aligned_loads_use_memory_views():
    inst = load(VIEWS)
    memory = inst.memories[0]
    memory.data[:256] = bytes(range(256))
    for pages in (1, 2):
        size = len(memory.data)
        for name, (fmt, offset) in VIEW_LOADS.items():
            f = getattr(inst.exports, name)
            for addr in list(range(0, 20)) + list(range(size - 24, size + 2)):
                if addr + offset + struct.calcsize(fmt) <= size:
                    assert (
                        f(addr)
                        == struct.unpack_from(fmt, memory.data, addr + offset)[0]
                    )
                else:
                    with pytest.raises(TrapError):
                        f(addr)
        assert inst.exports.const() == struct.unpack_from("<i", memory.data, 8)[0]
        # after growing, the views see the new memory
        assert inst.exports.grow(1) == pages
        memory.data[-16:] = bytes(range(100, 116))
    for j, name in enumerate(["u32", "u32_4", "u32_5", "u16", "u64", "u64_32", "f64"]):
        source = python_source(inst.functions[j])
        view = {"u16": "_M16[", "u64": "_M64[", "f64": "_MD["}.get(name, "_M32[")
        assert view in source


ADD_CONSTANTS = [1, 5, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFE, 0xFFFFFFFF]
ADD_VALUES = [0, 1, 4, 5, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFA, 0xFFFFFFFF]


@pytest.mark.parametrize("t,bits", [("i32", 32), ("i64", 64)])
@pytest.mark.parametrize("operation", ["add", "sub"])
def test_adding_constants_wraps_without_masking(t, bits, operation):
    full = 1 << bits
    constants = [c if bits == 32 else c | (c << 32) for c in ADD_CONSTANTS]
    values = [v if bits == 32 else v | (v << 32) for v in ADD_VALUES]
    funcs = "".join(
        f'(func (export "f{j}") (param {t}) (result {t})'
        f" ({t}.{operation} (local.get 0) ({t}.const {c - full if c >= full // 2 else c})))"
        for j, c in enumerate(constants)
    )
    inst = load(f"(module {funcs})")
    for j, c in enumerate(constants):
        f = getattr(inst.exports, f"f{j}")
        for v in values:
            expected = (v + c if operation == "add" else v - c) % full
            signed = expected - full if expected >= full // 2 else expected
            assert f(v - full if v >= full // 2 else v) == signed
        source = python_source(inst.functions[j])
        assert "& 0xFFFF" not in source
        assert " if l0 " in source


OFFSET_SWITCH = """(module (func (export "f") (param i32) (result i32)
  (block $d
    (block $c2
      (block $c1
        (block $c0
          (br_table $c0 $c1 $c2 $d (INDEX)))
        (return (i32.const 100)))
      (return (i32.const 101)))
    (return (i32.const 102)))
  (i32.const 999)))"""


@pytest.mark.parametrize(
    "index",
    [
        "local.get 0",
        "i32.sub (local.get 0) (i32.const 3)",
        "i32.add (local.get 0) (i32.const -3)",
    ],
)
@pytest.mark.parametrize("force", [False, True])
def test_br_table_indexes(monkeypatch, index, force):
    import pwasm.codegen

    monkeypatch.setattr(pwasm.codegen, "FORCE_CHAINS", force)
    text = OFFSET_SWITCH.replace("INDEX", index)
    expected = [load(text, mode="interpret").exports.f(n) for n in range(-2, 9)]
    inst = load(text)
    assert [inst.exports.f(n) for n in range(-2, 9)] == expected
    source = python_source(inst.functions[0])
    # the table absorbs the offset, and the local is used directly
    assert "l0 - 3" not in source
    assert "t1 = l0" not in source


def test_deeply_nested_loops_use_a_state_machine():
    inst = load(nested_loops_module(25))
    assert inst.exports.f(10) == 10
    assert inst.exports.f(0) == 1
    assert "pc = " in python_source(inst.functions[0])


def test_out_of_bounds_access_in_compiled_code_traps():
    inst = load("""(module (memory 1)
      (func (export "f") (param i32) (result i32) (i32.load (local.get 0))))""")
    assert inst.exports.f(0) == 0
    with pytest.raises(TrapError, match="out of bounds memory access"):
        inst.exports.f(65534)


def test_index_error_from_a_host_function_propagates_unchanged():
    def host():
        raise IndexError("from the host")

    inst = load(
        """(module (import "env" "host" (func $host))
          (memory 1)
          (func (export "f") (call $host)))""",
        imports={"env": {"host": host}},
    )
    with pytest.raises(IndexError, match="from the host"):
        inst.exports.f()


@pytest.mark.parametrize("mode", ["interpret", "compile"])
def test_fuel_accounting_is_the_same_in_every_mode(mode):
    limits = Limits(fuel=10**6)
    inst = load(COUNT, mode=mode, limits=limits)
    assert inst.exports.count(100) == 100
    assert limits.fuel_consumed == 1 + 101


def test_compiled_code_is_shared_between_instances():
    module = decode_module(wat2wasm(ADD))
    a = instantiate(module, mode="compile")
    b = instantiate(module, mode="compile")
    a.exports.add(1, 2)
    b.exports.add(1, 2)
    assert a.functions[0].pyfunc is not b.functions[0].pyfunc
    assert a.functions[0].pyfunc.__code__ is b.functions[0].pyfunc.__code__


def test_callers_switch_to_compiled_callees():
    inst = load(
        """(module
          (func $sq (param i32) (result i32) (i32.mul (local.get 0) (local.get 0)))
          (func (export "f") (param i32) (result i32) (i32.add (call $sq (local.get 0)) (i32.const 1))))""",
        mode="auto",
    )
    for _ in range(5):
        assert inst.exports.f(3) == 10
    assert inst.functions[0].pyfunc is not None
    assert inst.functions[1].pyfunc is not None


def test_tracebacks_show_generated_source():
    inst = load("""(module (func (export "f") (param i32) (result i32)
      (i32.div_u (i32.const 1) (local.get 0))))""")
    with pytest.raises(TrapError) as info:
        inst.exports.f(0)
    assert "i32_div_u" in "".join(traceback.format_tb(info.value.__traceback__))


def test_auto_mode_uses_cached_code_on_first_call():
    module = decode_module(wat2wasm(ADD))
    first = instantiate(module, mode="auto")
    first.exports.add(1, 1)
    first.exports.add(1, 1)
    assert first.functions[0].pyfunc is not None
    # a new instance can use the code compiled for the first one straight away
    second = instantiate(module, mode="auto")
    second.exports.add(1, 1)
    assert second.functions[0].pyfunc is not None
