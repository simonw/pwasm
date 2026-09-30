"""Compiling WebAssembly functions to Python source code."""

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
