"""Imported functions: Python callables and functions from other instances."""

import pytest

from pwasm import LinkError, TrapError
from wat import load


def test_imported_function_shifts_function_index_space():
    inst = load(
        """(module (import "env" "seven" (func $seven (result i32)))
          (func $forty (result i32) (i32.const 40))
          (func (export "f") (result i32) (call $forty)))""",
        {"env": {"seven": lambda: 7}},
    )
    assert inst.exports.f() == 40


def test_call_imported_python_function():
    calls = []

    def add(a, b):
        calls.append((a, b))
        return a + b

    inst = load(
        """(module (import "env" "add" (func $add (param i32 i32) (result i32)))
          (func (export "f") (param i32) (result i32)
            (call $add (local.get 0) (i32.const -3))))""",
        {"env": {"add": add}},
    )
    assert inst.exports.f(10) == 7
    # host functions see signed Python ints
    assert calls == [(10, -3)]


def test_host_function_result_is_wrapped_to_i32():
    inst = load(
        """(module (import "env" "big" (func $big (result i32)))
          (func (export "f") (result i32) (call $big)))""",
        {"env": {"big": lambda: 2**32 + 5}},
    )
    assert inst.exports.f() == 5


def test_host_function_with_no_results():
    seen = []
    inst = load(
        """(module (import "env" "log" (func $log (param i64 f64)))
          (func (export "f") (call $log (i64.const -1) (f64.const 2.5))))""",
        {"env": {"log": lambda a, b: seen.append((a, b))}},
    )
    assert inst.exports.f() is None
    assert seen == [(-1, 2.5)]


def test_host_function_with_multiple_results():
    inst = load(
        """(module (import "env" "pair" (func $pair (result i32 i32)))
          (func (export "f") (result i32) (i32.sub (call $pair))))""",
        {"env": {"pair": lambda: (10, 3)}},
    )
    assert inst.exports.f() == 7


def test_exceptions_propagate_from_host_functions():
    class Boom(Exception):
        pass

    def boom():
        raise Boom("bang")

    inst = load(
        """(module (import "env" "boom" (func $boom))
          (func (export "f") (result i32) (call $boom) (i32.const 1)))""",
        {"env": {"boom": boom}},
    )
    with pytest.raises(Boom):
        inst.exports.f()


def test_exported_import_is_callable():
    inst = load(
        """(module (import "env" "twice" (func $twice (param i32) (result i32)))
          (export "twice" (func $twice)))""",
        {"env": {"twice": lambda x: x * 2}},
    )
    assert inst.exports.twice(21) == 42


def test_start_function_can_call_imports():
    seen = []
    load(
        """(module (import "env" "hello" (func $hello (param i32)))
          (func $start (call $hello (i32.const 99)))
          (start $start))""",
        {"env": {"hello": seen.append}},
    )
    assert seen == [99]


def test_unresolved_import_raises_link_error():
    with pytest.raises(LinkError, match="env.missing"):
        load("""(module (import "env" "missing" (func)))""", {"env": {}})


def test_missing_import_module_raises_link_error():
    with pytest.raises(LinkError, match="env.missing"):
        load("""(module (import "env" "missing" (func)))""")


def test_function_imported_from_another_instance():
    lib = load("""(module (func (export "sq") (param i32) (result i32)
      (i32.mul (local.get 0) (local.get 0))))""")
    app = load(
        """(module (import "lib" "sq" (func $sq (param i32) (result i32)))
          (func (export "f") (param i32) (result i32)
            (i32.add (call $sq (local.get 0)) (i32.const 1))))""",
        {"lib": {"sq": lib.exports.sq}},
    )
    assert app.exports.f(-4) == 17


def test_imported_wasm_function_with_wrong_type_raises_link_error():
    lib = load("""(module (func (export "g") (param i64)))""")
    with pytest.raises(LinkError, match="incompatible import type"):
        load(
            """(module (import "lib" "g" (func (param i32))))""",
            {"lib": {"g": lib.exports.g}},
        )


def test_trap_inside_host_callback_chain_is_trap_error():
    inst = load(
        """(module (import "env" "id" (func $id (param i32) (result i32)))
          (func (export "f") (result i32)
            (i32.div_s (i32.const 1) (call $id (i32.const 0)))))""",
        {"env": {"id": lambda x: x}},
    )
    with pytest.raises(TrapError, match="integer divide by zero"):
        inst.exports.f()
