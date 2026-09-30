"""Values have one consistent internal representation, and i32/i64 results
are returned to Python as signed integers."""

import pytest
from wat import load

pytestmark = pytest.mark.usefixtures("each_mode")


def test_div_u_result_equals_signed_constant():
    inst = load("""(module (func (export "f") (result i32)
      (i32.eq (i32.div_u (i32.const -2) (i32.const 1)) (i32.const -2))))""")
    assert inst.exports.f() == 1


def test_rem_u_result_equals_signed_constant():
    inst = load("""(module (func (export "f") (result i32)
      (i32.eq (i32.rem_u (i32.const -1) (i32.const -2)) (i32.const 1))))""")
    assert inst.exports.f() == 1


def test_unsigned_compare_of_negative_constant():
    inst = load("""(module (func (export "f") (param i32) (result i32)
      (i32.lt_u (local.get 0) (i32.const -1))))""")
    assert inst.exports.f(5) == 1
    assert inst.exports.f(-1) == 0


def test_i32_results_are_signed():
    inst = load("""(module (func (export "f") (result i32) (i32.const 0xFFFFFFFF)))""")
    assert inst.exports.f() == -1


def test_i32_arguments_accept_signed_and_unsigned():
    inst = load("""(module (func (export "f") (param i32) (result i32)
      (i32.add (local.get 0) (i32.const 1))))""")
    assert inst.exports.f(-1) == 0
    assert inst.exports.f(0xFFFFFFFF) == 0


def test_i32_results_are_plain_ints():
    inst = load("""(module (func (export "f") (param i32 i32) (result i32)
      (i32.lt_s (local.get 0) (local.get 1))))""")
    result = inst.exports.f(1, 2)
    assert result == 1 and type(result) is int


def test_shr_s_on_negative():
    inst = load("""(module (func (export "f") (param i32) (result i32)
      (i32.shr_s (local.get 0) (i32.const 1))))""")
    assert inst.exports.f(-8) == -4


def test_signed_comparisons():
    inst = load("""(module (func (export "f") (param i32 i32) (result i32)
      (i32.lt_s (local.get 0) (local.get 1))))""")
    assert inst.exports.f(-1, 0) == 1
    assert inst.exports.f(0, -1) == 0
