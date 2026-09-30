"""Branches must unwind the operand stack to the target label's height."""

import pytest

from wat import load


def test_br_discards_extra_values():
    inst = load("""(module (func (export "f") (result i32)
      (i32.const 10)
      (block (result i32) (i32.const 1) (i32.const 2) (br 0))
      (i32.add)))""")
    assert inst.exports.f() == 12


def test_br_if_discards_extra_values():
    inst = load("""(module (func (export "f") (param i32) (result i32)
      (i32.const 10)
      (block (result i32)
        (i32.const 1) (i32.const 2) (i32.const 3)
        (br_if 0 (local.get 0))
        (drop) (drop))
      (i32.add)))""")
    assert inst.exports.f(1) == 13
    assert inst.exports.f(0) == 11


def test_br_table_discards_extra_values():
    inst = load("""(module (func (export "f") (param i32) (result i32)
      (i32.const 100)
      (block $outer (result i32)
        (block $inner (result i32)
          (i32.const 7) (i32.const 1) (i32.const 2)
          (br_table $inner $outer (local.get 0)))
        (i32.const 1000) (i32.add))
      (i32.add)))""")
    assert inst.exports.f(0) == 1102
    assert inst.exports.f(1) == 102
    assert inst.exports.f(5) == 102


def test_br_out_of_nested_loops():
    inst = load("""(module (func (export "f") (param i32) (result i32) (local i32)
      (block $done
        (loop $outer
          (loop $inner
            (local.set 1 (i32.add (local.get 1) (i32.const 1)))
            (br_if $done (i32.ge_u (local.get 1) (local.get 0)))
            (br_if $inner (i32.and (local.get 1) (i32.const 1)))
            (br $outer))))
      (local.get 1)))""")
    assert inst.exports.f(10) == 10


def test_return_from_nested_blocks_discards_values():
    inst = load("""(module (func (export "f") (result i32)
      (i32.const 1)
      (block (i32.const 2) (block (i32.const 3) (i32.const 42) (return)) (drop))
      (drop) (i32.const 0)))""")
    assert inst.exports.f() == 42


def test_multi_value_block_and_function_results():
    inst = load("""(module (func (export "f") (result i32 i32)
      (block (result i32 i32) (i32.const 1) (i32.const 2))))""")
    assert inst.exports.f() == (1, 2)


def test_block_with_params():
    inst = load("""(module (func (export "f") (result i32)
      (i32.const 5)
      (block (param i32) (result i32) (i32.const 3) (i32.sub))))""")
    assert inst.exports.f() == 2


def test_loop_with_params_branches_carry_values():
    # sum 1..n using a loop whose parameter is the running total
    inst = load("""(module (func (export "f") (param i32) (result i32)
      (i32.const 0)
      (loop $l (param i32) (result i32)
        (i32.add (local.get 0))
        (local.set 0 (i32.sub (local.get 0) (i32.const 1)))
        (br_if $l (local.get 0)))))""")
    assert inst.exports.f(4) == 10


def test_if_else_with_results():
    inst = load("""(module (func (export "f") (param i32) (result i32)
      (if (result i32) (local.get 0) (then (i32.const 7)) (else (i32.const 9)))))""")
    assert inst.exports.f(1) == 7
    assert inst.exports.f(0) == 9


def test_unreachable_code_after_br_is_ignored():
    inst = load("""(module (func (export "f") (result i32)
      (block (result i32) (i32.const 3) (br 0) (i32.add) (unreachable))))""")
    assert inst.exports.f() == 3
