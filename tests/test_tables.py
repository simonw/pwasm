"""Tables, element segments, call_indirect and reference instructions."""

import pytest

from pwasm import TrapError
from wat import load

pytestmark = pytest.mark.usefixtures("each_mode")

DISPATCH = """(module
  (type $binop (func (param i32 i32) (result i32)))
  (type $unop (func (param i32) (result i32)))
  (table $t (export "table") 4 funcref)
  (elem (i32.const 0) $add $sub $neg)
  (func $add (type $binop) (i32.add (local.get 0) (local.get 1)))
  (func $sub (type $binop) (i32.sub (local.get 0) (local.get 1)))
  (func $neg (type $unop) (i32.sub (i32.const 0) (local.get 0)))
  (func (export "binop") (param i32 i32 i32) (result i32)
    (call_indirect (type $binop) (local.get 1) (local.get 2) (local.get 0))))"""


def test_call_indirect_dispatches_through_table():
    inst = load(DISPATCH)
    assert inst.exports.binop(0, 10, 3) == 13
    assert inst.exports.binop(1, 10, 3) == 7


def test_call_indirect_traps():
    inst = load(DISPATCH)
    with pytest.raises(TrapError, match="indirect call type mismatch"):
        inst.exports.binop(2, 1, 1)
    with pytest.raises(TrapError, match="uninitialized element"):
        inst.exports.binop(3, 1, 1)
    with pytest.raises(TrapError, match="undefined element"):
        inst.exports.binop(4, 1, 1)


def test_exported_table_from_python():
    inst = load(DISPATCH)
    table = inst.exports.table
    assert table.size == 4
    assert table.get(3) is None
    assert table.get(0) is inst.functions[0]


def test_table_instructions():
    inst = load(
        """(module
      (table $t 2 10 funcref)
      (func $f (result i32) (i32.const 42))
      (elem declare func $f)
      (func (export "size") (result i32) (table.size $t))
      (func (export "grow") (param i32) (result i32) (table.grow $t (ref.null func) (local.get 0)))
      (func (export "set_f") (param i32) (table.set $t (local.get 0) (ref.func $f)))
      (func (export "is_null") (param i32) (result i32) (ref.is_null (table.get $t (local.get 0))))
      (func (export "fill_f") (param i32 i32) (table.fill $t (local.get 0) (ref.func $f) (local.get 1)))
      (func (export "copy") (param i32 i32 i32) (table.copy $t $t (local.get 0) (local.get 1) (local.get 2)))
      (func (export "call") (param i32) (result i32) (call_indirect $t (result i32) (local.get 0))))"""
    )
    e = inst.exports
    assert e.size() == 2
    assert e.grow(3) == 2
    assert e.size() == 5
    assert e.grow(6) == -1
    assert e.is_null(1) == 1
    e.set_f(1)
    assert e.is_null(1) == 0
    assert e.call(1) == 42
    e.fill_f(2, 2)
    assert [e.is_null(i) for i in range(5)] == [1, 0, 0, 0, 1]
    e.copy(3, 0, 2)
    assert [e.is_null(i) for i in range(5)] == [1, 0, 0, 1, 0]
    with pytest.raises(TrapError, match="out of bounds table access"):
        e.is_null(5)
    with pytest.raises(TrapError, match="out of bounds table access"):
        e.fill_f(4, 2)


def test_passive_element_segments():
    inst = load(
        """(module
      (table $t 4 funcref)
      (func $a (result i32) (i32.const 1))
      (func $b (result i32) (i32.const 2))
      (elem $seg func $a $b)
      (func (export "init") (param i32) (table.init $t $seg (local.get 0) (i32.const 0) (i32.const 2)))
      (func (export "drop") (elem.drop $seg))
      (func (export "call") (param i32) (result i32) (call_indirect $t (result i32) (local.get 0))))"""
    )
    e = inst.exports
    e.init(1)
    assert e.call(1) == 1
    assert e.call(2) == 2
    e.drop()
    with pytest.raises(TrapError, match="out of bounds table access"):
        e.init(0)


def test_element_segment_with_expressions_and_ref_func_global():
    inst = load("""(module
      (table $t 3 funcref)
      (func $a (result i32) (i32.const 7))
      (global $g funcref (ref.func $a))
      (elem (table $t) (i32.const 1) funcref (ref.func $a) (ref.null func))
      (func (export "call") (param i32) (result i32) (call_indirect $t (result i32) (local.get 0)))
      (func (export "g_is_null") (result i32) (ref.is_null (global.get $g))))""")
    assert inst.exports.call(1) == 7
    with pytest.raises(TrapError, match="uninitialized element"):
        inst.exports.call(2)
    assert inst.exports.g_is_null() == 0


def test_out_of_bounds_active_segment_traps_at_instantiation():
    with pytest.raises(TrapError, match="out of bounds table access"):
        load("""(module (table 1 funcref) (func $f) (elem (i32.const 1) $f))""")


def test_typed_select_with_references():
    inst = load("""(module
      (func (export "f") (param i32) (result i32)
        (ref.is_null (select (result funcref) (ref.null func) (ref.func 0) (local.get 0))))
      (elem declare func 0))""")
    assert inst.exports.f(1) == 1
    assert inst.exports.f(0) == 0


def test_uninitialized_element_trap_names_the_index():
    inst = load(DISPATCH)
    with pytest.raises(TrapError, match="uninitialized element 3"):
        inst.exports.binop(3, 1, 1)
