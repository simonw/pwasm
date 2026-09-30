"""Emscripten-style setjmp/longjmp: invoke_* trampolines implemented on the
host, unwinding with a Python exception."""

import pytest

from pwasm import TrapError, decode_module, instantiate
from pwasm.emscripten import EmscriptenSjLj
from wat import wat2wasm

# Mimics the ABI: invoke_vi calls a table entry; the callee may "longjmp"
# by calling _emscripten_throw_longjmp; the trampoline then restores the
# stack pointer and calls setThrew(1, 0).
GUEST = """(module
  (import "env" "invoke_vi" (func $invoke_vi (param i32 i32)))
  (import "env" "invoke_ii" (func $invoke_ii (param i32 i32) (result i32)))
  (import "env" "_emscripten_throw_longjmp" (func $throw))
  (global $__stack_pointer (export "__stack_pointer") (mut i32) (i32.const 1000))
  (global $threw (mut i32) (i32.const 0))
  (table (export "__indirect_function_table") 3 funcref)
  (elem (i32.const 0) $maybe_throw $nest $overflow_target)
  (func $setThrew (export "setThrew") (param i32 i32) (global.set $threw (local.get 0)))
  (func $maybe_throw (param i32)
    (global.set $__stack_pointer (i32.const 500))
    (if (local.get 0) (then (call $throw))))
  ;; recurses through the trampoline n times
  (func $nest (param i32) (result i32)
    (if (result i32) (local.get 0)
      (then (i32.add (i32.const 1) (call $invoke_ii (i32.const 1) (i32.sub (local.get 0) (i32.const 1)))))
      (else (i32.const 0))))
  (func $overflow_target (param i32))
  (func (export "overflow") (call $throw))
  (func (export "run") (param i32) (result i32)
    (global.set $threw (i32.const 0))
    (call $invoke_vi (i32.const 0) (local.get 0))
    (i32.add (i32.mul (global.get $threw) (i32.const 10000)) (global.get $__stack_pointer)))
  (func (export "nest") (param i32) (result i32)
    (global.set $threw (i32.const 0))
    (i32.add (call $invoke_ii (i32.const 1) (local.get 0)) (i32.mul (global.get $threw) (i32.const 10000)))))"""


def setup(**kwargs):
    module = decode_module(wat2wasm(GUEST))
    sjlj = EmscriptenSjLj(**kwargs)
    instance = instantiate(module, {"env": sjlj.imports(module)})
    sjlj.bind(instance)
    return sjlj, instance.exports


def test_invoke_without_longjmp_returns_normally():
    sjlj, e = setup()
    assert e.run(0) == 500  # stack pointer left as the callee set it
    assert (sjlj.invokes, sjlj.unwinds) == (1, 0)


def test_longjmp_unwinds_to_the_invoke_and_sets_threw():
    sjlj, e = setup()
    assert e.run(1) == 10000 + 1000  # threw, stack pointer restored
    assert (sjlj.invokes, sjlj.unwinds) == (1, 1)


def test_nested_invokes():
    sjlj, e = setup()
    assert e.nest(50) == 50
    assert sjlj.invokes == 51


def test_deep_nesting_calls_the_overflow_export():
    sjlj, e = setup(overflow_export="overflow", max_depth=20)
    # the 21st nested invoke calls "overflow" instead, which longjmps back
    # to that invoke: 20 levels each add 1, and setThrew was called
    assert e.nest(50) == 10000 + 20
    assert sjlj.overflows == 1


def test_imports_only_covers_emscripten_functions():
    module = decode_module(
        wat2wasm(
            """(module (import "env" "invoke_v" (func (param i32)))
          (import "env" "other" (func)) (import "wasi_snapshot_preview1" "fd_close" (func (param i32) (result i32))))"""
        )
    )
    assert set(EmscriptenSjLj().imports(module)) == {"invoke_v"}


def test_stack_save_and_restore_exports_are_used_when_there_is_no_stack_pointer():
    module = decode_module(wat2wasm("""(module
      (import "env" "invoke_v" (func $invoke_v (param i32)))
      (import "env" "_emscripten_throw_longjmp" (func $throw))
      (global $sp (mut i32) (i32.const 100))
      (global $threw (mut i32) (i32.const 0))
      (table (export "__indirect_function_table") 1 funcref)
      (elem (i32.const 0) $f)
      (func $f (global.set $sp (i32.const 5)) (call $throw))
      (func (export "stackSave") (result i32) (global.get $sp))
      (func (export "stackRestore") (param i32) (global.set $sp (local.get 0)))
      (func (export "setThrew") (param i32 i32) (global.set $threw (local.get 0)))
      (func (export "run") (result i32)
        (call $invoke_v (i32.const 0))
        (i32.add (global.get $sp) (i32.mul (global.get $threw) (i32.const 1000)))))"""))
    sjlj = EmscriptenSjLj()
    instance = instantiate(module, {"env": sjlj.imports(module)})
    sjlj.bind(instance)
    assert instance.exports.run() == 1100


def test_other_exceptions_propagate_through_invoke():
    module = decode_module(wat2wasm("""(module
      (import "env" "invoke_v" (func $invoke_v (param i32)))
      (global (export "__stack_pointer") (mut i32) (i32.const 0))
      (table (export "__indirect_function_table") 1 funcref)
      (elem (i32.const 0) $f)
      (func $f unreachable)
      (func (export "setThrew") (param i32 i32))
      (func (export "run") (call $invoke_v (i32.const 0))))"""))
    sjlj = EmscriptenSjLj()
    instance = instantiate(module, {"env": sjlj.imports(module)})
    sjlj.bind(instance)
    with pytest.raises(TrapError, match="unreachable"):
        instance.exports.run()
