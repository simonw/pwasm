"""Sandbox: load a module with WASI-lite, limits and memory helpers."""

import pytest

from pwasm import LinkError, OutOfFuel, Timeout
from pwasm.sandbox import Sandbox
from wat import wat2wasm

# A reactor with a bump allocator, a WASI write and some busy loops
REACTOR = wat2wasm("""(module
  (import "wasi_snapshot_preview1" "fd_write" (func $fd_write (param i32 i32 i32 i32) (result i32)))
  (import "env" "add" (func $add (param i32 i32) (result i32)))
  (memory (export "memory") 1)
  (global $heap (mut i32) (i32.const 4096))
  (global $initialized (mut i32) (i32.const 0))
  (table (export "__indirect_function_table") 1 funcref)
  (elem (i32.const 0) $triple)
  (func $triple (param i32) (result i32) (i32.mul (local.get 0) (i32.const 3)))
  (func (export "_initialize") (global.set $initialized (i32.const 1)))
  (func (export "initialized") (result i32) (global.get $initialized))
  (func (export "malloc") (param i32) (result i32)
    (global.get $heap)
    (global.set $heap (i32.add (global.get $heap) (local.get 0))))
  (func (export "free") (param i32))
  ;; write(ptr, len) -> writes the bytes to stdout
  (func (export "write") (param i32 i32) (result i32)
    (i32.store (i32.const 0) (local.get 0))
    (i32.store (i32.const 4) (local.get 1))
    (call $fd_write (i32.const 1) (i32.const 0) (i32.const 1) (i32.const 8)))
  (func (export "add") (param i32 i32) (result i32) (call $add (local.get 0) (local.get 1)))
  (func (export "spin") (loop $l (br $l)))
  (func (export "grow") (param i32) (result i32) (memory.grow (local.get 0))))""")


def make(**kwargs):
    kwargs.setdefault("imports", {"env": {"add": lambda a, b: a + b}})
    return Sandbox(REACTOR, **kwargs)


def test_initialize_is_called_and_imports_resolved():
    sb = make()
    assert sb.call("initialized") == 1
    assert sb.call("add", 2, 3) == 5


def test_resolver_function_for_imports():
    sb = Sandbox(
        REACTOR,
        imports=lambda module, name, imp: (
            (lambda a, b: a * b) if name == "add" else None
        ),
    )
    assert sb.call("add", 4, 5) == 20


def test_unresolved_imports_are_listed():
    with pytest.raises(LinkError, match="env.add"):
        Sandbox(REACTOR, imports={})


def test_wasi_stdout_and_memory_helpers():
    sb = make()
    ptr = sb.alloc(b"hello\n")
    assert sb.read(ptr, 6) == b"hello\n"
    assert sb.call("write", ptr, 6) == 0
    assert sb.stdout == "hello\n"
    ptr2 = sb.alloc(b"abc", nul=True)
    assert sb.read_cstr(ptr2) == b"abc"
    sb.write_u32(100, 0xDEADBEEF)
    assert sb.read_u32(100) == 0xDEADBEEF
    sb.free(ptr)


def test_call_indirect():
    sb = make()
    assert sb.call_indirect(0, 14) == 42


def test_timeout():
    sb = make(timeout=0.2)
    with pytest.raises(Timeout):
        sb.call("spin")
    # per-call override
    with pytest.raises(Timeout):
        sb.call("spin", timeout=0.05)


def test_fuel():
    sb = make(fuel=5000)
    with pytest.raises(OutOfFuel):
        sb.call("spin")
    assert sb.fuel_consumed == 5000
    sb.fuel = 100
    assert sb.call("add", 1, 1) == 2
    assert sb.fuel < 100


def test_max_memory():
    sb = make(max_memory=2 * 65536)
    assert sb.call("grow", 1) == 1
    assert sb.call("grow", 1) == -1
    assert sb.memory_size == 2 * 65536


def test_accepts_a_path(tmp_path):
    path = tmp_path / "reactor.wasm"
    path.write_bytes(REACTOR)
    sb = Sandbox(str(path), imports={"env": {"add": lambda a, b: 0}})
    assert sb.call("initialized") == 1
