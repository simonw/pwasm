"""Importing memories, globals and tables, and host functions that call
back into WebAssembly."""

import pytest

from pwasm import LinkError, TrapError
from pwasm.runtime import GlobalInstance, MemoryInstance, TableInstance
from pwasm.types import GlobalType
from wat import load

pytestmark = pytest.mark.usefixtures("each_mode")


def test_import_memory_created_in_python():
    memory = MemoryInstance(1, 2)
    inst = load(
        """(module (import "env" "mem" (memory 1))
          (func (export "store") (param i32 i32) (i32.store (local.get 0) (local.get 1)))
          (func (export "grow") (result i32) (memory.grow (i32.const 1))))""",
        {"env": {"mem": memory}},
    )
    inst.exports.store(8, 0x01020304)
    assert memory.read(8, 4) == b"\x04\x03\x02\x01"
    assert inst.exports.grow() == 1
    assert memory.size == 2
    assert inst.exports.grow() == -1


def test_memory_shared_between_instances():
    lib = load("""(module (memory (export "mem") 1)
      (func (export "load") (param i32) (result i32) (i32.load (local.get 0))))""")
    app = load(
        """(module (import "lib" "mem" (memory 1))
          (data (i32.const 16) "\\2a\\00\\00\\00"))""",
        {"lib": {"mem": lib.exports.mem}},
    )
    assert app is not None
    assert lib.exports.load(16) == 42


def test_incompatible_memory_limits_raise_link_error():
    with pytest.raises(LinkError, match="incompatible import type"):
        load(
            """(module (import "env" "mem" (memory 2)))""",
            {"env": {"mem": MemoryInstance(1)}},
        )
    with pytest.raises(LinkError, match="incompatible import type"):
        load(
            """(module (import "env" "mem" (memory 1 2)))""",
            {"env": {"mem": MemoryInstance(1, 3)}},
        )


def test_import_globals():
    counter = GlobalInstance(GlobalType("i32", True), 5)
    inst = load(
        """(module
          (import "env" "base" (global $base i64))
          (import "env" "counter" (global $counter (mut i32)))
          (global $derived i64 (global.get $base))
          (func (export "derived") (result i64) (global.get $derived))
          (func (export "bump") (global.set $counter (i32.add (global.get $counter) (i32.const 1)))))""",
        {"env": {"base": -7, "counter": counter}},
    )
    assert inst.exports.derived() == -7
    inst.exports.bump()
    assert counter.value == 6


def test_global_type_mismatch_raises_link_error():
    with pytest.raises(LinkError, match="incompatible import type"):
        load(
            """(module (import "env" "g" (global (mut i32))))""",
            {"env": {"g": GlobalInstance(GlobalType("i32", False), 1)}},
        )
    with pytest.raises(LinkError, match="incompatible import type"):
        load(
            """(module (import "env" "g" (global i64)))""",
            {"env": {"g": GlobalInstance(GlobalType("i32", False), 1)}},
        )


def test_exported_global_from_python():
    inst = load("""(module (global (export "g") (mut i32) (i32.const -1))
      (global (export "c") f64 (f64.const 2.5)))""")
    assert inst.exports.g.value == -1
    inst.exports.g.value = 0xFFFFFFFE
    assert inst.exports.g.value == -2
    assert inst.exports.c.value == 2.5
    with pytest.raises(AttributeError):
        inst.exports.c.value = 1.0


def test_table_shared_between_instances():
    lib = load(
        """(module (table (export "tab") 2 funcref)
      (func (export "call") (param i32) (result i32) (call_indirect (result i32) (local.get 0))))"""
    )
    load(
        """(module (import "lib" "tab" (table 2 funcref))
          (func $seven (result i32) (i32.const 7))
          (elem (i32.const 1) $seven))""",
        {"lib": {"tab": lib.exports.tab}},
    )
    assert lib.exports.call(1) == 7


def test_import_table_created_in_python():
    table = TableInstance("funcref", 1)
    with pytest.raises(LinkError, match="incompatible import type"):
        load("""(module (import "env" "t" (table 2 funcref)))""", {"env": {"t": table}})
    with pytest.raises(LinkError, match="incompatible import type"):
        load(
            """(module (import "env" "t" (table 1 externref)))""", {"env": {"t": table}}
        )


def test_wrong_kind_of_import_raises_link_error():
    with pytest.raises(LinkError):
        load("""(module (import "env" "m" (memory 1)))""", {"env": {"m": lambda: 1}})


def test_host_function_can_call_back_into_exports():
    exports = {}

    def host(n):
        # re-enters the module while a call is in progress
        return exports["double"](n) + 1

    inst = load(
        """(module (import "env" "host" (func $host (param i32) (result i32)))
          (func (export "double") (param i32) (result i32) (i32.mul (local.get 0) (i32.const 2)))
          (func (export "f") (param i32) (result i32) (call $host (local.get 0))))""",
        {"env": {"host": lambda n: host(n)}},
    )
    exports["double"] = inst.exports.double
    assert inst.exports.f(20) == 41


def test_host_function_can_call_table_entries_and_read_memory():
    state = {}

    def call_via_table(index, value):
        func = state["inst"].exports.table.get(index)
        return func(value)

    inst = load(
        """(module
          (import "env" "call_via_table" (func $cvt (param i32 i32) (result i32)))
          (memory (export "memory") 1)
          (table (export "table") 1 funcref)
          (func $square (param i32) (result i32)
            (i32.store (i32.const 0) (local.get 0))
            (i32.mul (local.get 0) (local.get 0)))
          (elem (i32.const 0) $square)
          (func (export "f") (param i32) (result i32) (call $cvt (i32.const 0) (local.get 0))))""",
        {"env": {"call_via_table": call_via_table}},
    )
    state["inst"] = inst
    assert inst.exports.f(-9) == 81
    assert inst.exports.memory.read(0, 4) == (-9).to_bytes(4, "little", signed=True)


def test_exception_unwinds_nested_host_and_wasm_frames():
    class Unwind(Exception):
        pass

    def thrower():
        raise Unwind()

    def trampoline(index):
        # like emscripten's invoke_*: call into wasm, catch the unwind
        try:
            return inst.exports.table.get(index)()
        except Unwind:
            return -1

    inst = load(
        """(module
          (import "env" "throw" (func $throw))
          (import "env" "invoke" (func $invoke (param i32) (result i32)))
          (table (export "table") 2 funcref)
          (func $deep (result i32) (call $throw) (i32.const 1))
          (func $ok (result i32) (i32.const 5))
          (elem (i32.const 0) $deep $ok)
          (func (export "f") (result i32)
            (i32.add (call $invoke (i32.const 0)) (call $invoke (i32.const 1)))))""",
        {"env": {"throw": thrower, "invoke": trampoline}},
    )
    assert inst.exports.f() == 4


def test_deep_recursion_traps_as_call_stack_exhausted():
    inst = load("""(module (func $f (export "f") (param i32) (result i32)
      (i32.add (call $f (local.get 0)) (i32.const 1))))""")
    with pytest.raises(TrapError, match="call stack exhausted"):
        inst.exports.f(0)
