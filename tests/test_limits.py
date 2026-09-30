"""Resource limits: fuel, deadlines and memory caps."""

import time

import pytest

from pwasm import LinkError, OutOfFuel, Timeout, TrapError, decode_module, instantiate
from pwasm.runtime import Limits
from wat import wat2wasm

pytestmark = pytest.mark.usefixtures("each_mode")

LOOPS = """(module
  (memory 1)
  (func (export "spin") (loop $l (br $l)))
  (func (export "count") (param i32) (result i32) (local i32)
    (block $done
      (loop $l
        (br_if $done (i32.ge_u (local.get 1) (local.get 0)))
        (local.set 1 (i32.add (local.get 1) (i32.const 1)))
        (br $l)))
    (local.get 1))
  (func $recurse (export "recurse") (param i32) (result i32)
    (if (result i32) (local.get 0)
      (then (call $recurse (i32.sub (local.get 0) (i32.const 1))))
      (else (i32.const 0))))
  (func (export "grow") (param i32) (result i32) (memory.grow (local.get 0))))"""


def load(limits=None):
    return instantiate(decode_module(wat2wasm(LOOPS)), limits=limits)


def test_out_of_fuel_stops_an_infinite_loop():
    limits = Limits(fuel=10_000)
    inst = load(limits)
    with pytest.raises(OutOfFuel):
        inst.exports.spin()
    assert limits.fuel_consumed == 10_000


def test_out_of_fuel_is_a_trap():
    assert issubclass(OutOfFuel, TrapError)
    assert issubclass(Timeout, TrapError)


def test_fuel_is_charged_per_loop_iteration_and_call_deterministically():
    limits = Limits(fuel=10**9)
    inst = load(limits)
    assert inst.exports.count(100) == 100
    first = limits.fuel_consumed
    inst.exports.count(100)
    assert limits.fuel_consumed == 2 * first
    # one unit for the call, one per loop header visit
    assert first == 1 + 101
    before = limits.fuel_consumed
    inst.exports.recurse(10)
    assert limits.fuel_consumed - before == 11


def test_fuel_can_be_topped_up():
    limits = Limits(fuel=50)
    inst = load(limits)
    with pytest.raises(OutOfFuel):
        inst.exports.count(100)
    limits.set_fuel(1000)
    assert inst.exports.count(100) == 100


def test_deadline_stops_an_infinite_loop():
    limits = Limits()
    inst = load(limits)
    limits.set_deadline(time.monotonic() + 0.2)
    start = time.monotonic()
    with pytest.raises(Timeout):
        inst.exports.spin()
    assert time.monotonic() - start < 2
    limits.set_deadline(None)
    assert inst.exports.count(10) == 10


def test_unlimited_instances_compile_without_ticks():
    module = decode_module(wat2wasm(LOOPS))
    unlimited = instantiate(module, mode="interpret")
    unlimited.exports.count(3)
    limited = instantiate(module, limits=Limits(fuel=100), mode="interpret")
    limited.exports.count(3)
    assert len(limited.functions[1].code.ops) > len(unlimited.functions[1].code.ops)


def test_max_memory_caps_memory_grow():
    inst = load(Limits(max_memory=3 * 65536))
    assert inst.exports.grow(2) == 1
    assert inst.exports.grow(1) == -1


def test_initial_memory_over_max_memory_is_a_link_error():
    with pytest.raises(LinkError, match="max_memory"):
        instantiate(
            decode_module(wat2wasm("(module (memory 2))")),
            limits=Limits(max_memory=65536),
        )
