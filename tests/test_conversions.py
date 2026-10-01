"""Numeric conversions, checked against wasmtime."""

import math
import struct

import pytest

from differential import Pair
from pwasm.numeric import F32NaN, f32_to_bits
from wat import load

pytestmark = pytest.mark.usefixtures("each_mode")

FLOATS = [
    0.0,
    -0.0,
    0.9,
    -0.9,
    -1.0,
    1.5,
    -1.5,
    2147483647.0,
    2147483648.0,
    -2147483648.0,
    -2147483649.0,
    -2147483904.0,
    4294967295.0,
    4294967296.0,
    9.2e18,
    9.223372036854775807e18,
    float("-9.223372036854775808e18"),  # PyPy folds the literal to an int
    -9.3e18,
    1.8446744073709552e19,
    1.9e19,
    math.inf,
    -math.inf,
    math.nan,
    1e-40,
]
INTS = [
    0,
    1,
    -1,
    0x7FFFFFFF,
    -0x80000000,
    0xFFFFFF,
    0x1000001,
    0x1000003,
    0x7FFFFFFFFFFFFFFF,
    -0x8000000000000000,
    0x8000008000000001 - 2**64,
    0x20000010000001,
    0x20000030000000,
    0xFFFFFFFF,
    0x100000001,
    123456789012345678,
]

TRUNCS = [
    f"{i}.trunc{sat}_{f}_{s}"
    for i in ("i32", "i64")
    for sat in ("", "_sat")
    for f in ("f32", "f64")
    for s in ("s", "u")
]
CONVERTS = [
    f"{f}.convert_{i}_{s}"
    for f in ("f32", "f64")
    for i in ("i32", "i64")
    for s in ("s", "u")
]


def conversion_module() -> str:
    funcs = []
    for op in TRUNCS:
        result, rest = op.split(".")
        source = rest.split("_")[-2]
        funcs.append(
            f'(func (export "{op}") (param {source}) (result {result}) '
            f"({op} (local.get 0)))"
        )
    for op in CONVERTS:
        result, rest = op.split(".")
        source = rest.split("_")[1]
        funcs.append(
            f'(func (export "{op}") (param {source}) (result {result}) '
            f"({op} (local.get 0)))"
        )
    funcs.append(
        '(func (export "demote") (param f64) (result f32) (f32.demote_f64 (local.get 0)))'
    )
    funcs.append(
        '(func (export "promote") (param f32) (result f64) (f64.promote_f32 (local.get 0)))'
    )
    return "(module " + "\n".join(funcs) + ")"


@pytest.fixture(scope="module")
def pair(each_mode):
    return Pair(conversion_module())


@pytest.mark.parametrize("op", TRUNCS)
def test_truncations(pair, op):
    for value in FLOATS:
        pair.check(op, value)


@pytest.mark.parametrize("op", CONVERTS)
def test_int_to_float(pair, op):
    for value in INTS:
        if "i32" in op:
            value = value & 0xFFFFFFFF
            value = value - 2**32 if value >= 2**31 else value
        pair.check(op, value)


def test_demote_and_promote(pair):
    for value in FLOATS + [3.4028235677973366e38, 1e39, 1.00000005960464477539]:
        pair.check("demote", value)
        pair.check("promote", value)


def test_trunc_traps_have_spec_messages():
    inst = load("""(module
      (func (export "f") (param f64) (result i32) (i32.trunc_f64_s (local.get 0))))""")
    from pwasm import TrapError

    with pytest.raises(TrapError, match="invalid conversion to integer"):
        inst.exports.f(math.nan)
    with pytest.raises(TrapError, match="integer overflow"):
        inst.exports.f(3e9)


def test_reinterpret_round_trips_nan_bits():
    inst = load(
        """(module
      (func (export "f32_bits") (param i32) (result i32)
        (i32.reinterpret_f32 (f32.reinterpret_i32 (local.get 0))))
      (func (export "f64_bits") (param i64) (result i64)
        (i64.reinterpret_f64 (f64.reinterpret_i64 (local.get 0))))
      (func (export "to_f32") (param i32) (result f32) (f32.reinterpret_i32 (local.get 0)))
      (func (export "from_f64") (param f64) (result i64) (i64.reinterpret_f64 (local.get 0))))"""
    )
    assert inst.exports.f32_bits(0x7FA00001) == 0x7FA00001
    assert inst.exports.f32_bits(-1) == -1
    assert inst.exports.f64_bits(0x7FF4000000000001) == 0x7FF4000000000001
    assert f32_to_bits(inst.exports.to_f32(0x7FA00001)) == 0x7FA00001
    assert inst.exports.to_f32(0x3FC00000) == 1.5
    assert (
        inst.exports.from_f64(-2.0) == struct.unpack("<q", struct.pack("<d", -2.0))[0]
    )


def test_demote_quiets_signalling_nans():
    # Python 3.14's struct keeps signalling NaNs when narrowing to f32;
    # WebAssembly requires demote to produce a quiet (arithmetic) NaN
    inst = load("""(module (func (export "f") (param i64) (result i32)
      (i32.reinterpret_f32 (f32.demote_f64 (f64.reinterpret_i64 (local.get 0))))))""")
    for bits in (0x7FF4000000000000, 0xFFF4000000000000 - 2**64):
        assert inst.exports.f(bits) & 0x7FC00000 == 0x7FC00000
