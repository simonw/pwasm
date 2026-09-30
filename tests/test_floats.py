"""Floating point operations, checked against wasmtime over edge cases."""

import itertools
import math

import pytest

from differential import Pair

pytestmark = pytest.mark.usefixtures("each_mode")

INF = math.inf
NAN = math.nan
VALUES = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.5,
    -0.5,
    1.5,
    2.5,
    -2.5,
    3.5,
    -0.4,
    0.6,
    INF,
    -INF,
    NAN,
    1.1,
    -7.25,
    math.pi,
    1e-45,
    1.401298464324817e-45,
    3.4028234663852886e38,
    1e38,
    2.0**24 + 1,
    2.0**52 + 0.5,
    1e300,
    5e-324,
    123456.789,
]

BINARY = [
    "add",
    "sub",
    "mul",
    "div",
    "min",
    "max",
    "copysign",
    "eq",
    "ne",
    "lt",
    "gt",
    "le",
    "ge",
]
UNARY = ["abs", "neg", "sqrt", "ceil", "floor", "trunc", "nearest"]
COMPARISONS = {"eq", "ne", "lt", "gt", "le", "ge"}


def module_for(t: str) -> str:
    funcs = []
    for op in BINARY:
        result = "i32" if op in COMPARISONS else t
        funcs.append(
            f'(func (export "{op}") (param {t} {t}) (result {result}) '
            f"({t}.{op} (local.get 0) (local.get 1)))"
        )
    for op in UNARY:
        funcs.append(
            f'(func (export "{op}") (param {t}) (result {t}) '
            f"({t}.{op} (local.get 0)))"
        )
    return "(module " + "\n".join(funcs) + ")"


@pytest.fixture(scope="module", params=["f32", "f64"])
def floats(request, each_mode):
    return Pair(module_for(request.param))


@pytest.mark.parametrize("op", BINARY)
def test_binary(floats, op):
    for a, b in itertools.product(VALUES, repeat=2):
        floats.check(op, a, b)


@pytest.mark.parametrize("op", UNARY)
def test_unary(floats, op):
    for a in VALUES:
        floats.check(op, a)


def test_f32_nan_argument_and_result_keep_their_bits():
    from pwasm.numeric import F32NaN, f32_to_bits
    from wat import load

    inst = load('(module (func (export "f") (param f32) (result f32) (local.get 0)))')
    result = inst.exports.f(F32NaN(0x7FA00001))
    assert f32_to_bits(result) == 0x7FA00001


def test_f32_arguments_are_rounded_to_single_precision():
    from wat import load

    inst = load('(module (func (export "f") (param f32) (result f32) (local.get 0)))')
    assert inst.exports.f(1.1) == 1.100000023841858


def test_sqrt_quiets_signalling_nans():
    # PyPy's math.sqrt returns a signalling NaN unchanged; WebAssembly
    # requires a quiet (arithmetic) NaN
    from wat import load

    inst = load("""(module (func (export "f") (param i64) (result i64)
      (i64.reinterpret_f64 (f64.sqrt (f64.reinterpret_i64 (local.get 0))))))""")
    result = inst.exports.f(0x7FF4000000000000)
    assert result & 0x7FF8000000000000 == 0x7FF8000000000000
