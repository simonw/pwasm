"""Integer operations, checked against wasmtime over edge-case values."""

import itertools

import pytest

from differential import Pair

I32_VALUES = [
    0,
    1,
    -1,
    2,
    7,
    31,
    32,
    33,
    0x7FFFFFFF,
    -0x80000000,
    -0x7FFFFFFF,
    0xFFFF,
    0x10000,
    0x80,
    0xFF,
    123456789,
    -123456789,
]
I64_VALUES = [
    0,
    1,
    -1,
    2,
    7,
    63,
    64,
    65,
    0x7FFFFFFF,
    0x80000000,
    0xFFFFFFFF,
    0x100000000,
    0x7FFFFFFFFFFFFFFF,
    -0x8000000000000000,
    -0x7FFFFFFFFFFFFFFF,
    0x123456789ABCDEF0,
    -0x123456789,
    0x80,
    0x8000,
]

BINARY = [
    "add",
    "sub",
    "mul",
    "div_s",
    "div_u",
    "rem_s",
    "rem_u",
    "and",
    "or",
    "xor",
    "shl",
    "shr_s",
    "shr_u",
    "rotl",
    "rotr",
    "eq",
    "ne",
    "lt_s",
    "lt_u",
    "gt_s",
    "gt_u",
    "le_s",
    "le_u",
    "ge_s",
    "ge_u",
]
UNARY = ["clz", "ctz", "popcnt", "eqz", "extend8_s", "extend16_s"]
COMPARISONS = {
    "eq",
    "ne",
    "lt_s",
    "lt_u",
    "gt_s",
    "gt_u",
    "le_s",
    "le_u",
    "ge_s",
    "ge_u",
    "eqz",
}


def module_for(t: str) -> str:
    funcs = []
    for op in BINARY:
        result = "i32" if op in COMPARISONS else t
        funcs.append(
            f'(func (export "{op}") (param {t} {t}) (result {result}) '
            f"({t}.{op} (local.get 0) (local.get 1)))"
        )
    unary = UNARY + (["extend32_s"] if t == "i64" else [])
    for op in unary:
        result = "i32" if op in COMPARISONS else t
        funcs.append(
            f'(func (export "{op}") (param {t}) (result {result}) '
            f"({t}.{op} (local.get 0)))"
        )
    return "(module " + "\n".join(funcs) + ")"


@pytest.fixture(scope="module")
def i32():
    return Pair(module_for("i32"))


@pytest.fixture(scope="module")
def i64():
    return Pair(module_for("i64"))


@pytest.mark.parametrize("op", BINARY)
def test_i32_binary(i32, op):
    for a, b in itertools.product(I32_VALUES, repeat=2):
        i32.check(op, a, b)


@pytest.mark.parametrize("op", UNARY)
def test_i32_unary(i32, op):
    for a in I32_VALUES:
        i32.check(op, a)


@pytest.mark.parametrize("op", BINARY)
def test_i64_binary(i64, op):
    for a, b in itertools.product(I64_VALUES, repeat=2):
        i64.check(op, a, b)


@pytest.mark.parametrize("op", UNARY + ["extend32_s"])
def test_i64_unary(i64, op):
    for a in I64_VALUES:
        i64.check(op, a)


@pytest.fixture(scope="module")
def conversions():
    return Pair(
        """(module
      (func (export "wrap") (param i64) (result i32) (i32.wrap_i64 (local.get 0)))
      (func (export "extend_s") (param i32) (result i64) (i64.extend_i32_s (local.get 0)))
      (func (export "extend_u") (param i32) (result i64) (i64.extend_i32_u (local.get 0))))"""
    )


def test_wrap(conversions):
    for a in I64_VALUES:
        conversions.check("wrap", a)


@pytest.mark.parametrize("op", ["extend_s", "extend_u"])
def test_extend(conversions, op):
    for a in I32_VALUES:
        conversions.check(op, a)
