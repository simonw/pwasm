"""Numeric helpers shared by the decoder, compiler and executor.

i32/i64 values are unsigned ints; signed views are computed on demand.
f32 values are Python floats rounded to single precision. A NaN f32 whose
exact bit pattern matters (from a constant, a load or a reinterpret) is kept
as an F32NaN, a float subclass that remembers its bits, because converting a
signalling NaN from single to double precision can quiet it.
"""

from __future__ import annotations

import struct

from .errors import TrapError

MASK_32 = 0xFFFFFFFF
MASK_64 = 0xFFFFFFFFFFFFFFFF
SIGN_32 = 0x80000000
SIGN_64 = 0x8000000000000000

_pack_f32 = struct.Struct("<f").pack
_unpack_f32 = struct.Struct("<f").unpack
_pack_u32 = struct.Struct("<I").pack
_unpack_u32 = struct.Struct("<I").unpack

INF = float("inf")


class F32NaN(float):
    """A NaN f32 value that carries its exact 32-bit pattern."""

    __slots__ = ("bits",)

    def __new__(cls, bits: int) -> "F32NaN":
        self = float.__new__(cls, "nan" if not bits & 0x80000000 else "-nan")
        self.bits = bits
        return self

    def __repr__(self) -> str:
        return f"F32NaN(0x{self.bits:08x})"


def f32_round(x: float) -> float:
    """Round a double to the nearest single-precision value."""
    try:
        return _unpack_f32(_pack_f32(x))[0]
    except OverflowError:
        # struct refuses values that round to infinity
        return INF if x > 0 else -INF


def f32_from_bits(bits: int) -> float:
    value = _unpack_f32(_pack_u32(bits))[0]
    if value != value:
        return F32NaN(bits)
    return value


def f32_to_bits(value: float) -> int:
    if type(value) is F32NaN:
        return value.bits
    return _unpack_u32(_pack_f32(value))[0]


def i32_signed(v: int) -> int:
    return (v ^ SIGN_32) - SIGN_32


def i64_signed(v: int) -> int:
    return (v ^ SIGN_64) - SIGN_64


# --- i32 operations that are too rare to inline in the dispatch loop ---


def i32_div_s(a: int, b: int) -> int:
    if b == 0:
        raise TrapError("integer divide by zero")
    sa = (a ^ SIGN_32) - SIGN_32
    sb = (b ^ SIGN_32) - SIGN_32
    if sa == -SIGN_32 and sb == -1:
        raise TrapError("integer overflow")
    q = abs(sa) // abs(sb)
    if (sa < 0) != (sb < 0):
        q = -q
    return q & MASK_32


def i32_div_u(a: int, b: int) -> int:
    if b == 0:
        raise TrapError("integer divide by zero")
    return a // b


def i32_rem_s(a: int, b: int) -> int:
    if b == 0:
        raise TrapError("integer divide by zero")
    sa = (a ^ SIGN_32) - SIGN_32
    sb = (b ^ SIGN_32) - SIGN_32
    r = abs(sa) % abs(sb)
    if sa < 0:
        r = -r
    return r & MASK_32


def i32_rem_u(a: int, b: int) -> int:
    if b == 0:
        raise TrapError("integer divide by zero")
    return a % b


def i32_shr_s(a: int, b: int) -> int:
    return (((a ^ SIGN_32) - SIGN_32) >> (b & 31)) & MASK_32


def i32_rotl(a: int, b: int) -> int:
    k = b & 31
    return ((a << k) | (a >> (32 - k))) & MASK_32


def i32_rotr(a: int, b: int) -> int:
    k = b & 31
    return ((a >> k) | (a << (32 - k))) & MASK_32


def i32_clz(a: int) -> int:
    return 32 - a.bit_length()


def i32_ctz(a: int) -> int:
    return (a & -a).bit_length() - 1 if a else 32


def i32_popcnt(a: int) -> int:
    return bin(a).count("1")


def i64_add(a: int, b: int) -> int:
    return (a + b) & MASK_64


def f32_add(a: float, b: float) -> float:
    return f32_round(a + b)
