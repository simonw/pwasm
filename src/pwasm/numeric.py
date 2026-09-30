"""Numeric helpers shared by the decoder, compiler and executor.

i32/i64 values are unsigned ints; signed views are computed on demand.
f32 values are Python floats rounded to single precision. A NaN f32 whose
exact bit pattern matters (from a constant, a load or a reinterpret) is kept
as an F32NaN, a float subclass that remembers its bits, because converting a
signalling NaN from single to double precision can quiet it.
"""

from __future__ import annotations

import math
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


def i32_extend8_s(a: int) -> int:
    return (((a & 0xFF) ^ 0x80) - 0x80) & MASK_32


def i32_extend16_s(a: int) -> int:
    return (((a & 0xFFFF) ^ 0x8000) - 0x8000) & MASK_32


def i32_wrap_i64(a: int) -> int:
    return a & MASK_32


def i64_extend_i32_s(a: int) -> int:
    return ((a ^ SIGN_32) - SIGN_32) & MASK_64


# --- i64 operations ---


def i64_add(a: int, b: int) -> int:
    return (a + b) & MASK_64


def i64_sub(a: int, b: int) -> int:
    return (a - b) & MASK_64


def i64_mul(a: int, b: int) -> int:
    return (a * b) & MASK_64


def i64_div_s(a: int, b: int) -> int:
    if b == 0:
        raise TrapError("integer divide by zero")
    sa = (a ^ SIGN_64) - SIGN_64
    sb = (b ^ SIGN_64) - SIGN_64
    if sa == -SIGN_64 and sb == -1:
        raise TrapError("integer overflow")
    q = abs(sa) // abs(sb)
    if (sa < 0) != (sb < 0):
        q = -q
    return q & MASK_64


def i64_div_u(a: int, b: int) -> int:
    if b == 0:
        raise TrapError("integer divide by zero")
    return a // b


def i64_rem_s(a: int, b: int) -> int:
    if b == 0:
        raise TrapError("integer divide by zero")
    sa = (a ^ SIGN_64) - SIGN_64
    sb = (b ^ SIGN_64) - SIGN_64
    r = abs(sa) % abs(sb)
    if sa < 0:
        r = -r
    return r & MASK_64


def i64_rem_u(a: int, b: int) -> int:
    if b == 0:
        raise TrapError("integer divide by zero")
    return a % b


def i64_shl(a: int, b: int) -> int:
    return (a << (b & 63)) & MASK_64


def i64_shr_s(a: int, b: int) -> int:
    return (((a ^ SIGN_64) - SIGN_64) >> (b & 63)) & MASK_64


def i64_shr_u(a: int, b: int) -> int:
    return a >> (b & 63)


def i64_rotl(a: int, b: int) -> int:
    k = b & 63
    return ((a << k) | (a >> (64 - k))) & MASK_64


def i64_rotr(a: int, b: int) -> int:
    k = b & 63
    return ((a >> k) | (a << (64 - k))) & MASK_64


def i64_lt_s(a: int, b: int) -> int:
    return 1 if (a ^ SIGN_64) < (b ^ SIGN_64) else 0


def i64_gt_s(a: int, b: int) -> int:
    return 1 if (a ^ SIGN_64) > (b ^ SIGN_64) else 0


def i64_le_s(a: int, b: int) -> int:
    return 1 if (a ^ SIGN_64) <= (b ^ SIGN_64) else 0


def i64_ge_s(a: int, b: int) -> int:
    return 1 if (a ^ SIGN_64) >= (b ^ SIGN_64) else 0


def i64_clz(a: int) -> int:
    return 64 - a.bit_length()


def i64_ctz(a: int) -> int:
    return (a & -a).bit_length() - 1 if a else 64


def i64_eqz(a: int) -> int:
    return 1 if a == 0 else 0


def i64_extend8_s(a: int) -> int:
    return (((a & 0xFF) ^ 0x80) - 0x80) & MASK_64


def i64_extend16_s(a: int) -> int:
    return (((a & 0xFFFF) ^ 0x8000) - 0x8000) & MASK_64


def i64_extend32_s(a: int) -> int:
    return (((a & MASK_32) ^ SIGN_32) - SIGN_32) & MASK_64


# --- floating point ---
#
# Python float arithmetic is IEEE 754 double precision, except that division
# by zero raises and math functions raise instead of returning NaN. f32
# results are computed in double precision and rounded once, which gives
# correctly rounded results for +, -, *, / and sqrt.

NAN = float("nan")


def f32_add(a: float, b: float) -> float:
    return f32_round(a + b)


def f32_sub(a: float, b: float) -> float:
    return f32_round(a - b)


def f32_mul(a: float, b: float) -> float:
    return f32_round(a * b)


def f32_div(a: float, b: float) -> float:
    return f32_round(f64_div(a, b))


def f32_sqrt(a: float) -> float:
    return f32_round(f64_sqrt(a))


def f64_add(a: float, b: float) -> float:
    return a + b


def f64_sub(a: float, b: float) -> float:
    return a - b


def f64_mul(a: float, b: float) -> float:
    return a * b


def f64_div(a: float, b: float) -> float:
    try:
        return a / b
    except ZeroDivisionError:
        if a != a:
            return a + 0.0  # quiet a signalling NaN
        if a == 0.0:
            return NAN
        return (
            INF if (math.copysign(1.0, a) > 0) == (math.copysign(1.0, b) > 0) else -INF
        )


def f64_sqrt(a: float) -> float:
    try:
        return math.sqrt(a)
    except ValueError:
        return NAN


def f64_min(a: float, b: float) -> float:
    if a != a or b != b:
        return a + b  # a (quiet) NaN
    if a == b:
        # min(-0.0, 0.0) is -0.0
        return a if math.copysign(1.0, a) < 0 else b
    return a if a < b else b


def f64_max(a: float, b: float) -> float:
    if a != a or b != b:
        return a + b
    if a == b:
        return b if math.copysign(1.0, a) < 0 else a
    return a if a > b else b


def f64_ceil(a: float) -> float:
    if a != a:
        return a + 0.0
    if a in (INF, -INF) or a == 0.0:
        return a
    return math.copysign(float(math.ceil(a)), a)


def f64_floor(a: float) -> float:
    if a != a:
        return a + 0.0
    if a in (INF, -INF) or a == 0.0:
        return a
    return math.copysign(float(math.floor(a)), a)


def f64_trunc(a: float) -> float:
    if a != a:
        return a + 0.0
    if a in (INF, -INF) or a == 0.0:
        return a
    return math.copysign(float(math.trunc(a)), a)


def f64_nearest(a: float) -> float:
    if a != a:
        return a + 0.0
    if a in (INF, -INF) or a == 0.0:
        return a
    # round() on a float rounds half to even
    return math.copysign(float(round(a)), a)


def f64_abs(a: float) -> float:
    return math.fabs(a)


def f64_neg(a: float) -> float:
    return -a


def f64_copysign(a: float, b: float) -> float:
    return math.copysign(a, b)


def f32_abs(a: float) -> float:
    if type(a) is F32NaN:
        return F32NaN(a.bits & 0x7FFFFFFF)
    return math.fabs(a)


def f32_neg(a: float) -> float:
    if type(a) is F32NaN:
        return F32NaN(a.bits ^ 0x80000000)
    return -a


def f32_copysign(a: float, b: float) -> float:
    if type(b) is F32NaN:
        negative = b.bits >> 31
    else:
        negative = math.copysign(1.0, b) < 0
    if type(a) is F32NaN:
        return F32NaN((a.bits & 0x7FFFFFFF) | (0x80000000 if negative else 0))
    return math.copysign(a, -1.0 if negative else 1.0)


def f32_min(a: float, b: float) -> float:
    return float(f64_min(a, b))


def f32_max(a: float, b: float) -> float:
    return float(f64_max(a, b))
