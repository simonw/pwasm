"""Run WebAssembly spec test scripts (.wast) against pwasm.

Module text is compiled with wasmtime.wat2wasm; the script commands are
parsed and executed here. Values are compared bit for bit, including NaN
payloads, using pwasm's internal value representation.

Usage: python tests/spec_runner.py tests/spec/i32.wast [...]
"""

from __future__ import annotations

import re
import struct
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import wasmtime

from pwasm import decode_module
from pwasm.errors import DecodeError, LinkError, TrapError, WasmError
from pwasm.executor import instantiate, invoke
from pwasm.numeric import f32_from_bits, f32_to_bits
from pwasm.runtime import GlobalInstance, HostFunction, MemoryInstance
from pwasm.types import FuncType, GlobalType

try:
    from pwasm.runtime import TableInstance
except ImportError:  # tables not implemented yet
    TableInstance = None

MASKS = {"i32": 0xFFFFFFFF, "i64": 0xFFFFFFFFFFFFFFFF}

# --- S-expression parsing ------------------------------------------------

_TOKEN = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<line_comment>;;[^\n]*)
  | (?P<block_comment>\(;)
  | (?P<open>\()
  | (?P<close>\))
  | (?P<string>"(?:[^"\\]|\\.)*")
  | (?P<atom>[^\s()";]+)
    """,
    re.VERBOSE | re.DOTALL,
)


class SExpr(list):
    """A parenthesised list, remembering where it came from in the source."""

    start: int = 0
    end: int = 0


def _skip_block_comment(text: str, pos: int) -> int:
    depth = 1
    pos += 2
    while depth:
        nxt_open = text.find("(;", pos)
        nxt_close = text.find(";)", pos)
        if nxt_close < 0:
            raise ValueError("unterminated block comment")
        if 0 <= nxt_open < nxt_close:
            depth += 1
            pos = nxt_open + 2
        else:
            depth -= 1
            pos = nxt_close + 2
    return pos


def parse_sexprs(text: str) -> list[SExpr]:
    stack: list[SExpr] = [SExpr()]
    pos = 0
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if m is None:
            raise ValueError(f"cannot tokenize at {pos}: {text[pos:pos + 20]!r}")
        kind = m.lastgroup
        if kind == "block_comment":
            pos = _skip_block_comment(text, pos)
            continue
        pos = m.end()
        if kind in ("ws", "line_comment"):
            continue
        if kind == "open":
            node = SExpr()
            node.start = m.start()
            stack.append(node)
        elif kind == "close":
            node = stack.pop()
            node.end = pos
            stack[-1].append(node)
        else:
            stack[-1].append(m.group())
    return stack[0]


def parse_string(token: str) -> bytes:
    """Decode a WAT string literal (with \\hh, \\n, \\t, \\u{...} escapes)."""
    body = token[1:-1]
    out = bytearray()
    i = 0
    while i < len(body):
        c = body[i]
        if c != "\\":
            out += c.encode()
            i += 1
            continue
        nxt = body[i + 1]
        simple = {"n": b"\n", "t": b"\t", "r": b"\r", '"': b'"', "'": b"'", "\\": b"\\"}
        if nxt in simple:
            out += simple[nxt]
            i += 2
        elif nxt == "u":
            close = body.index("}", i)
            out += chr(int(body[i + 3 : close], 16)).encode()
            i = close + 1
        else:
            out.append(int(body[i + 1 : i + 3], 16))
            i += 3
    return bytes(out)


# --- Numbers --------------------------------------------------------------


def parse_int(text: str, bits: int) -> int:
    text = text.replace("_", "")
    sign = 1
    if text[0] in "+-":
        sign = -1 if text[0] == "-" else 1
        text = text[1:]
    value = int(text, 16) if text.lower().startswith("0x") else int(text, 10)
    return (sign * value) & ((1 << bits) - 1)


def _round_to_float_bits(value: Fraction, sign: int, mant_bits: int, exp_bits: int):
    width = 1 + exp_bits + mant_bits
    bias = (1 << (exp_bits - 1)) - 1
    sign_bit = sign << (width - 1)
    if value == 0:
        return sign_bit
    n, d = value.numerator, value.denominator
    e = n.bit_length() - d.bit_length()
    # make 2**e <= value < 2**(e+1)
    if (n << max(0, -e)) < (d << max(0, e)):
        e -= 1
    emin = 1 - bias
    if e < emin:
        e = emin
    shift = e - mant_bits
    scaled = value / (Fraction(2) ** shift)
    q, r = divmod(scaled.numerator, scaled.denominator)
    if 2 * r > scaled.denominator or (2 * r == scaled.denominator and q & 1):
        q += 1
    if q >> (mant_bits + 1):  # rounding carried into a new binade
        q >>= 1
        e += 1
    if e > bias:
        return sign_bit | (((1 << exp_bits) - 1) << mant_bits)  # infinity
    if q < (1 << mant_bits):
        return sign_bit | q  # subnormal
    return sign_bit | ((e + bias) << mant_bits) | (q - (1 << mant_bits))


def parse_float_bits(text: str, bits: int) -> int:
    """Exact bit pattern of a WAT float literal."""
    mant_bits, exp_bits = (23, 8) if bits == 32 else (52, 11)
    exp_all = (1 << exp_bits) - 1
    text = text.replace("_", "")
    sign = 0
    if text[0] in "+-":
        sign = 1 if text[0] == "-" else 0
        text = text[1:]
    sign_bit = sign << (bits - 1)
    if text == "inf":
        return sign_bit | (exp_all << mant_bits)
    if text.startswith("nan"):
        payload = int(text[4:], 16) if text.startswith("nan:") else 1 << (mant_bits - 1)
        return sign_bit | (exp_all << mant_bits) | payload
    if text.lower().startswith("0x"):
        body = text[2:]
        exp = 0
        if "p" in body.lower():
            body, exp_text = re.split("[pP]", body)
            exp = int(exp_text)
        int_part, _, frac_part = body.partition(".")
        mantissa = int((int_part + frac_part) or "0", 16)
        value = Fraction(mantissa, 16 ** len(frac_part)) * Fraction(2) ** exp
    else:
        value = Fraction(text)
    return _round_to_float_bits(value, sign, mant_bits, exp_bits)


# wat2wasm refuses these "likely-confusing" characters even inside strings,
# where the spec tests use them, so pass them as \u{...} escapes instead.
_CONFUSING = re.compile("[\u061c\u200e\u200f\u202a-\u202e\u2066-\u206f\ufeff]")


def wat2wasm(text: str) -> bytes:
    text = _CONFUSING.sub(lambda m: "\\u{%x}" % ord(m.group()), text)
    return bytes(wasmtime.wat2wasm(text))


def f64_from_bits(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def f64_to_bits(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", value))[0]


@dataclass(frozen=True)
class ExternRef:
    """Host reference used by (ref.extern N)."""

    n: int


class Skip(Exception):
    pass


def parse_value(expr: SExpr) -> Any:
    """An argument: (i32.const 1) etc. -> internal pwasm value."""
    op = expr[0]
    if op == "i32.const":
        return parse_int(expr[1], 32)
    if op == "i64.const":
        return parse_int(expr[1], 64)
    if op == "f32.const":
        return f32_from_bits(parse_float_bits(expr[1], 32))
    if op == "f64.const":
        return f64_from_bits(parse_float_bits(expr[1], 64))
    if op == "ref.null":
        return None
    if op == "ref.extern":
        return ExternRef(int(expr[1]))
    raise Skip(f"unsupported value {op}")


def parse_expected(expr: SExpr) -> tuple:
    """An expected result -> (kind, payload)."""
    op = expr[0]
    if op in ("i32.const", "i64.const"):
        return (op[:3], parse_int(expr[1], int(op[1:3])))
    if op in ("f32.const", "f64.const"):
        bits = int(op[1:3])
        if expr[1] in ("nan:canonical", "nan:arithmetic"):
            return (op[:3], expr[1])
        return (op[:3], parse_float_bits(expr[1], bits))
    if op == "ref.null":
        return ("ref.null", None)
    if op == "ref.extern":
        return ("ref.extern", ExternRef(int(expr[1])))
    if op == "ref.func":
        return ("ref.func", None)
    raise Skip(f"unsupported result {op}")


def matches(expected: tuple, actual: Any) -> bool:
    kind, want = expected
    if kind in MASKS:
        return isinstance(actual, int) and (int(actual) & MASKS[kind]) == want
    if kind in ("f32", "f64"):
        if not isinstance(actual, float):
            return False
        if kind == "f32":
            bits, quiet, canonical = f32_to_bits(actual), 0x7FC00000, 0x7FC00000
            magnitude = bits & 0x7FFFFFFF
        else:
            bits = f64_to_bits(actual)
            quiet = canonical = 0x7FF8000000000000
            magnitude = bits & 0x7FFFFFFFFFFFFFFF
        if want == "nan:canonical":
            return magnitude == canonical
        if want == "nan:arithmetic":
            return magnitude & quiet == quiet
        return bits == want
    if kind == "ref.null":
        return actual is None
    if kind == "ref.extern":
        return actual == want
    if kind == "ref.func":
        return actual is not None
    return False


def describe(value: Any) -> str:
    if isinstance(value, float):
        try:
            return f"{value!r} (f32 bits 0x{f32_to_bits(value):08x}, f64 bits 0x{f64_to_bits(value):016x})"
        except OverflowError:
            return f"{value!r} (f64 bits 0x{f64_to_bits(value):016x})"
    return repr(value)


# --- Running scripts --------------------------------------------------------


def spectest_imports() -> dict:
    def host(params, fn=lambda *a: None):
        return HostFunction(FuncType(tuple(params), ()), fn)

    imports = {
        "print": host([]),
        "print_i32": host(["i32"]),
        "print_i64": host(["i64"]),
        "print_f32": host(["f32"]),
        "print_f64": host(["f64"]),
        "print_i32_f32": host(["i32", "f32"]),
        "print_f64_f64": host(["f64", "f64"]),
        "global_i32": GlobalInstance(GlobalType("i32", False), 666),
        "global_i64": GlobalInstance(GlobalType("i64", False), 666),
        "global_f32": GlobalInstance(
            GlobalType("f32", False), f32_from_bits(parse_float_bits("666.6", 32))
        ),
        "global_f64": GlobalInstance(GlobalType("f64", False), 666.6),
        "memory": MemoryInstance(1, 2),
    }
    if TableInstance is not None:
        imports["table"] = TableInstance("funcref", 10, 20)
    return imports


@dataclass
class Result:
    path: str
    passed: int = 0
    skipped: int = 0
    failures: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{Path(self.path).name}: {self.passed} passed, "
            f"{len(self.failures)} failed, {self.skipped} skipped"
        )


class Runner:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self.text = Path(path).read_text()
        self.result = Result(self.path)
        self.registry: dict[str, dict] = {"spectest": spectest_imports()}
        self.instances: dict[str, Any] = {}
        self.current: Any = None

    def line(self, expr: SExpr) -> int:
        return self.text.count("\n", 0, expr.start) + 1

    def fail(self, expr: SExpr, message: str) -> None:
        self.result.failures.append(f"line {self.line(expr)}: {message}")

    # modules

    def compile(self, expr: SExpr) -> bytes:
        words = [w for w in expr[1:3] if isinstance(w, str)]
        if "quote" in words:
            start = expr.index("quote") + 1
            body = b"".join(parse_string(s) for s in expr[start:]).decode()
            return wat2wasm("(module " + body + ")")
        if "binary" in words:
            start = expr.index("binary") + 1
            return b"".join(parse_string(s) for s in expr[start:])
        return wat2wasm(self.text[expr.start : expr.end])

    def instantiate(self, expr: SExpr):
        module = decode_module(self.compile(expr))
        return instantiate(module, self.registry)

    def define_module(self, expr: SExpr) -> None:
        self.current = None
        instance = self.instantiate(expr)
        self.current = instance
        if len(expr) > 1 and isinstance(expr[1], str) and expr[1].startswith("$"):
            self.instances[expr[1]] = instance

    # actions

    def action(self, expr: SExpr) -> list:
        kind = expr[0]
        rest = list(expr[1:])
        instance = self.current
        if rest and isinstance(rest[0], str) and rest[0].startswith("$"):
            instance = self.instances[rest.pop(0)]
        if instance is None:
            raise WasmError("no current module")
        name = parse_string(rest[0]).decode()
        export = instance.exports[name]
        if kind == "get":
            return [export._value]
        args = [parse_value(a) for a in rest[1:]]
        func = export.func
        result = invoke(func, args)
        if func.n_results == 0:
            return []
        if func.n_results == 1:
            return [result]
        return list(result)

    # commands

    def run(self) -> Result:
        for expr in parse_sexprs(self.text):
            try:
                self.command(expr)
            except Skip:
                self.result.skipped += 1
            except Exception as e:  # anything unexpected is a failure
                self.fail(expr, f"{expr[0]}: {type(e).__name__}: {e}")
        return self.result

    def command(self, expr: SExpr) -> None:
        head = expr[0]
        if head == "module":
            self.define_module(expr)
            self.result.passed += 1
        elif head == "register":
            name = parse_string(expr[1]).decode()
            instance = self.instances[expr[2]] if len(expr) > 2 else self.current
            self.registry[name] = {n: instance.exports[n] for n in instance.exports}
        elif head in ("invoke", "get"):
            self.action(expr)
            self.result.passed += 1
        elif head == "assert_return":
            expected = [parse_expected(e) for e in expr[2:]]
            actual = self.action(expr[1])
            if len(actual) != len(expected) or not all(
                matches(e, a) for e, a in zip(expected, actual)
            ):
                self.fail(
                    expr,
                    f"{self.text[expr[1].start:expr[1].end]} returned "
                    f"{[describe(a) for a in actual]}, expected {expected}",
                )
            else:
                self.result.passed += 1
        elif head in ("assert_trap", "assert_exhaustion"):
            message = parse_string(expr[2]).decode()
            target = expr[1]
            try:
                if target[0] == "module":
                    self.instantiate(target)
                else:
                    self.action(target)
            except TrapError as e:
                if message not in str(e):
                    self.fail(expr, f"trapped with {str(e)!r}, expected {message!r}")
                else:
                    self.result.passed += 1
            except RecursionError:
                if head == "assert_exhaustion":
                    self.result.passed += 1
                else:
                    self.fail(expr, f"RecursionError, expected trap {message!r}")
            else:
                self.fail(expr, f"did not trap, expected {message!r}")
        elif head == "assert_unlinkable":
            try:
                self.instantiate(expr[1])
            except LinkError:
                self.result.passed += 1
            else:
                self.fail(expr, "module linked, expected a LinkError")
        elif head == "assert_uninstantiable":
            try:
                self.instantiate(expr[1])
            except TrapError:
                self.result.passed += 1
            else:
                self.fail(expr, "module instantiated, expected a trap")
        elif head in ("assert_invalid", "assert_malformed"):
            raise Skip(head)
        else:
            raise Skip(head)


def run_wast(path: str | Path) -> Result:
    return Runner(path).run()


if __name__ == "__main__":
    verbose = "-v" in sys.argv
    totals = [0, 0, 0]
    for arg in sys.argv[1:]:
        if arg == "-v":
            continue
        result = run_wast(arg)
        print(result.summary())
        if verbose:
            for failure in result.failures[:20]:
                print("   ", failure)
        totals[0] += result.passed
        totals[1] += len(result.failures)
        totals[2] += result.skipped
    print(f"TOTAL: {totals[0]} passed, {totals[1]} failed, {totals[2]} skipped")
