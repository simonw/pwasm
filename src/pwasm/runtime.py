"""Runtime objects: memories, globals, tables and function instances.

Internally i32 and i64 values are unsigned Python ints (0 <= v < 2**32 or
2**64), f32/f64 values are Python floats (f32 values always hold a value
that is exactly representable as a 32-bit float) and references are either
None or a function object. Values are converted to signed ints only at the
boundary with Python code.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from .errors import OutOfFuel, Timeout, TrapError
from .numeric import F32NaN, f32_round
from .types import FuncType, GlobalType, Function

MASK_32 = 0xFFFFFFFF
MASK_64 = 0xFFFFFFFFFFFFFFFF
PAGE_SIZE = 65536
MAX_PAGES = 65536  # 4 GiB, the wasm32 limit


def _i32_to_python(v: int) -> int:
    return v - 0x100000000 if v & 0x80000000 else int(v)


def _i64_to_python(v: int) -> int:
    return v - 0x10000000000000000 if v & 0x8000000000000000 else int(v)


def _i32_from_python(v: Any) -> int:
    return int(v) & MASK_32


def _i64_from_python(v: Any) -> int:
    return int(v) & MASK_64


def _identity(v: Any) -> Any:
    return v


TO_PYTHON: dict[str, Callable[[Any], Any]] = {
    "i32": _i32_to_python,
    "i64": _i64_to_python,
    "f32": _identity,
    "f64": _identity,
    "funcref": _identity,
    "externref": _identity,
}


def _f32_from_python(v: Any) -> float:
    if type(v) is F32NaN:
        return v
    return f32_round(float(v))


FROM_PYTHON: dict[str, Callable[[Any], Any]] = {
    "i32": _i32_from_python,
    "i64": _i64_from_python,
    "f32": _f32_from_python,
    "f64": float,
    "funcref": _identity,
    "externref": _identity,
}

ZERO: dict[str, Any] = {
    "i32": 0,
    "i64": 0,
    "f32": 0.0,
    "f64": 0.0,
    "funcref": None,
    "externref": None,
}


class MemoryInstance:
    """A linear memory: a bytearray that grows in 64 KiB pages."""

    PAGE_SIZE = PAGE_SIZE

    def __init__(self, min_pages: int, max_pages: int | None = None) -> None:
        self.data = bytearray(min_pages * PAGE_SIZE)
        self.max_pages = max_pages
        # An embedder-imposed cap (see Limits.max_memory), separate from the
        # maximum the module declared
        self.limit_pages: int | None = None

    @property
    def size(self) -> int:
        """Current size in pages."""
        return len(self.data) // PAGE_SIZE

    def grow(self, delta: int) -> int:
        """Grow by delta pages. Returns the old size in pages, or -1."""
        old = len(self.data) // PAGE_SIZE
        new = old + delta
        limit = MAX_PAGES if self.max_pages is None else self.max_pages
        if self.limit_pages is not None:
            limit = min(limit, self.limit_pages)
        if new > limit:
            return -1
        if delta:
            self.data.extend(bytes(delta * PAGE_SIZE))
        return old

    def read(self, ptr: int, length: int) -> bytes:
        """Read length bytes starting at ptr."""
        if ptr < 0 or length < 0 or ptr + length > len(self.data):
            raise TrapError("out of bounds memory access")
        return bytes(self.data[ptr : ptr + length])

    def write(self, ptr: int, data: bytes) -> None:
        """Write bytes starting at ptr."""
        if ptr < 0 or ptr + len(data) > len(self.data):
            raise TrapError("out of bounds memory access")
        self.data[ptr : ptr + len(data)] = data


class Limits:
    """Resource limits for an instance, passed to instantiate().

    fuel: budget of work units (None for unlimited). One unit is charged
        for every function call and every iteration of a loop, so the cost
        of running the same code is deterministic.
    max_memory: cap in bytes on the size of the instance's memories.

    The deadline (a time.monotonic() value, see set_deadline) and the fuel
    budget are checked every `check_interval` units.
    """

    __slots__ = (
        "fuel",
        "deadline",
        "max_memory",
        "check_interval",
        "countdown",
        "_granted",
        "_consumed",
    )

    def __init__(
        self,
        fuel: int | None = None,
        max_memory: int | None = None,
        check_interval: int = 1000,
    ) -> None:
        self.fuel = fuel
        self.deadline: float | None = None
        self.max_memory = max_memory
        self.check_interval = check_interval
        self._consumed = 0
        self._granted = 0
        self.countdown = 0
        self._grant()

    @property
    def fuel_consumed(self) -> int:
        return self._consumed + self._granted - self.countdown

    @property
    def fuel_remaining(self) -> int | None:
        if self.fuel is None:
            return None
        return self.fuel - self.fuel_consumed

    def set_fuel(self, fuel: int | None) -> None:
        """Set the total budget (fuel consumed so far still counts)."""
        self._settle()
        self.fuel = None if fuel is None else self.fuel_consumed + fuel
        self._grant()

    def set_deadline(self, deadline: float | None) -> None:
        """Stop execution once time.monotonic() passes deadline."""
        self._settle()
        self.deadline = deadline
        self._grant()

    def _settle(self) -> None:
        self._consumed += self._granted - self.countdown
        self._granted = self.countdown = 0

    def _grant(self) -> None:
        grant = self.check_interval
        if self.fuel is not None:
            grant = max(0, min(grant, self.fuel - self._consumed))
        self._granted = self.countdown = grant

    def refill(self) -> None:
        """Called by the executor when a unit is charged with the countdown
        already at zero (it has gone to -1): check the limits, then grant
        another slice and charge the unit to it."""
        self.countdown += 1
        self._settle()
        if self.fuel is not None and self._consumed >= self.fuel:
            raise OutOfFuel(f"fuel budget of {self.fuel} units exhausted")
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise Timeout("execution timed out")
        self._grant()
        self.countdown -= 1


MAX_TABLE_SIZE = 10_000_000  # implementation limit on table.grow


class TableInstance:
    """A table of references (functions, or host values for externref)."""

    def __init__(
        self, element_type: str, min_size: int, max_size: int | None = None
    ) -> None:
        self.element_type = element_type
        self.elements: list = [None] * min_size
        self.max_size = max_size

    @property
    def size(self) -> int:
        return len(self.elements)

    def grow(self, delta: int, init: Any = None) -> int:
        """Grow by delta elements. Returns the old size, or -1."""
        old = len(self.elements)
        limit = MAX_TABLE_SIZE if self.max_size is None else self.max_size
        if old + delta > min(limit, MAX_TABLE_SIZE):
            return -1
        self.elements.extend([init] * delta)
        return old

    def get(self, index: int) -> Any:
        if not 0 <= index < len(self.elements):
            raise TrapError("out of bounds table access")
        return self.elements[index]

    def set(self, index: int, value: Any) -> None:
        if not 0 <= index < len(self.elements):
            raise TrapError("out of bounds table access")
        self.elements[index] = value

    def fill(self, index: int, value: Any, n: int) -> None:
        if index + n > len(self.elements):
            raise TrapError("out of bounds table access")
        self.elements[index : index + n] = [value] * n

    def init(self, dest: int, refs: list, src: int, n: int) -> None:
        """table.init: copy n references from refs[src:] to dest."""
        if src + n > len(refs) or dest + n > len(self.elements):
            raise TrapError("out of bounds table access")
        self.elements[dest : dest + n] = refs[src : src + n]


class GlobalInstance:
    """A global variable. `value` converts to and from Python values."""

    __slots__ = ("type", "_value")

    def __init__(self, type: GlobalType, value: Any = None) -> None:
        self.type = type
        self._value = ZERO[type.valtype] if value is None else value

    @property
    def mutable(self) -> bool:
        return self.type.mutable

    @property
    def value(self) -> Any:
        return TO_PYTHON[self.type.valtype](self._value)

    @value.setter
    def value(self, v: Any) -> None:
        if not self.type.mutable:
            raise AttributeError("global is immutable")
        self._value = FROM_PYTHON[self.type.valtype](v)


class WasmFunction:
    """A function defined by a module instance; compiled on first call."""

    __slots__ = ("type", "instance", "func", "index", "code", "n_params", "n_results")

    def __init__(
        self, type: FuncType, instance: Any, func: Function, index: int
    ) -> None:
        self.type = type
        self.instance = instance
        self.func = func
        self.index = index
        self.code = None
        self.n_params = len(type.params)
        self.n_results = len(type.results)

    def compile(self):
        from .compiler import compile_function

        self.code = compile_function(self)
        return self.code

    def __call__(self, *args: Any) -> Any:
        """Call with Python values, like an exported function."""
        from .executor import call_with_python_values

        return call_with_python_values(self, args)

    def __repr__(self) -> str:
        return f"<WasmFunction {self.index} {self.type}>"


class HostFunction:
    """A Python callable exposed to WebAssembly with a given signature.

    The callable receives Python values (signed ints for i32/i64) and returns
    None, a single value, or a tuple/list of values for multiple results.
    """

    __slots__ = (
        "type",
        "fn",
        "raw",
        "n_params",
        "n_results",
        "_to_python",
        "_from_python",
    )

    def __init__(
        self, type: FuncType, fn: Callable[..., Any], raw: bool = False
    ) -> None:
        self.type = type
        self.fn = fn
        # raw functions receive and return internal values (unsigned ints)
        # without conversion: faster, for trampolines that only pass
        # values through
        self.raw = raw
        self.n_params = len(type.params)
        self.n_results = len(type.results)
        self._to_python = [TO_PYTHON[t] for t in type.params]
        self._from_python = [FROM_PYTHON[t] for t in type.results]

    def call(self, args: list) -> Any:
        """Call with internal values; returns internal results in the same
        shape as wasm functions (None, a value, or a list)."""
        if self.raw:
            return self.fn(*args)
        result = self.fn(*[conv(a) for conv, a in zip(self._to_python, args)])
        n = self.n_results
        if n == 0:
            return None
        if n == 1:
            if isinstance(result, (tuple, list)):
                (result,) = result
            return self._from_python[0](result)
        if not isinstance(result, (tuple, list)) or len(result) != n:
            raise TypeError(f"host function must return {n} values")
        return [conv(r) for conv, r in zip(self._from_python, result)]

    def __call__(self, *args: Any) -> Any:
        from .executor import call_with_python_values

        return call_with_python_values(self, args)

    def __repr__(self) -> str:
        return f"<HostFunction {getattr(self.fn, '__name__', self.fn)} {self.type}>"
