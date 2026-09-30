"""Untrusted Python: MicroPython compiled to WebAssembly.

MicroPython raises exceptions with setjmp/longjmp, which this build
implements with emscripten-style invoke_* trampolines (see
pwasm.emscripten). Guest code can call registered Python functions with
`import host; host.call("name", *args)`; arguments and results travel as
JSON.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from ..emscripten import EmscriptenSjLj
from ..sandbox import _DEFAULT, Sandbox
from . import load_guest


class PythonError(Exception):
    """An uncaught exception inside the guest (the traceback text)."""


class MicroPython:
    """A MicroPython interpreter running inside pwasm.

    heap_size: initial MicroPython heap (it grows on demand up to max_memory)
    max_memory: cap in bytes on the WebAssembly memory
    fuel / timeout: CPU limits (raise pwasm.OutOfFuel / pwasm.Timeout)
    functions: {"name": callable} available as host.call("name", ...)
    mode: "interpret", "compile" or "auto" (see pwasm.executor.instantiate)
    max_depth: nesting depth of guest calls at which the guest raises
        RuntimeError: maximum recursion depth exceeded
    """

    def __init__(
        self,
        *,
        wasm_path: str | None = None,
        heap_size: int = 256 * 1024,
        max_memory: int = 64 * 1024 * 1024,
        fuel: int | None = None,
        timeout: float | None = None,
        mode: str | None = None,
        functions: dict[str, Callable[..., Any]] | None = None,
        max_depth: int = 300,
    ) -> None:
        self.functions: dict[str, Callable[..., Any]] = dict(functions or {})
        self.output = bytearray()
        self.error_output = bytearray()
        self._pending = b""
        module = load_guest("micropython.wasm", wasm_path)
        self.sjlj = EmscriptenSjLj(
            overflow_export="mp_sandbox_recursion_error", max_depth=max_depth
        )
        env = {
            "host_write": self._host_write,
            "host_call": self._host_call,
            "host_take": self._host_take,
            **self.sjlj.imports(module),
        }
        self.sb = Sandbox(
            module,
            imports={"env": env},
            max_memory=max_memory,
            fuel=fuel,
            timeout=timeout,
            mode=mode,
            initialize=False,
        )
        self.sjlj.bind(self.sb.instance)
        self.sb.call("_initialize")
        status = self.sb.call("mp_sandbox_init", heap_size)
        if status != 0:
            raise RuntimeError(f"mp_sandbox_init failed with status {status}")

    # host side

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        self.functions[name] = fn

    def function(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator: make fn available as host.call(fn.__name__, ...)."""
        self.functions[fn.__name__] = fn
        return fn

    def _host_write(self, fd: int, ptr: int, length: int) -> None:
        data = self.sb.read(ptr, length)
        (self.output if fd == 1 else self.error_output).extend(data)

    def _host_call(
        self, name_ptr: int, name_len: int, args_ptr: int, args_len: int
    ) -> int:
        name = self.sb.read(name_ptr, name_len).decode()
        try:
            args = json.loads(self.sb.read(args_ptr, args_len))
            fn = self.functions.get(name)
            if fn is None:
                raise NameError(f"no host function named {name!r}")
            self._pending = json.dumps(fn(*args)).encode()
            return len(self._pending)
        except Exception as e:  # reported to the guest as an exception
            self._pending = f"{type(e).__name__}: {e}".encode()
            return -len(self._pending)

    def _host_take(self, ptr: int, length: int) -> None:
        self.sb.write(ptr, self._pending[:length])

    # guest side

    def exec(
        self, code: str, *, timeout: Any = _DEFAULT, fuel: int | None = None
    ) -> str:
        """Run Python source in the guest; returns what it printed. Raises
        PythonError for uncaught guest exceptions, and pwasm.Timeout,
        pwasm.OutOfFuel or pwasm.TrapError for hard limits.

        fuel: if given, sets the remaining fuel budget first."""
        if fuel is not None:
            self.sb.fuel = fuel
        self.error_output.clear()
        source = code.encode()
        ptr = self.sb.alloc(source, nul=True)
        try:
            status = self.sb.call("mp_sandbox_exec", ptr, len(source), timeout=timeout)
        finally:
            self.sb.free(ptr)
        if status != 0:
            raise PythonError(bytes(self.error_output).decode("utf-8", "replace"))
        return self.take_output()

    def take_output(self) -> str:
        out = bytes(self.output).decode("utf-8", "replace")
        self.output.clear()
        return out

    def collect(self) -> None:
        """Run MicroPython's garbage collector."""
        self.sb.call("mp_sandbox_collect")

    @property
    def memory_size(self) -> int:
        return self.sb.memory_size

    @property
    def fuel_consumed(self) -> int:
        return self.sb.fuel_consumed
