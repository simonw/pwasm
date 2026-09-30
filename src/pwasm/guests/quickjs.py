"""Untrusted JavaScript: quickjs-ng compiled to WebAssembly.

The guest is a small reactor around quickjs-ng with no std/os modules, so
JavaScript code has no filesystem or network access. `host.<name>(...)`
inside JavaScript calls a registered Python function (a Proxy that sends
the name and JSON-encoded arguments to the host).
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable

from ..sandbox import _DEFAULT, Sandbox
from . import load_guest


class JSError(Exception):
    """A JavaScript exception escaped from the guest."""


class QuickJS:
    """A QuickJS interpreter running inside pwasm.

    max_memory: cap in bytes on the WebAssembly memory
    js_memory_limit: QuickJS's own allocation limit in bytes (0 for none)
    stack_size: QuickJS stack limit, so deep recursion is a RangeError
    fuel / timeout: CPU limits (raise pwasm.OutOfFuel / pwasm.Timeout)
    mode: "interpret", "compile" or "auto" (see pwasm.executor.instantiate)
    functions: {"name": callable} available in JavaScript as host.name()
    """

    def __init__(
        self,
        *,
        wasm_path: str | None = None,
        max_memory: int = 64 * 1024 * 1024,
        js_memory_limit: int = 0,
        stack_size: int = 512 * 1024,
        fuel: int | None = None,
        timeout: float | None = None,
        mode: str | None = None,
        functions: dict[str, Callable[..., Any]] | None = None,
    ) -> None:
        self.functions: dict[str, Callable[..., Any]] = dict(functions or {})
        self.output = bytearray()
        self.error_output = bytearray()
        self._result = bytearray()
        self._exception = bytearray()
        self._pending = b""
        self._soft_deadline: float | None = None
        env = {
            "host_write": self._host_write,
            "host_call": self._host_call,
            "host_take": self._host_take,
            "host_interrupt": self._host_interrupt,
        }
        self.sb = Sandbox(
            load_guest("quickjs.wasm", wasm_path),
            imports={"env": env},
            max_memory=max_memory,
            fuel=fuel,
            timeout=timeout,
            mode=mode,
        )
        status = self.sb.call("qjs_init", js_memory_limit, stack_size)
        if status != 0:
            raise RuntimeError(f"qjs_init failed with status {status}")

    # host side

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        self.functions[name] = fn

    def function(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator: expose fn to JavaScript as host.<fn.__name__>()."""
        self.functions[fn.__name__] = fn
        return fn

    def _host_write(self, fd: int, ptr: int, length: int) -> None:
        # fd 1/2: stdout/stderr, 3: JSON completion value, 4: exception text
        buffers = {1: self.output, 2: self.error_output, 3: self._result}
        buffers.get(fd, self._exception).extend(self.sb.read(ptr, length))

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
        except Exception as e:  # becomes a JavaScript exception
            self._pending = f"{type(e).__name__}: {e}".encode()
            return -len(self._pending)

    def _host_take(self, ptr: int, length: int) -> None:
        self.sb.write(ptr, self._pending[:length])

    def _host_interrupt(self) -> int:
        deadline = self._soft_deadline
        return 1 if deadline is not None and time.monotonic() > deadline else 0

    # guest side

    def eval(
        self,
        code: str,
        *,
        timeout: Any = _DEFAULT,
        fuel: int | None = None,
        soft_timeout: float | None = None,
    ) -> Any:
        """Evaluate JavaScript; returns the completion value (decoded from
        JSON when possible). Raises JSError for JavaScript exceptions, and
        pwasm.Timeout / pwasm.OutOfFuel / pwasm.TrapError for hard limits.

        soft_timeout: cooperative limit using QuickJS's interrupt handler;
        the guest sees an InternalError and stays consistent."""
        if fuel is not None:
            self.sb.fuel = fuel
        self._result.clear()
        self._exception.clear()
        if soft_timeout:
            self._soft_deadline = time.monotonic() + soft_timeout
        source = code.encode()
        ptr = self.sb.alloc(source, nul=True)
        try:
            status = self.sb.call("qjs_eval", ptr, len(source), timeout=timeout)
        finally:
            self._soft_deadline = None
            self.sb.free(ptr)
        if status != 0:
            raise JSError(bytes(self._exception).decode("utf-8", "replace"))
        text = bytes(self._result).decode("utf-8", "replace")
        try:
            return json.loads(text)
        except ValueError:
            return text

    def take_output(self) -> str:
        out = bytes(self.output).decode("utf-8", "replace")
        self.output.clear()
        return out

    def gc(self) -> None:
        self.sb.call("qjs_gc")

    @property
    def js_memory_usage(self) -> int:
        return self.sb.call("qjs_memory_usage")

    @property
    def memory_size(self) -> int:
        return self.sb.memory_size

    @property
    def fuel_consumed(self) -> int:
        return self.sb.fuel_consumed
