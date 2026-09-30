"""Untrusted JavaScript: Micro QuickJS (mquickjs), an emscripten build.

mquickjs runs an ES5-style subset of JavaScript inside a fixed-size heap
(memory_limit). It has no host bridge: eval() returns the completion value
converted to a string by the guest, which is parsed back into None, a
bool, an int, a float or a str.
"""

from __future__ import annotations

from typing import Any

from ..emscripten import EmscriptenSjLj
from ..errors import TrapError
from ..sandbox import _DEFAULT, Sandbox
from . import load_guest
from .quickjs import JSError


class MQuickJS:
    """A Micro QuickJS interpreter running inside pwasm.

    memory_limit: size in bytes of the JavaScript heap (fixed at startup)
    max_memory: cap in bytes on the WebAssembly memory
    fuel / timeout: CPU limits (raise pwasm.OutOfFuel / pwasm.Timeout)
    """

    def __init__(
        self,
        *,
        wasm_path: str | None = None,
        memory_limit: int = 1024 * 1024,
        max_memory: int = 64 * 1024 * 1024,
        fuel: int | None = None,
        timeout: float | None = None,
    ) -> None:
        module = load_guest("mquickjs.wasm", wasm_path)
        self.sjlj = EmscriptenSjLj()
        self._temp_ret0 = 0
        env = {
            "abort": self._abort,
            "__assert_fail": self._assert_fail,
            "setTempRet0": self._set_temp_ret0,
            "getTempRet0": lambda: self._temp_ret0,
            "emscripten_resize_heap": self._resize_heap,
            "emscripten_memcpy_big": self._memcpy_big,
            **self.sjlj.imports(module),
        }
        self.sb = Sandbox(
            module,
            imports={"env": env},
            max_memory=max_memory,
            fuel=fuel,
            timeout=timeout,
            wasi=False,
            initialize=False,
        )
        self.sjlj.bind(self.sb.instance)
        self.sb.call("__wasm_call_ctors")
        if not self.sb.call("sandbox_init", memory_limit):
            raise RuntimeError("sandbox_init failed")

    # emscripten runtime imports

    def _abort(self) -> None:
        raise TrapError("abort() called")

    def _assert_fail(self, condition: int, filename: int, line: int, func: int) -> None:
        message = self.sb.read_cstr(condition).decode("utf-8", "replace")
        raise TrapError(f"assertion failed: {message}")

    def _set_temp_ret0(self, value: int) -> None:
        self._temp_ret0 = value

    def _resize_heap(self, requested: int) -> int:
        memory = self.sb.memory
        requested &= 0xFFFFFFFF
        if requested <= len(memory.data):
            return 1
        pages = (requested - len(memory.data) + 65535) // 65536
        return 0 if memory.grow(pages) == -1 else 1

    def _memcpy_big(self, dest: int, src: int, n: int) -> int:
        self.sb.write(dest, self.sb.read(src, n))
        return dest

    # guest side

    def eval(
        self, code: str, *, timeout: Any = _DEFAULT, fuel: int | None = None
    ) -> Any:
        """Evaluate JavaScript; returns the completion value as None, a
        bool, an int, a float or a str. Raises JSError for JavaScript
        exceptions, and pwasm.Timeout / pwasm.OutOfFuel for hard limits."""
        if fuel is not None:
            self.sb.fuel = fuel
        ptr = self.sb.alloc(code.encode(), nul=True)
        try:
            result = self.sb.call("sandbox_eval", ptr, timeout=timeout)
        finally:
            self.sb.free(ptr)
        if result == 0:
            error = self.sb.read_cstr(self.sb.call("sandbox_get_error"))
            raise JSError(error.decode("utf-8", "replace"))
        return _parse(self.sb.read_cstr(result).decode("utf-8", "replace"))

    def close(self) -> None:
        self.sb.call("sandbox_free")

    @property
    def memory_size(self) -> int:
        return self.sb.memory_size

    @property
    def fuel_consumed(self) -> int:
        return self.sb.fuel_consumed


def _parse(text: str) -> Any:
    if text in ("undefined", "null"):
        return None
    if text in ("true", "false"):
        return text == "true"
    for convert in (int, float):
        try:
            return convert(text)
        except ValueError:
            pass
    return text
