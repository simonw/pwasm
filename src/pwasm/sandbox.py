"""Sandbox: run a WebAssembly module with WASI-lite, resource limits and
memory helpers. Ported from the wasmi-python-sandbox research project.

    sb = Sandbox("guest.wasm", imports={"env": {"log": print}},
                 max_memory=32 << 20, timeout=2.0)
    ptr = sb.alloc(b"input", nul=True)   # uses the guest's malloc
    result = sb.call("process", ptr)
    print(sb.stdout)
"""

from __future__ import annotations

import struct
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Union

from .decoder import decode_module
from .errors import LinkError
from .executor import instantiate
from .runtime import Limits
from .types import Module
from .wasi import WASI_MODULES, WasiLite

ImportResolver = Callable[[str, str, Any], Any]
Imports = Union[Mapping[str, Mapping[str, Any]], ImportResolver]

_DEFAULT = object()
_u32 = struct.Struct("<I")
_u64 = struct.Struct("<Q")


class Sandbox:
    """A module instance with resource limits.

    wasm: module bytes, a path, or a decoded Module.
    imports: {"module": {"name": value}} or resolver(module, name, import)
        returning a value or None. WASI imports not otherwise resolved are
        served by WasiLite (unless wasi=False).
    max_memory: cap in bytes on linear memory growth.
    fuel: work budget (one unit per function call or loop iteration).
    timeout: default wall-clock limit in seconds for each call().
    recursion_limit: Python's recursion limit is raised to at least this,
        since each WebAssembly call is a Python call.
    mode: "interpret", "compile" or "auto" (see pwasm.executor.instantiate)
    """

    def __init__(
        self,
        wasm: bytes | str | Path | Module,
        *,
        imports: Imports | None = None,
        max_memory: int | None = None,
        fuel: int | None = None,
        timeout: float | None = None,
        wasi: bool = True,
        stdin: bytes = b"",
        args: Any = ("guest",),
        env: Mapping[str, str] | None = None,
        initialize: bool = True,
        recursion_limit: int = 20_000,
        mode: str | None = None,
    ) -> None:
        if isinstance(wasm, Module):
            module = wasm
        elif isinstance(wasm, (bytes, bytearray, memoryview)):
            module = decode_module(bytes(wasm))
        else:
            module = decode_module(Path(wasm).read_bytes())
        self.module = module
        self.timeout = timeout
        self.limits = Limits(fuel=fuel, max_memory=max_memory)
        self.wasi = WasiLite(stdin=stdin, args=args, env=env) if wasi else None
        _raise_recursion_limit(recursion_limit)

        if imports is None:
            resolver: ImportResolver = lambda m, n, i: None
        elif callable(imports):
            resolver = imports
        else:
            resolver = lambda m, n, i, _d=imports: _d.get(m, {}).get(n)

        resolved: dict[str, dict[str, Any]] = {}
        missing = []
        for imp in module.imports:
            value = resolver(imp.module, imp.name, imp)
            if value is None and self.wasi is not None and imp.module in WASI_MODULES:
                if imp.kind == "func":
                    has_result = bool(module.types[imp.desc].results)
                    value = self.wasi.lookup(imp.name, has_result)
            if value is None:
                missing.append(f"{imp.module}.{imp.name}")
                continue
            resolved.setdefault(imp.module, {})[imp.name] = value
        if missing:
            raise LinkError("unresolved imports: " + ", ".join(missing))

        self.instance = instantiate(module, resolved, limits=self.limits, mode=mode)
        self.exports = self.instance.exports
        if self.wasi is not None and self.instance.memories:
            self.wasi.bind(self.instance)
        if initialize and "_initialize" in self.exports:
            self.call("_initialize")

    # calls

    def call(self, name: str, *args: Any, timeout: Any = _DEFAULT) -> Any:
        """Call an export with Python values. timeout overrides the default
        wall-clock limit for this call (None for no limit)."""
        return self._with_timeout(self.exports[name], args, timeout)

    def call_indirect(
        self,
        index: int,
        *args: Any,
        table: str = "__indirect_function_table",
        timeout: Any = _DEFAULT,
    ) -> Any:
        """Call the function at `index` of an exported table."""
        return self._with_timeout(self.exports[table].get(index), args, timeout)

    def _with_timeout(self, func: Any, args: tuple, timeout: Any) -> Any:
        if timeout is _DEFAULT:
            timeout = self.timeout
        if timeout is None:
            # keep any deadline set by an enclosing call
            return func(*args)
        limits = self.limits
        previous = limits.deadline
        deadline = time.monotonic() + timeout
        if previous is not None:
            deadline = min(deadline, previous)
        limits.set_deadline(deadline)
        try:
            return func(*args)
        finally:
            limits.set_deadline(previous)

    # fuel

    @property
    def fuel(self) -> int | None:
        """Remaining fuel (None for unlimited)."""
        return self.limits.fuel_remaining

    @fuel.setter
    def fuel(self, value: int | None) -> None:
        self.limits.set_fuel(value)

    @property
    def fuel_consumed(self) -> int:
        return self.limits.fuel_consumed

    # memory

    @property
    def memory(self):
        return self.instance.memories[0]

    @property
    def memory_size(self) -> int:
        """Size of the linear memory in bytes."""
        return len(self.memory.data)

    def read(self, ptr: int, length: int) -> bytes:
        return self.memory.read(ptr, length)

    def write(self, ptr: int, data: bytes) -> None:
        self.memory.write(ptr, bytes(data))

    def read_cstr(self, ptr: int, max_length: int = 1 << 20) -> bytes:
        """Read a NUL-terminated string."""
        data = self.memory.data
        end = data.find(0, ptr, min(len(data), ptr + max_length))
        if end < 0:
            raise ValueError("string is not NUL-terminated")
        return bytes(data[ptr:end])

    def read_u32(self, ptr: int) -> int:
        return _u32.unpack(self.read(ptr, 4))[0]

    def write_u32(self, ptr: int, value: int) -> None:
        self.write(ptr, _u32.pack(value & 0xFFFFFFFF))

    def write_u64(self, ptr: int, value: int) -> None:
        self.write(ptr, _u64.pack(value & 0xFFFFFFFFFFFFFFFF))

    def alloc(self, data: bytes, nul: bool = False) -> int:
        """Copy data into guest memory allocated with the guest's malloc."""
        payload = bytes(data) + (b"\0" if nul else b"")
        ptr = self.call("malloc", len(payload), timeout=None)
        if ptr == 0:
            raise MemoryError("guest malloc failed")
        self.write(ptr, payload)
        return ptr

    def free(self, ptr: int) -> None:
        self.call("free", ptr, timeout=None)

    # stdio

    @property
    def stdout(self) -> str:
        return bytes(self.wasi.stdout).decode("utf-8", "replace") if self.wasi else ""

    @property
    def stderr(self) -> str:
        return bytes(self.wasi.stderr).decode("utf-8", "replace") if self.wasi else ""


def _raise_recursion_limit(limit: int) -> None:
    # Python 3.10 and PyPy use the C stack for Python-to-Python calls, and
    # crash rather than raise RecursionError much beyond 8,000 frames
    if sys.version_info < (3, 11) or sys.implementation.name != "cpython":
        limit = min(limit, 8000)
    if sys.getrecursionlimit() < limit:
        sys.setrecursionlimit(limit)
