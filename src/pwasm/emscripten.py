"""Emscripten-style setjmp/longjmp support, implemented on the host.

Code compiled with LLVM's emscripten setjmp/longjmp lowering (used by
emscripten, and by `-mllvm -enable-emscripten-sjlj` with wasi-sdk) calls
functions through imported `invoke_<sig>(table_index, args...)` trampolines,
and implements longjmp by calling `_emscripten_throw_longjmp()`.

The host side: `_emscripten_throw_longjmp` raises a Python exception, which
unwinds every WebAssembly frame up to the nearest `invoke_*` trampoline.
That trampoline restores the stack pointer and calls the module's exported
`setThrew(1, 0)`; the compiled code then finds the matching setjmp.

    sjlj = EmscriptenSjLj()
    instance = instantiate(module, {"env": {**sjlj.imports(module), ...}})
    sjlj.bind(instance)
"""

from __future__ import annotations

from typing import Any

from .errors import LinkError, TrapError
from .executor import execute, invoke
from .runtime import HostFunction


class LongjmpUnwind(Exception):
    """Raised by _emscripten_throw_longjmp to unwind to an invoke_*."""


class EmscriptenSjLj:
    """Provides the invoke_* and _emscripten_throw_longjmp imports.

    overflow_export: name of an exported function to call instead of the
        target once invokes are nested more than max_depth deep. It should
        raise a guest-level error by longjmp-ing (MicroPython's sandbox
        guest exports mp_sandbox_recursion_error for this), turning runaway
        recursion into an exception the guest can catch.
    """

    def __init__(self, overflow_export: str | None = None, max_depth: int = 400):
        self.overflow_export = overflow_export
        self.max_depth = max_depth
        self.invokes = 0
        self.unwinds = 0
        self.overflows = 0
        self.depth = 0
        self._instance: Any = None

    def imports(self, module: Any) -> dict[str, HostFunction]:
        """Host functions for the emscripten imports the module declares."""
        functions = {}
        for imp in module.imports:
            if imp.kind != "func" or imp.module != "env":
                continue
            ftype = module.types[imp.desc]
            if imp.name.startswith("invoke_"):
                functions[imp.name] = HostFunction(
                    ftype, self._make_invoke(bool(ftype.results)), raw=True
                )
            elif imp.name == "_emscripten_throw_longjmp":
                functions[imp.name] = HostFunction(ftype, self._throw, raw=True)
        return functions

    def bind(self, instance: Any) -> None:
        exports = instance.exports
        self._instance = instance
        if "__indirect_function_table" not in exports:
            raise LinkError("module does not export __indirect_function_table")
        self._table = exports["__indirect_function_table"].elements
        if "setThrew" not in exports:
            raise LinkError("module does not export setThrew")
        self._set_threw = exports["setThrew"].func
        self._stack_pointer = None
        self._stack_save = self._stack_restore = None
        if "__stack_pointer" in exports:
            self._stack_pointer = exports["__stack_pointer"]
        else:
            for save, restore in (
                ("stackSave", "stackRestore"),
                ("stack_save", "stack_restore"),
            ):
                if save in exports and restore in exports:
                    self._stack_save = exports[save].func
                    self._stack_restore = exports[restore].func
                    break
            else:
                raise LinkError("module exports neither __stack_pointer nor stackSave")
        self._overflow = None
        if self.overflow_export is not None:
            self._overflow = exports[self.overflow_export].func

    def _throw(self) -> None:
        self.unwinds += 1
        raise LongjmpUnwind()

    def _make_invoke(self, has_result: bool):
        def invoke_trampoline(index: int, *args: Any) -> Any:
            self.invokes += 1
            stack_pointer = self._stack_pointer
            if stack_pointer is not None:
                saved = stack_pointer._value
            else:
                saved = execute(self._stack_save, [])
            self.depth += 1
            try:
                if self.depth > self.max_depth and self._overflow is not None:
                    self.overflows += 1
                    execute(self._overflow, [])
                    return 0 if has_result else None
                if index >= len(self._table) or self._table[index] is None:
                    raise TrapError(f"uninitialized element {index}")
                return invoke(self._table[index], list(args))
            except LongjmpUnwind:
                if stack_pointer is not None:
                    stack_pointer._value = saved
                else:
                    execute(self._stack_restore, [saved])
                execute(self._set_threw, [1, 0])
                return 0 if has_result else None
            finally:
                self.depth -= 1

        return invoke_trampoline
