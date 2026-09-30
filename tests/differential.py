"""Differential testing helpers: run the same export in pwasm and wasmtime."""

import wasmtime

from pwasm import TrapError, decode_module, instantiate
from wat import wat2wasm


class Pair:
    """A module instantiated in both pwasm and wasmtime."""

    def __init__(self, text: str) -> None:
        binary = wat2wasm(text)
        self.pwasm = instantiate(decode_module(binary))
        self.store = wasmtime.Store()
        module = wasmtime.Module(self.store.engine, binary)
        self.wasmtime = wasmtime.Instance(self.store, module, []).exports(self.store)

    def expected(self, name: str, *args):
        try:
            return self.wasmtime[name](self.store, *args)
        except wasmtime.Trap as e:
            return ("trap", str(e).split("\n")[0])

    def actual(self, name: str, *args):
        try:
            return self.pwasm.exports[name](*args)
        except TrapError as e:
            return ("trap", str(e))

    def check(self, name: str, *args) -> None:
        expected = self.expected(name, *args)
        actual = self.actual(name, *args)
        if isinstance(expected, tuple) and expected[0] == "trap":
            assert (
                isinstance(actual, tuple) and actual[0] == "trap"
            ), f"{name}{args}: expected trap {expected[1]!r}, got {actual!r}"
        else:
            assert (
                actual == expected
            ), f"{name}{args}: expected {expected!r}, got {actual!r}"
