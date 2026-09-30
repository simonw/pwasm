"""Micro QuickJS (mquickjs) compiled to WebAssembly with emscripten."""

import time

import pytest

from pwasm import OutOfFuel, Timeout
from pwasm.guests import JSError, MQuickJS


@pytest.fixture(scope="module")
def js():
    return MQuickJS(timeout=30)


@pytest.mark.parametrize(
    "code,expected",
    [
        ("1 + 2", 3),
        ("'hello' + ' world'", "hello world"),
        ("true && false", False),
        ("Math.abs(-5)", 5),
        ("[1, 2, 3].length", 3),
        ("0.5 + 0.25", 0.75),
        ("undefined", None),
        ("null", None),
        ("JSON.stringify({a: [1, 2]})", '{"a":[1,2]}'),
    ],
)
def test_eval_values(js, code, expected):
    assert js.eval(code) == expected


def test_state_persists(js):
    js.eval("var x = 5")
    assert js.eval("x * 2") == 10


def test_functions_and_recursion(js):
    assert (
        js.eval(
            "function fib(n) { return n < 2 ? n : fib(n - 1) + fib(n - 2) } fib(12)"
        )
        == 144
    )


def test_errors(js):
    with pytest.raises(JSError, match="TypeError"):
        js.eval("null.x")
    with pytest.raises(JSError, match="SyntaxError"):
        js.eval("function (")
    assert js.eval("1 + 1") == 2


def test_no_host_access(js):
    # results come back as strings, so compare inside JavaScript
    assert js.eval("typeof require === 'undefined'") is True
    assert js.eval("typeof fetch === 'undefined'") is True


def test_js_memory_limit():
    js = MQuickJS(memory_limit=64 * 1024)
    with pytest.raises(JSError, match="out of memory"):
        js.eval("var a = []; while (true) { a.push([1, 2, 3, 4, 5, 6, 7, 8]) }")


def test_timeout():
    js = MQuickJS()
    start = time.monotonic()
    with pytest.raises(Timeout):
        js.eval("while (true) {}", timeout=0.3)
    assert time.monotonic() - start < 5


def test_fuel():
    js = MQuickJS(fuel=100_000)
    with pytest.raises(OutOfFuel):
        js.eval("var i = 0; while (true) { i++ }")
