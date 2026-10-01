"""quickjs-ng compiled to WebAssembly, running inside pwasm."""

import time

import pytest

from pwasm import OutOfFuel, Timeout
from pwasm.guests import JSError, QuickJS


@pytest.fixture(scope="module")
def js():
    return QuickJS(max_memory=32 << 20, timeout=60)


def test_eval_values(js):
    assert js.eval("1 + 2") == 3
    assert js.eval("({a: [1, 2, 3], b: 'hi'})") == {"a": [1, 2, 3], "b": "hi"}
    assert js.eval("'hello'.toUpperCase()") == "HELLO"
    assert js.eval("undefined") == "undefined"
    assert js.eval("[1.5, true, null]") == [1.5, True, None]


def test_state_persists_between_evals(js):
    js.eval("var counter = 10")
    js.eval("counter += 5")
    assert js.eval("counter") == 15


def test_output_and_host_functions(js):
    js.eval("console.log('hi', 42); print({x: 1})")
    assert js.take_output() == "hi 42\n[object Object]\n"

    @js.function
    def add(a, b):
        return a + b

    js.register("lookup", lambda key: {"key": key, "value": key[::-1]})
    assert js.eval("host.add(40, 2)") == 42
    assert js.eval("host.lookup('abc')") == {"key": "abc", "value": "cba"}
    with pytest.raises(JSError, match="no host function"):
        js.eval("host.missing()")
    # host exceptions become JS exceptions the guest can catch
    js.register("fail", lambda: 1 / 0)
    caught = js.eval("try { host.fail() } catch (e) { 'caught: ' + e.message }")
    assert caught.startswith("caught: ZeroDivisionError")


def test_exception(js):
    with pytest.raises(JSError, match="TypeError"):
        js.eval("null.x")


def test_no_filesystem_or_network(js):
    assert js.eval("typeof require") == "undefined"
    assert js.eval("typeof std") == "undefined"
    assert js.eval("typeof os") == "undefined"
    assert js.eval("typeof fetch") == "undefined"


def test_deep_recursion_is_a_catchable_range_error(js):
    out = js.eval(
        "let d = 0; function f() { d++; return f() + 1 }; try { f() } catch (e) { e.name }"
    )
    assert out == "RangeError"
    assert js.eval("d > 100") is True


def test_hard_timeout():
    js = QuickJS()
    start = time.monotonic()
    with pytest.raises(Timeout):
        js.eval("while (true) {}", timeout=0.3)
    assert time.monotonic() - start < 5
    assert js.eval("1 + 1") == 2  # the runtime happens to survive this


def test_soft_timeout():
    js = QuickJS()
    with pytest.raises(JSError, match="interrupted"):
        js.eval("while (true) {}", soft_timeout=0.2)
    assert js.eval("2 + 2") == 4


def test_fuel():
    js = QuickJS(fuel=100_000)
    with pytest.raises(OutOfFuel):
        js.eval("function fib(n){return n<2?n:fib(n-1)+fib(n-2)}; fib(30)")


def test_memory_limit():
    # (ArrayBuffer rather than 'x'.repeat(n): quickjs-ng fills repeated
    # strings a character at a time, which is slow to interpret)
    js = QuickJS(max_memory=8 << 20)
    with pytest.raises(JSError, match="out of memory"):
        js.eval("let a = []; while (true) { a.push(new ArrayBuffer(1 << 20)) }")
    assert js.memory_size <= 8 << 20
    assert js.eval("a = null; 1 + 1") == 2


def test_instances_share_the_decoded_module_and_compiled_code():
    first = QuickJS(mode="compile")
    second = QuickJS(mode="compile")
    assert first.sb.module is second.sb.module
    first.eval("1 + 1")
    second.eval("1 + 1")
    code_objects = first.sb.module._python_code
    assert code_objects and all(code is not None for code in code_objects.values())
