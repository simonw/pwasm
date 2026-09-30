"""MicroPython compiled to WebAssembly, running inside pwasm."""

import time

import pytest

from pwasm import OutOfFuel, Timeout
from pwasm.guests import MicroPython, PythonError


@pytest.fixture(scope="module")
def mp():
    return MicroPython(max_memory=32 << 20, timeout=60)


def test_exec(mp):
    assert (
        mp.exec("print('hi', 1 + 2, [x * x for x in range(4)])")
        == "hi 3 [0, 1, 4, 9]\n"
    )
    assert mp.exec("x = 5") == ""
    assert mp.exec("print(x * 2)") == "10\n"  # globals persist between execs


def test_exceptions(mp):
    out = mp.exec(
        "try:\n    1/0\nexcept ZeroDivisionError as e:\n    print('caught', e)"
    )
    assert out == "caught divide by zero\n"
    with pytest.raises(PythonError, match="ValueError: boom"):
        mp.exec("raise ValueError('boom')")
    assert mp.sjlj.unwinds >= 2
    assert mp.exec("print('still alive')") == "still alive\n"


def test_json_and_classes(mp):
    out = mp.exec(
        "import json\n"
        "class Point:\n"
        "    def __init__(self, x, y):\n"
        "        self.x, self.y = x, y\n"
        "    def __repr__(self):\n"
        "        return 'Point(%d, %d)' % (self.x, self.y)\n"
        "print(Point(1, 2), json.dumps({'a': [1, 2.5, None]}))"
    )
    assert out == 'Point(1, 2) {"a": [1, 2.5, null]}\n'


def test_host_functions(mp):
    mp.register("add", lambda a, b: a + b)

    @mp.function
    def info():
        return {"answer": 42, "items": [1, "two"]}

    out = mp.exec(
        "import host\nprint(host.call('add', 40, 2))\nprint(host.call('info')['items'][1])"
    )
    assert out == "42\ntwo\n"
    with pytest.raises(PythonError, match="no host function"):
        mp.exec("import host\nhost.call('nope')")
    out = mp.exec(
        "import host\ntry:\n    host.call('nope')\nexcept RuntimeError as e:\n    print('caught', e)"
    )
    assert out.startswith("caught NameError")


def test_no_filesystem_or_os(mp):
    with pytest.raises(PythonError, match="OSError"):
        mp.exec("open('/etc/passwd')")
    with pytest.raises(PythonError, match="ImportError"):
        mp.exec("import os")


def test_unbounded_recursion_is_caught_in_guest(mp):
    out = mp.exec(
        "def f():\n    return f() + 1\ntry:\n    f()\nexcept RuntimeError as e:\n    print('caught:', e)"
    )
    assert out == "caught: maximum recursion depth exceeded\n"
    assert mp.exec("print(sum(range(10)))") == "45\n"


def test_timeout():
    mp = MicroPython()
    start = time.monotonic()
    with pytest.raises(Timeout):
        mp.exec("while True:\n    pass", timeout=0.5)
    assert time.monotonic() - start < 5


def test_fuel():
    mp = MicroPython(fuel=200_000)
    with pytest.raises(OutOfFuel):
        mp.exec("x = 0\nwhile True:\n    x += 1")


def test_memory_limit():
    # (bytearray rather than 'x' * n: MicroPython builds repeated strings
    # one memcpy per repetition, which is slow to interpret)
    mp = MicroPython(max_memory=2 << 20)
    with pytest.raises(PythonError, match="MemoryError"):
        mp.exec("a = []\nwhile True:\n    a.append(bytearray(65536))")
    assert mp.memory_size <= 2 << 20
    assert mp.exec("a = None\nprint('alive')") == "alive\n"
