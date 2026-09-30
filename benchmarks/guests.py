"""Time the bundled guest interpreters running typical small workloads.

uv run python benchmarks/guests.py
"""

import sys
import time

from pwasm.guests import MicroPython, MQuickJS, QuickJS

PYTHON = [
    ("print(1 + 1)", "print(1 + 1)"),
    (
        "json round trip",
        "import json\nd = {'a': [1, 2, 3], 'b': 'hello'}\ns = json.dumps(d)\nprint(s, json.loads(s)['b'].upper())",
    ),
    (
        "100 x try/except",
        "n = 0\nfor i in range(100):\n    try:\n        1/0\n    except ZeroDivisionError:\n        n += 1\nprint(n)",
    ),
    (
        "fib(15)",
        "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)\nprint(fib(15))",
    ),
    ("1,000 iteration loop", "t = 0\nfor i in range(1000):\n    t += i\nprint(t)"),
]

JAVASCRIPT = [
    ("1 + 1", "1 + 1"),
    ("JSON.stringify", "JSON.stringify({a: [1, 2, 3], b: 'hello'.toUpperCase()})"),
    (
        "fib(15)",
        "function fib(n) { return n < 2 ? n : fib(n - 1) + fib(n - 2) } fib(15)",
    ),
    ("1,000 iteration loop", "var t = 0; for (var i = 0; i < 1000; i++) t += i; t"),
]


def timed(fn):
    start = time.perf_counter()
    result = fn()
    return time.perf_counter() - start, result


def report(name, make, run, workloads):
    elapsed, guest = timed(make)
    print(f"{name}")
    print(f"  {'startup':24s} {elapsed * 1000:8.0f} ms")
    for label, code in workloads:
        elapsed, result = timed(lambda: run(guest, code))
        print(
            f"  {label:24s} {elapsed * 1000:8.0f} ms   -> {str(result).strip()[:40]!r}"
        )


if __name__ == "__main__":
    print(f"Python {sys.version.split()[0]} ({sys.implementation.name})\n")
    report("MicroPython", MicroPython, lambda g, c: g.exec(c), PYTHON)
    report("QuickJS (quickjs-ng)", QuickJS, lambda g, c: g.eval(c), JAVASCRIPT)
    report("Micro QuickJS", MQuickJS, lambda g, c: g.eval(c), JAVASCRIPT)
