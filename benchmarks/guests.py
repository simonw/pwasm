"""Time the bundled guest interpreters running typical small workloads.

uv run python benchmarks/guests.py

Start-up is timed in new processes, with an empty on-disk code cache
("cold") and then with the cache that run filled ("warm"). Workloads are
timed on their first run (which includes compiling the functions they
use) and at their best over three more runs.
"""

import os
import subprocess
import sys
import tempfile
import time

import pwasm.codegen
from pwasm.guests import MicroPython, MQuickJS, QuickJS

GUESTS = {
    "MicroPython": (MicroPython, "exec"),
    "QuickJS (quickjs-ng)": (QuickJS, "eval"),
    "Micro QuickJS": (MQuickJS, "eval"),
}

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

WORKLOADS = {
    "MicroPython": PYTHON,
    "QuickJS (quickjs-ng)": JAVASCRIPT,
    "Micro QuickJS": JAVASCRIPT,
}

# run in a new process: start a guest and evaluate its first workload
START = """
import sys, time
sys.path.insert(0, {here!r})
from guests import GUESTS, WORKLOADS
cls, method = GUESTS[{name!r}]
start = time.perf_counter()
guest = cls()
started = time.perf_counter()
getattr(guest, method)(WORKLOADS[{name!r}][0][1])
print(started - start, time.perf_counter() - started)
"""


def start_up(name, cache_dir):
    code = START.format(here=os.path.dirname(os.path.abspath(__file__)), name=name)
    env = dict(os.environ, PWASM_CACHE_DIR=cache_dir)
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    if out.returncode:
        raise RuntimeError(out.stderr)
    return [float(x) for x in out.stdout.split()]


def ms(seconds):
    return f"{seconds * 1000:6.0f} ms"


def report(name):
    cls, method = GUESTS[name]
    print(name)
    with tempfile.TemporaryDirectory() as cache_dir:
        for state in ("cold", "warm"):
            startup, first = start_up(name, cache_dir)
            print(
                f"  {'start up + first eval (' + state + ')':34s} {ms(startup + first)}"
            )
    guest = cls()
    run = getattr(guest, method)
    for label, code in WORKLOADS[name]:
        start = time.perf_counter()
        result = run(code)
        first = time.perf_counter() - start
        best = first
        for _ in range(3):
            start = time.perf_counter()
            run(code)
            best = min(best, time.perf_counter() - start)
        result = str(result).strip()[:30]
        print(f"  {label:22s} first {ms(first)}, then {ms(best)}   -> {result!r}")


if __name__ == "__main__":
    # workloads here are timed without the disk cache, so that their first
    # run always includes compiling
    pwasm.codegen.CACHE_DIR = None
    print(f"Python {sys.version.split()[0]} ({sys.implementation.name})\n")
    for name in GUESTS:
        report(name)
