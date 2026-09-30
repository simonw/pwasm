"""Run the WebAssembly spec test suite (tests/spec/*.wast).

Files listed in KNOWN_FAILURES are expected to fail; the xfail is strict,
so a file that starts passing must be removed from the list.
"""

from pathlib import Path

import pytest

from spec_runner import run_wast

SPEC_DIR = Path(__file__).parent / "spec"

KNOWN_FAILURES = {
    "binary",
    "binary-leb128",
    "block",
    "br",
    "br_if",
    "br_table",
    "bulk",
    "call",
    "call_indirect",
    "comments",
    "conversions",
    "data",
    "elem",
    "endianness",
    "f32",
    "f32_bitwise",
    "f32_cmp",
    "f64",
    "f64_bitwise",
    "f64_cmp",
    "float_exprs",
    "float_literals",
    "float_misc",
    "func",
    "func_ptrs",
    "global",
    "if",
    "imports",
    "left-to-right",
    "linking",
    "load",
    "local_get",
    "local_set",
    "local_tee",
    "loop",
    "memory",
    "memory_copy",
    "memory_fill",
    "memory_grow",
    "memory_init",
    "names",
    "nop",
    "ref_func",
    "ref_is_null",
    "ref_null",
    "return",
    "select",
    "table",
    "table_copy",
    "table_fill",
    "table_get",
    "table_grow",
    "table_init",
    "table_set",
    "table_size",
    "traps",
    "unreachable",
}


def spec_files():
    for path in sorted(SPEC_DIR.glob("*.wast")):
        marks = []
        if path.stem in KNOWN_FAILURES:
            marks.append(pytest.mark.xfail(strict=True, reason="not implemented yet"))
        yield pytest.param(path, id=path.stem, marks=marks)


@pytest.mark.parametrize("path", list(spec_files()))
def test_spec(path):
    result = run_wast(path)
    if result.failures:
        shown = "\n".join(result.failures[:25])
        pytest.fail(f"{result.summary()}\n{shown}", pytrace=False)


def test_known_failures_exist():
    names = {p.stem for p in SPEC_DIR.glob("*.wast")}
    assert KNOWN_FAILURES <= names
