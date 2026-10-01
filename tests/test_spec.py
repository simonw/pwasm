"""Run the WebAssembly spec test suite (tests/spec/*.wast).

Files listed in KNOWN_FAILURES are expected to fail; the xfail is strict,
so a file that starts passing must be removed from the list.
"""

from pathlib import Path

import pytest

from spec_runner import run_wast

pytestmark = pytest.mark.usefixtures("each_mode")

SPEC_DIR = Path(__file__).parent / "spec"

KNOWN_FAILURES: set[str] = set()


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
