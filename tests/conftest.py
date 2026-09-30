import pytest

import pwasm.codegen
import pwasm.executor

# Every way pwasm can run code: the interpreter, compiled to structured
# Python, and compiled with every function forced into a state machine
MODES = ["interpret", "compile", "state-machine"]


@pytest.fixture(scope="module", params=MODES)
def each_mode(request):
    """Run the tests of a module once per mode (sets the default mode)."""
    with pytest.MonkeyPatch.context() as mp:
        mode = request.param
        if mode == "state-machine":
            mp.setattr(pwasm.codegen, "FORCE_STATE_MACHINE", True)
            mode = "compile"
        mp.setattr(pwasm.executor, "DEFAULT_MODE", mode)
        yield request.param
