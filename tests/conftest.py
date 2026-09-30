import pytest

import pwasm.codegen
import pwasm.executor

# Every way pwasm can run code: the interpreter, compiled to structured
# Python, compiled with every chain of blocks lowered to a dispatch loop,
# and compiled with every function forced into a state machine
MODES = ["interpret", "compile", "chains", "state-machine"]


@pytest.fixture(scope="module", params=MODES)
def each_mode(request):
    """Run the tests of a module once per mode (sets the default mode)."""
    with pytest.MonkeyPatch.context() as mp:
        mode = request.param
        if mode == "state-machine":
            mp.setattr(pwasm.codegen, "FORCE_STATE_MACHINE", True)
            mode = "compile"
        elif mode == "chains":
            mp.setattr(pwasm.codegen, "FORCE_CHAINS", True)
            mode = "compile"
        mp.setattr(pwasm.executor, "DEFAULT_MODE", mode)
        yield request.param
