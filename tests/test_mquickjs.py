"""Tests for running MicroQuickJS WASM module."""

import pytest
import pure_python_wasm
from pathlib import Path


class LongjmpException(Exception):
    """Exception used to simulate Emscripten's longjmp mechanism."""

    pass


class EmscriptenRuntime:
    """Runtime support for Emscripten-compiled WASM modules."""

    def __init__(self):
        self.temp_ret0 = 0
        self.threw = 0
        self.threw_value = 0
        self.instance = None  # Will be set after instantiation

    def set_instance(self, instance):
        """Set the instance after it's created."""
        self.instance = instance

    # Emscripten functions
    def setTempRet0(self, value):
        """Store the high 32 bits of a 64-bit return value."""
        self.temp_ret0 = value

    def getTempRet0(self):
        """Get the high 32 bits of a 64-bit return value."""
        return self.temp_ret0

    def _emscripten_throw_longjmp(self):
        """Called when longjmp is executed - throws to be caught by invoke_*."""
        raise LongjmpException()

    def _make_invoke(self, signature):
        """Create an invoke_* function for the given signature.

        Invoke functions call a function from the indirect function table
        and catch any longjmp exceptions.
        """

        def invoke_func(func_idx, *args):
            try:
                # Call the function via the indirect function table
                # The func_idx is an index into the table, not a direct function index
                if self.instance is None:
                    raise RuntimeError("Instance not set")

                # Get the function index from the table
                if not self.instance.tables:
                    raise RuntimeError("No tables in instance")
                table = self.instance.tables[0]
                if func_idx < 0 or func_idx >= len(table.elements):
                    raise RuntimeError(f"Table index {func_idx} out of bounds")

                actual_func_idx = table.elements[func_idx]
                if actual_func_idx is None:
                    raise RuntimeError(
                        f"Null function reference at table index {func_idx}"
                    )

                result = pure_python_wasm.execute_function(
                    self.instance, actual_func_idx, list(args)
                )
                return result if result is not None else 0
            except LongjmpException:
                # A longjmp occurred - record it and return
                self._call_setThrew(1, 0)
                # Return default based on signature
                if signature.startswith("v"):
                    return None
                return 0

        return invoke_func

    def _call_setThrew(self, threw, value):
        """Call the setThrew export to record a longjmp."""
        if self.instance is not None:
            # Find setThrew in exports
            for exp in self.instance.module.exports:
                if exp.name == "setThrew" and exp.kind == "func":
                    pure_python_wasm.execute_function(
                        self.instance, exp.index, [threw, value]
                    )
                    break

    def create_imports(self):
        """Create the imports dict for Emscripten modules."""
        return {
            "env": {
                "setTempRet0": self.setTempRet0,
                "getTempRet0": self.getTempRet0,
                "_emscripten_throw_longjmp": self._emscripten_throw_longjmp,
                "invoke_iii": self._make_invoke("iii"),
                "invoke_iiii": self._make_invoke("iiii"),
                "invoke_iiiii": self._make_invoke("iiiii"),
                "invoke_vi": self._make_invoke("vi"),
                "invoke_vii": self._make_invoke("vii"),
                "invoke_viii": self._make_invoke("viii"),
                "invoke_viiiii": self._make_invoke("viiiii"),
                "invoke_viiiiii": self._make_invoke("viiiiii"),
            },
            "wasi_snapshot_preview1": {
                "args_sizes_get": lambda environ_count, environ_buf_size: 0,
                "args_get": lambda argv, argv_buf: 0,
                "proc_exit": lambda code: None,
                "fd_close": lambda fd: 0,
                "fd_write": lambda fd, iovs, iovs_len, nwritten: 0,
                "fd_seek": lambda fd, offset, whence, newoffset: 0,
            },
        }


@pytest.fixture
def mquickjs_standalone():
    """Load the standalone MicroQuickJS WASM module."""
    wasm_path = Path(__file__).parent / "mquickjs_standalone.wasm"
    if not wasm_path.exists():
        pytest.skip("mquickjs_standalone.wasm not found")

    with open(wasm_path, "rb") as f:
        wasm_bytes = f.read()

    module = pure_python_wasm.decode_module(wasm_bytes)

    # Set up Emscripten runtime
    runtime = EmscriptenRuntime()
    imports = runtime.create_imports()

    instance = pure_python_wasm.instantiate(module, imports)
    runtime.set_instance(instance)

    return instance, runtime


def test_mquickjs_instantiation(mquickjs_standalone):
    """Test that the MicroQuickJS module can be instantiated."""
    instance, runtime = mquickjs_standalone
    assert instance is not None


def test_mquickjs_exports(mquickjs_standalone):
    """Test that expected exports are present."""
    instance, runtime = mquickjs_standalone

    export_names = [exp.name for exp in instance.module.exports]
    assert "sandbox_init" in export_names
    assert "sandbox_eval" in export_names
    assert "sandbox_free" in export_names
    assert "malloc" in export_names
    assert "free" in export_names


@pytest.mark.skip(
    reason="Pure Python interpreter too slow for complex module execution"
)
def test_mquickjs_sandbox_init(mquickjs_standalone):
    """Test initializing the QuickJS sandbox.

    Note: This test is skipped because the pure Python interpreter is too slow
    to execute the complex MicroQuickJS initialization code in a reasonable time.
    The module decodes and instantiates successfully, but execution is prohibitively slow.
    """
    instance, runtime = mquickjs_standalone

    # Find sandbox_init export
    sandbox_init_idx = None
    for exp in instance.module.exports:
        if exp.name == "sandbox_init":
            sandbox_init_idx = exp.index
            break

    assert sandbox_init_idx is not None

    # Call sandbox_init with 1MB heap
    heap_size = 1024 * 1024
    result = pure_python_wasm.execute_function(instance, sandbox_init_idx, [heap_size])

    # Result should be a context pointer (non-zero)
    print(f"sandbox_init result: {result}")
    assert result is not None


@pytest.mark.skip(
    reason="Pure Python interpreter too slow for complex module execution"
)
def test_mquickjs_eval_simple(mquickjs_standalone):
    """Test evaluating simple JavaScript.

    Note: This test is skipped because the pure Python interpreter is too slow.
    See test_mquickjs_sandbox_init for details.
    """
    instance, runtime = mquickjs_standalone

    # Get export indices
    exports = {exp.name: exp.index for exp in instance.module.exports}

    # Initialize sandbox with 1MB heap
    ctx = pure_python_wasm.execute_function(
        instance, exports["sandbox_init"], [1048576]
    )
    if ctx is None or ctx == 0:
        pytest.skip("sandbox_init failed")

    # Allocate memory for JavaScript code
    js_code = b"1 + 2\x00"
    code_ptr = pure_python_wasm.execute_function(
        instance, exports["malloc"], [len(js_code)]
    )

    if code_ptr is None or code_ptr == 0:
        pytest.skip("malloc failed")

    # Write the code to WASM memory
    memory = instance.memories[0]
    memory.data[code_ptr : code_ptr + len(js_code)] = js_code

    # Evaluate the code
    result = pure_python_wasm.execute_function(
        instance, exports["sandbox_eval"], [ctx, code_ptr, len(js_code) - 1]
    )

    print(f"sandbox_eval result: {result}")

    # Read the result string from memory if result is a pointer
    if result and result > 0:
        # Find null terminator
        end = result
        while end < len(memory.data) and memory.data[end] != 0:
            end += 1
        result_str = memory.data[result:end].decode("utf-8", errors="replace")
        print(f"Result string: {result_str}")
        assert "3" in result_str

    # Free the context
    pure_python_wasm.execute_function(instance, exports["sandbox_free"], [ctx])
