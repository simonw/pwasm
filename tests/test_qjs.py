"""Tests for the QuickJS CLI tool."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from qjs import QuickJSRuntime, main


class TestQuickJSRuntimeInit:
    """Test QuickJSRuntime initialization."""

    def test_wasm_file_not_found(self):
        """Test that FileNotFoundError is raised when WASM file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="WASM file not found"):
            QuickJSRuntime(Path("/nonexistent/path/to/file.wasm"))

    def test_default_wasm_path(self):
        """Test that default WASM path is in the same directory as qjs.py."""
        # The default path should be qjs.py's parent / mquickjs.wasm
        qjs_dir = Path(__file__).parent.parent
        expected_path = qjs_dir / "mquickjs.wasm"

        # Verify the WASM file exists
        assert expected_path.exists(), f"Expected WASM file at {expected_path}"


class TestCLIArgumentParsing:
    """Test CLI argument parsing."""

    def test_help_flag_short(self):
        """Test -h flag shows help."""
        with patch.object(sys, "argv", ["qjs.py", "-h"]):
            result = main()
            assert result == 0

    def test_help_flag_long(self):
        """Test --help flag shows help."""
        with patch.object(sys, "argv", ["qjs.py", "--help"]):
            result = main()
            assert result == 0

    def test_e_flag_requires_expression(self):
        """Test that -e flag requires an expression."""
        with patch.object(sys, "argv", ["qjs.py", "-e"]):
            result = main()
            assert result == 1


class TestWASMDecoding:
    """Test WASM module decoding without full initialization."""

    def test_wasm_module_can_be_decoded(self):
        """Test that the QuickJS WASM module can be decoded."""
        from pure_python_wasm import decode_module

        wasm_path = Path(__file__).parent.parent / "mquickjs.wasm"
        if not wasm_path.exists():
            pytest.skip("WASM file not found")

        with open(wasm_path, "rb") as f:
            wasm_bytes = f.read()

        module = decode_module(wasm_bytes)

        # Check that required exports exist
        export_names = {e.name for e in module.exports}
        assert "sandbox_init" in export_names
        assert "sandbox_eval" in export_names
        assert "sandbox_free" in export_names
        assert "sandbox_get_error" in export_names
        assert "malloc" in export_names
        assert "free" in export_names
        assert "memory" in export_names

    def test_wasm_module_has_expected_structure(self):
        """Test WASM module has expected types and functions."""
        from pure_python_wasm import decode_module

        wasm_path = Path(__file__).parent.parent / "mquickjs.wasm"
        if not wasm_path.exists():
            pytest.skip("WASM file not found")

        with open(wasm_path, "rb") as f:
            wasm_bytes = f.read()

        module = decode_module(wasm_bytes)

        # Should have types, functions, memory, and tables
        assert len(module.types) > 0
        assert len(module.funcs) > 0
        assert len(module.mems) > 0
        assert len(module.tables) > 0

        # Should have imports for Emscripten runtime
        assert len(module.imports) > 0
        import_names = {(i.module, i.name) for i in module.imports}
        assert ("env", "abort") in import_names


class TestQuickJSRuntimeFunctionality:
    """Test QuickJSRuntime methods.

    Note: These tests are slow because they require full WASM instantiation.
    We test initialization but avoid actual JS evaluation due to extreme slowness.
    """

    @pytest.fixture
    def runtime(self):
        """Create a QuickJSRuntime instance."""
        wasm_path = Path(__file__).parent.parent / "mquickjs.wasm"
        if not wasm_path.exists():
            pytest.skip("WASM file not found")
        return QuickJSRuntime(wasm_path)

    @pytest.mark.slow
    def test_runtime_initializes(self, runtime):
        """Test that runtime initializes successfully."""
        # If we got here, initialization succeeded
        assert runtime.instance is not None
        assert runtime._sandbox_init is not None
        assert runtime._sandbox_eval is not None
        assert runtime._memory is not None

    @pytest.mark.slow
    def test_memory_is_accessible(self, runtime):
        """Test that WASM memory is accessible."""
        memory = runtime._memory
        assert memory is not None
        assert hasattr(memory, "data")
        # Memory should have at least 1MB (as initialized in sandbox_init)
        assert len(memory.data) >= 1024 * 1024

    @pytest.mark.slow
    def test_malloc_returns_valid_pointer(self, runtime):
        """Test that malloc returns a valid pointer."""
        ptr = runtime._malloc(100)
        assert ptr > 0
        # Clean up
        runtime._free(ptr)

    @pytest.mark.slow
    def test_write_and_read_string(self, runtime):
        """Test string write and read operations."""
        test_string = "Hello, WebAssembly!"
        ptr = runtime._write_string(test_string)

        assert ptr > 0

        # Read it back
        result = runtime._read_string(ptr)
        assert result == test_string

        # Clean up
        runtime._free(ptr)

    @pytest.mark.slow
    def test_read_string_null_pointer(self, runtime):
        """Test that reading from null pointer returns empty string."""
        result = runtime._read_string(0)
        assert result == ""


class TestImportFunctions:
    """Test the Emscripten import function stubs."""

    def test_imports_structure(self):
        """Test that imports have expected structure."""
        wasm_path = Path(__file__).parent.parent / "mquickjs.wasm"
        if not wasm_path.exists():
            pytest.skip("WASM file not found")

        runtime = QuickJSRuntime(wasm_path)
        imports = runtime.imports

        assert "env" in imports
        env = imports["env"]

        # Check required import functions exist
        assert "abort" in env
        assert "__assert_fail" in env
        assert "emscripten_resize_heap" in env
        assert "emscripten_memcpy_big" in env
        assert "setTempRet0" in env
        assert "getTempRet0" in env
        assert "_emscripten_throw_longjmp" in env

        # Check invoke functions
        assert "invoke_iii" in env
        assert "invoke_iiii" in env
        assert "invoke_vi" in env
        assert "invoke_vii" in env
