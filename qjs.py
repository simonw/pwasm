#!/usr/bin/env python3
"""
QuickJS JavaScript interpreter running via pure Python WebAssembly runtime.

NOTE: This pure Python WASM interpreter is educational and demonstrates the
concept of running WebAssembly in pure Python. However, it is EXTREMELY SLOW
for complex WASM modules like the QuickJS JavaScript engine, which requires
millions of instruction executions. For production use, consider native
WASM runtimes like wasmer or wasmtime.

This tool provides:
- Interactive REPL mode for running JavaScript line by line
- Direct execution mode for running JavaScript code from command line

Usage:
    uv run qjs.py                    # Start interactive REPL
    uv run qjs.py 'js code here'     # Execute JavaScript directly
    uv run qjs.py -e 'js code'       # Execute JavaScript directly (explicit flag)
"""

import sys
from pathlib import Path

from pure_python_wasm import decode_module, instantiate


class QuickJSRuntime:
    """Manages the QuickJS WebAssembly runtime."""

    def __init__(self, wasm_path: Path | None = None):
        """Initialize the QuickJS runtime.

        Args:
            wasm_path: Path to the mquickjs.wasm file. If None, looks in
                       the same directory as this script.
        """
        if wasm_path is None:
            wasm_path = Path(__file__).parent / "mquickjs.wasm"

        if not wasm_path.exists():
            raise FileNotFoundError(f"WASM file not found: {wasm_path}")

        # Load and decode the WASM module
        with open(wasm_path, "rb") as f:
            wasm_bytes = f.read()

        self.module = decode_module(wasm_bytes)

        # Provide Emscripten runtime imports
        self._temp_ret0 = 0
        self.imports = self._create_imports()

        # Instantiate the module
        self.instance = instantiate(self.module, self.imports)

        # Get exported functions
        self._sandbox_init = self.instance.exports.sandbox_init
        self._sandbox_free = self.instance.exports.sandbox_free
        self._sandbox_eval = self.instance.exports.sandbox_eval
        self._sandbox_get_error = self.instance.exports.sandbox_get_error
        self._malloc = self.instance.exports.malloc
        self._free = self.instance.exports.free
        self._memory = self.instance.exports.memory

        # Initialize the sandbox with 1MB of memory
        result = self._sandbox_init(1024 * 1024)
        if not result:
            raise RuntimeError("Failed to initialize QuickJS sandbox")

    def _create_imports(self) -> dict:
        """Create the import functions required by Emscripten."""

        def abort():
            raise RuntimeError("abort() called")

        def assert_fail(condition, filename, line, func):
            raise RuntimeError(f"Assertion failed at line {line}")

        def emscripten_resize_heap(requested_size):
            return 0

        def emscripten_memcpy_big(dest, src, num):
            mem = self.instance.memories[0].data
            mem[dest : dest + num] = bytes(mem[src : src + num])
            return dest

        def set_temp_ret0(val):
            self._temp_ret0 = val

        def get_temp_ret0():
            return self._temp_ret0

        def emscripten_throw_longjmp():
            raise RuntimeError("longjmp called")

        def invoke_iii(index, a1, a2):
            try:
                table = self.instance.tables[0]
                func_idx = table.elements[index]
                if func_idx is not None:
                    from pure_python_wasm.executor import execute_function

                    return execute_function(self.instance, func_idx, [a1, a2])
            except Exception:
                pass
            return 0

        def invoke_iiii(index, a1, a2, a3):
            try:
                table = self.instance.tables[0]
                func_idx = table.elements[index]
                if func_idx is not None:
                    from pure_python_wasm.executor import execute_function

                    return execute_function(self.instance, func_idx, [a1, a2, a3])
            except Exception:
                pass
            return 0

        def invoke_iiiii(index, a1, a2, a3, a4):
            try:
                table = self.instance.tables[0]
                func_idx = table.elements[index]
                if func_idx is not None:
                    from pure_python_wasm.executor import execute_function

                    return execute_function(self.instance, func_idx, [a1, a2, a3, a4])
            except Exception:
                pass
            return 0

        def invoke_vi(index, a1):
            try:
                table = self.instance.tables[0]
                func_idx = table.elements[index]
                if func_idx is not None:
                    from pure_python_wasm.executor import execute_function

                    execute_function(self.instance, func_idx, [a1])
            except Exception:
                pass

        def invoke_vii(index, a1, a2):
            try:
                table = self.instance.tables[0]
                func_idx = table.elements[index]
                if func_idx is not None:
                    from pure_python_wasm.executor import execute_function

                    execute_function(self.instance, func_idx, [a1, a2])
            except Exception:
                pass

        def invoke_viii(index, a1, a2, a3):
            try:
                table = self.instance.tables[0]
                func_idx = table.elements[index]
                if func_idx is not None:
                    from pure_python_wasm.executor import execute_function

                    execute_function(self.instance, func_idx, [a1, a2, a3])
            except Exception:
                pass

        def invoke_viiiii(index, a1, a2, a3, a4, a5):
            try:
                table = self.instance.tables[0]
                func_idx = table.elements[index]
                if func_idx is not None:
                    from pure_python_wasm.executor import execute_function

                    execute_function(self.instance, func_idx, [a1, a2, a3, a4, a5])
            except Exception:
                pass

        def invoke_viiiiii(index, a1, a2, a3, a4, a5, a6):
            try:
                table = self.instance.tables[0]
                func_idx = table.elements[index]
                if func_idx is not None:
                    from pure_python_wasm.executor import execute_function

                    execute_function(self.instance, func_idx, [a1, a2, a3, a4, a5, a6])
            except Exception:
                pass

        imports = {
            "env": {
                "abort": abort,
                "__assert_fail": assert_fail,
                "emscripten_resize_heap": emscripten_resize_heap,
                "emscripten_memcpy_big": emscripten_memcpy_big,
                "setTempRet0": set_temp_ret0,
                "getTempRet0": get_temp_ret0,
                "_emscripten_throw_longjmp": emscripten_throw_longjmp,
                "invoke_iii": invoke_iii,
                "invoke_iiii": invoke_iiii,
                "invoke_iiiii": invoke_iiiii,
                "invoke_vi": invoke_vi,
                "invoke_vii": invoke_vii,
                "invoke_viii": invoke_viii,
                "invoke_viiiii": invoke_viiiii,
                "invoke_viiiiii": invoke_viiiiii,
            },
        }

        return imports

    def _read_string(self, ptr: int) -> str:
        """Read a null-terminated string from WASM memory."""
        if ptr == 0:
            return ""

        mem = self._memory.data
        end = ptr
        while end < len(mem) and mem[end] != 0:
            end += 1

        return bytes(mem[ptr:end]).decode("utf-8", errors="replace")

    def _write_string(self, s: str) -> int:
        """Write a string to WASM memory and return its pointer."""
        encoded = s.encode("utf-8") + b"\x00"
        ptr = self._malloc(len(encoded))
        if ptr == 0:
            raise MemoryError("Failed to allocate memory for string")

        mem = self._memory.data
        mem[ptr : ptr + len(encoded)] = encoded
        return ptr

    def eval(self, code: str) -> str | None:
        """Evaluate JavaScript code and return the result.

        Args:
            code: JavaScript code to evaluate

        Returns:
            The result as a string, or None if there was an error.
            Call get_error() to get the error message.

        Note:
            This is EXTREMELY SLOW due to the pure Python WASM interpreter.
            Even simple expressions may take minutes to evaluate.
        """
        code_ptr = self._write_string(code)

        try:
            result_ptr = self._sandbox_eval(code_ptr)

            if result_ptr == 0:
                return None

            return self._read_string(result_ptr)

        finally:
            self._free(code_ptr)

    def get_error(self) -> str:
        """Get the last error message."""
        error_ptr = self._sandbox_get_error()
        if error_ptr == 0:
            return "Unknown error"
        return self._read_string(error_ptr)

    def reset(self):
        """Reset the sandbox to a clean state."""
        self._sandbox_free()
        result = self._sandbox_init(1024 * 1024)
        if not result:
            raise RuntimeError("Failed to reinitialize QuickJS sandbox")


def run_interactive():
    """Run the interactive REPL."""
    print("QuickJS JavaScript REPL (pure Python WebAssembly)")
    print()
    print("WARNING: This pure Python WASM interpreter is EXTREMELY SLOW.")
    print("Even simple expressions may take several minutes to evaluate.")
    print("This is intended as an educational demonstration, not for")
    print("production use.")
    print()
    print("Type JavaScript code and press Enter to execute.")
    print("Type 'exit' or 'quit' to exit, 'reset' to reset the sandbox.")
    print("-" * 60)

    print("Initializing QuickJS runtime (this may take a moment)...")
    try:
        runtime = QuickJSRuntime()
        print("Runtime initialized successfully!")
    except Exception as e:
        print(f"Error initializing runtime: {e}", file=sys.stderr)
        return 1

    while True:
        try:
            try:
                code = input("js> ")
            except EOFError:
                print()
                break

            code_lower = code.strip().lower()
            if code_lower in ("exit", "quit"):
                break
            if code_lower == "reset":
                runtime.reset()
                print("Sandbox reset.")
                continue
            if not code.strip():
                continue

            print("Evaluating (this may take a while)...")
            result = runtime.eval(code)

            if result is not None:
                print(result)
            else:
                error = runtime.get_error()
                print(f"Error: {error}", file=sys.stderr)

        except KeyboardInterrupt:
            print("\n(Use 'exit' to quit)")
            continue
        except Exception as e:
            print(f"Runtime error: {e}", file=sys.stderr)

    return 0


def run_code(code: str) -> int:
    """Execute JavaScript code directly and print the result."""
    print("WARNING: Pure Python WASM interpreter is EXTREMELY SLOW.", file=sys.stderr)
    print("Initializing QuickJS runtime...", file=sys.stderr)

    try:
        runtime = QuickJSRuntime()
    except Exception as e:
        print(f"Error initializing runtime: {e}", file=sys.stderr)
        return 1

    print("Evaluating (this may take several minutes)...", file=sys.stderr)
    result = runtime.eval(code)

    if result is not None:
        print(result)
        return 0
    else:
        error = runtime.get_error()
        print(f"Error: {error}", file=sys.stderr)
        return 1


def main():
    """Main entry point."""
    args = sys.argv[1:]

    if not args:
        return run_interactive()

    if args[0] == "-e":
        if len(args) < 2:
            print("Error: -e requires a JavaScript expression", file=sys.stderr)
            return 1
        code = args[1]
    elif args[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    else:
        code = " ".join(args)

    return run_code(code)


if __name__ == "__main__":
    sys.exit(main())
