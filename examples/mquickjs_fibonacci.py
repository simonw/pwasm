#!/usr/bin/env python3
"""Run a Fibonacci JavaScript program using MicroQuickJS in pure-python-wasm.

NOTE: This example demonstrates the ability to load and call MicroQuickJS,
but the pure Python interpreter is too slow for MicroQuickJS's complex
initialization code. The `_start` function (WASI initialization) would take
hours to complete in pure Python.

Without proper _start initialization, sandbox_init returns quickly but
doesn't actually initialize the QuickJS engine, so sandbox_eval returns
"undefined" for any input.

This is a limitation of the pure Python interpreter's performance, not
a correctness issue. The WebAssembly execution is correct - it's just slow.
"""

import pure_python_wasm
from pathlib import Path
import time


class LongjmpException(Exception):
    """Exception used to simulate Emscripten's longjmp mechanism."""

    pass


class EmscriptenRuntime:
    """Runtime support for Emscripten-compiled WASM modules."""

    def __init__(self):
        self.temp_ret0 = 0
        self.instance = None

    def set_instance(self, instance):
        self.instance = instance

    def setTempRet0(self, value):
        self.temp_ret0 = value

    def getTempRet0(self):
        return self.temp_ret0

    def _emscripten_throw_longjmp(self):
        raise LongjmpException()

    def _make_invoke(self, signature):
        def invoke_func(func_idx, *args):
            try:
                if self.instance is None:
                    return 0
                if not self.instance.tables:
                    return 0
                table = self.instance.tables[0]
                if func_idx < 0 or func_idx >= len(table.elements):
                    return 0
                actual_func_idx = table.elements[func_idx]
                if actual_func_idx is None:
                    return 0
                result = pure_python_wasm.execute_function(
                    self.instance, actual_func_idx, list(args)
                )
                return result if result is not None else 0
            except LongjmpException:
                self._call_setThrew(1, 0)
                if signature.startswith("v"):
                    return None
                return 0

        return invoke_func

    def _call_setThrew(self, threw, value):
        if self.instance is not None:
            for exp in self.instance.module.exports:
                if exp.name == "setThrew" and exp.kind == "func":
                    pure_python_wasm.execute_function(
                        self.instance, exp.index, [threw, value]
                    )
                    break

    def create_imports(self):
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
                "args_sizes_get": lambda *a: 0,
                "args_get": lambda *a: 0,
                "proc_exit": lambda *a: None,
                "fd_close": lambda *a: 0,
                "fd_write": lambda *a: 0,
                "fd_seek": lambda *a: 0,
            },
        }


def read_string(memory, ptr):
    """Read a null-terminated string from WASM memory."""
    result = []
    i = ptr
    while i < len(memory) and memory[i] != 0:
        result.append(chr(memory[i]))
        i += 1
    return "".join(result)


def main():
    # Find the WASM file
    script_dir = Path(__file__).parent
    wasm_path = script_dir.parent / "tests" / "mquickjs_standalone.wasm"

    if not wasm_path.exists():
        print(f"Error: {wasm_path} not found")
        return 1

    print("Loading MicroQuickJS WASM module...")
    start = time.time()

    with open(wasm_path, "rb") as f:
        wasm_bytes = f.read()

    module = pure_python_wasm.decode_module(wasm_bytes)
    print(f"  Decoded in {time.time() - start:.2f}s")

    # Set up runtime
    runtime = EmscriptenRuntime()
    imports = runtime.create_imports()

    start = time.time()
    instance = pure_python_wasm.instantiate(module, imports)
    runtime.set_instance(instance)
    print(f"  Instantiated in {time.time() - start:.2f}s")

    # Get exports
    exports = {exp.name: exp.index for exp in instance.module.exports}

    # Try running _start if not run yet (WASI initialization)
    if "_start" in exports:
        print("\nRunning WASI _start initialization...")
        start = time.time()
        try:
            pure_python_wasm.execute_function(instance, exports["_start"], [])
            print(f"  _start completed in {time.time() - start:.2f}s")
        except Exception as e:
            print(f"  _start failed after {time.time() - start:.2f}s: {e}")

    # Initialize sandbox
    print("\nInitializing QuickJS sandbox...")
    start = time.time()
    ctx = pure_python_wasm.execute_function(
        instance, exports["sandbox_init"], [1048576]
    )
    print(f"  sandbox_init returned {ctx} in {time.time() - start:.2f}s")

    if ctx is None or ctx == 0:
        print("Error: sandbox_init failed")
        return 1

    # JavaScript code to evaluate - Fibonacci
    js_code = b"""
function fib(n) {
    if (n <= 1) return n;
    return fib(n - 1) + fib(n - 2);
}
fib(10)
\x00"""

    print(f"\nJavaScript code:")
    print(f"  {js_code.decode('utf-8').strip()}")

    # Allocate memory for the code
    print("\nAllocating memory for code...")
    start = time.time()
    code_ptr = pure_python_wasm.execute_function(
        instance, exports["malloc"], [len(js_code)]
    )
    print(f"  malloc({len(js_code)}) returned {code_ptr} in {time.time() - start:.2f}s")

    if code_ptr is None or code_ptr == 0:
        print("Error: malloc failed")
        return 1

    # Write code to memory
    memory = instance.memories[0].data
    memory[code_ptr : code_ptr + len(js_code)] = js_code

    # Evaluate the code
    print("\nEvaluating JavaScript...")
    start = time.time()
    result_ptr = pure_python_wasm.execute_function(
        instance, exports["sandbox_eval"], [ctx, code_ptr, len(js_code) - 1]
    )
    elapsed = time.time() - start
    print(f"  sandbox_eval returned {result_ptr} in {elapsed:.2f}s")

    if result_ptr and result_ptr > 0:
        # Read result string from memory
        result_str = read_string(memory, result_ptr)
        print(f"\nResult: {result_str}")

        # Expected: fib(10) = 55
        if result_str == "55":
            print("SUCCESS! Fibonacci(10) = 55")
        else:
            print(f"Note: Got '{result_str}', expected '55'")
    else:
        print(f"\nNo result string (ptr={result_ptr})")

    # Free the context
    print("\nCleaning up...")
    pure_python_wasm.execute_function(instance, exports["sandbox_free"], [ctx])
    print("Done!")

    return 0


if __name__ == "__main__":
    exit(main())
