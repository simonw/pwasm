#!/usr/bin/env python3
"""Test the optimized MicroQuickJS WASM build.

This uses the franzenzenhofer/mquickjs-wasm build which has:
- Size-optimized code (-Os)
- Minified export names
- Simpler API than the original
"""

import pure_python_wasm
from pathlib import Path
import time


# Export name mapping from minified names
EXPORTS = {
    "q": "unknown",  # First export, likely module init
    "r": "_mquickjs_init",
    "s": "_mquickjs_cleanup",
    "t": "_mquickjs_clear_output",
    "u": "_mquickjs_get_output",
    "v": "_mquickjs_run",
    "w": "_mquickjs_reset",
    "x": "_mquickjs_version",
    "y": "_mquickjs_memory_size",
    "A": "_malloc",
    "B": "_free",
    "C": "_setThrew",
    "D": "__emscripten_stack_restore",
    "E": "__emscripten_stack_alloc",
    "F": "_emscripten_stack_get_current",
}


class LongjmpException(Exception):
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

    def _emscripten_throw_longjmp(self, *args):
        # longjmp takes jmp_buf and value
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
            except Exception:
                # Return default for other errors
                if signature.startswith("v"):
                    return None
                return 0

        return invoke_func

    def _call_setThrew(self, threw, value):
        if self.instance is not None:
            exports = {exp.name: exp.index for exp in self.instance.module.exports}
            if "C" in exports:  # setThrew is export 'C'
                try:
                    pure_python_wasm.execute_function(
                        self.instance, exports["C"], [threw, value]
                    )
                except Exception:
                    pass


def main():
    script_dir = Path(__file__).parent.parent
    wasm_path = script_dir / "tests" / "mquickjs_optimized.wasm"

    if not wasm_path.exists():
        print(f"Error: {wasm_path} not found")
        print("Download from: https://github.com/franzenzenhofer/mquickjs-wasm")
        return 1

    print("Loading optimized MicroQuickJS WASM module...")
    start = time.monotonic()
    with open(wasm_path, "rb") as f:
        wasm_bytes = f.read()

    module = pure_python_wasm.decode_module(wasm_bytes)
    decode_time = time.monotonic() - start
    print(f"  Decoded in {decode_time:.2f}s ({len(wasm_bytes):,} bytes)")
    print(f"  Functions: {len(module.funcs)}")
    print(f"  Imports: {len(module.imports)}")

    # Create Emscripten runtime
    runtime = EmscriptenRuntime()

    # Create imports for the optimized build
    # The "a" module contains env-like functions (minified names)
    imports = {
        "a": {
            "a": runtime.setTempRet0,  # setTempRet0
            "b": runtime.getTempRet0,  # getTempRet0
            "c": runtime._emscripten_throw_longjmp,  # _emscripten_throw_longjmp
            "d": runtime._make_invoke("iii"),  # invoke_iii
            "e": runtime._make_invoke("iiii"),  # invoke_iiii
            "f": runtime._make_invoke("iiiii"),  # invoke_iiiii
            "g": runtime._make_invoke("vi"),  # invoke_vi
            "h": runtime._make_invoke("vii"),  # invoke_vii
            "i": runtime._make_invoke("viii"),  # invoke_viii
            "j": runtime._make_invoke("viiiii"),  # invoke_viiiii
            "k": runtime._make_invoke("viiiiii"),  # invoke_viiiiii
            "l": lambda *a: 0,  # wasi: args_sizes_get
            "m": lambda *a: 0,  # wasi: args_get
            "n": lambda *a: None,  # wasi: proc_exit
            "o": lambda *a: 0,  # wasi: fd_close / fd_write / fd_seek
        }
    }

    print("\nInstantiating module...")
    start = time.monotonic()
    instance = pure_python_wasm.instantiate(module, imports)
    runtime.set_instance(instance)
    inst_time = time.monotonic() - start
    print(f"  Instantiated in {inst_time:.2f}s")

    # Get export indices by minified name
    exports = {exp.name: exp.index for exp in instance.module.exports}
    print(f"\nExports: {list(exports.keys())}")

    # Try to initialize MicroQuickJS
    print("\nCalling _mquickjs_init (export 'r')...")
    start = time.monotonic()
    try:
        result = pure_python_wasm.execute_function(instance, exports["r"], [])
        elapsed = time.monotonic() - start
        print(f"  _mquickjs_init returned {result} in {elapsed:.2f}s")
    except Exception as e:
        elapsed = time.monotonic() - start
        print(f"  Error after {elapsed:.2f}s: {e}")
        return 1

    # Get version
    print("\nCalling _mquickjs_version (export 'x')...")
    start = time.monotonic()
    try:
        result = pure_python_wasm.execute_function(instance, exports["x"], [])
        elapsed = time.monotonic() - start
        print(f"  _mquickjs_version returned {result} in {elapsed:.2f}s")

        # If result is a pointer, read the string from memory
        if result and result > 0:
            memory = instance.memories[0].data
            version = []
            i = result
            while i < len(memory) and memory[i] != 0:
                version.append(chr(memory[i]))
                i += 1
            if version:
                print(f"  Version string: {''.join(version)}")
    except Exception as e:
        elapsed = time.monotonic() - start
        print(f"  Error after {elapsed:.2f}s: {e}")

    # Try running simple JavaScript
    print("\nRunning JavaScript: 1 + 2")
    js_code = b"1 + 2\x00"

    # Allocate memory for the code
    malloc_idx = exports.get("A")
    if malloc_idx:
        try:
            code_ptr = pure_python_wasm.execute_function(
                instance, malloc_idx, [len(js_code)]
            )
            print(f"  malloc({len(js_code)}) returned {code_ptr}")
            if code_ptr and code_ptr > 0:
                # Write code to memory
                memory = instance.memories[0].data
                memory[code_ptr : code_ptr + len(js_code)] = js_code
                print(f"  Wrote code to memory at {code_ptr}")

                # Call _mquickjs_run
                run_idx = exports.get("v")
                if run_idx:
                    print(f"  Calling _mquickjs_run({code_ptr}, {len(js_code) - 1})...")
                    start = time.monotonic()
                    try:
                        result = pure_python_wasm.execute_function(
                            instance, run_idx, [code_ptr, len(js_code) - 1]
                        )
                        elapsed = time.monotonic() - start
                        print(f"  _mquickjs_run returned {result} in {elapsed:.2f}s")
                    except LongjmpException:
                        elapsed = time.monotonic() - start
                        print(
                            f"  _mquickjs_run: caught longjmp after {elapsed:.2f}s (this is expected)"
                        )
                        # After longjmp, we need to handle the setjmp return
                        # For now, just continue and get output
                        result = None
                    except Exception as e:
                        elapsed = time.monotonic() - start
                        print(f"  _mquickjs_run error after {elapsed:.2f}s: {e}")
                        import traceback

                        traceback.print_exc()
                        return 1

                    # Get output
                    get_output_idx = exports.get("u")
                    if get_output_idx:
                        print("  Getting output...")
                        try:
                            output_ptr = pure_python_wasm.execute_function(
                                instance, get_output_idx, []
                            )
                            print(f"  _mquickjs_get_output returned {output_ptr}")
                            if output_ptr and output_ptr > 0:
                                output = []
                                i = output_ptr
                                while i < len(memory) and memory[i] != 0:
                                    output.append(chr(memory[i]))
                                    i += 1
                                if output:
                                    print(f"  Output: {''.join(output)}")
                                else:
                                    print("  Output: (empty string)")
                            else:
                                print("  Output: (null pointer)")
                        except LongjmpException:
                            print("  _mquickjs_get_output: caught longjmp")
                        except Exception as e:
                            print(f"  _mquickjs_get_output error: {e}")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback

            traceback.print_exc()

    return 0


if __name__ == "__main__":
    exit(main())
