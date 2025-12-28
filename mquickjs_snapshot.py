#!/usr/bin/env python3
"""Create a snapshot of MicroQuickJS state after _start initialization.

This script runs _start (which takes a long time) and saves the resulting
memory and global state to a file. The snapshot can then be loaded to
skip the _start initialization.
"""

import pure_python_wasm
from pathlib import Path
import time
import pickle
import sys


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


def save_snapshot(instance, path):
    """Save instance state to a file."""
    snapshot = {
        # Memory contents (bytearray -> bytes for pickling)
        "memories": [bytes(mem.data) for mem in instance.memories],
        # Global values
        "globals": [g.value for g in instance.globals],
        # Table elements (function indices)
        "tables": [list(t.elements) for t in instance.tables],
        # Runtime temp_ret0 value
        "temp_ret0": 0,
    }
    with open(path, "wb") as f:
        pickle.dump(snapshot, f)
    print(
        f"Saved snapshot to {path} ({len(snapshot['memories'][0]):,} bytes of memory)"
    )


def load_snapshot(instance, path):
    """Load instance state from a file."""
    with open(path, "rb") as f:
        snapshot = pickle.load(f)

    # Restore memory
    for i, mem_data in enumerate(snapshot["memories"]):
        instance.memories[i].data[: len(mem_data)] = bytearray(mem_data)

    # Restore globals
    for i, val in enumerate(snapshot["globals"]):
        instance.globals[i].value = val

    # Restore tables
    for i, elems in enumerate(snapshot["tables"]):
        instance.tables[i].elements = elems

    print(f"Loaded snapshot from {path}")


def create_snapshot():
    """Run _start and create a snapshot."""
    script_dir = Path(__file__).parent
    wasm_path = script_dir / "tests" / "mquickjs_standalone.wasm"
    snapshot_path = script_dir / "tests" / "mquickjs_snapshot.pkl"

    if not wasm_path.exists():
        print(f"Error: {wasm_path} not found")
        return 1

    print("Loading MicroQuickJS WASM module...")
    with open(wasm_path, "rb") as f:
        wasm_bytes = f.read()

    module = pure_python_wasm.decode_module(wasm_bytes)

    runtime = EmscriptenRuntime()
    imports = runtime.create_imports()

    instance = pure_python_wasm.instantiate(module, imports)
    runtime.set_instance(instance)

    exports = {exp.name: exp.index for exp in instance.module.exports}

    print("\nRunning _start (this may take several minutes)...")
    print("Progress will be printed every ~30 seconds.")

    start = time.time()
    try:
        pure_python_wasm.execute_function(instance, exports["_start"], [])
        elapsed = time.time() - start
        print(f"\n_start completed in {elapsed:.1f} seconds")
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n_start failed after {elapsed:.1f}s: {e}")
        return 1

    # Save snapshot
    save_snapshot(instance, snapshot_path)

    return 0


def test_with_snapshot():
    """Test loading a snapshot and running QuickJS."""
    script_dir = Path(__file__).parent
    wasm_path = script_dir / "tests" / "mquickjs_standalone.wasm"
    snapshot_path = script_dir / "tests" / "mquickjs_snapshot.pkl"

    if not snapshot_path.exists():
        print(f"Error: Snapshot not found at {snapshot_path}")
        print("Run: python mquickjs_snapshot.py create")
        return 1

    print("Loading MicroQuickJS WASM module...")
    with open(wasm_path, "rb") as f:
        wasm_bytes = f.read()

    module = pure_python_wasm.decode_module(wasm_bytes)

    runtime = EmscriptenRuntime()
    imports = runtime.create_imports()

    print("Instantiating...")
    instance = pure_python_wasm.instantiate(module, imports)
    runtime.set_instance(instance)

    # Load snapshot (skip _start!)
    print("Loading snapshot (skipping _start)...")
    load_snapshot(instance, snapshot_path)

    exports = {exp.name: exp.index for exp in instance.module.exports}

    # Now test QuickJS
    print("\nInitializing QuickJS sandbox...")
    start = time.time()
    ctx = pure_python_wasm.execute_function(
        instance, exports["sandbox_init"], [1048576]
    )
    print(f"  sandbox_init returned {ctx} in {time.time() - start:.2f}s")

    if ctx is None or ctx == 0:
        print("Error: sandbox_init failed")
        return 1

    # JavaScript code
    js_code = b"1 + 2\x00"

    # Allocate memory for code
    code_ptr = pure_python_wasm.execute_function(
        instance, exports["malloc"], [len(js_code)]
    )

    # Write code to memory
    memory = instance.memories[0].data
    memory[code_ptr : code_ptr + len(js_code)] = js_code

    # Evaluate
    print(f"\nEvaluating: {js_code.decode().strip()}")
    start = time.time()
    result_ptr = pure_python_wasm.execute_function(
        instance, exports["sandbox_eval"], [ctx, code_ptr, len(js_code) - 1]
    )
    print(f"  sandbox_eval returned {result_ptr} in {time.time() - start:.2f}s")

    if result_ptr and result_ptr > 0:
        result = []
        i = result_ptr
        while i < len(memory) and memory[i] != 0:
            result.append(chr(memory[i]))
            i += 1
        result_str = "".join(result)
        print(f"\nResult: {result_str}")

    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "create":
        exit(create_snapshot())
    else:
        exit(test_with_snapshot())
