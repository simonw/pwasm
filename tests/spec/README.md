# WebAssembly spec tests

The `.wast` files in this directory are the core test suite from the
[WebAssembly specification repository](https://github.com/WebAssembly/spec),
tag `wg-2.0` (commit `fffc6e12fa454e475455a7b58d3b5dc343980c10`), minus the
SIMD tests. They are licensed under the Apache License 2.0, see `LICENSE`.

`tests/spec_runner.py` runs them; `tests/test_spec.py` runs every file as a
pytest test. To see a summary for some files:

    uv run python tests/spec_runner.py tests/spec/i32.wast tests/spec/i64.wast

Module text is compiled to binary with `wasmtime.wat2wasm` (a dev
dependency). `assert_invalid` and `assert_malformed` commands are skipped
because pwasm does not validate modules.
