# Guest interpreters

Prebuilt WebAssembly builds of three small interpreters, used by
`pwasm.guests` to run untrusted Python and JavaScript inside pwasm.

| File | Interpreter | Built by |
|---|---|---|
| `micropython.wasm` | [MicroPython](https://github.com/micropython/micropython) (embed port + a `host` module) | [simonw/research `wasmi-python-sandbox`](https://github.com/simonw/research/tree/fe91d6d50a0cacfebae78ab186417109af3446ee/wasmi-python-sandbox/guests/micropython), commit `fe91d6d`, with wasi-sdk 27 and LLVM's emscripten-style setjmp/longjmp lowering |
| `quickjs.wasm` | [quickjs-ng](https://github.com/quickjs-ng/quickjs) with a small reactor (`qjs_init`, `qjs_eval`, `host.*` calls) | [simonw/research `wasmi-python-sandbox`](https://github.com/simonw/research/tree/fe91d6d50a0cacfebae78ab186417109af3446ee/wasmi-python-sandbox/guests/quickjs), commit `fe91d6d`, with wasi-sdk 27 |
| `mquickjs.wasm` | [Micro QuickJS](https://github.com/bellard/mquickjs) with a `sandbox_eval` wrapper | [simonw/research `mquickjs-sandbox/build_wasm.py`](https://github.com/simonw/research/tree/4d8ae472f622668e51df55f1c138b1cd4a787096/mquickjs-sandbox), commit `4d8ae47`, with emscripten |

`micropython.wasm` and `quickjs.wasm` had their DWARF debug sections (and
the `producers` and `target_features` custom sections) removed; the `name`
section was kept.

All three interpreters are MIT licensed, see `licenses/`. The binaries also
contain the C runtime they were linked against (wasi-libc, or emscripten's
musl-based libc), which are available under permissive licenses.
