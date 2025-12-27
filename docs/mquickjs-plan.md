# Plan: Getting MicroQuickJS Working

This document outlines the steps needed to run MicroQuickJS (Emscripten-compiled JavaScript engine) in pure-python-wasm.

## Current Status

- **WASM decoding**: Working (228KB module, 273 functions)
- **Instantiation**: Working (all 124 opcodes implemented)
- **Execution**: Fails during `sandbox_init()` due to missing Emscripten runtime support

## The Problem

MicroQuickJS is compiled with Emscripten and uses setjmp/longjmp for error handling. Emscripten implements this pattern through:

1. **`invoke_*` functions** - Imported functions that call through the indirect function table while catching longjmp exceptions
2. **`setThrew` export** - Called when a longjmp occurs to record the jump target
3. **`saveSetjmp` export** - Saves setjmp state for later longjmp
4. **`_emscripten_throw_longjmp`** - Triggers the longjmp unwind

When C code calls `setjmp()`, Emscripten saves state. When `longjmp()` is called, it throws a special exception that gets caught by the `invoke_*` wrapper, which then calls `setThrew` and returns control to the setjmp point.

## Implementation Plan

### Phase 1: Implement Emscripten Memory Management

The module needs proper memory operations:

1. **`emscripten_memcpy_big(dest, src, num)`** - Copy memory regions
   - Current: Stub that returns dest
   - Needed: Actually copy `num` bytes from `src` to `dest` in WASM memory

2. **`emscripten_resize_heap(requested_size)`** - Grow memory
   - Current: Returns 0 (failure)
   - Needed: Call `memory.grow` and return success/failure

### Phase 2: Implement setjmp/longjmp Support

This is the core challenge. We need to implement the Emscripten exception handling pattern:

1. **Track setjmp state** - When `saveSetjmp` is called, record the current execution context

2. **Implement `invoke_*` functions** - These wrap indirect calls:
   ```
   invoke_iii(index, a1, a2) -> i32
   invoke_iiii(index, a1, a2, a3) -> i32
   invoke_iiiii(index, a1, a2, a3, a4) -> i32
   invoke_vi(index, a1) -> void
   invoke_vii(index, a1, a2) -> void
   invoke_viii(index, a1, a2, a3) -> void
   invoke_viiiii(index, a1, a2, a3, a4, a5) -> void
   invoke_viiiiii(index, a1, a2, a3, a4, a5, a6) -> void
   ```

   Each invoke function should:
   - Save the current stack state
   - Call the function at `table[index]` with the given args
   - If `_emscripten_throw_longjmp` is called during execution, catch it
   - Call `setThrew(1, 0)` to indicate a longjmp occurred
   - Return 0 (or appropriate default)

3. **Implement `_emscripten_throw_longjmp`** - Raise a special exception that invoke functions catch

### Phase 3: Implement TempRet0 for 64-bit Returns

Emscripten uses TempRet0 to return the high 32 bits of 64-bit values:

1. **`setTempRet0(value)`** - Store the high 32 bits
2. **`getTempRet0()`** - Retrieve the high 32 bits

Current implementation stores in a Python list - this should work, but verify it's being used correctly.

### Phase 4: Wire Up the Runtime

Create a proper Emscripten runtime class:

```python
class EmscriptenRuntime:
    def __init__(self, instance):
        self.instance = instance
        self.temp_ret0 = 0
        self.threw = False
        self.threw_value = 0

    def create_imports(self):
        return {
            'env': {
                'abort': self.abort,
                '__assert_fail': self.assert_fail,
                'invoke_iii': self.make_invoke('iii'),
                # ... etc
                'setTempRet0': self.set_temp_ret0,
                'getTempRet0': self.get_temp_ret0,
                'emscripten_memcpy_big': self.memcpy_big,
                'emscripten_resize_heap': self.resize_heap,
                '_emscripten_throw_longjmp': self.throw_longjmp,
            }
        }
```

### Phase 5: Test with MicroQuickJS

Once the runtime is implemented:

1. Call `__wasm_call_ctors()` to initialize
2. Call `sandbox_init(heap_size)` with sufficient heap (e.g., 1MB)
3. Allocate a string in WASM memory using `malloc`
4. Call `sandbox_eval(ctx, code_ptr, code_len)`
5. Read the result from memory

## Testing Strategy

### Unit Tests

1. Test `emscripten_memcpy_big` with various sizes
2. Test `invoke_*` functions with simple table calls
3. Test setjmp/longjmp simulation with a minimal test case

### Integration Test

Create a test that:
1. Loads mquickjs.wasm
2. Initializes the sandbox
3. Evaluates simple JavaScript: `1 + 2`
4. Verifies the result is `3`

## Files to Create/Modify

1. **`src/pure_python_wasm/emscripten.py`** - New file with Emscripten runtime support
2. **`tests/test_emscripten.py`** - Tests for Emscripten runtime
3. **`tests/test_mquickjs.py`** - Integration tests with MicroQuickJS
4. **`examples/mquickjs_example.py`** - Example usage

## Estimated Complexity

| Phase | Complexity | Description |
|-------|------------|-------------|
| 1 | Low | Memory operations are straightforward |
| 2 | High | setjmp/longjmp requires careful exception handling |
| 3 | Low | Already partially implemented |
| 4 | Medium | Wiring everything together |
| 5 | Low | Testing and validation |

## Open Questions

1. **Stack unwinding**: Do we need to properly unwind the WASM stack when longjmp occurs, or can we just return from the invoke function?

2. **Multiple setjmp points**: Does MicroQuickJS use nested setjmp? If so, we need to track multiple jump points.

3. **Memory layout**: Does MicroQuickJS expect specific memory layout or can we just provide sufficient heap?

## References

- [Emscripten setjmp implementation](https://github.com/emscripten-core/emscripten/blob/main/system/lib/compiler-rt/emscripten_setjmp.c)
- [MicroQuickJS source](https://github.com/bellard/mquickjs)
- [WebAssembly exception handling proposal](https://github.com/WebAssembly/exception-handling)
