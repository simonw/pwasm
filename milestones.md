# Pure Python WASM Runtime Milestones

## Milestone 1: Foundation & Binary Decoder
*Goal: Parse WASM binary format into an AST*

- [x] Set up project structure and error types
- [x] Implement LEB128 encoding/decoding (unsigned and signed)
- [x] Implement binary reader with position tracking
- [x] Parse WASM magic number and version
- [x] Parse type section (function signatures)
- [x] Parse function section (type indices)
- [x] Parse export section
- [x] Parse code section (function bodies with instructions)
- [x] Parse import section
- [x] Parse memory section
- [x] Parse global section
- [x] Parse table section
- [x] Parse data section
- [x] Parse element section
- [x] Parse start section
- [x] Parse custom sections (for names, etc.)

## Milestone 2: Core Types & Module Structure
*Goal: Represent decoded modules in Python*

- [x] Define value types (i32, i64, f32, f64)
- [x] Define reference types (funcref, externref)
- [x] Implement function type representation
- [x] Implement limits (for memory/tables)
- [x] Implement global type (valtype + mutability)
- [x] Implement module structure with all sections
- [x] Implement instruction AST representation
- [x] Define all opcodes with their immediates

## Milestone 3: Minimal Interpreter (i32 only)
*Goal: Execute simple functions with i32 arithmetic*

- [x] Implement value stack
- [x] Implement call stack (frames)
- [x] Implement local variable storage
- [x] Execute `i32.const`
- [x] Execute `local.get`, `local.set`, `local.tee`
- [x] Execute `i32.add`, `i32.sub`, `i32.mul`
- [x] Execute `i32.div_s`, `i32.div_u`, `i32.rem_s`, `i32.rem_u`
- [x] Execute `i32.and`, `i32.or`, `i32.xor`
- [x] Execute `i32.shl`, `i32.shr_s`, `i32.shr_u`, `i32.rotl`, `i32.rotr`
- [x] Execute `i32.clz`, `i32.ctz`, `i32.popcnt`
- [x] Execute `i32.eqz`, `i32.eq`, `i32.ne`
- [x] Execute `i32.lt_s`, `i32.lt_u`, `i32.gt_s`, `i32.gt_u`
- [x] Execute `i32.le_s`, `i32.le_u`, `i32.ge_s`, `i32.ge_u`
- [x] Execute `drop`, `select`
- [x] Execute `nop`, `unreachable`
- [x] Execute `return`
- [x] Execute `call` (direct function calls)

## Milestone 4: Control Flow
*Goal: Handle blocks, loops, branches*

- [x] Implement label stack for structured control flow
- [x] Execute `block` instruction
- [x] Execute `loop` instruction
- [x] Execute `if`/`else`/`end`
- [x] Execute `br` (unconditional branch)
- [x] Execute `br_if` (conditional branch)
- [x] Execute `br_table` (branch table)
- [x] Handle multi-value block results

## Milestone 5: i64 and Integer Conversions
*Goal: Complete integer support*

- [x] Implement i64 value type (parsing and `i64.const` work)
- [x] Execute all i64 arithmetic operations (like i32)
- [x] Execute `i32.wrap_i64`
- [x] Execute `i64.extend_i32_s`, `i64.extend_i32_u`
- [x] Execute `i32.extend8_s`, `i32.extend16_s`
- [x] Execute `i64.extend8_s`, `i64.extend16_s`, `i64.extend32_s`

## Milestone 6: Floating Point
*Goal: IEEE 754 float support*

- [x] Implement f32 and f64 value types with proper bit representation
- [x] Execute `f32.const`, `f64.const`
- [x] Execute `f32.add`, `f32.sub`, `f32.mul`, `f32.div` (and f64 variants)
- [x] Execute `f32.abs`, `f32.neg`, `f32.sqrt`
- [x] Execute `f32.ceil`, `f32.floor`, `f32.trunc`, `f32.nearest`
- [x] Execute `f32.min`, `f32.max`
- [x] Execute `f32.copysign`
- [x] Execute all f32 comparison ops
- [x] Execute all f64 operations (mirrors f32)
- [x] Execute integer-float conversions (i32/i64 <-> f32/f64)
- [x] Execute `f32.reinterpret_i32`, `f64.reinterpret_i64`
- [x] Execute `i32.reinterpret_f32`, `i64.reinterpret_f64`
- [x] Handle NaN canonicalization

## Milestone 7: Linear Memory
*Goal: Load/store operations with memory*

- [x] Implement MemoryInstance with bytearray storage
- [x] Implement little-endian load/store helpers
- [x] Execute `memory.size`, `memory.grow`
- [x] Execute `i32.load`, `i32.load8_s`, `i32.load8_u`, `i32.load16_s`, `i32.load16_u`
- [x] Execute `i64.load`, `i64.load8_s`, `i64.load8_u`, `i64.load16_s`, `i64.load16_u`, `i64.load32_s`, `i64.load32_u`
- [x] Execute `f32.load`, `f64.load`
- [x] Execute `i32.store`, `i32.store8`, `i32.store16`
- [x] Execute `i64.store`, `i64.store8`, `i64.store16`, `i64.store32`
- [x] Execute `f32.store`, `f64.store`
- [x] Implement memory bounds checking (trap on out-of-bounds)
- [x] Initialize memory from data segments

## Milestone 8: Globals
*Goal: Global variable support*

- [x] Implement GlobalInstance
- [x] Parse and evaluate constant expressions for initializers
- [x] Execute `global.get`, `global.set`
- [x] Validate mutability constraints

## Milestone 9: Tables and Indirect Calls
*Goal: Function pointers via tables*

- [x] Implement TableInstance
- [x] Initialize tables from element segments
- [x] Execute `call_indirect`
- [x] Validate indirect call type signatures
- [x] Execute `table.get`, `table.set` (if targeting reference types)
- [x] Execute `table.size`, `table.grow` (if targeting reference types)

## Milestone 10: Imports and Exports
*Goal: Module linking and Python interop*

- [x] Implement import resolution
- [x] Support imported functions (Python callables)
- [x] Support imported memories
- [x] Support imported globals
- [x] Support imported tables
- [x] Implement export namespace
- [x] Create Pythonic export accessors

## Milestone 11: Validation
*Goal: Static type checking*

- [ ] Validate type section structure
- [ ] Validate function indices in bounds
- [ ] Implement stack-based instruction type checking
- [ ] Validate memory and table indices
- [ ] Validate global mutability in contexts
- [ ] Validate start function signature
- [x] Validate import/export matching

## Milestone 12: Public API Polish
*Goal: User-friendly Python interface*

- [x] Implement `decode_module()` from bytes/file/path
- [ ] Implement `validate()` as standalone function
- [x] Implement `instantiate()` with imports dict (basic version exists)
- [x] Add memory read/write helpers for Python
- [x] Add type annotations throughout
- [ ] Write comprehensive docstrings
- [ ] Create usage examples

## Milestone 13: WAST Test Runner
*Goal: Run official spec tests*

`tests/spec_runner.py` runs the vendored `wg-2.0` core tests in `tests/spec/`, compiling module text with `wasmtime.wat2wasm`.

- [x] Implement WAST S-expression parser
- [x] Handle `(module ...)` declarations
- [x] Handle `(assert_return ...)` tests
- [x] Handle `(assert_trap ...)` tests
- [ ] Handle `(assert_invalid ...)` tests (skipped: no validator)
- [ ] Handle `(assert_malformed ...)` tests (skipped)
- [x] Handle `(invoke ...)` commands
- [x] Handle `(register ...)` for module linking

## Milestone 14: Spec Test Compliance
*Goal: Pass official test suite*

**Core Integer Tests:**
- [x] Pass `i32.wast`
- [x] Pass `i64.wast`
- [x] Pass `int_literals.wast`
- [x] Pass `int_exprs.wast`

**Control Flow Tests:**
- [x] Pass `block.wast`
- [x] Pass `loop.wast`
- [x] Pass `if.wast`
- [x] Pass `br.wast`
- [x] Pass `br_if.wast`
- [x] Pass `br_table.wast`
- [x] Pass `return.wast`
- [x] Pass `unreachable.wast`
- [x] Pass `nop.wast`

**Function Tests:**
- [x] Pass `func.wast`
- [x] Pass `call.wast`
- [x] Pass `call_indirect.wast`
- [x] Pass `fac.wast` (factorial)

**Variable Tests:**
- [x] Pass `local_get.wast`
- [x] Pass `local_set.wast`
- [x] Pass `local_tee.wast`
- [x] Pass `global.wast`

**Memory Tests:**
- [x] Pass `memory.wast`
- [x] Pass `memory_size.wast`
- [x] Pass `memory_grow.wast`
- [x] Pass `memory_trap.wast`
- [x] Pass `address.wast`
- [x] Pass `align.wast`
- [x] Pass `load.wast`
- [x] Pass `store.wast`
- [x] Pass `endianness.wast`

**Table Tests:**
- [x] Pass `table.wast`
- [x] Pass `elem.wast`
- [x] Pass `func_ptrs.wast`

**Float Tests:**
- [x] Pass `f32.wast`
- [x] Pass `f64.wast`
- [x] Pass `f32_cmp.wast`
- [x] Pass `f64_cmp.wast`
- [x] Pass `f32_bitwise.wast`
- [x] Pass `f64_bitwise.wast`
- [x] Pass `float_literals.wast`
- [x] Pass `float_exprs.wast`
- [x] Pass `float_misc.wast`
- [x] Pass `float_memory.wast`
- [x] Pass `conversions.wast`

**Validation Tests:**
- [x] Pass `type.wast`
- [x] Pass `exports.wast`
- [x] Pass `imports.wast`
- [x] Pass `data.wast`
- [x] Pass `start.wast`
- [x] Pass `binary.wast`
- [x] Pass `binary-leb128.wast`
- [x] Pass `custom.wast`

**Miscellaneous Tests:**
- [x] Pass `select.wast`
- [x] Pass `stack.wast`
- [x] Pass `traps.wast`
- [x] Pass `unwind.wast`
- [x] Pass `labels.wast`
- [x] Pass `forward.wast`
- [x] Pass `names.wast`
- [x] Pass `comments.wast`
- [x] Pass `token.wast`
- [x] Pass `const.wast`
- [x] Pass `switch.wast`
- [x] Pass `left-to-right.wast`
- [x] Pass `linking.wast`

## Current Focus: Running real programs

Milestones 1-10 and 13 are complete, and every non-SIMD WebAssembly 2.0 core
spec test passes (tests/spec, 25,000+ assertions; assert_invalid and
assert_malformed are skipped because there is no validator yet).

Next priorities:
- Run larger programs compiled from C: MicroPython and QuickJS guests
- WASI support for guests that use it
- Resource limits (memory, call depth, fuel and timeouts) for sandboxing
- Milestone 11: Validation
