# Pure Python WASM Runtime Milestones

## Milestone 1: Foundation & Binary Decoder
*Goal: Parse WASM binary format into an AST*

- [ ] Set up project structure and error types
- [ ] Implement LEB128 encoding/decoding (unsigned and signed)
- [ ] Implement binary reader with position tracking
- [ ] Parse WASM magic number and version
- [ ] Parse type section (function signatures)
- [ ] Parse function section (type indices)
- [ ] Parse export section
- [ ] Parse code section (function bodies with instructions)
- [ ] Parse import section
- [ ] Parse memory section
- [ ] Parse global section
- [ ] Parse table section
- [ ] Parse data section
- [ ] Parse element section
- [ ] Parse start section
- [ ] Parse custom sections (for names, etc.)

## Milestone 2: Core Types & Module Structure
*Goal: Represent decoded modules in Python*

- [ ] Define value types (i32, i64, f32, f64)
- [ ] Define reference types (funcref, externref)
- [ ] Implement function type representation
- [ ] Implement limits (for memory/tables)
- [ ] Implement global type (valtype + mutability)
- [ ] Implement module structure with all sections
- [ ] Implement instruction AST representation
- [ ] Define all opcodes with their immediates

## Milestone 3: Minimal Interpreter (i32 only)
*Goal: Execute simple functions with i32 arithmetic*

- [ ] Implement value stack
- [ ] Implement call stack (frames)
- [ ] Implement local variable storage
- [ ] Execute `i32.const`
- [ ] Execute `local.get`, `local.set`, `local.tee`
- [ ] Execute `i32.add`, `i32.sub`, `i32.mul`
- [ ] Execute `i32.div_s`, `i32.div_u`, `i32.rem_s`, `i32.rem_u`
- [ ] Execute `i32.and`, `i32.or`, `i32.xor`
- [ ] Execute `i32.shl`, `i32.shr_s`, `i32.shr_u`, `i32.rotl`, `i32.rotr`
- [ ] Execute `i32.clz`, `i32.ctz`, `i32.popcnt`
- [ ] Execute `i32.eqz`, `i32.eq`, `i32.ne`
- [ ] Execute `i32.lt_s`, `i32.lt_u`, `i32.gt_s`, `i32.gt_u`
- [ ] Execute `i32.le_s`, `i32.le_u`, `i32.ge_s`, `i32.ge_u`
- [ ] Execute `drop`, `select`
- [ ] Execute `nop`, `unreachable`
- [ ] Execute `return`
- [ ] Execute `call` (direct function calls)

## Milestone 4: Control Flow
*Goal: Handle blocks, loops, branches*

- [ ] Implement label stack for structured control flow
- [ ] Execute `block` instruction
- [ ] Execute `loop` instruction
- [ ] Execute `if`/`else`/`end`
- [ ] Execute `br` (unconditional branch)
- [ ] Execute `br_if` (conditional branch)
- [ ] Execute `br_table` (branch table)
- [ ] Handle multi-value block results

## Milestone 5: i64 and Integer Conversions
*Goal: Complete integer support*

- [ ] Implement i64 value type
- [ ] Execute all i64 arithmetic operations (like i32)
- [ ] Execute `i32.wrap_i64`
- [ ] Execute `i64.extend_i32_s`, `i64.extend_i32_u`
- [ ] Execute `i32.extend8_s`, `i32.extend16_s`
- [ ] Execute `i64.extend8_s`, `i64.extend16_s`, `i64.extend32_s`

## Milestone 6: Floating Point
*Goal: IEEE 754 float support*

- [ ] Implement f32 and f64 value types with proper bit representation
- [ ] Execute `f32.const`, `f64.const`
- [ ] Execute `f32.add`, `f32.sub`, `f32.mul`, `f32.div` (and f64 variants)
- [ ] Execute `f32.abs`, `f32.neg`, `f32.sqrt`
- [ ] Execute `f32.ceil`, `f32.floor`, `f32.trunc`, `f32.nearest`
- [ ] Execute `f32.min`, `f32.max`
- [ ] Execute `f32.copysign`
- [ ] Execute all f32 comparison ops
- [ ] Execute all f64 operations (mirrors f32)
- [ ] Execute integer-float conversions (i32/i64 <-> f32/f64)
- [ ] Execute `f32.reinterpret_i32`, `f64.reinterpret_i64`
- [ ] Execute `i32.reinterpret_f32`, `i64.reinterpret_f64`
- [ ] Handle NaN canonicalization

## Milestone 7: Linear Memory
*Goal: Load/store operations with memory*

- [ ] Implement MemoryInstance with bytearray storage
- [ ] Implement little-endian load/store helpers
- [ ] Execute `memory.size`, `memory.grow`
- [ ] Execute `i32.load`, `i32.load8_s`, `i32.load8_u`, `i32.load16_s`, `i32.load16_u`
- [ ] Execute `i64.load`, `i64.load8_s`, `i64.load8_u`, `i64.load16_s`, `i64.load16_u`, `i64.load32_s`, `i64.load32_u`
- [ ] Execute `f32.load`, `f64.load`
- [ ] Execute `i32.store`, `i32.store8`, `i32.store16`
- [ ] Execute `i64.store`, `i64.store8`, `i64.store16`, `i64.store32`
- [ ] Execute `f32.store`, `f64.store`
- [ ] Implement memory bounds checking (trap on out-of-bounds)
- [ ] Initialize memory from data segments

## Milestone 8: Globals
*Goal: Global variable support*

- [ ] Implement GlobalInstance
- [ ] Parse and evaluate constant expressions for initializers
- [ ] Execute `global.get`, `global.set`
- [ ] Validate mutability constraints

## Milestone 9: Tables and Indirect Calls
*Goal: Function pointers via tables*

- [ ] Implement TableInstance
- [ ] Initialize tables from element segments
- [ ] Execute `call_indirect`
- [ ] Validate indirect call type signatures
- [ ] Execute `table.get`, `table.set` (if targeting reference types)
- [ ] Execute `table.size`, `table.grow` (if targeting reference types)

## Milestone 10: Imports and Exports
*Goal: Module linking and Python interop*

- [ ] Implement import resolution
- [ ] Support imported functions (Python callables)
- [ ] Support imported memories
- [ ] Support imported globals
- [ ] Support imported tables
- [ ] Implement export namespace
- [ ] Create Pythonic export accessors

## Milestone 11: Validation
*Goal: Static type checking*

- [ ] Validate type section structure
- [ ] Validate function indices in bounds
- [ ] Implement stack-based instruction type checking
- [ ] Validate memory and table indices
- [ ] Validate global mutability in contexts
- [ ] Validate start function signature
- [ ] Validate import/export matching

## Milestone 12: Public API Polish
*Goal: User-friendly Python interface*

- [ ] Implement `load_module()` from bytes/file/path
- [ ] Implement `validate()` as standalone function
- [ ] Implement `instantiate()` with imports dict
- [ ] Add memory read/write helpers for Python
- [ ] Add type annotations throughout
- [ ] Write comprehensive docstrings
- [ ] Create usage examples

## Milestone 13: WAST Test Runner
*Goal: Run official spec tests*

- [ ] Implement WAST S-expression parser
- [ ] Handle `(module ...)` declarations
- [ ] Handle `(assert_return ...)` tests
- [ ] Handle `(assert_trap ...)` tests
- [ ] Handle `(assert_invalid ...)` tests
- [ ] Handle `(assert_malformed ...)` tests
- [ ] Handle `(invoke ...)` commands
- [ ] Handle `(register ...)` for module linking

## Milestone 14: Spec Test Compliance
*Goal: Pass official test suite*

**Core Integer Tests:**
- [ ] Pass `i32.wast`
- [ ] Pass `i64.wast`
- [ ] Pass `int_literals.wast`
- [ ] Pass `int_exprs.wast`

**Control Flow Tests:**
- [ ] Pass `block.wast`
- [ ] Pass `loop.wast`
- [ ] Pass `if.wast`
- [ ] Pass `br.wast`
- [ ] Pass `br_if.wast`
- [ ] Pass `br_table.wast`
- [ ] Pass `return.wast`
- [ ] Pass `unreachable.wast`
- [ ] Pass `nop.wast`

**Function Tests:**
- [ ] Pass `func.wast`
- [ ] Pass `call.wast`
- [ ] Pass `call_indirect.wast`
- [ ] Pass `fac.wast` (factorial)

**Variable Tests:**
- [ ] Pass `local_get.wast`
- [ ] Pass `local_set.wast`
- [ ] Pass `local_tee.wast`
- [ ] Pass `global.wast`

**Memory Tests:**
- [ ] Pass `memory.wast`
- [ ] Pass `memory_size.wast`
- [ ] Pass `memory_grow.wast`
- [ ] Pass `memory_trap.wast`
- [ ] Pass `address.wast`
- [ ] Pass `align.wast`
- [ ] Pass `load.wast`
- [ ] Pass `store.wast`
- [ ] Pass `endianness.wast`

**Table Tests:**
- [ ] Pass `table.wast`
- [ ] Pass `elem.wast`
- [ ] Pass `func_ptrs.wast`

**Float Tests:**
- [ ] Pass `f32.wast`
- [ ] Pass `f64.wast`
- [ ] Pass `f32_cmp.wast`
- [ ] Pass `f64_cmp.wast`
- [ ] Pass `f32_bitwise.wast`
- [ ] Pass `f64_bitwise.wast`
- [ ] Pass `float_literals.wast`
- [ ] Pass `float_exprs.wast`
- [ ] Pass `float_misc.wast`
- [ ] Pass `float_memory.wast`
- [ ] Pass `conversions.wast`

**Validation Tests:**
- [ ] Pass `type.wast`
- [ ] Pass `exports.wast`
- [ ] Pass `imports.wast`
- [ ] Pass `data.wast`
- [ ] Pass `start.wast`
- [ ] Pass `binary.wast`
- [ ] Pass `binary-leb128.wast`
- [ ] Pass `custom.wast`

**Miscellaneous Tests:**
- [ ] Pass `select.wast`
- [ ] Pass `stack.wast`
- [ ] Pass `traps.wast`
- [ ] Pass `unwind.wast`
- [ ] Pass `labels.wast`
- [ ] Pass `forward.wast`
- [ ] Pass `names.wast`
- [ ] Pass `comments.wast`
- [ ] Pass `token.wast`
- [ ] Pass `const.wast`
- [ ] Pass `switch.wast`
- [ ] Pass `left-to-right.wast`
- [ ] Pass `linking.wast`

## Current Focus: Milestones 1-3

For the initial implementation, we're targeting a minimal subset:
- Binary parsing (all sections, but initially focus on type, func, code, export)
- Simple i32 arithmetic functions without control flow
- Direct function calls
- No memory, tables, or imports initially

This allows us to load and execute a `.wasm` file like:
```wasm
(module
  (func (export "add") (param i32 i32) (result i32)
    local.get 0
    local.get 1
    i32.add
  )
)
```

And call it from Python:
```python
module = load_module(wasm_bytes)
instance = instantiate(module)
assert instance.exports.add(2, 3) == 5
```
