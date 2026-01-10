"""WebAssembly opcode definitions."""

# Control instructions
UNREACHABLE = 0x00
NOP = 0x01
BLOCK = 0x02
LOOP = 0x03
IF = 0x04
ELSE = 0x05
END = 0x0B
BR = 0x0C
BR_IF = 0x0D
BR_TABLE = 0x0E
RETURN = 0x0F
CALL = 0x10
CALL_INDIRECT = 0x11

# Parametric instructions
DROP = 0x1A
SELECT = 0x1B
SELECT_T = 0x1C  # typed select (post-MVP)

# Variable instructions
LOCAL_GET = 0x20
LOCAL_SET = 0x21
LOCAL_TEE = 0x22
GLOBAL_GET = 0x23
GLOBAL_SET = 0x24

# Table instructions (reference types)
TABLE_GET = 0x25
TABLE_SET = 0x26

# Memory instructions
I32_LOAD = 0x28
I64_LOAD = 0x29
F32_LOAD = 0x2A
F64_LOAD = 0x2B
I32_LOAD8_S = 0x2C
I32_LOAD8_U = 0x2D
I32_LOAD16_S = 0x2E
I32_LOAD16_U = 0x2F
I64_LOAD8_S = 0x30
I64_LOAD8_U = 0x31
I64_LOAD16_S = 0x32
I64_LOAD16_U = 0x33
I64_LOAD32_S = 0x34
I64_LOAD32_U = 0x35
I32_STORE = 0x36
I64_STORE = 0x37
F32_STORE = 0x38
F64_STORE = 0x39
I32_STORE8 = 0x3A
I32_STORE16 = 0x3B
I64_STORE8 = 0x3C
I64_STORE16 = 0x3D
I64_STORE32 = 0x3E
MEMORY_SIZE = 0x3F
MEMORY_GROW = 0x40

# Numeric instructions - constants
I32_CONST = 0x41
I64_CONST = 0x42
F32_CONST = 0x43
F64_CONST = 0x44

# Numeric instructions - i32 comparison
I32_EQZ = 0x45
I32_EQ = 0x46
I32_NE = 0x47
I32_LT_S = 0x48
I32_LT_U = 0x49
I32_GT_S = 0x4A
I32_GT_U = 0x4B
I32_LE_S = 0x4C
I32_LE_U = 0x4D
I32_GE_S = 0x4E
I32_GE_U = 0x4F

# Numeric instructions - i64 comparison
I64_EQZ = 0x50
I64_EQ = 0x51
I64_NE = 0x52
I64_LT_S = 0x53
I64_LT_U = 0x54
I64_GT_S = 0x55
I64_GT_U = 0x56
I64_LE_S = 0x57
I64_LE_U = 0x58
I64_GE_S = 0x59
I64_GE_U = 0x5A

# Numeric instructions - f32 comparison
F32_EQ = 0x5B
F32_NE = 0x5C
F32_LT = 0x5D
F32_GT = 0x5E
F32_LE = 0x5F
F32_GE = 0x60

# Numeric instructions - f64 comparison
F64_EQ = 0x61
F64_NE = 0x62
F64_LT = 0x63
F64_GT = 0x64
F64_LE = 0x65
F64_GE = 0x66

# Numeric instructions - i32 unary
I32_CLZ = 0x67
I32_CTZ = 0x68
I32_POPCNT = 0x69

# Numeric instructions - i32 binary
I32_ADD = 0x6A
I32_SUB = 0x6B
I32_MUL = 0x6C
I32_DIV_S = 0x6D
I32_DIV_U = 0x6E
I32_REM_S = 0x6F
I32_REM_U = 0x70
I32_AND = 0x71
I32_OR = 0x72
I32_XOR = 0x73
I32_SHL = 0x74
I32_SHR_S = 0x75
I32_SHR_U = 0x76
I32_ROTL = 0x77
I32_ROTR = 0x78

# Numeric instructions - i64 unary
I64_CLZ = 0x79
I64_CTZ = 0x7A
I64_POPCNT = 0x7B

# Numeric instructions - i64 binary
I64_ADD = 0x7C
I64_SUB = 0x7D
I64_MUL = 0x7E
I64_DIV_S = 0x7F
I64_DIV_U = 0x80
I64_REM_S = 0x81
I64_REM_U = 0x82
I64_AND = 0x83
I64_OR = 0x84
I64_XOR = 0x85
I64_SHL = 0x86
I64_SHR_S = 0x87
I64_SHR_U = 0x88
I64_ROTL = 0x89
I64_ROTR = 0x8A

# Numeric instructions - f32 unary
F32_ABS = 0x8B
F32_NEG = 0x8C
F32_CEIL = 0x8D
F32_FLOOR = 0x8E
F32_TRUNC = 0x8F
F32_NEAREST = 0x90
F32_SQRT = 0x91

# Numeric instructions - f32 binary
F32_ADD = 0x92
F32_SUB = 0x93
F32_MUL = 0x94
F32_DIV = 0x95
F32_MIN = 0x96
F32_MAX = 0x97
F32_COPYSIGN = 0x98

# Numeric instructions - f64 unary
F64_ABS = 0x99
F64_NEG = 0x9A
F64_CEIL = 0x9B
F64_FLOOR = 0x9C
F64_TRUNC = 0x9D
F64_NEAREST = 0x9E
F64_SQRT = 0x9F

# Numeric instructions - f64 binary
F64_ADD = 0xA0
F64_SUB = 0xA1
F64_MUL = 0xA2
F64_DIV = 0xA3
F64_MIN = 0xA4
F64_MAX = 0xA5
F64_COPYSIGN = 0xA6

# Numeric instructions - conversions
I32_WRAP_I64 = 0xA7
I32_TRUNC_F32_S = 0xA8
I32_TRUNC_F32_U = 0xA9
I32_TRUNC_F64_S = 0xAA
I32_TRUNC_F64_U = 0xAB
I64_EXTEND_I32_S = 0xAC
I64_EXTEND_I32_U = 0xAD
I64_TRUNC_F32_S = 0xAE
I64_TRUNC_F32_U = 0xAF
I64_TRUNC_F64_S = 0xB0
I64_TRUNC_F64_U = 0xB1
F32_CONVERT_I32_S = 0xB2
F32_CONVERT_I32_U = 0xB3
F32_CONVERT_I64_S = 0xB4
F32_CONVERT_I64_U = 0xB5
F32_DEMOTE_F64 = 0xB6
F64_CONVERT_I32_S = 0xB7
F64_CONVERT_I32_U = 0xB8
F64_CONVERT_I64_S = 0xB9
F64_CONVERT_I64_U = 0xBA
F64_PROMOTE_F32 = 0xBB
I32_REINTERPRET_F32 = 0xBC
I64_REINTERPRET_F64 = 0xBD
F32_REINTERPRET_I32 = 0xBE
F64_REINTERPRET_I64 = 0xBF

# Sign extension (post-MVP but widely supported)
I32_EXTEND8_S = 0xC0
I32_EXTEND16_S = 0xC1
I64_EXTEND8_S = 0xC2
I64_EXTEND16_S = 0xC3
I64_EXTEND32_S = 0xC4

# Reference instructions
REF_NULL = 0xD0
REF_IS_NULL = 0xD1
REF_FUNC = 0xD2

# Opcode to name mapping
OPCODE_NAMES = {
    UNREACHABLE: "unreachable",
    NOP: "nop",
    BLOCK: "block",
    LOOP: "loop",
    IF: "if",
    ELSE: "else",
    END: "end",
    BR: "br",
    BR_IF: "br_if",
    BR_TABLE: "br_table",
    RETURN: "return",
    CALL: "call",
    CALL_INDIRECT: "call_indirect",
    DROP: "drop",
    SELECT: "select",
    SELECT_T: "select",
    LOCAL_GET: "local.get",
    LOCAL_SET: "local.set",
    LOCAL_TEE: "local.tee",
    GLOBAL_GET: "global.get",
    GLOBAL_SET: "global.set",
    TABLE_GET: "table.get",
    TABLE_SET: "table.set",
    I32_LOAD: "i32.load",
    I64_LOAD: "i64.load",
    F32_LOAD: "f32.load",
    F64_LOAD: "f64.load",
    I32_LOAD8_S: "i32.load8_s",
    I32_LOAD8_U: "i32.load8_u",
    I32_LOAD16_S: "i32.load16_s",
    I32_LOAD16_U: "i32.load16_u",
    I64_LOAD8_S: "i64.load8_s",
    I64_LOAD8_U: "i64.load8_u",
    I64_LOAD16_S: "i64.load16_s",
    I64_LOAD16_U: "i64.load16_u",
    I64_LOAD32_S: "i64.load32_s",
    I64_LOAD32_U: "i64.load32_u",
    I32_STORE: "i32.store",
    I64_STORE: "i64.store",
    F32_STORE: "f32.store",
    F64_STORE: "f64.store",
    I32_STORE8: "i32.store8",
    I32_STORE16: "i32.store16",
    I64_STORE8: "i64.store8",
    I64_STORE16: "i64.store16",
    I64_STORE32: "i64.store32",
    MEMORY_SIZE: "memory.size",
    MEMORY_GROW: "memory.grow",
    I32_CONST: "i32.const",
    I64_CONST: "i64.const",
    F32_CONST: "f32.const",
    F64_CONST: "f64.const",
    I32_EQZ: "i32.eqz",
    I32_EQ: "i32.eq",
    I32_NE: "i32.ne",
    I32_LT_S: "i32.lt_s",
    I32_LT_U: "i32.lt_u",
    I32_GT_S: "i32.gt_s",
    I32_GT_U: "i32.gt_u",
    I32_LE_S: "i32.le_s",
    I32_LE_U: "i32.le_u",
    I32_GE_S: "i32.ge_s",
    I32_GE_U: "i32.ge_u",
    I64_EQZ: "i64.eqz",
    I64_EQ: "i64.eq",
    I64_NE: "i64.ne",
    I64_LT_S: "i64.lt_s",
    I64_LT_U: "i64.lt_u",
    I64_GT_S: "i64.gt_s",
    I64_GT_U: "i64.gt_u",
    I64_LE_S: "i64.le_s",
    I64_LE_U: "i64.le_u",
    I64_GE_S: "i64.ge_s",
    I64_GE_U: "i64.ge_u",
    F32_EQ: "f32.eq",
    F32_NE: "f32.ne",
    F32_LT: "f32.lt",
    F32_GT: "f32.gt",
    F32_LE: "f32.le",
    F32_GE: "f32.ge",
    F64_EQ: "f64.eq",
    F64_NE: "f64.ne",
    F64_LT: "f64.lt",
    F64_GT: "f64.gt",
    F64_LE: "f64.le",
    F64_GE: "f64.ge",
    I32_CLZ: "i32.clz",
    I32_CTZ: "i32.ctz",
    I32_POPCNT: "i32.popcnt",
    I32_ADD: "i32.add",
    I32_SUB: "i32.sub",
    I32_MUL: "i32.mul",
    I32_DIV_S: "i32.div_s",
    I32_DIV_U: "i32.div_u",
    I32_REM_S: "i32.rem_s",
    I32_REM_U: "i32.rem_u",
    I32_AND: "i32.and",
    I32_OR: "i32.or",
    I32_XOR: "i32.xor",
    I32_SHL: "i32.shl",
    I32_SHR_S: "i32.shr_s",
    I32_SHR_U: "i32.shr_u",
    I32_ROTL: "i32.rotl",
    I32_ROTR: "i32.rotr",
    I64_CLZ: "i64.clz",
    I64_CTZ: "i64.ctz",
    I64_POPCNT: "i64.popcnt",
    I64_ADD: "i64.add",
    I64_SUB: "i64.sub",
    I64_MUL: "i64.mul",
    I64_DIV_S: "i64.div_s",
    I64_DIV_U: "i64.div_u",
    I64_REM_S: "i64.rem_s",
    I64_REM_U: "i64.rem_u",
    I64_AND: "i64.and",
    I64_OR: "i64.or",
    I64_XOR: "i64.xor",
    I64_SHL: "i64.shl",
    I64_SHR_S: "i64.shr_s",
    I64_SHR_U: "i64.shr_u",
    I64_ROTL: "i64.rotl",
    I64_ROTR: "i64.rotr",
    F32_ABS: "f32.abs",
    F32_NEG: "f32.neg",
    F32_CEIL: "f32.ceil",
    F32_FLOOR: "f32.floor",
    F32_TRUNC: "f32.trunc",
    F32_NEAREST: "f32.nearest",
    F32_SQRT: "f32.sqrt",
    F32_ADD: "f32.add",
    F32_SUB: "f32.sub",
    F32_MUL: "f32.mul",
    F32_DIV: "f32.div",
    F32_MIN: "f32.min",
    F32_MAX: "f32.max",
    F32_COPYSIGN: "f32.copysign",
    F64_ABS: "f64.abs",
    F64_NEG: "f64.neg",
    F64_CEIL: "f64.ceil",
    F64_FLOOR: "f64.floor",
    F64_TRUNC: "f64.trunc",
    F64_NEAREST: "f64.nearest",
    F64_SQRT: "f64.sqrt",
    F64_ADD: "f64.add",
    F64_SUB: "f64.sub",
    F64_MUL: "f64.mul",
    F64_DIV: "f64.div",
    F64_MIN: "f64.min",
    F64_MAX: "f64.max",
    F64_COPYSIGN: "f64.copysign",
    I32_WRAP_I64: "i32.wrap_i64",
    I32_TRUNC_F32_S: "i32.trunc_f32_s",
    I32_TRUNC_F32_U: "i32.trunc_f32_u",
    I32_TRUNC_F64_S: "i32.trunc_f64_s",
    I32_TRUNC_F64_U: "i32.trunc_f64_u",
    I64_EXTEND_I32_S: "i64.extend_i32_s",
    I64_EXTEND_I32_U: "i64.extend_i32_u",
    I64_TRUNC_F32_S: "i64.trunc_f32_s",
    I64_TRUNC_F32_U: "i64.trunc_f32_u",
    I64_TRUNC_F64_S: "i64.trunc_f64_s",
    I64_TRUNC_F64_U: "i64.trunc_f64_u",
    F32_CONVERT_I32_S: "f32.convert_i32_s",
    F32_CONVERT_I32_U: "f32.convert_i32_u",
    F32_CONVERT_I64_S: "f32.convert_i64_s",
    F32_CONVERT_I64_U: "f32.convert_i64_u",
    F32_DEMOTE_F64: "f32.demote_f64",
    F64_CONVERT_I32_S: "f64.convert_i32_s",
    F64_CONVERT_I32_U: "f64.convert_i32_u",
    F64_CONVERT_I64_S: "f64.convert_i64_s",
    F64_CONVERT_I64_U: "f64.convert_i64_u",
    F64_PROMOTE_F32: "f64.promote_f32",
    I32_REINTERPRET_F32: "i32.reinterpret_f32",
    I64_REINTERPRET_F64: "i64.reinterpret_f64",
    F32_REINTERPRET_I32: "f32.reinterpret_i32",
    F64_REINTERPRET_I64: "f64.reinterpret_i64",
    I32_EXTEND8_S: "i32.extend8_s",
    I32_EXTEND16_S: "i32.extend16_s",
    I64_EXTEND8_S: "i64.extend8_s",
    I64_EXTEND16_S: "i64.extend16_s",
    I64_EXTEND32_S: "i64.extend32_s",
    REF_NULL: "ref.null",
    REF_IS_NULL: "ref.is_null",
    REF_FUNC: "ref.func",
}

# Opcodes with no immediate
NO_IMMEDIATE = {
    UNREACHABLE,
    NOP,
    RETURN,
    DROP,
    SELECT,
    I32_EQZ,
    I32_EQ,
    I32_NE,
    I32_LT_S,
    I32_LT_U,
    I32_GT_S,
    I32_GT_U,
    I32_LE_S,
    I32_LE_U,
    I32_GE_S,
    I32_GE_U,
    I64_EQZ,
    I64_EQ,
    I64_NE,
    I64_LT_S,
    I64_LT_U,
    I64_GT_S,
    I64_GT_U,
    I64_LE_S,
    I64_LE_U,
    I64_GE_S,
    I64_GE_U,
    F32_EQ,
    F32_NE,
    F32_LT,
    F32_GT,
    F32_LE,
    F32_GE,
    F64_EQ,
    F64_NE,
    F64_LT,
    F64_GT,
    F64_LE,
    F64_GE,
    I32_CLZ,
    I32_CTZ,
    I32_POPCNT,
    I32_ADD,
    I32_SUB,
    I32_MUL,
    I32_DIV_S,
    I32_DIV_U,
    I32_REM_S,
    I32_REM_U,
    I32_AND,
    I32_OR,
    I32_XOR,
    I32_SHL,
    I32_SHR_S,
    I32_SHR_U,
    I32_ROTL,
    I32_ROTR,
    I64_CLZ,
    I64_CTZ,
    I64_POPCNT,
    I64_ADD,
    I64_SUB,
    I64_MUL,
    I64_DIV_S,
    I64_DIV_U,
    I64_REM_S,
    I64_REM_U,
    I64_AND,
    I64_OR,
    I64_XOR,
    I64_SHL,
    I64_SHR_S,
    I64_SHR_U,
    I64_ROTL,
    I64_ROTR,
    F32_ABS,
    F32_NEG,
    F32_CEIL,
    F32_FLOOR,
    F32_TRUNC,
    F32_NEAREST,
    F32_SQRT,
    F32_ADD,
    F32_SUB,
    F32_MUL,
    F32_DIV,
    F32_MIN,
    F32_MAX,
    F32_COPYSIGN,
    F64_ABS,
    F64_NEG,
    F64_CEIL,
    F64_FLOOR,
    F64_TRUNC,
    F64_NEAREST,
    F64_SQRT,
    F64_ADD,
    F64_SUB,
    F64_MUL,
    F64_DIV,
    F64_MIN,
    F64_MAX,
    F64_COPYSIGN,
    I32_WRAP_I64,
    I32_TRUNC_F32_S,
    I32_TRUNC_F32_U,
    I32_TRUNC_F64_S,
    I32_TRUNC_F64_U,
    I64_EXTEND_I32_S,
    I64_EXTEND_I32_U,
    I64_TRUNC_F32_S,
    I64_TRUNC_F32_U,
    I64_TRUNC_F64_S,
    I64_TRUNC_F64_U,
    F32_CONVERT_I32_S,
    F32_CONVERT_I32_U,
    F32_CONVERT_I64_S,
    F32_CONVERT_I64_U,
    F32_DEMOTE_F64,
    F64_CONVERT_I32_S,
    F64_CONVERT_I32_U,
    F64_CONVERT_I64_S,
    F64_CONVERT_I64_U,
    F64_PROMOTE_F32,
    I32_REINTERPRET_F32,
    I64_REINTERPRET_F64,
    F32_REINTERPRET_I32,
    F64_REINTERPRET_I64,
    I32_EXTEND8_S,
    I32_EXTEND16_S,
    I64_EXTEND8_S,
    I64_EXTEND16_S,
    I64_EXTEND32_S,
    REF_IS_NULL,
    END,
    ELSE,
}

# Opcodes with single u32 immediate (label/index)
U32_IMMEDIATE = {
    BR,
    BR_IF,
    CALL,
    LOCAL_GET,
    LOCAL_SET,
    LOCAL_TEE,
    GLOBAL_GET,
    GLOBAL_SET,
    TABLE_GET,
    TABLE_SET,
    REF_FUNC,
}

# Opcodes with signed i32 immediate
I32_IMMEDIATE = {
    I32_CONST,
}

# Opcodes with signed i64 immediate
I64_IMMEDIATE = {
    I64_CONST,
}

# Opcodes with f32 immediate
F32_IMMEDIATE = {
    F32_CONST,
}

# Opcodes with f64 immediate
F64_IMMEDIATE = {
    F64_CONST,
}

# Memory opcodes (align + offset)
MEMORY_IMMEDIATE = {
    I32_LOAD,
    I64_LOAD,
    F32_LOAD,
    F64_LOAD,
    I32_LOAD8_S,
    I32_LOAD8_U,
    I32_LOAD16_S,
    I32_LOAD16_U,
    I64_LOAD8_S,
    I64_LOAD8_U,
    I64_LOAD16_S,
    I64_LOAD16_U,
    I64_LOAD32_S,
    I64_LOAD32_U,
    I32_STORE,
    I64_STORE,
    F32_STORE,
    F64_STORE,
    I32_STORE8,
    I32_STORE16,
    I64_STORE8,
    I64_STORE16,
    I64_STORE32,
}

# Block type opcodes
BLOCK_TYPE = {
    BLOCK,
    LOOP,
    IF,
}
