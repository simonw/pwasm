"""Tests for the WebAssembly executor/interpreter."""

import pytest
from pwism import decode_module
from pwism.executor import instantiate, Instance
from pwism.errors import TrapError


def make_simple_func(name: str, instr_bytes: bytes, num_params: int = 2) -> bytes:
    """Helper to create a simple module with one exported function.

    Creates: (func (export name) (param i32...) (result i32) ...)
    """
    name_bytes = name.encode("utf-8")
    name_len = len(name_bytes)

    # Body: 0 locals + instructions
    body_size = 1 + len(instr_bytes)  # 1 byte for local count (0)

    # Type section: (i32, i32, ...) -> i32 or (i32) -> i32 or () -> i32
    if num_params == 0:
        type_section = bytes([0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7F])
    elif num_params == 1:
        type_section = bytes([0x01, 0x06, 0x01, 0x60, 0x01, 0x7F, 0x01, 0x7F])
    else:  # num_params == 2
        type_section = bytes([0x01, 0x07, 0x01, 0x60, 0x02, 0x7F, 0x7F, 0x01, 0x7F])

    # Function section
    func_section = bytes([0x03, 0x02, 0x01, 0x00])

    # Export section
    export_size = 1 + 1 + name_len + 1 + 1  # count + name_len + name + kind + idx
    export_section = (
        bytes([0x07, export_size, 0x01, name_len]) + name_bytes + bytes([0x00, 0x00])
    )

    # Code section
    code_section = bytes([0x0A, 2 + body_size, 0x01, body_size, 0x00]) + instr_bytes

    return (
        bytes([0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00])
        + type_section
        + func_section
        + export_section
        + code_section
    )


class TestBasicArithmetic:
    """Test basic i32 arithmetic operations."""

    def test_add_two_numbers(self):
        # local.get 0, local.get 1, i32.add, end
        wasm = make_simple_func("add", bytes([0x20, 0x00, 0x20, 0x01, 0x6A, 0x0B]))
        module = decode_module(wasm)
        instance = instantiate(module)
        assert instance.exports.add(2, 3) == 5
        assert instance.exports.add(0, 0) == 0
        assert instance.exports.add(-1, 1) == 0

    def test_subtract(self):
        wasm = make_simple_func("sub", bytes([0x20, 0x00, 0x20, 0x01, 0x6B, 0x0B]))
        module = decode_module(wasm)
        instance = instantiate(module)
        assert instance.exports.sub(5, 3) == 2
        assert instance.exports.sub(3, 5) == -2

    def test_multiply(self):
        wasm = make_simple_func("mul", bytes([0x20, 0x00, 0x20, 0x01, 0x6C, 0x0B]))
        module = decode_module(wasm)
        instance = instantiate(module)
        assert instance.exports.mul(6, 7) == 42
        assert instance.exports.mul(-3, 4) == -12

    def test_div_signed(self):
        wasm = make_simple_func("div_s", bytes([0x20, 0x00, 0x20, 0x01, 0x6D, 0x0B]))
        module = decode_module(wasm)
        instance = instantiate(module)
        assert instance.exports.div_s(10, 3) == 3
        assert instance.exports.div_s(-7, 3) == -2

    def test_div_by_zero_traps(self):
        wasm = make_simple_func("div_s", bytes([0x20, 0x00, 0x20, 0x01, 0x6D, 0x0B]))
        module = decode_module(wasm)
        instance = instantiate(module)
        with pytest.raises(TrapError, match="divide by zero"):
            instance.exports.div_s(10, 0)


class TestConst:
    """Test constant instructions."""

    def test_i32_const(self):
        # i32.const 42, end
        wasm = make_simple_func("const42", bytes([0x41, 0x2A, 0x0B]), num_params=0)
        module = decode_module(wasm)
        instance = instantiate(module)
        assert instance.exports.const42() == 42

    def test_i32_const_negative(self):
        # i32.const -42 (encoded as 0x56 in signed LEB128), end
        wasm = make_simple_func("neg", bytes([0x41, 0x56, 0x0B]), num_params=0)
        module = decode_module(wasm)
        instance = instantiate(module)
        assert instance.exports.neg() == -42


class TestLocals:
    """Test local variable operations."""

    def test_local_get_returns_param(self):
        # Simple identity: local.get 0, end
        wasm = make_simple_func("id", bytes([0x20, 0x00, 0x0B]), num_params=1)
        module = decode_module(wasm)
        instance = instantiate(module)
        assert instance.exports.id(42) == 42
        assert instance.exports.id(-1) == -1

    def test_local_tee(self):
        # i32.const 100, local.tee 0, end
        # 100 in signed LEB128 needs continuation byte to avoid sign extension
        # 100 = 0b01100100, bit 6 is set so we need: 0xE4 0x00
        wasm = make_simple_func(
            "tee", bytes([0x41, 0xE4, 0x00, 0x22, 0x00, 0x0B]), num_params=1
        )
        module = decode_module(wasm)
        instance = instantiate(module)
        assert instance.exports.tee(5) == 100


class TestComparison:
    """Test comparison operations."""

    def test_i32_eqz(self):
        # local.get 0, i32.eqz, end
        wasm = make_simple_func("eqz", bytes([0x20, 0x00, 0x45, 0x0B]), num_params=1)
        module = decode_module(wasm)
        instance = instantiate(module)
        assert instance.exports.eqz(0) == 1
        assert instance.exports.eqz(1) == 0
        assert instance.exports.eqz(-1) == 0

    def test_i32_lt_s(self):
        # local.get 0, local.get 1, i32.lt_s, end
        wasm = make_simple_func("lt_s", bytes([0x20, 0x00, 0x20, 0x01, 0x48, 0x0B]))
        module = decode_module(wasm)
        instance = instantiate(module)
        assert instance.exports.lt_s(1, 2) == 1
        assert instance.exports.lt_s(2, 1) == 0
        assert instance.exports.lt_s(-1, 1) == 1


class TestControlFlow:
    """Test control flow instructions."""

    def test_simple_return(self):
        # local.get 0, return, end
        wasm = make_simple_func("ret", bytes([0x20, 0x00, 0x0F, 0x0B]), num_params=1)
        module = decode_module(wasm)
        instance = instantiate(module)
        assert instance.exports.ret(42) == 42

    def test_block_and_br(self):
        # block (result i32), local.get 0, br 0, end, end
        # 0x02 0x7F = block (result i32)
        # 0x20 0x00 = local.get 0
        # 0x0C 0x00 = br 0
        # 0x0B = end block
        # 0x0B = end func
        wasm = make_simple_func(
            "blk", bytes([0x02, 0x7F, 0x20, 0x00, 0x0C, 0x00, 0x0B, 0x0B]), num_params=1
        )
        module = decode_module(wasm)
        instance = instantiate(module)
        assert instance.exports.blk(123) == 123

    def test_if_else(self):
        # local.get 0, if (result i32), i32.const 1, else, i32.const 0, end, end
        instr = bytes(
            [
                0x20,
                0x00,  # local.get 0
                0x04,
                0x7F,  # if (result i32)
                0x41,
                0x01,  # i32.const 1
                0x05,  # else
                0x41,
                0x00,  # i32.const 0
                0x0B,  # end if
                0x0B,  # end func
            ]
        )
        wasm = make_simple_func("ife", instr, num_params=1)
        module = decode_module(wasm)
        instance = instantiate(module)
        assert instance.exports.ife(1) == 1
        assert instance.exports.ife(0) == 0
        assert instance.exports.ife(42) == 1


class TestFunctionCalls:
    """Test function call instructions."""

    def test_call(self):
        # Module with two functions:
        # func 0: add 10 to param
        # func 1: call func 0 (exported)
        wasm = bytes(
            [
                0x00,
                0x61,
                0x73,
                0x6D,
                0x01,
                0x00,
                0x00,
                0x00,  # magic + version
                # Type section: (i32) -> i32
                0x01,
                0x06,
                0x01,
                0x60,
                0x01,
                0x7F,
                0x01,
                0x7F,
                # Function section: 2 functions of type 0
                0x03,
                0x03,
                0x02,
                0x00,
                0x00,
                # Export section: export func 1 as "f"
                0x07,
                0x05,
                0x01,
                0x01,
                0x66,
                0x00,
                0x01,  # "f"
                # Code section: size = 1 (count) + 8 (func0) + 7 (func1) = 16 = 0x10
                0x0A,
                0x10,
                0x02,
                # func 0: body_size=7, 0 locals, local.get 0, i32.const 10, i32.add, end
                0x07,
                0x00,
                0x20,
                0x00,
                0x41,
                0x0A,
                0x6A,
                0x0B,
                # func 1: body_size=6, 0 locals, local.get 0, call 0, end
                0x06,
                0x00,
                0x20,
                0x00,
                0x10,
                0x00,
                0x0B,
            ]
        )
        module = decode_module(wasm)
        instance = instantiate(module)
        assert instance.exports.f(5) == 15
        assert instance.exports.f(32) == 42


class TestBitwise:
    """Test bitwise operations."""

    def test_i32_and(self):
        # Use bracket notation since "and" is a keyword
        wasm = make_simple_func("and", bytes([0x20, 0x00, 0x20, 0x01, 0x71, 0x0B]))
        module = decode_module(wasm)
        instance = instantiate(module)
        assert instance.exports["and"](0xFF, 0x0F) == 0x0F
        assert instance.exports["and"](0b1010, 0b1100) == 0b1000

    def test_i32_or(self):
        wasm = make_simple_func("or", bytes([0x20, 0x00, 0x20, 0x01, 0x72, 0x0B]))
        module = decode_module(wasm)
        instance = instantiate(module)
        assert instance.exports["or"](0b1010, 0b0101) == 0b1111

    def test_i32_xor(self):
        wasm = make_simple_func("xor", bytes([0x20, 0x00, 0x20, 0x01, 0x73, 0x0B]))
        module = decode_module(wasm)
        instance = instantiate(module)
        assert instance.exports.xor(0b1010, 0b0110) == 0b1100

    def test_i32_shl(self):
        wasm = make_simple_func("shl", bytes([0x20, 0x00, 0x20, 0x01, 0x74, 0x0B]))
        module = decode_module(wasm)
        instance = instantiate(module)
        assert instance.exports.shl(1, 4) == 16
        assert instance.exports.shl(0xFF, 8) == 0xFF00


class TestDrop:
    """Test drop instruction."""

    def test_drop(self):
        # i32.const 999, drop, i32.const 42, end
        instr = bytes(
            [
                0x41,
                0xE7,
                0x07,  # i32.const 999
                0x1A,  # drop
                0x41,
                0x2A,  # i32.const 42
                0x0B,  # end
            ]
        )
        wasm = make_simple_func("drp", instr, num_params=0)
        module = decode_module(wasm)
        instance = instantiate(module)
        assert instance.exports.drp() == 42


class TestSelect:
    """Test select instruction."""

    def test_select(self):
        # local.get 0, local.get 1, local.get 0, select, end
        # selects first if third is nonzero, else second
        instr = bytes(
            [
                0x20,
                0x00,  # local.get 0 (val1)
                0x20,
                0x01,  # local.get 1 (val2)
                0x20,
                0x00,  # local.get 0 (condition)
                0x1B,  # select
                0x0B,  # end
            ]
        )
        wasm = make_simple_func("sel", instr)
        module = decode_module(wasm)
        instance = instantiate(module)
        assert instance.exports.sel(1, 99) == 1  # condition=1, select val1
        assert instance.exports.sel(0, 99) == 99  # condition=0, select val2


class TestI64Arithmetic:
    """Test i64 arithmetic operations."""

    def test_i64_add(self):
        # Module with (i64, i64) -> i64 function
        wasm = bytes(
            [
                0x00,
                0x61,
                0x73,
                0x6D,
                0x01,
                0x00,
                0x00,
                0x00,  # magic + version
                # Type section: (i64, i64) -> i64
                0x01,
                0x07,
                0x01,
                0x60,
                0x02,
                0x7E,
                0x7E,
                0x01,
                0x7E,
                # Function section
                0x03,
                0x02,
                0x01,
                0x00,
                # Export section
                0x07,
                0x05,
                0x01,
                0x01,
                0x66,
                0x00,
                0x00,  # "f"
                # Code section: local.get 0, local.get 1, i64.add, end
                0x0A,
                0x09,
                0x01,
                0x07,
                0x00,
                0x20,
                0x00,
                0x20,
                0x01,
                0x7C,
                0x0B,
            ]
        )
        module = decode_module(wasm)
        instance = instantiate(module)
        assert instance.exports.f(10, 20) == 30
        assert instance.exports.f(0, 0) == 0
        # Test large values that would overflow i32
        assert instance.exports.f(0x100000000, 1) == 0x100000001

    def test_i64_add_negative(self):
        wasm = bytes(
            [
                0x00,
                0x61,
                0x73,
                0x6D,
                0x01,
                0x00,
                0x00,
                0x00,
                0x01,
                0x07,
                0x01,
                0x60,
                0x02,
                0x7E,
                0x7E,
                0x01,
                0x7E,
                0x03,
                0x02,
                0x01,
                0x00,
                0x07,
                0x05,
                0x01,
                0x01,
                0x66,
                0x00,
                0x00,
                0x0A,
                0x09,
                0x01,
                0x07,
                0x00,
                0x20,
                0x00,
                0x20,
                0x01,
                0x7C,
                0x0B,
            ]
        )
        module = decode_module(wasm)
        instance = instantiate(module)
        assert instance.exports.f(-1, 1) == 0
        assert instance.exports.f(-10, -20) == -30


class TestF32Arithmetic:
    """Test f32 floating point operations."""

    def test_f32_add(self):
        import struct

        # Module with (f32, f32) -> f32 function
        wasm = bytes(
            [
                0x00,
                0x61,
                0x73,
                0x6D,
                0x01,
                0x00,
                0x00,
                0x00,  # magic + version
                # Type section: (f32, f32) -> f32
                0x01,
                0x07,
                0x01,
                0x60,
                0x02,
                0x7D,
                0x7D,
                0x01,
                0x7D,
                # Function section
                0x03,
                0x02,
                0x01,
                0x00,
                # Export section
                0x07,
                0x05,
                0x01,
                0x01,
                0x66,
                0x00,
                0x00,  # "f"
                # Code section: local.get 0, local.get 1, f32.add, end
                0x0A,
                0x09,
                0x01,
                0x07,
                0x00,
                0x20,
                0x00,
                0x20,
                0x01,
                0x92,
                0x0B,
            ]
        )
        module = decode_module(wasm)
        instance = instantiate(module)
        result = instance.exports.f(1.5, 2.5)
        assert abs(result - 4.0) < 0.0001
        result = instance.exports.f(-1.0, 1.0)
        assert abs(result - 0.0) < 0.0001


class TestMemoryLoad:
    """Test memory load operations."""

    def test_i32_load(self):
        # Module with memory and a function that loads from address
        wasm = bytes(
            [
                0x00,
                0x61,
                0x73,
                0x6D,
                0x01,
                0x00,
                0x00,
                0x00,  # magic + version
                # Type section: (i32) -> i32
                0x01,
                0x06,
                0x01,
                0x60,
                0x01,
                0x7F,
                0x01,
                0x7F,
                # Function section
                0x03,
                0x02,
                0x01,
                0x00,
                # Memory section: 1 page min
                0x05,
                0x03,
                0x01,
                0x00,
                0x01,
                # Export section
                0x07,
                0x05,
                0x01,
                0x01,
                0x66,
                0x00,
                0x00,  # "f"
                # Data section: init memory at offset 0 with bytes [0x2A, 0x00, 0x00, 0x00] = 42
                0x0B,
                0x0A,
                0x01,
                0x00,
                0x41,
                0x00,
                0x0B,
                0x04,
                0x2A,
                0x00,
                0x00,
                0x00,
                # Code section: local.get 0, i32.load (align=2, offset=0), end
                0x0A,
                0x09,
                0x01,
                0x07,
                0x00,
                0x20,
                0x00,
                0x28,
                0x02,
                0x00,
                0x0B,
            ]
        )
        module = decode_module(wasm)
        instance = instantiate(module)
        assert instance.exports.f(0) == 42

    def test_i32_load_with_offset(self):
        # Store 0x12345678 at address 4 in little-endian
        wasm = bytes(
            [
                0x00,
                0x61,
                0x73,
                0x6D,
                0x01,
                0x00,
                0x00,
                0x00,
                0x01,
                0x06,
                0x01,
                0x60,
                0x01,
                0x7F,
                0x01,
                0x7F,
                0x03,
                0x02,
                0x01,
                0x00,
                0x05,
                0x03,
                0x01,
                0x00,
                0x01,
                0x07,
                0x05,
                0x01,
                0x01,
                0x66,
                0x00,
                0x00,
                # Data: offset 4, data = [0x78, 0x56, 0x34, 0x12] (little-endian 0x12345678)
                0x0B,
                0x0A,
                0x01,
                0x00,
                0x41,
                0x04,
                0x0B,
                0x04,
                0x78,
                0x56,
                0x34,
                0x12,
                # Code: local.get 0, i32.load (align=2, offset=0), end
                0x0A,
                0x09,
                0x01,
                0x07,
                0x00,
                0x20,
                0x00,
                0x28,
                0x02,
                0x00,
                0x0B,
            ]
        )
        module = decode_module(wasm)
        instance = instantiate(module)
        assert instance.exports.f(4) == 0x12345678
