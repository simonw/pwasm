"""Tests for the WebAssembly binary decoder."""

import pytest
from pure_python_wasm.decoder import (
    BinaryReader,
    decode_module,
    decode_unsigned_leb128,
    decode_signed_leb128,
)
from pure_python_wasm.errors import DecodeError


class TestLEB128:
    """Test LEB128 variable-length integer encoding."""

    def test_decode_unsigned_zero(self):
        reader = BinaryReader(bytes([0x00]))
        assert decode_unsigned_leb128(reader) == 0

    def test_decode_unsigned_single_byte(self):
        reader = BinaryReader(bytes([0x01]))
        assert decode_unsigned_leb128(reader) == 1

        reader = BinaryReader(bytes([0x7F]))
        assert decode_unsigned_leb128(reader) == 127

    def test_decode_unsigned_multibyte(self):
        # 128 = 0x80 0x01
        reader = BinaryReader(bytes([0x80, 0x01]))
        assert decode_unsigned_leb128(reader) == 128

        # 624485 = 0xE5 0x8E 0x26
        reader = BinaryReader(bytes([0xE5, 0x8E, 0x26]))
        assert decode_unsigned_leb128(reader) == 624485

    def test_decode_unsigned_max_u32(self):
        # 2^32 - 1 = 0xFF 0xFF 0xFF 0xFF 0x0F
        reader = BinaryReader(bytes([0xFF, 0xFF, 0xFF, 0xFF, 0x0F]))
        assert decode_unsigned_leb128(reader) == 0xFFFFFFFF

    def test_decode_signed_zero(self):
        reader = BinaryReader(bytes([0x00]))
        assert decode_signed_leb128(reader) == 0

    def test_decode_signed_positive(self):
        reader = BinaryReader(bytes([0x01]))
        assert decode_signed_leb128(reader) == 1

        reader = BinaryReader(bytes([0x3F]))
        assert decode_signed_leb128(reader) == 63

    def test_decode_signed_negative(self):
        # -1 = 0x7F
        reader = BinaryReader(bytes([0x7F]))
        assert decode_signed_leb128(reader) == -1

        # -123456 = 0xC0 0xBB 0x78
        reader = BinaryReader(bytes([0xC0, 0xBB, 0x78]))
        assert decode_signed_leb128(reader) == -123456

    def test_decode_signed_min_i32(self):
        # -2^31 = 0x80 0x80 0x80 0x80 0x78
        reader = BinaryReader(bytes([0x80, 0x80, 0x80, 0x80, 0x78]))
        assert decode_signed_leb128(reader) == -(2**31)


class TestBinaryReader:
    """Test the binary reader helper class."""

    def test_read_bytes(self):
        reader = BinaryReader(bytes([1, 2, 3, 4, 5]))
        assert reader.read_bytes(3) == bytes([1, 2, 3])
        assert reader.read_bytes(2) == bytes([4, 5])

    def test_read_byte(self):
        reader = BinaryReader(bytes([0xAB, 0xCD]))
        assert reader.read_byte() == 0xAB
        assert reader.read_byte() == 0xCD

    def test_position_tracking(self):
        reader = BinaryReader(bytes([1, 2, 3, 4, 5]))
        assert reader.position == 0
        reader.read_byte()
        assert reader.position == 1
        reader.read_bytes(2)
        assert reader.position == 3

    def test_eof(self):
        reader = BinaryReader(bytes([1, 2]))
        assert not reader.eof()
        reader.read_bytes(2)
        assert reader.eof()

    def test_read_past_eof_raises(self):
        reader = BinaryReader(bytes([1]))
        reader.read_byte()
        with pytest.raises(DecodeError):
            reader.read_byte()


class TestDecodeModule:
    """Test complete module decoding."""

    def test_decode_minimal_module(self):
        # Minimal valid WASM: magic + version + empty
        wasm = bytes(
            [
                0x00,
                0x61,
                0x73,
                0x6D,  # magic: \0asm
                0x01,
                0x00,
                0x00,
                0x00,  # version: 1
            ]
        )
        module = decode_module(wasm)
        assert module is not None
        assert module.types == []
        assert module.funcs == []

    def test_decode_invalid_magic(self):
        wasm = bytes(
            [
                0x00,
                0x00,
                0x00,
                0x00,  # wrong magic
                0x01,
                0x00,
                0x00,
                0x00,
            ]
        )
        with pytest.raises(DecodeError, match="magic"):
            decode_module(wasm)

    def test_decode_invalid_version(self):
        wasm = bytes(
            [
                0x00,
                0x61,
                0x73,
                0x6D,  # magic
                0x02,
                0x00,
                0x00,
                0x00,  # version 2 (unsupported)
            ]
        )
        with pytest.raises(DecodeError, match="version"):
            decode_module(wasm)

    def test_decode_type_section(self):
        # Module with one function type: (i32, i32) -> i32
        wasm = bytes(
            [
                0x00,
                0x61,
                0x73,
                0x6D,  # magic
                0x01,
                0x00,
                0x00,
                0x00,  # version
                # Type section
                0x01,  # section id: type
                0x07,  # section size: 7 bytes
                0x01,  # number of types: 1
                0x60,  # func type marker
                0x02,  # number of params: 2
                0x7F,  # param 1: i32
                0x7F,  # param 2: i32
                0x01,  # number of results: 1
                0x7F,  # result: i32
            ]
        )
        module = decode_module(wasm)
        assert len(module.types) == 1
        assert module.types[0].params == ("i32", "i32")
        assert module.types[0].results == ("i32",)

    def test_decode_function_and_code_sections(self):
        # Module with one function that adds two i32s
        wasm = bytes(
            [
                0x00,
                0x61,
                0x73,
                0x6D,  # magic
                0x01,
                0x00,
                0x00,
                0x00,  # version
                # Type section
                0x01,  # section id: type
                0x07,  # section size
                0x01,  # 1 type
                0x60,  # func
                0x02,
                0x7F,
                0x7F,  # (i32, i32)
                0x01,
                0x7F,  # -> i32
                # Function section
                0x03,  # section id: function
                0x02,  # section size
                0x01,  # 1 function
                0x00,  # type index 0
                # Code section
                0x0A,  # section id: code
                0x09,  # section size
                0x01,  # 1 function body
                0x07,  # body size
                0x00,  # local count: 0
                0x20,
                0x00,  # local.get 0
                0x20,
                0x01,  # local.get 1
                0x6A,  # i32.add
                0x0B,  # end
            ]
        )
        module = decode_module(wasm)
        assert len(module.funcs) == 1
        func = module.funcs[0]
        assert func.type_idx == 0
        assert func.locals == ()
        assert len(func.body) == 4  # local.get, local.get, i32.add, end

    def test_decode_export_section(self):
        # Module with exported function "add"
        wasm = bytes(
            [
                0x00,
                0x61,
                0x73,
                0x6D,  # magic
                0x01,
                0x00,
                0x00,
                0x00,  # version
                # Type section
                0x01,
                0x07,
                0x01,
                0x60,
                0x02,
                0x7F,
                0x7F,
                0x01,
                0x7F,
                # Function section
                0x03,
                0x02,
                0x01,
                0x00,
                # Export section
                0x07,  # section id: export
                0x07,  # section size
                0x01,  # 1 export
                0x03,  # name length: 3
                0x61,
                0x64,
                0x64,  # "add"
                0x00,  # export kind: func
                0x00,  # func index: 0
                # Code section
                0x0A,
                0x09,
                0x01,
                0x07,
                0x00,
                0x20,
                0x00,
                0x20,
                0x01,
                0x6A,
                0x0B,
            ]
        )
        module = decode_module(wasm)
        assert len(module.exports) == 1
        export = module.exports[0]
        assert export.name == "add"
        assert export.kind == "func"
        assert export.index == 0


class TestDecodeInstructions:
    """Test instruction parsing."""

    def test_decode_i32_const(self):
        # Just the code section with i32.const 42
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
                # Type: () -> i32
                0x01,
                0x05,
                0x01,
                0x60,
                0x00,
                0x01,
                0x7F,
                # Function
                0x03,
                0x02,
                0x01,
                0x00,
                # Code
                0x0A,
                0x06,
                0x01,
                0x04,
                0x00,
                0x41,
                0x2A,  # i32.const 42
                0x0B,  # end
            ]
        )
        module = decode_module(wasm)
        func = module.funcs[0]
        assert func.body[0].opcode == "i32.const"
        assert func.body[0].operand == 42

    def test_decode_local_get(self):
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
                # Type: (i32) -> i32
                0x01,
                0x06,
                0x01,
                0x60,
                0x01,
                0x7F,
                0x01,
                0x7F,
                # Function
                0x03,
                0x02,
                0x01,
                0x00,
                # Code
                0x0A,
                0x06,
                0x01,
                0x04,
                0x00,
                0x20,
                0x00,  # local.get 0
                0x0B,  # end
            ]
        )
        module = decode_module(wasm)
        func = module.funcs[0]
        assert func.body[0].opcode == "local.get"
        assert func.body[0].operand == 0
