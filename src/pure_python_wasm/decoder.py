"""WebAssembly binary format decoder."""

import struct
from typing import BinaryIO
from pathlib import Path

from .errors import DecodeError
from .types import (
    Module,
    FuncType,
    Function,
    Export,
    Import,
    Memory,
    Table,
    Global,
    GlobalType,
    Element,
    Data,
    Limits,
    Instruction,
    VALTYPE_ENCODING,
    EXPORT_KIND_ENCODING,
)
from . import opcodes


# WASM magic number and version
WASM_MAGIC = b"\x00asm"
WASM_VERSION = 1

# Section IDs
SECTION_CUSTOM = 0
SECTION_TYPE = 1
SECTION_IMPORT = 2
SECTION_FUNCTION = 3
SECTION_TABLE = 4
SECTION_MEMORY = 5
SECTION_GLOBAL = 6
SECTION_EXPORT = 7
SECTION_START = 8
SECTION_ELEMENT = 9
SECTION_CODE = 10
SECTION_DATA = 11
SECTION_DATA_COUNT = 12
SECTION_TAG = 13  # Exception handling proposal

# Block type encoding
BLOCK_TYPE_EMPTY = 0x40


class BinaryReader:
    """A reader for binary data with position tracking."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.position = 0

    def read_byte(self) -> int:
        """Read a single byte."""
        if self.position >= len(self.data):
            raise DecodeError(f"Unexpected end of data at position {self.position}")
        byte = self.data[self.position]
        self.position += 1
        return byte

    def read_bytes(self, n: int) -> bytes:
        """Read n bytes."""
        if self.position + n > len(self.data):
            raise DecodeError(
                f"Unexpected end of data: wanted {n} bytes at position {self.position}"
            )
        result = self.data[self.position : self.position + n]
        self.position += n
        return result

    def eof(self) -> bool:
        """Check if at end of data."""
        return self.position >= len(self.data)

    def remaining(self) -> int:
        """Return number of remaining bytes."""
        return len(self.data) - self.position


def decode_unsigned_leb128(reader: BinaryReader, max_bits: int = 32) -> int:
    """Decode an unsigned LEB128 integer."""
    result = 0
    shift = 0
    while True:
        byte = reader.read_byte()
        result |= (byte & 0x7F) << shift
        if (byte & 0x80) == 0:
            break
        shift += 7
        if shift >= max_bits + 7:  # Allow some slack for encoding
            raise DecodeError("LEB128 integer too long")
    return result


def decode_signed_leb128(reader: BinaryReader, max_bits: int = 32) -> int:
    """Decode a signed LEB128 integer."""
    result = 0
    shift = 0
    byte = 0
    while True:
        byte = reader.read_byte()
        result |= (byte & 0x7F) << shift
        shift += 7
        if (byte & 0x80) == 0:
            break
        if shift >= max_bits + 7:
            raise DecodeError("LEB128 integer too long")

    # Sign extend if the sign bit (bit 6 of the last byte) is set
    if byte & 0x40:
        result |= -(1 << shift)

    return result


def decode_name(reader: BinaryReader) -> str:
    """Decode a UTF-8 name (length-prefixed byte vector)."""
    length = decode_unsigned_leb128(reader)
    data = reader.read_bytes(length)
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as e:
        raise DecodeError(f"Invalid UTF-8 in name: {e}") from e


def decode_valtype(reader: BinaryReader) -> str:
    """Decode a value type."""
    byte = reader.read_byte()
    if byte not in VALTYPE_ENCODING:
        raise DecodeError(f"Unknown value type: 0x{byte:02x}")
    return VALTYPE_ENCODING[byte]


def decode_limits(reader: BinaryReader) -> Limits:
    """Decode limits (min, optional max)."""
    flags = reader.read_byte()
    min_val = decode_unsigned_leb128(reader)
    max_val = None
    if flags & 0x01:
        max_val = decode_unsigned_leb128(reader)
    return Limits(min=min_val, max=max_val)


def decode_blocktype(reader: BinaryReader) -> tuple | str | int:
    """Decode a block type (empty, valtype, or type index)."""
    byte = reader.read_byte()
    if byte == BLOCK_TYPE_EMPTY:
        return ()  # Empty result
    if byte in VALTYPE_ENCODING:
        return (VALTYPE_ENCODING[byte],)  # Single result type
    # Otherwise it's a signed type index (for multi-value)
    # Need to put the byte back and read as signed LEB128
    reader.position -= 1
    return decode_signed_leb128(reader)


def decode_extended_instruction(sub_opcode: int, reader: BinaryReader) -> Instruction:
    """Decode an extended instruction (0xFC prefix)."""
    # Create virtual opcode for internal use
    virtual_opcode = 0xFC00 | sub_opcode

    if virtual_opcode not in opcodes.OPCODE_NAMES:
        raise DecodeError(f"Unknown extended opcode: 0xFC 0x{sub_opcode:02x}")

    name = opcodes.OPCODE_NAMES[virtual_opcode]

    # Saturating truncation operations (no immediates)
    if sub_opcode <= 0x07:
        return Instruction(name)

    # Bulk memory operations
    if sub_opcode == 0x08:  # memory.init
        data_idx = decode_unsigned_leb128(reader)
        mem_idx = decode_unsigned_leb128(reader)
        return Instruction(name, (data_idx, mem_idx))

    if sub_opcode == 0x09:  # data.drop
        data_idx = decode_unsigned_leb128(reader)
        return Instruction(name, data_idx)

    if sub_opcode == 0x0A:  # memory.copy
        dst_mem = decode_unsigned_leb128(reader)
        src_mem = decode_unsigned_leb128(reader)
        return Instruction(name, (dst_mem, src_mem))

    if sub_opcode == 0x0B:  # memory.fill
        mem_idx = decode_unsigned_leb128(reader)
        return Instruction(name, mem_idx)

    # table.size takes a table index
    if sub_opcode == 0x10:  # table.size
        table_idx = decode_unsigned_leb128(reader)
        return Instruction(name, table_idx)

    # table.grow takes table index
    if sub_opcode == 0x0F:  # table.grow
        table_idx = decode_unsigned_leb128(reader)
        return Instruction(name, table_idx)

    # table.fill takes table index
    if sub_opcode == 0x11:  # table.fill
        table_idx = decode_unsigned_leb128(reader)
        return Instruction(name, table_idx)

    # table.copy takes two table indices
    if sub_opcode == 0x0E:  # table.copy
        dst_table = decode_unsigned_leb128(reader)
        src_table = decode_unsigned_leb128(reader)
        return Instruction(name, (dst_table, src_table))

    # table.init takes elem index and table index
    if sub_opcode == 0x0C:  # table.init
        elem_idx = decode_unsigned_leb128(reader)
        table_idx = decode_unsigned_leb128(reader)
        return Instruction(name, (elem_idx, table_idx))

    # elem.drop takes elem index
    if sub_opcode == 0x0D:  # elem.drop
        elem_idx = decode_unsigned_leb128(reader)
        return Instruction(name, elem_idx)

    raise DecodeError(f"Unhandled extended opcode: 0xFC 0x{sub_opcode:02x}")


def decode_instruction(reader: BinaryReader) -> Instruction:
    """Decode a single instruction."""
    opcode = reader.read_byte()

    # Handle extended opcode prefix (0xFC)
    if opcode == opcodes.EXTENDED_PREFIX:
        sub_opcode = decode_unsigned_leb128(reader)
        return decode_extended_instruction(sub_opcode, reader)

    # Get opcode name
    if opcode not in opcodes.OPCODE_NAMES:
        raise DecodeError(f"Unknown opcode: 0x{opcode:02x}")
    name = opcodes.OPCODE_NAMES[opcode]

    # Handle different immediate types
    if opcode in opcodes.NO_IMMEDIATE:
        return Instruction(name)

    if opcode in opcodes.U32_IMMEDIATE:
        operand = decode_unsigned_leb128(reader)
        return Instruction(name, operand)

    if opcode in opcodes.I32_IMMEDIATE:
        operand = decode_signed_leb128(reader, 32)
        return Instruction(name, operand)

    if opcode in opcodes.I64_IMMEDIATE:
        operand = decode_signed_leb128(reader, 64)
        return Instruction(name, operand)

    if opcode in opcodes.F32_IMMEDIATE:
        data = reader.read_bytes(4)
        operand = struct.unpack("<f", data)[0]
        return Instruction(name, operand)

    if opcode in opcodes.F64_IMMEDIATE:
        data = reader.read_bytes(8)
        operand = struct.unpack("<d", data)[0]
        return Instruction(name, operand)

    if opcode in opcodes.MEMORY_IMMEDIATE:
        align = decode_unsigned_leb128(reader)
        offset = decode_unsigned_leb128(reader)
        return Instruction(name, (align, offset))

    if opcode in opcodes.BLOCK_TYPE:
        blocktype = decode_blocktype(reader)
        return Instruction(name, blocktype)

    if opcode == opcodes.BR_TABLE:
        # Vector of labels + default label
        count = decode_unsigned_leb128(reader)
        labels = [decode_unsigned_leb128(reader) for _ in range(count)]
        default = decode_unsigned_leb128(reader)
        return Instruction(name, (labels, default))

    if opcode == opcodes.CALL_INDIRECT:
        type_idx = decode_unsigned_leb128(reader)
        table_idx = decode_unsigned_leb128(reader)
        return Instruction(name, (type_idx, table_idx))

    if opcode == opcodes.MEMORY_SIZE or opcode == opcodes.MEMORY_GROW:
        # Memory index (always 0 in MVP)
        _ = reader.read_byte()  # Reserved byte, must be 0
        return Instruction(name)

    if opcode == opcodes.SELECT_T:
        # Typed select with result types
        count = decode_unsigned_leb128(reader)
        types = tuple(decode_valtype(reader) for _ in range(count))
        return Instruction(name, types)

    if opcode == opcodes.REF_NULL:
        reftype = decode_valtype(reader)
        return Instruction(name, reftype)

    raise DecodeError(f"Unhandled opcode: 0x{opcode:02x} ({name})")


def decode_expr(reader: BinaryReader) -> list[Instruction]:
    """Decode an expression (instruction sequence ending with END)."""
    instructions = []
    depth = 0
    while True:
        instr = decode_instruction(reader)
        instructions.append(instr)

        # Track nesting for block/loop/if
        if instr.opcode in ("block", "loop", "if"):
            depth += 1
        elif instr.opcode == "end":
            if depth == 0:
                break
            depth -= 1

    return instructions


def decode_func_type(reader: BinaryReader) -> FuncType:
    """Decode a function type."""
    marker = reader.read_byte()
    if marker != 0x60:
        raise DecodeError(f"Expected function type marker 0x60, got 0x{marker:02x}")

    # Parameters
    param_count = decode_unsigned_leb128(reader)
    params = tuple(decode_valtype(reader) for _ in range(param_count))

    # Results
    result_count = decode_unsigned_leb128(reader)
    results = tuple(decode_valtype(reader) for _ in range(result_count))

    return FuncType(params, results)


def decode_type_section(reader: BinaryReader, module: Module) -> None:
    """Decode the type section."""
    count = decode_unsigned_leb128(reader)
    for _ in range(count):
        module.types.append(decode_func_type(reader))


def decode_import_section(reader: BinaryReader, module: Module) -> None:
    """Decode the import section."""
    count = decode_unsigned_leb128(reader)
    for _ in range(count):
        mod_name = decode_name(reader)
        name = decode_name(reader)
        kind = reader.read_byte()

        if kind == 0x00:  # func
            type_idx = decode_unsigned_leb128(reader)
            module.imports.append(Import(mod_name, name, "func", type_idx))
        elif kind == 0x01:  # table
            elem_type = decode_valtype(reader)
            limits = decode_limits(reader)
            module.imports.append(Import(mod_name, name, "table", (elem_type, limits)))
        elif kind == 0x02:  # memory
            limits = decode_limits(reader)
            module.imports.append(Import(mod_name, name, "memory", limits))
        elif kind == 0x03:  # global
            valtype = decode_valtype(reader)
            mutable = reader.read_byte() != 0
            module.imports.append(
                Import(mod_name, name, "global", GlobalType(valtype, mutable))
            )
        else:
            raise DecodeError(f"Unknown import kind: {kind}")


def decode_function_section(reader: BinaryReader, module: Module) -> None:
    """Decode the function section (just type indices)."""
    count = decode_unsigned_leb128(reader)
    # We store the type indices temporarily; bodies come from code section
    module._func_type_indices = [decode_unsigned_leb128(reader) for _ in range(count)]


def decode_table_section(reader: BinaryReader, module: Module) -> None:
    """Decode the table section."""
    count = decode_unsigned_leb128(reader)
    for _ in range(count):
        elem_type = decode_valtype(reader)
        limits = decode_limits(reader)
        module.tables.append(Table(elem_type, limits))


def decode_memory_section(reader: BinaryReader, module: Module) -> None:
    """Decode the memory section."""
    count = decode_unsigned_leb128(reader)
    for _ in range(count):
        limits = decode_limits(reader)
        module.mems.append(Memory(limits))


def decode_global_section(reader: BinaryReader, module: Module) -> None:
    """Decode the global section."""
    count = decode_unsigned_leb128(reader)
    for _ in range(count):
        valtype = decode_valtype(reader)
        mutable = reader.read_byte() != 0
        init = decode_expr(reader)
        module.globals.append(Global(GlobalType(valtype, mutable), init))


def decode_export_section(reader: BinaryReader, module: Module) -> None:
    """Decode the export section."""
    count = decode_unsigned_leb128(reader)
    for _ in range(count):
        name = decode_name(reader)
        kind_byte = reader.read_byte()
        if kind_byte not in EXPORT_KIND_ENCODING:
            raise DecodeError(f"Unknown export kind: {kind_byte}")
        kind = EXPORT_KIND_ENCODING[kind_byte]
        index = decode_unsigned_leb128(reader)
        module.exports.append(Export(name, kind, index))


def decode_start_section(reader: BinaryReader, module: Module) -> None:
    """Decode the start section."""
    module.start = decode_unsigned_leb128(reader)


def decode_element_section(reader: BinaryReader, module: Module) -> None:
    """Decode the element section.

    Element segment flags:
    - 0: Active, table 0, funcref, expr offset, vec(funcidx)
    - 1: Passive, elemkind, vec(funcidx)
    - 2: Active, tableidx, elemkind, expr offset, vec(funcidx)
    - 3: Declarative, elemkind, vec(funcidx)
    - 4: Active, table 0, expr offset, vec(expr)
    - 5: Passive, reftype, vec(expr)
    - 6: Active, tableidx, reftype, expr offset, vec(expr)
    - 7: Declarative, reftype, vec(expr)
    """
    count = decode_unsigned_leb128(reader)
    for _ in range(count):
        flags = decode_unsigned_leb128(reader)

        if flags == 0:
            # Active segment for table 0, funcref, vec(funcidx)
            offset = decode_expr(reader)
            func_count = decode_unsigned_leb128(reader)
            func_indices = [decode_unsigned_leb128(reader) for _ in range(func_count)]
            module.elem.append(Element(0, offset, func_indices))

        elif flags == 1:
            # Passive segment, elemkind, vec(funcidx)
            _elemkind = reader.read_byte()  # Must be 0x00 for funcref
            func_count = decode_unsigned_leb128(reader)
            func_indices = [decode_unsigned_leb128(reader) for _ in range(func_count)]
            # Passive segments have table_idx = -1 (not active)
            module.elem.append(Element(-1, [], func_indices))

        elif flags == 2:
            # Active segment with table index, elemkind, vec(funcidx)
            table_idx = decode_unsigned_leb128(reader)
            offset = decode_expr(reader)
            _elemkind = reader.read_byte()  # Must be 0x00 for funcref
            func_count = decode_unsigned_leb128(reader)
            func_indices = [decode_unsigned_leb128(reader) for _ in range(func_count)]
            module.elem.append(Element(table_idx, offset, func_indices))

        elif flags == 3:
            # Declarative segment, elemkind, vec(funcidx)
            _elemkind = reader.read_byte()
            func_count = decode_unsigned_leb128(reader)
            func_indices = [decode_unsigned_leb128(reader) for _ in range(func_count)]
            # Declarative segments declare functions for validation, not for tables
            module.elem.append(Element(-2, [], func_indices))

        elif flags == 4:
            # Active segment for table 0, vec(expr)
            offset = decode_expr(reader)
            elem_count = decode_unsigned_leb128(reader)
            # Each element is an expression (we only handle ref.func for now)
            func_indices = []
            for _ in range(elem_count):
                expr = decode_expr(reader)
                # Extract function index from ref.func instruction
                for instr in expr:
                    if instr.opcode == "ref.func":
                        func_indices.append(instr.operand)
                        break
                    elif instr.opcode == "ref.null":
                        func_indices.append(None)
                        break
            module.elem.append(Element(0, offset, func_indices))

        elif flags == 5:
            # Passive segment, reftype, vec(expr)
            _reftype = decode_valtype(reader)
            elem_count = decode_unsigned_leb128(reader)
            func_indices = []
            for _ in range(elem_count):
                expr = decode_expr(reader)
                for instr in expr:
                    if instr.opcode == "ref.func":
                        func_indices.append(instr.operand)
                        break
                    elif instr.opcode == "ref.null":
                        func_indices.append(None)
                        break
            module.elem.append(Element(-1, [], func_indices))

        elif flags == 6:
            # Active segment with table index, reftype, vec(expr)
            table_idx = decode_unsigned_leb128(reader)
            offset = decode_expr(reader)
            _reftype = decode_valtype(reader)
            elem_count = decode_unsigned_leb128(reader)
            func_indices = []
            for _ in range(elem_count):
                expr = decode_expr(reader)
                for instr in expr:
                    if instr.opcode == "ref.func":
                        func_indices.append(instr.operand)
                        break
                    elif instr.opcode == "ref.null":
                        func_indices.append(None)
                        break
            module.elem.append(Element(table_idx, offset, func_indices))

        elif flags == 7:
            # Declarative segment, reftype, vec(expr)
            _reftype = decode_valtype(reader)
            elem_count = decode_unsigned_leb128(reader)
            func_indices = []
            for _ in range(elem_count):
                expr = decode_expr(reader)
                for instr in expr:
                    if instr.opcode == "ref.func":
                        func_indices.append(instr.operand)
                        break
                    elif instr.opcode == "ref.null":
                        func_indices.append(None)
                        break
            module.elem.append(Element(-2, [], func_indices))

        else:
            raise DecodeError(f"Unsupported element segment flags: {flags}")


def decode_code_section(reader: BinaryReader, module: Module) -> None:
    """Decode the code section."""
    count = decode_unsigned_leb128(reader)

    if not hasattr(module, "_func_type_indices"):
        raise DecodeError("Code section without function section")

    if count != len(module._func_type_indices):
        raise DecodeError(
            f"Code section count ({count}) != function section count "
            f"({len(module._func_type_indices)})"
        )

    for i in range(count):
        body_size = decode_unsigned_leb128(reader)
        body_start = reader.position

        # Local declarations
        local_count = decode_unsigned_leb128(reader)
        locals_list: list[str] = []
        for _ in range(local_count):
            n = decode_unsigned_leb128(reader)
            valtype = decode_valtype(reader)
            locals_list.extend([valtype] * n)

        # Instructions
        body = decode_expr(reader)

        # Verify we consumed exactly body_size bytes
        consumed = reader.position - body_start
        if consumed != body_size:
            raise DecodeError(
                f"Function body size mismatch: expected {body_size}, got {consumed}"
            )

        module.funcs.append(
            Function(
                type_idx=module._func_type_indices[i],
                locals=tuple(locals_list),
                body=body,
            )
        )


def decode_data_section(reader: BinaryReader, module: Module) -> None:
    """Decode the data section."""
    count = decode_unsigned_leb128(reader)
    for _ in range(count):
        flags = decode_unsigned_leb128(reader)

        if flags == 0:
            # Active segment for memory 0
            offset = decode_expr(reader)
            length = decode_unsigned_leb128(reader)
            init = reader.read_bytes(length)
            module.data.append(Data(0, offset, init))
        elif flags == 1:
            # Passive segment
            length = decode_unsigned_leb128(reader)
            init = reader.read_bytes(length)
            module.data.append(Data(-1, [], init))  # -1 indicates passive
        elif flags == 2:
            # Active segment with explicit memory index
            mem_idx = decode_unsigned_leb128(reader)
            offset = decode_expr(reader)
            length = decode_unsigned_leb128(reader)
            init = reader.read_bytes(length)
            module.data.append(Data(mem_idx, offset, init))
        else:
            raise DecodeError(f"Unsupported data segment flags: {flags}")


def decode_section(reader: BinaryReader, module: Module) -> None:
    """Decode a single section."""
    section_id = reader.read_byte()
    section_size = decode_unsigned_leb128(reader)
    section_end = reader.position + section_size

    # Create a sub-reader for the section content
    section_data = reader.read_bytes(section_size)
    section_reader = BinaryReader(section_data)

    if section_id == SECTION_CUSTOM:
        # Skip custom sections for now (could parse names section later)
        pass
    elif section_id == SECTION_TYPE:
        decode_type_section(section_reader, module)
    elif section_id == SECTION_IMPORT:
        decode_import_section(section_reader, module)
    elif section_id == SECTION_FUNCTION:
        decode_function_section(section_reader, module)
    elif section_id == SECTION_TABLE:
        decode_table_section(section_reader, module)
    elif section_id == SECTION_MEMORY:
        decode_memory_section(section_reader, module)
    elif section_id == SECTION_GLOBAL:
        decode_global_section(section_reader, module)
    elif section_id == SECTION_EXPORT:
        decode_export_section(section_reader, module)
    elif section_id == SECTION_START:
        decode_start_section(section_reader, module)
    elif section_id == SECTION_ELEMENT:
        decode_element_section(section_reader, module)
    elif section_id == SECTION_CODE:
        decode_code_section(section_reader, module)
    elif section_id == SECTION_DATA:
        decode_data_section(section_reader, module)
    elif section_id == SECTION_DATA_COUNT:
        # Data count section (for bulk memory)
        _ = decode_unsigned_leb128(section_reader)  # Just read and ignore for now
    elif section_id == SECTION_TAG:
        # Tag section (for exception handling proposal)
        # Skip for now - would need to decode tag types for try/catch support
        pass
    else:
        raise DecodeError(f"Unknown section id: {section_id}")


def decode_module(source: bytes | BinaryIO | Path) -> Module:
    """Decode a WebAssembly module from binary format.

    Args:
        source: WASM bytes, file-like object, or path to .wasm file

    Returns:
        Decoded Module object

    Raises:
        DecodeError: If the binary format is invalid
    """
    # Handle different source types
    if isinstance(source, Path):
        data = source.read_bytes()
    elif isinstance(source, bytes):
        data = source
    else:
        # Assume file-like object
        data = source.read()

    reader = BinaryReader(data)

    # Check magic number
    magic = reader.read_bytes(4)
    if magic != WASM_MAGIC:
        raise DecodeError(
            f"Invalid WASM magic number: expected {WASM_MAGIC!r}, got {magic!r}"
        )

    # Check version
    version_bytes = reader.read_bytes(4)
    version = int.from_bytes(version_bytes, "little")
    if version != WASM_VERSION:
        raise DecodeError(f"Unsupported WASM version: {version}")

    # Create module and decode sections
    module = Module()

    while not reader.eof():
        decode_section(reader, module)

    return module
