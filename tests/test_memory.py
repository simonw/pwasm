"""Linear memory: loads, stores, memory.size and memory.grow."""

import struct

import pytest

from pwasm import TrapError
from wat import load

LOADS_AND_STORES = """(module
  (memory (export "memory") 1 3)
  (func (export "store32") (param i32 i32) (i32.store (local.get 0) (local.get 1)))
  (func (export "store8") (param i32 i32) (i32.store8 (local.get 0) (local.get 1)))
  (func (export "store16") (param i32 i32) (i32.store16 (local.get 0) (local.get 1)))
  (func (export "store64") (param i32 i64) (i64.store (local.get 0) (local.get 1)))
  (func (export "store64_8") (param i32 i64) (i64.store8 (local.get 0) (local.get 1)))
  (func (export "store64_16") (param i32 i64) (i64.store16 (local.get 0) (local.get 1)))
  (func (export "store64_32") (param i32 i64) (i64.store32 (local.get 0) (local.get 1)))
  (func (export "storef32") (param i32 f32) (f32.store (local.get 0) (local.get 1)))
  (func (export "storef64") (param i32 f64) (f64.store (local.get 0) (local.get 1)))
  (func (export "load32") (param i32) (result i32) (i32.load (local.get 0)))
  (func (export "load8_s") (param i32) (result i32) (i32.load8_s (local.get 0)))
  (func (export "load8_u") (param i32) (result i32) (i32.load8_u (local.get 0)))
  (func (export "load16_s") (param i32) (result i32) (i32.load16_s (local.get 0)))
  (func (export "load16_u") (param i32) (result i32) (i32.load16_u (local.get 0)))
  (func (export "load64") (param i32) (result i64) (i64.load (local.get 0)))
  (func (export "load64_8_s") (param i32) (result i64) (i64.load8_s (local.get 0)))
  (func (export "load64_8_u") (param i32) (result i64) (i64.load8_u (local.get 0)))
  (func (export "load64_16_s") (param i32) (result i64) (i64.load16_s (local.get 0)))
  (func (export "load64_16_u") (param i32) (result i64) (i64.load16_u (local.get 0)))
  (func (export "load64_32_s") (param i32) (result i64) (i64.load32_s (local.get 0)))
  (func (export "load64_32_u") (param i32) (result i64) (i64.load32_u (local.get 0)))
  (func (export "loadf32") (param i32) (result f32) (f32.load (local.get 0)))
  (func (export "loadf64") (param i32) (result f64) (f64.load (local.get 0)))
  (func (export "load_offset") (param i32) (result i32) (i32.load offset=4 (local.get 0)))
  (func (export "size") (result i32) (memory.size))
  (func (export "grow") (param i32) (result i32) (memory.grow (local.get 0)))
  (func (export "copy_f32_bits") (param i32 i32)
    (f32.store (local.get 1) (f32.load (local.get 0)))))"""


@pytest.fixture
def m():
    return load(LOADS_AND_STORES).exports


def test_store_and_load_i32(m):
    m.store32(8, -2)
    assert m.load32(8) == -2
    assert bytes(m.memory.data[8:12]) == b"\xfe\xff\xff\xff"


def test_narrow_stores_and_signed_loads(m):
    m.store8(0, 0x1FF)
    assert m.load8_u(0) == 0xFF
    assert m.load8_s(0) == -1
    m.store16(2, 0x18000)
    assert m.load16_u(2) == 0x8000
    assert m.load16_s(2) == -0x8000


def test_i64_loads_and_stores(m):
    m.store64(16, -3)
    assert m.load64(16) == -3
    m.store64_8(32, 0x180)
    assert m.load64_8_u(32) == 0x80
    assert m.load64_8_s(32) == -0x80
    m.store64_16(40, 0x8001)
    assert m.load64_16_s(40) == -0x7FFF
    assert m.load64_16_u(40) == 0x8001
    m.store64_32(48, 0x1_8000_0000)
    assert m.load64_32_s(48) == -0x80000000
    assert m.load64_32_u(48) == 0x80000000


def test_float_loads_and_stores(m):
    m.storef32(0, 1.5)
    assert m.loadf32(0) == 1.5
    m.storef64(8, -2.25)
    assert m.loadf64(8) == -2.25
    assert struct.unpack("<d", bytes(m.memory.data[8:16]))[0] == -2.25


def test_f32_nan_bits_survive_load_and_store(m):
    m.memory.data[0:4] = struct.pack("<I", 0x7FA00001)  # signalling NaN
    m.copy_f32_bits(0, 4)
    assert bytes(m.memory.data[4:8]) == struct.pack("<I", 0x7FA00001)


def test_offset_immediate(m):
    m.store32(12, 99)
    assert m.load_offset(8) == 99


def test_out_of_bounds_traps(m):
    with pytest.raises(TrapError, match="out of bounds memory access"):
        m.load32(65534)
    with pytest.raises(TrapError, match="out of bounds memory access"):
        m.store8(65536, 1)
    with pytest.raises(TrapError, match="out of bounds memory access"):
        m.load_offset(-1)  # 0xFFFFFFFF + 4 does not wrap around


def test_memory_size_and_grow(m):
    assert m.size() == 1
    assert m.grow(1) == 1
    assert m.size() == 2
    m.store32(65536 + 100, 7)
    assert m.load32(65536 + 100) == 7
    assert m.grow(2) == -1  # max is 3 pages
    assert m.grow(1) == 2
    assert m.size() == 3
