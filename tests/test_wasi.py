"""WasiLite: a small WASI preview1 subset with no filesystem access."""

import struct

import pytest

from pwasm import decode_module, instantiate
from pwasm.wasi import Exit, WasiLite
from wat import wat2wasm

pytestmark = pytest.mark.usefixtures("each_mode")

MODULE = """(module
  (import "wasi_snapshot_preview1" "fd_write" (func $fd_write (param i32 i32 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "fd_read" (func $fd_read (param i32 i32 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "args_sizes_get" (func $args_sizes_get (param i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "args_get" (func $args_get (param i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "environ_sizes_get" (func $environ_sizes_get (param i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "environ_get" (func $environ_get (param i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "clock_time_get" (func $clock_time_get (param i32 i64 i32) (result i32)))
  (import "wasi_snapshot_preview1" "random_get" (func $random_get (param i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "path_open"
    (func $path_open (param i32 i32 i32 i32 i32 i64 i64 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "sock_accept" (func $sock_accept (param i32 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "proc_exit" (func $proc_exit (param i32)))
  (memory (export "memory") 1)
  ;; iovecs at 0: two buffers, "hello " at 100 and "world\\n" at 200
  (data (i32.const 0) "\\64\\00\\00\\00\\06\\00\\00\\00\\c8\\00\\00\\00\\06\\00\\00\\00")
  (data (i32.const 100) "hello ")
  (data (i32.const 200) "world\\n")
  (func (export "write") (param i32) (result i32)
    (call $fd_write (local.get 0) (i32.const 0) (i32.const 2) (i32.const 300)))
  (func (export "read") (result i32)
    (call $fd_read (i32.const 0) (i32.const 0) (i32.const 1) (i32.const 300)))
  (func (export "args_sizes") (result i32) (call $args_sizes_get (i32.const 400) (i32.const 404)))
  (func (export "args") (result i32) (call $args_get (i32.const 500) (i32.const 600)))
  (func (export "environ_sizes") (result i32) (call $environ_sizes_get (i32.const 400) (i32.const 404)))
  (func (export "environ") (result i32) (call $environ_get (i32.const 500) (i32.const 600)))
  (func (export "clock") (result i32) (call $clock_time_get (i32.const 0) (i64.const 1) (i32.const 400)))
  (func (export "random") (result i32) (call $random_get (i32.const 700) (i32.const 16)))
  (func (export "open") (result i32)
    (call $path_open (i32.const 3) (i32.const 0) (i32.const 0) (i32.const 0) (i32.const 0)
      (i64.const 0) (i64.const 0) (i32.const 0) (i32.const 400)))
  (func (export "sock") (result i32) (call $sock_accept (i32.const 0) (i32.const 0) (i32.const 0)))
  (func (export "exit") (param i32) (call $proc_exit (local.get 0))))"""


def setup(**kwargs):
    module = decode_module(wat2wasm(MODULE))
    wasi = WasiLite(**kwargs)
    instance = instantiate(module, {"wasi_snapshot_preview1": wasi.imports(module)})
    wasi.bind(instance)
    return wasi, instance.exports


def u32(memory, ptr):
    return struct.unpack("<I", memory.read(ptr, 4))[0]


def test_fd_write_captures_stdout_and_stderr():
    wasi, e = setup()
    assert e.write(1) == 0
    assert u32(e.memory, 300) == 12
    assert e.write(2) == 0
    assert wasi.stdout == b"hello world\n"
    assert wasi.stderr == b"hello world\n"
    assert e.write(7) == WasiLite.EBADF


def test_fd_write_sinks():
    seen = []
    wasi, e = setup()
    wasi.sinks[1] = seen.append
    e.write(1)
    assert seen == [b"hello world\n"]
    assert wasi.stdout == b""


def test_fd_read_from_stdin():
    wasi, e = setup(stdin=b"typed")
    assert e.read() == 0
    assert u32(e.memory, 300) == 5
    assert e.memory.read(100, 5) == b"typed"


def test_args_and_environ():
    wasi, e = setup(args=["prog", "-x"], env={"A": "1"})
    assert e.args_sizes() == 0
    assert (u32(e.memory, 400), u32(e.memory, 404)) == (2, 8)
    assert e.args() == 0
    first, second = u32(e.memory, 500), u32(e.memory, 504)
    assert e.memory.read(first, 5) == b"prog\0"
    assert e.memory.read(second, 3) == b"-x\0"
    assert e.environ_sizes() == 0
    assert (u32(e.memory, 400), u32(e.memory, 404)) == (1, 4)
    assert e.environ() == 0
    assert e.memory.read(u32(e.memory, 500), 4) == b"A=1\0"


def test_clock_and_random():
    wasi, e = setup()
    assert e.clock() == 0
    assert struct.unpack("<Q", e.memory.read(400, 8))[0] > 1_600_000_000 * 10**9
    assert e.random() == 0
    assert e.memory.read(700, 16) != bytes(16)


def test_no_filesystem_and_unknown_calls():
    wasi, e = setup()
    assert e.open() == WasiLite.ENOTCAPABLE
    assert e.sock() == WasiLite.ENOSYS
    assert wasi.unsupported_calls == ["sock_accept"]


def test_proc_exit():
    wasi, e = setup()
    with pytest.raises(Exit) as info:
        e.exit(3)
    assert info.value.code == 3
