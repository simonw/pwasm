"""A deliberately small WASI preview1 implementation.

WasiLite gives a guest captured stdio, clocks, randomness, args and
environment variables - and nothing else. There is no filesystem and no
network: path and socket calls answer ENOTCAPABLE or ENOSYS. Unknown calls
answer ENOSYS and are recorded in `unsupported_calls`.

Usage:

    wasi = WasiLite(args=["prog"], stdin=b"...")
    instance = instantiate(module, {"wasi_snapshot_preview1": wasi.imports(module)})
    wasi.bind(instance)
"""

from __future__ import annotations

import os
import struct
import time
from typing import Any, Callable, Mapping

from .errors import WasmError

WASI_MODULES = ("wasi_snapshot_preview1", "wasi_unstable")

_u32 = struct.Struct("<I")
_u64 = struct.Struct("<Q")
_iovec = struct.Struct("<II")


class Exit(WasmError):
    """Raised when the guest calls proc_exit."""

    def __init__(self, code: int) -> None:
        super().__init__(f"guest exited with code {code}")
        self.code = code


class WasiLite:
    ESUCCESS = 0
    EBADF = 8
    EINVAL = 28
    ENOSYS = 52
    ESPIPE = 70
    ENOTCAPABLE = 76

    def __init__(
        self,
        stdin: bytes = b"",
        args: Any = ("guest",),
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.stdout = bytearray()
        self.stderr = bytearray()
        self.stdin = bytes(stdin)
        self.stdin_pos = 0
        self.args = [str(a).encode() + b"\0" for a in args]
        self.env = [f"{k}={v}".encode() + b"\0" for k, v in (env or {}).items()]
        # fd -> callable receiving written bytes, instead of capturing
        self.sinks: dict[int, Callable[[bytes], Any]] = {}
        self.unsupported_calls: list[str] = []
        self.memory: Any = None

    def bind(self, instance: Any) -> None:
        """Use the instance's (first) memory for all calls."""
        self.memory = instance.memories[0]

    def imports(self, module: Any) -> dict[str, Callable[..., Any]]:
        """Functions for every WASI import the module declares."""
        functions = {}
        for imp in module.imports:
            if imp.kind == "func" and imp.module in WASI_MODULES:
                has_result = bool(module.types[imp.desc].results)
                functions[imp.name] = self.lookup(imp.name, has_result)
        return functions

    def lookup(self, name: str, has_result: bool = True) -> Callable[..., Any]:
        fn = getattr(self, "wasi_" + name, None)
        if fn is not None:
            return fn

        def unsupported(*args: Any) -> int | None:
            self.unsupported_calls.append(name)
            return self.ENOSYS if has_result else None

        unsupported.__name__ = name
        return unsupported

    # helpers

    def _write_u32(self, ptr: int, value: int) -> None:
        _u32.pack_into(self.memory.data, ptr, value & 0xFFFFFFFF)

    def _write_u64(self, ptr: int, value: int) -> None:
        _u64.pack_into(self.memory.data, ptr, value & 0xFFFFFFFFFFFFFFFF)

    def _strings_sizes(self, items: list, count_ptr: int, size_ptr: int) -> int:
        self._write_u32(count_ptr, len(items))
        self._write_u32(size_ptr, sum(len(item) for item in items))
        return self.ESUCCESS

    def _strings_get(self, items: list, ptrs: int, buf: int) -> int:
        for i, item in enumerate(items):
            self._write_u32(ptrs + 4 * i, buf)
            self.memory.write(buf, item)
            buf += len(item)
        return self.ESUCCESS

    # WASI functions (prefixed with wasi_)

    def wasi_args_sizes_get(self, argc_ptr: int, size_ptr: int) -> int:
        return self._strings_sizes(self.args, argc_ptr, size_ptr)

    def wasi_args_get(self, argv_ptr: int, buf_ptr: int) -> int:
        return self._strings_get(self.args, argv_ptr, buf_ptr)

    def wasi_environ_sizes_get(self, count_ptr: int, size_ptr: int) -> int:
        return self._strings_sizes(self.env, count_ptr, size_ptr)

    def wasi_environ_get(self, environ_ptr: int, buf_ptr: int) -> int:
        return self._strings_get(self.env, environ_ptr, buf_ptr)

    def wasi_clock_res_get(self, clock_id: int, res_ptr: int) -> int:
        self._write_u64(res_ptr, 1000)
        return self.ESUCCESS

    def wasi_clock_time_get(self, clock_id: int, precision: int, time_ptr: int) -> int:
        now = time.time_ns() if clock_id == 0 else time.monotonic_ns()
        self._write_u64(time_ptr, now)
        return self.ESUCCESS

    def wasi_fd_write(
        self, fd: int, iovs: int, iovs_len: int, nwritten_ptr: int
    ) -> int:
        if fd not in (1, 2) and fd not in self.sinks:
            return self.EBADF
        mem = self.memory.data
        chunks = []
        for i in range(iovs_len):
            ptr, length = _iovec.unpack_from(mem, iovs + 8 * i)
            chunks.append(self.memory.read(ptr, length))
        data = b"".join(chunks)
        sink = self.sinks.get(fd)
        if sink is not None:
            sink(data)
        elif fd == 1:
            self.stdout += data
        else:
            self.stderr += data
        self._write_u32(nwritten_ptr, len(data))
        return self.ESUCCESS

    def wasi_fd_read(self, fd: int, iovs: int, iovs_len: int, nread_ptr: int) -> int:
        if fd != 0:
            return self.EBADF
        mem = self.memory.data
        total = 0
        for i in range(iovs_len):
            ptr, length = _iovec.unpack_from(mem, iovs + 8 * i)
            chunk = self.stdin[self.stdin_pos : self.stdin_pos + length]
            if not chunk:
                break
            self.memory.write(ptr, chunk)
            self.stdin_pos += len(chunk)
            total += len(chunk)
        self._write_u32(nread_ptr, total)
        return self.ESUCCESS

    def wasi_fd_close(self, fd: int) -> int:
        return self.ESUCCESS if fd in (0, 1, 2) else self.EBADF

    def wasi_fd_fdstat_get(self, fd: int, buf: int) -> int:
        if fd not in (0, 1, 2):
            return self.EBADF
        # filetype character device (2), no flags, all rights
        self.memory.write(buf, struct.pack("<BxHxxxxQQ", 2, 0, 2**64 - 1, 2**64 - 1))
        return self.ESUCCESS

    def wasi_fd_fdstat_set_flags(self, fd: int, flags: int) -> int:
        return self.ESUCCESS if fd in (0, 1, 2) else self.EBADF

    def wasi_fd_seek(
        self, fd: int, offset: int, whence: int, newoffset_ptr: int
    ) -> int:
        return self.ESPIPE if fd in (0, 1, 2) else self.EBADF

    def wasi_fd_prestat_get(self, fd: int, buf: int) -> int:
        return self.EBADF  # no preopened directories

    def wasi_fd_prestat_dir_name(self, fd: int, path: int, path_len: int) -> int:
        return self.EBADF

    def wasi_fd_filestat_get(self, fd: int, buf: int) -> int:
        return self.EBADF

    def wasi_fd_sync(self, fd: int) -> int:
        return self.ESUCCESS

    def wasi_path_open(self, *args: int) -> int:
        return self.ENOTCAPABLE

    def wasi_path_filestat_get(self, *args: int) -> int:
        return self.ENOTCAPABLE

    def wasi_path_unlink_file(self, *args: int) -> int:
        return self.ENOTCAPABLE

    def wasi_path_create_directory(self, *args: int) -> int:
        return self.ENOTCAPABLE

    def wasi_poll_oneoff(self, *args: int) -> int:
        return self.ENOSYS

    def wasi_proc_exit(self, code: int) -> None:
        raise Exit(code)

    def wasi_random_get(self, buf: int, length: int) -> int:
        self.memory.write(buf, os.urandom(length))
        return self.ESUCCESS

    def wasi_sched_yield(self) -> int:
        return self.ESUCCESS
