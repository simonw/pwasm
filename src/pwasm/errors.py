"""Exception classes for the WebAssembly runtime."""


class WasmError(Exception):
    """Base class for all WebAssembly runtime errors."""

    pass


class DecodeError(WasmError):
    """Error during binary format decoding."""

    pass


class ValidationError(WasmError):
    """Error during module validation."""

    pass


class TrapError(WasmError):
    """Runtime trap (division by zero, unreachable, etc.)."""

    pass


class LinkError(WasmError):
    """Error during module instantiation/linking."""

    pass
