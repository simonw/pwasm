Use uv. Run tests like this:

    uv run pytest

Run the the library directly like this:

    uv run python -c '
    import pure_python_wasm
    '
Always practice TDD: write a failing test, watch it fail, then make it pass.

Commit early and often. Commits should bundle the test, implementation, and documentation changes together.

Run Black to format code before you commit:

    uv run black .

## WebAssembly Spec Tests

For running spec tests, ensure a fresh checkout of the WebAssembly spec repository:

    git clone https://github.com/WebAssembly/spec.git spec/

Install wabt (WebAssembly Binary Toolkit) which provides `wast2json`:

    # On macOS with Homebrew:
    brew install wabt

    # On Ubuntu/Debian:
    sudo apt-get install wabt

    # Or build from source: https://github.com/WebAssembly/wabt

Verify wast2json is available:

    wast2json --version
