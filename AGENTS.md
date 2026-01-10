Use uv. Run tests like this:

    uv run pytest

Run the the library directly like this:

    uv run python -c '
    import pwism
    '
Always practice TDD: write a failing test, watch it fail, then make it pass.

Commit early and often. Commits should bundle the test, implementation, and documentation changes together.

Run Black to format code before you commit:

    uv run black .
