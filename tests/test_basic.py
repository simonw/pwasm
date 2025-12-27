from pure_python_wasm import hello


def test_hello():
    assert hello() == "Hello from pure-python-wasm!"
