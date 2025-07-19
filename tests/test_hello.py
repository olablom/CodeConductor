import pytest
import importlib.util
import sys
from pathlib import Path


def load_module_from_file(file_path: Path):
    """Dynamiskt ladda Python-modul från fil"""
    module_name = file_path.stem
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_hello_world_exists(tmp_path):
    """Testa att hello_world funktionen existerar"""
    code_file = tmp_path / "generated.py"
    code_file.write_text("""
def hello_world():
    return "Hello, Test!"
""")

    module = load_module_from_file(code_file)
    assert hasattr(module, "hello_world")
    assert callable(module.hello_world)


def test_hello_world_returns_string(tmp_path):
    """Testa att hello_world returnerar en sträng"""
    code_file = tmp_path / "generated.py"
    code_file.write_text("""
def hello_world():
    return "Hello, World!"
""")

    module = load_module_from_file(code_file)
    result = module.hello_world()
    assert isinstance(result, str)
    assert len(result) > 0
