import doctest
import glob
import importlib.util
import os


def _latest_generated():
    runs = sorted(glob.glob("artifacts/runs/*"))
    for r in reversed(runs):
        p = os.path.join(r, "after", "generated.py")
        if os.path.isfile(p):
            return p
    raise FileNotFoundError("No generated.py found under artifacts/runs")


def _load_module(path):
    spec = importlib.util.spec_from_file_location("generated_latest", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


import pytest


@pytest.mark.skip(reason="Requires local artifacts/runs directory with generated.py files")
def test_doctest_passes():
    path = _latest_generated()
    mod = _load_module(path)
    failed, _ = doctest.testmod(mod, verbose=False)
    assert failed == 0


@pytest.mark.skip(reason="Requires local artifacts/runs directory with generated.py files")
def test_print_output(capsys):
    path = _latest_generated()
    mod = _load_module(path)
    if hasattr(mod, "print_hello_world"):
        mod.print_hello_world()
        out = capsys.readouterr().out.strip()
        assert out == "Hello, World!"
