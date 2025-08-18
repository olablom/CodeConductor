from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from .base import AgentPlugin


def load_plugins(plugins_dir: str | Path = "plugins") -> list[AgentPlugin]:
    """
    Autoload plugins from a directory. Returns instantiated plugins.
    """
    dir_path = Path(plugins_dir)
    if not dir_path.exists():
        return []

    plugins: list[AgentPlugin] = []
    for py_file in dir_path.glob("*.py"):
        if py_file.name in {"base.py", "loader.py", "__init__.py"}:
            continue
        spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
        if not spec or not spec.loader:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[py_file.stem] = module
        try:
            spec.loader.exec_module(module)  # type: ignore[attr-defined]
        except Exception:
            continue
        # Find subclasses of AgentPlugin
        for attr in dir(module):
            obj = getattr(module, attr)
            try:
                if (
                    isinstance(obj, type)
                    and issubclass(obj, AgentPlugin)
                    and obj is not AgentPlugin
                ):
                    plugins.append(obj())
            except Exception:
                continue
    return plugins
