"""
Root pytest configuration that excludes problematic directories
"""

import pytest
from pathlib import Path


def pytest_ignore_collect(collection_path):
    """Exclude problematic directories from test collection"""
    path_str = str(collection_path)

    # Exclude ai-project-advisor and other problematic dirs
    if any(
        exclude in path_str
        for exclude in ["ai-project-advisor", "venv", ".git", "node_modules", "__pycache__"]
    ):
        return True

    return False
