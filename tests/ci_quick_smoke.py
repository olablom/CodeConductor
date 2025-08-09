import os
import subprocess
import sys


def main() -> int:
    # Ensure UTF-8 logs and mock mode
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("CC_QUICK_CI", "1")
    os.environ.setdefault("CC_QUICK_CI_SEED", "42")

    # Run the focused script directly so it produces JSON/YAML artifacts
    cmd = [sys.executable, "tests/test_codeconductor_2agents_focused.py"]
    completed = subprocess.run(cmd)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
