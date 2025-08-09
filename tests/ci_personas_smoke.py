import os
import sys
import subprocess
from pathlib import Path


def main() -> int:
    # Ensure UTF-8 logs and mock mode so no backends are touched
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("CC_QUICK_CI", "1")
    os.environ.setdefault("CC_QUICK_CI_SEED", "42")

    personas_path = Path("agents/personas.yaml")
    if not personas_path.exists():
        print("SKIP: personas.yaml not found; skipping personas smoke")
        return 0

    cmd = [
        sys.executable,
        "-m",
        "codeconductor.cli",
        "run",
        "--personas",
        str(personas_path),
        "--agents",
        "architect,coder",
        "--prompt",
        "ping",
        "--rounds",
        "1",
        "--timeout-per-turn",
        "10",
    ]
    completed = subprocess.run(cmd)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
