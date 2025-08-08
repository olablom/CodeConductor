import os
import sys
from pathlib import Path
import subprocess


def run_cli_in_cwd(argv, cwd: Path):
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore"
    repo_root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = (
        str((repo_root / "src").resolve()) + os.pathsep + env.get("PYTHONPATH", "")
    )
    cli_path = (repo_root / "src" / "codeconductor" / "cli.py").resolve()
    code = (
        "import runpy, sys;"
        f"sys.path.insert(0, r'{str((repo_root / 'src').resolve())}');"
        "sys.argv=['codeconductor']+sys.argv[1:];"
        f"runpy.run_path(r'{str(cli_path)}', run_name='__main__')"
    )
    cmd = [sys.executable, "-c", code] + argv
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, env=env)
    return proc.returncode, proc.stdout + proc.stderr


def test_diag_cursor_missing_log(tmp_path: Path):
    project = tmp_path
    # No artifacts/diagnostics directory created => missing log
    code, out = run_cli_in_cwd(["diag", "cursor"], project)
    assert code == 0
    assert "diagnose_latest.txt not found" in out


def test_diag_cursor_detected_port(tmp_path: Path):
    project = tmp_path
    diag_dir = project / "artifacts" / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    content = (
        "Diagnosing Cursor / Local API\n"
        "[3/5] Port checks\n"
        " Port 3000 -> TcpTestSucceeded=True\n"
        "  GET /api/health -> 200\n"
        "[5/5] Suggested fixes\n"
        " - Detected Cursor API port: 5123\n"
    )
    (diag_dir / "diagnose_latest.txt").write_text(content, encoding="utf-8")

    code, out = run_cli_in_cwd(["diag", "cursor"], project)
    assert code == 0
    assert "Detected Cursor API port: 5123" in out
    assert "SetEnvironmentVariable('CURSOR_API_BASE','http://127.0.0.1:5123'" in out


def test_diag_cursor_no_port(tmp_path: Path):
    project = tmp_path
    diag_dir = project / "artifacts" / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    content = (
        "Diagnosing Cursor / Local API\n"
        "[3/5] Port checks\n"
        " Port 3000 -> TcpTestSucceeded=False\n"
        "  No health endpoint responded\n"
        "[5/5] Suggested fixes\n"
    )
    (diag_dir / "diagnose_latest.txt").write_text(content, encoding="utf-8")

    code, out = run_cli_in_cwd(["diag", "cursor"], project)
    assert code == 0
    assert "Cursor API    : NOT LISTENING" in out
