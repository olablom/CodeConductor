#!/usr/bin/env python3
"""
Test Dashboard för CodeConductor
Kör alla tester och genererar rapport
"""

import subprocess
import sys
import webbrowser
from datetime import datetime
from pathlib import Path


def generate_test_report():
    """Kör alla tester och generera rapport"""

    print("🧪 CodeConductor Test Dashboard")
    print("=" * 50)
    print(f"Tidpunkt: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Aktivera venv om det finns
    python_cmd = "python"
    if Path("venv/Scripts/python.exe").exists():
        python_cmd = "venv/Scripts/python.exe"
    elif Path("venv/bin/python").exists():
        python_cmd = "venv/bin/python"

    try:
        # Kör tester med coverage
        print("📊 Kör tester med coverage...")
        result = subprocess.run(
            [
                python_cmd,
                "-m",
                "pytest",
                "tests/",
                "--tb=short",
                "--cov=codeconductor",
                "--cov-report=html",
                "--cov-report=term-missing",
                "-v",
                "--ignore=tests/test_master_full_suite.py",
                "--ignore=tests/test_vllm_integration.py",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        # Analysera output
        output = result.stdout + result.stderr

        # Räkna tester
        passed = output.count("PASSED")
        failed = output.count("FAILED")
        skipped = output.count("SKIPPED")
        errors = output.count("ERROR")

        print("\n📊 Test Summary:")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"⏭️ Skipped: {skipped}")
        print(f"💥 Errors: {errors}")

        # Visa coverage info
        if "TOTAL" in output:
            coverage_lines = [line for line in output.split("\n") if "TOTAL" in line]
            if coverage_lines:
                print(f"\n📈 Coverage: {coverage_lines[-1]}")

        # Visa specifika failures
        if failed > 0:
            print("\n❌ Failed Tests:")
            for line in output.split("\n"):
                if "FAILED" in line and "tests/" in line:
                    print(f"  {line.strip()}")

        # Öppna coverage report i browser
        if Path("htmlcov/index.html").exists():
            print("\n🌐 Öppnar coverage report...")
            try:
                webbrowser.open("htmlcov/index.html")
            except:
                print("  Kunde inte öppna browser, öppna htmlcov/index.html manuellt")

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print("⏰ Tester tog för lång tid (>5 min)")
        return False
    except Exception as e:
        print(f"💥 Fel vid körning av tester: {e}")
        return False


if __name__ == "__main__":
    success = generate_test_report()
    sys.exit(0 if success else 1)
