#!/usr/bin/env python3
"""
Check test status and update README badge
"""

import subprocess
import re


def run_tests():
    """Run tests and get results"""
    try:
        # Run tests with minimal output
        result = subprocess.run(
            ["pytest", "tests/", "-q", "--tb=no"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        return result.stdout
    except Exception as e:
        print("❌ Error running tests:", e)
        return None


def parse_test_results(output):
    """Parse test results from pytest output"""
    if not output:
        return None, None, None

    # Look for the summary line
    lines = output.split("\n")
    for line in lines:
        if "passed" in line and "failed" in line:
            # Extract numbers from line like "275 passed, 61 failed in 45.23s"
            match = re.search(r"(\d+) passed, (\d+) failed", line)
            if match:
                passed = int(match.group(1))
                failed = int(match.group(2))
                total = passed + failed
                return total, passed, failed

    return None, None, None


def update_readme_badge(total, passed):
    """Update the test badge in README.md"""
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            content = f.read()

        # Update the test badge
        old_badge = r"!\[Tests\]\(https://img\.shields\.io/badge/Tests-\d+%2F\d+%20passing-brightgreen\)"
        new_badge = f"![Tests](https://img.shields.io/badge/Tests-{passed}%2F{total}%20passing-brightgreen)"

        updated_content = re.sub(old_badge, new_badge, content)

        with open("README.md", "w", encoding="utf-8") as f:
            f.write(updated_content)

        print(f"✅ Updated README badge to {passed}/{total}")

    except Exception as e:
        print(f"❌ Error updating README: {e}")


def main():
    print("🧪 Checking test status...")

    output = run_tests()
    if not output:
        print("❌ Could not run tests")
        return

    total, passed, failed = parse_test_results(output)

    if total is None:
        print("❌ Could not parse test results")
        print("Raw output:", output)
        return

    print(f"📊 Test Results:")
    print(f"   Total: {total}")
    print(f"   Passed: {passed}")
    print(f"   Failed: {failed}")
    print(f"   Success Rate: {(passed / total) * 100:.1f}%")

    if failed == 0:
        print("🎉 ALL TESTS PASSING!")
        update_readme_badge(total, passed)
    else:
        print(f"⚠️  {failed} tests are failing")


if __name__ == "__main__":
    main()
