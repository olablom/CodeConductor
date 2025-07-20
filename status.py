#!/usr/bin/env python3
"""
Quick CI/CD Status Check
Shows current status of all workflows
"""

import subprocess
import sys
from datetime import datetime


def run_command(cmd):
    """Run a command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"


def main():
    print("🎼 CodeConductor CI/CD Status Check")
    print("=" * 50)
    print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Check GitHub Actions
    print("📊 GitHub Actions Status:")
    print("-" * 30)

    output = run_command("python check_github_actions.py")
    if output:
        # Extract just the workflow status lines
        lines = output.split("\n")
        workflow_lines = []

        for line in lines:
            if any(x in line for x in ["✅", "❌", "🔄", "⏳"]):
                if (
                    "Microservices CI/CD Pipeline" in line
                    or "CodeConductor CI/CD" in line
                    or "Main CodeConductor CI" in line
                    or "CodeConductor Analysis" in line
                ):
                    workflow_lines.append(line)

        # Show only the 4 most recent workflow statuses
        for i, line in enumerate(workflow_lines[:4]):
            print(line)

    print()
    print("🐳 Local Services Status:")
    print("-" * 30)

    # Check local services
    local_output = run_command("python check_ci.py")
    if local_output:
        # Extract service status lines
        lines = local_output.split("\n")
        for line in lines:
            if "✅" in line and "service" in line:
                print(line)

    print()
    print("💡 Summary:")
    print("-" * 30)

    # Count successes and failures
    if output:
        success_count = output.count("✅")
        failure_count = output.count("❌")
        running_count = output.count("🔄")

        print(f"✅ Successful runs: {success_count}")
        print(f"❌ Failed runs: {failure_count}")
        print(f"🔄 Running: {running_count}")

        if failure_count == 0 and running_count == 0:
            print("\n🎉 ALL WORKFLOWS ARE GREEN!")
        elif running_count > 0:
            print(f"\n⏳ {running_count} workflows still running...")
        else:
            print(f"\n⚠️ {failure_count} workflows failed")

    print()
    print("🔗 GitHub Actions: https://github.com/olablom/CodeConductor/actions")


if __name__ == "__main__":
    main()
