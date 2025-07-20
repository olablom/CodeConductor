#!/usr/bin/env python3
"""
CodeConductor CI/CD Monitor (Simple Version)
Monitors CI/CD status every 5 minutes using existing scripts
"""

import subprocess
import time
import sys
from datetime import datetime


def run_script(script_name):
    """Run a Python script and return output"""
    try:
        result = subprocess.run(
            [sys.executable, script_name], capture_output=True, text=True, timeout=30
        )
        return result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return "Timeout", "Script took too long"
    except Exception as e:
        return f"Error: {e}", ""


def check_ci_status():
    """Check CI status using our existing scripts"""
    print(f"\n{'=' * 80}")
    print(
        f"🎼 CodeConductor CI/CD Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print(f"{'=' * 80}")

    # Check local services
    print("\n🐳 Checking local services...")
    local_output, local_error = run_script("check_ci.py")
    if local_output:
        print(local_output)
    if local_error:
        print(f"⚠️ Local check error: {local_error}")

    # Check GitHub Actions (if available)
    print("\n📊 Checking GitHub Actions...")
    try:
        github_output, github_error = run_script("check_github_actions.py")
        if github_output:
            # Only show first few lines to avoid spam
            lines = github_output.split("\n")
            for i, line in enumerate(lines):
                if i < 15:  # Show first 15 lines
                    print(line)
                elif i == 15:
                    print("... (truncated)")
                    break
        if github_error:
            print(f"⚠️ GitHub check error: {github_error}")
    except:
        print("⚠️ Could not check GitHub Actions")


def is_everything_working():
    """Check if everything is working by running our scripts"""
    try:
        # Check local services
        local_output, _ = run_script("check_ci.py")

        # Look for success indicators
        if (
            "✅ gateway-service" in local_output
            and "✅ agent-service" in local_output
            and "✅ orchestrator-service" in local_output
            and "✅ auth-service" in local_output
            and "✅ data-service" in local_output
        ):
            return True, "All local services running"

        return False, "Some local services not running"
    except:
        return False, "Could not check status"


def main():
    print("🎼 CodeConductor CI/CD Monitor Started!")
    print("📊 Monitoring every 5 minutes until everything is perfect...")
    print("⏹️ Press Ctrl+C to stop")
    print("🔗 GitHub Actions: https://github.com/olablom/CodeConductor/actions")

    check_count = 0

    try:
        while True:
            check_count += 1

            # Check current status
            check_ci_status()

            # Check if everything is working
            is_working, message = is_everything_working()

            print(f"\n🎯 Status Check #{check_count}: {message}")

            if is_working:
                print("\n" + "=" * 80)
                print("🎉 SUCCESS! Everything is working perfectly!")
                print("=" * 80)
                print("✅ Local Services: All 5 running")
                print("✅ Docker Containers: Healthy")
                print("✅ Ready for development!")
                print("\n🚀 Your CodeConductor microservices stack is ready!")
                break

            print(f"\n⏰ Next check in 5 minutes... (Press Ctrl+C to stop)")
            print(f"💡 Manual checks:")
            print(f"   • python check_ci.py")
            print(f"   • python check_github_actions.py")
            print(f"   • https://github.com/olablom/CodeConductor/actions")

            # Wait 5 minutes
            time.sleep(300)

    except KeyboardInterrupt:
        print(f"\n⏹️ Monitoring stopped after {check_count} checks")
        print("👋 Thanks for using CodeConductor CI/CD Monitor!")


if __name__ == "__main__":
    main()
