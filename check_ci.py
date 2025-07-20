#!/usr/bin/env python3
"""
Simple CI Status Checker
Shows the current status of your latest CI/CD runs
"""

import subprocess
import sys
import time


def run_command(cmd):
    """Run a command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"


def check_git_status():
    """Check current git status"""
    print("🔍 Checking current git status...")

    # Get current branch
    branch = run_command("git branch --show-current")
    print(f"🌿 Current branch: {branch}")

    # Get latest commit
    latest_commit = run_command("git log -1 --oneline")
    print(f"📝 Latest commit: {latest_commit}")

    # Check if there are unpushed commits
    unpushed = run_command("git log --oneline origin/main..HEAD")
    if unpushed:
        print(f"📤 Unpushed commits: {unpushed.count(chr(10)) + 1}")
    else:
        print("✅ All commits pushed")


def show_ci_tips():
    """Show tips for checking CI status"""
    print("\n🚀 CI/CD Status Tips:")
    print("=" * 50)
    print("📊 To check GitHub Actions status:")
    print("   1. Go to: https://github.com/olablom/CodeConductor/actions")
    print("   2. Look for 'Microservices CI/CD Pipeline'")
    print("   3. Click on the latest run to see details")
    print()
    print("🔍 Quick status check:")
    print("   • ✅ Green checkmark = Success")
    print("   • ❌ Red X = Failed")
    print("   • 🔄 Yellow circle = In progress")
    print("   • ⏳ Gray circle = Queued")
    print()
    print("📱 From terminal (if you have GitHub CLI):")
    print("   gh run list --repo olablom/CodeConductor")
    print("   gh run view --repo olablom/CodeConductor")
    print()
    print("🔗 Direct links:")
    print("   • All workflows: https://github.com/olablom/CodeConductor/actions")
    print(
        "   • Microservices CI/CD: https://github.com/olablom/CodeConductor/actions/workflows/microservices-ci-cd.yml"
    )


def check_local_services():
    """Check if local services are running"""
    print("\n🐳 Checking local Docker services...")

    # Check if Docker is running
    docker_status = run_command(
        "docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' | head -10"
    )
    if "Error" in docker_status:
        print("❌ Docker not running or not available")
        return

    print("📋 Running containers:")
    print(docker_status)

    # Check specific services
    services = [
        "gateway-service",
        "agent-service",
        "orchestrator-service",
        "auth-service",
        "data-service",
    ]
    print("\n🔍 Checking CodeConductor services:")

    for service in services:
        status = run_command(
            f"docker ps --filter name={service} --format '{{{{.Status}}}}'"
        )
        if status:
            print(f"✅ {service}: {status}")
        else:
            print(f"❌ {service}: Not running")


def main():
    print("🎼 CodeConductor CI/CD Status Checker")
    print("=" * 50)

    check_git_status()
    check_local_services()
    show_ci_tips()

    print("\n💡 Next steps:")
    print("   1. Check GitHub Actions in browser")
    print("   2. If CI fails, check the logs")
    print("   3. Fix any issues and push again")
    print("   4. Monitor the next run")


if __name__ == "__main__":
    main()
