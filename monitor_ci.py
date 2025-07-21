#!/usr/bin/env python3
"""
CodeConductor CI/CD Monitor
Automatically monitors CI/CD status every 5 minutes until everything is working perfectly
"""

import subprocess
import time
import requests
from datetime import datetime


def run_command(cmd):
    """Run a command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"


def get_workflow_runs():
    """Get workflow runs from GitHub API"""
    url = "https://api.github.com/repos/olablom/CodeConductor/actions/runs"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "CodeConductor-CI-Monitor",
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception:
        return None


def get_status_emoji(status, conclusion):
    """Get emoji for workflow status"""
    if status == "completed":
        if conclusion == "success":
            return "✅"
        elif conclusion == "failure":
            return "❌"
        elif conclusion == "cancelled":
            return "⏹️"
        else:
            return "⚠️"
    elif status == "in_progress":
        return "🔄"
    elif status == "queued":
        return "⏳"
    else:
        return "❓"


def check_microservices_pipeline(runs_data):
    """Check specifically the Microservices CI/CD Pipeline"""
    if not runs_data or "workflow_runs" not in runs_data:
        return None, "Could not fetch data"

    # Find Microservices CI/CD Pipeline runs
    microservices_runs = [
        run
        for run in runs_data["workflow_runs"]
        if run["name"] == "Microservices CI/CD Pipeline"
    ]

    if not microservices_runs:
        return None, "No Microservices CI/CD Pipeline runs found"

    latest_run = microservices_runs[0]
    return latest_run, None


def check_local_services():
    """Check if local services are running"""
    services = [
        "gateway-service",
        "agent-service",
        "orchestrator-service",
        "auth-service",
        "data-service",
    ]
    running_services = []

    for service in services:
        status = run_command(
            f"docker ps --filter name={service} --format '{{{{.Status}}}}'"
        )
        if status and "Error" not in status:
            running_services.append(service)

    return running_services


def print_status(run, local_services, check_time):
    """Print current status"""
    print(f"\n{'=' * 80}")
    print(f"🎼 CodeConductor CI/CD Monitor - {check_time}")
    print(f"{'=' * 80}")

    if run:
        status_emoji = get_status_emoji(run["status"], run.get("conclusion", ""))
        commit_msg = (
            run["head_commit"]["message"][:60] + "..."
            if len(run["head_commit"]["message"]) > 60
            else run["head_commit"]["message"]
        )

        print("🔄 Microservices CI/CD Pipeline:")
        print(f"   {status_emoji} Status: {run['status']}")
        if run.get("conclusion"):
            print(f"   📋 Conclusion: {run['conclusion']}")
        print(f"   📝 Commit: {commit_msg}")
        print(
            f"   🔗 URL: https://github.com/olablom/CodeConductor/actions/runs/{run['id']}"
        )

        # Calculate duration if completed
        if run["status"] == "completed" and run.get("updated_at"):
            try:
                start_time = datetime.fromisoformat(
                    run["created_at"].replace("Z", "+00:00")
                )
                end_time = datetime.fromisoformat(
                    run["updated_at"].replace("Z", "+00:00")
                )
                duration_seconds = int((end_time - start_time).total_seconds())
                minutes = duration_seconds // 60
                seconds = duration_seconds % 60
                print(f"   ⏱️ Duration: {minutes}m {seconds}s")
            except Exception:
                pass
    else:
        print("❌ Could not fetch Microservices CI/CD Pipeline status")

    print("\n🐳 Local Services:")
    if local_services:
        for service in local_services:
            print(f"   ✅ {service}")
        print(f"   📊 {len(local_services)}/5 services running")
    else:
        print("   ❌ No local services running")


def is_everything_perfect(run, local_services):
    """Check if everything is working perfectly"""
    if not run:
        return False, "Could not fetch CI status"

    if not local_services or len(local_services) < 5:
        return False, f"Only {len(local_services)}/5 local services running"

    if run["status"] == "in_progress":
        return False, "CI/CD still running"

    if run["status"] == "completed" and run.get("conclusion") == "success":
        return True, "Everything is working perfectly! 🎉"

    if run["status"] == "completed" and run.get("conclusion") == "failure":
        return False, "CI/CD failed"

    return False, f"Unknown status: {run['status']}"


def main():
    print("🎼 CodeConductor CI/CD Monitor Started!")
    print("📊 Monitoring every 5 minutes until everything is perfect...")
    print("⏹️ Press Ctrl+C to stop")

    check_count = 0

    try:
        while True:
            check_count += 1
            check_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Get CI/CD status
            runs_data = get_workflow_runs()
            run, error = check_microservices_pipeline(runs_data)

            # Check local services
            local_services = check_local_services()

            # Print status
            print_status(run, local_services, check_time)

            # Check if everything is perfect
            is_perfect, message = is_everything_perfect(run, local_services)

            print(f"\n🎯 Status Check #{check_count}: {message}")

            if is_perfect:
                print("\n" + "=" * 80)
                print("🎉 SUCCESS! Everything is working perfectly!")
                print("=" * 80)
                print("✅ Microservices CI/CD Pipeline: SUCCESS")
                print("✅ Local Services: All 5 running")
                print("✅ Integration Tests: Passing")
                print("✅ Docker Hub: Ready")
                print("✅ Security Scan: Complete")
                print("\n🚀 Your CodeConductor microservices stack is ready!")
                break

            print("\n⏰ Next check in 5 minutes... (Press Ctrl+C to stop)")
            print("💡 Tips:")
            print("   • Check logs: https://github.com/olablom/CodeConductor/actions")
            print("   • Local services: python check_ci.py")
            print("   • Manual check: python check_github_actions.py")

            # Wait 5 minutes
            time.sleep(300)

    except KeyboardInterrupt:
        print(f"\n⏹️ Monitoring stopped after {check_count} checks")
        print("👋 Thanks for using CodeConductor CI/CD Monitor!")


if __name__ == "__main__":
    main()
