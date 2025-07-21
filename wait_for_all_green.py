#!/usr/bin/env python3
"""
Wait for all GitHub Actions workflows to be green
"""

import time
import requests


def check_github_actions():
    """Check GitHub Actions status directly via API"""
    try:
        # Use GitHub API to get recent workflow runs
        url = "https://api.github.com/repos/olablom/CodeConductor/actions/runs"
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            print(f"❌ API Error: {response.status_code}")
            return None

        data = response.json()
        runs = data.get("workflow_runs", [])

        # Get the latest run for each workflow
        workflows = {}
        for run in runs[:20]:  # Check last 20 runs
            workflow_name = run["name"]
            if workflow_name not in workflows:
                workflows[workflow_name] = run

        return workflows

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def analyze_workflows(workflows):
    """Analyze workflow status"""
    if not workflows:
        return "❌ Could not fetch workflows"

    status_summary = []
    running_count = 0

    for name, run in workflows.items():
        status = run["conclusion"] or run["status"]
        duration = run.get("duration", 0)

        if status == "success":
            emoji = "✅"
            status_summary.append(f"{emoji} {name} - {duration}s")
        elif status == "failure":
            emoji = "❌"
            status_summary.append(f"{emoji} {name} - FAILED")
        elif status in ["in_progress", "queued", "waiting"]:
            emoji = "🔄"
            running_count += 1
            status_summary.append(f"{emoji} {name} - RUNNING")
        else:
            emoji = "❓"
            status_summary.append(f"{emoji} {name} - {status}")

    return "\n".join(status_summary), running_count


def main():
    print("🎯 Waiting for ALL workflows to be GREEN...")
    print("=" * 60)

    start_time = time.time()
    check_count = 0

    while True:
        check_count += 1
        elapsed = time.time() - start_time

        print(f"\n⏰ Check #{check_count} ({(elapsed / 60):.1f} min elapsed)")
        print("-" * 40)

        workflows = check_github_actions()
        if workflows:
            summary, running = analyze_workflows(workflows)
            print(summary)

            if running == 0:
                print("\n🎉 ALL WORKFLOWS ARE GREEN! 🎉")
                print("=" * 60)
                break
            else:
                print(f"\n⏳ Still waiting for {running} workflow(s) to complete...")
        else:
            print("❌ Could not check status")

        print("Sleeping 30 seconds...")
        time.sleep(30)


if __name__ == "__main__":
    main()
