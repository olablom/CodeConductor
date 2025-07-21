#!/usr/bin/env python3
"""
GitHub Actions Status Checker
Check the status of your GitHub Actions workflows without needing GitHub CLI
"""

import requests
import sys
from datetime import datetime, timezone


def get_workflow_runs(repo_owner, repo_name, token=None):
    """Get workflow runs from GitHub API"""
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/actions/runs"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "CodeConductor-CI-Checker",
    }

    if token:
        headers["Authorization"] = f"token {token}"

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching workflow runs: {e}")
        return None


def format_duration(seconds):
    """Format duration in seconds to human readable format"""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes}m {secs}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m"


def format_time_ago(created_at):
    """Format created_at timestamp to 'time ago' format"""
    try:
        created_time = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        diff = now - created_time.replace(tzinfo=timezone.utc)

        if diff.days > 0:
            return f"{diff.days} day{'s' if diff.days != 1 else ''} ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        else:
            return f"{diff.seconds} second{'s' if diff.seconds != 1 else ''} ago"
    except:
        return "unknown time ago"


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


def print_workflow_runs(runs_data, limit=10):
    """Print workflow runs in a nice format"""
    if not runs_data or "workflow_runs" not in runs_data:
        print("❌ No workflow runs found or error occurred")
        return

    runs = runs_data["workflow_runs"][:limit]

    print("\n🚀 GitHub Actions Status for CodeConductor")
    print("=" * 80)
    print(f"📊 Total runs: {runs_data['total_count']}")
    print("=" * 80)

    for run in runs:
        status_emoji = get_status_emoji(run["status"], run.get("conclusion", ""))
        workflow_name = run["name"]
        commit_msg = (
            run["head_commit"]["message"][:50] + "..."
            if len(run["head_commit"]["message"]) > 50
            else run["head_commit"]["message"]
        )
        branch = run["head_branch"]
        created_ago = format_time_ago(run["created_at"])

        # Calculate duration
        duration = "N/A"
        if run["status"] == "completed" and run.get("updated_at"):
            try:
                start_time = datetime.fromisoformat(
                    run["created_at"].replace("Z", "+00:00")
                )
                end_time = datetime.fromisoformat(
                    run["updated_at"].replace("Z", "+00:00")
                )
                duration_seconds = int((end_time - start_time).total_seconds())
                duration = format_duration(duration_seconds)
            except:
                duration = "N/A"

        print(f"{status_emoji} {workflow_name}")
        print(f"   📝 {commit_msg}")
        print(f"   🌿 {branch} • {created_ago} • ⏱️ {duration}")
        print(
            f"   🔗 https://github.com/olablom/CodeConductor/actions/runs/{run['id']}"
        )
        print()


def check_specific_workflow(runs_data, workflow_name):
    """Check status of a specific workflow"""
    if not runs_data or "workflow_runs" not in runs_data:
        return

    workflow_runs = [
        run for run in runs_data["workflow_runs"] if run["name"] == workflow_name
    ]

    if not workflow_runs:
        print(f"❌ No runs found for workflow: {workflow_name}")
        return

    latest_run = workflow_runs[0]
    status_emoji = get_status_emoji(
        latest_run["status"], latest_run.get("conclusion", "")
    )

    print(f"\n🎯 Latest {workflow_name} Status:")
    print(f"{status_emoji} Status: {latest_run['status']}")
    if latest_run.get("conclusion"):
        print(f"📋 Conclusion: {latest_run['conclusion']}")
    print(
        f"🔗 URL: https://github.com/olablom/CodeConductor/actions/runs/{latest_run['id']}"
    )


def main():
    repo_owner = "olablom"
    repo_name = "CodeConductor"

    # Check if GitHub token is provided
    token = None
    if len(sys.argv) > 1:
        token = sys.argv[1]

    print("🔍 Fetching GitHub Actions status...")
    runs_data = get_workflow_runs(repo_owner, repo_name, token)

    if not runs_data:
        print("❌ Failed to fetch workflow runs")
        print("💡 If you have a GitHub token, you can provide it as an argument:")
        print("   python check_github_actions.py YOUR_GITHUB_TOKEN")
        return

    # Check for specific workflow
    if len(sys.argv) > 2 and sys.argv[2] == "microservices":
        check_specific_workflow(runs_data, "Microservices CI/CD Pipeline")
    else:
        print_workflow_runs(runs_data, limit=15)

    print("\n💡 Tips:")
    print(
        "   • Run 'python check_github_actions.py YOUR_TOKEN microservices' to check only Microservices CI/CD"
    )
    print("   • Get a GitHub token from: https://github.com/settings/tokens")
    print("   • Click the URLs above to see detailed logs")


if __name__ == "__main__":
    main()
