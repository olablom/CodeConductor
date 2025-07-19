"""
GitHub Webhook Integration for CodeConductor

Automatically triggers CodeConductor pipeline on PR events and posts results back to GitHub.
"""

import json
import hmac
import hashlib
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import subprocess
import tempfile
import shutil
from datetime import datetime

import requests
from flask import Flask, request, jsonify
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


class GitHubWebhook:
    """GitHub webhook handler for CodeConductor integration"""

    def __init__(self, secret: str, github_token: str):
        self.secret = secret
        self.github_token = github_token
        self.config = OmegaConf.load("config/base.yaml")
        self.app = Flask(__name__)
        self.setup_routes()

    def setup_routes(self):
        """Setup Flask routes"""

        @self.app.route("/webhook", methods=["POST"])
        def webhook():
            """Handle GitHub webhook events"""
            try:
                # Verify webhook signature
                if not self.verify_signature(request):
                    return jsonify({"error": "Invalid signature"}), 401

                # Parse webhook payload
                payload = request.json
                event_type = request.headers.get("X-GitHub-Event")

                logger.info(f"Received {event_type} event from GitHub")

                # Handle different event types
                if event_type == "pull_request":
                    return self.handle_pull_request(payload)
                elif event_type == "push":
                    return self.handle_push(payload)
                elif event_type == "issues":
                    return self.handle_issue(payload)
                else:
                    logger.info(f"Ignoring {event_type} event")
                    return jsonify({"status": "ignored"}), 200

            except Exception as e:
                logger.error(f"Webhook processing failed: {e}")
                return jsonify({"error": str(e)}), 500

    def verify_signature(self, request) -> bool:
        """Verify GitHub webhook signature"""
        if not self.secret:
            return True  # Skip verification if no secret configured

        signature = request.headers.get("X-Hub-Signature-256")
        if not signature:
            return False

        expected_signature = (
            "sha256="
            + hmac.new(
                self.secret.encode("utf-8"), request.data, hashlib.sha256
            ).hexdigest()
        )

        return hmac.compare_digest(signature, expected_signature)

    def handle_pull_request(self, payload: Dict[str, Any]) -> tuple:
        """Handle pull request events"""
        action = payload.get("action")
        pr = payload.get("pull_request", {})

        if action in ["opened", "synchronize"]:
            return self.analyze_pull_request(pr)
        elif action == "closed" and pr.get("merged"):
            return self.handle_merged_pr(pr)

        return jsonify({"status": "ignored"}), 200

    def analyze_pull_request(self, pr: Dict[str, Any]) -> tuple:
        """Analyze pull request with CodeConductor"""
        try:
            # Extract PR information
            pr_number = pr.get("number")
            repo_name = pr.get("base", {}).get("repo", {}).get("full_name")
            pr_url = pr.get("html_url")
            pr_title = pr.get("title", "")
            pr_body = pr.get("body", "")

            logger.info(f"Analyzing PR #{pr_number}: {pr_title}")

            # Create analysis prompt
            prompt = self.create_analysis_prompt(pr_title, pr_body)

            # Run CodeConductor analysis
            analysis_result = self.run_codeconductor_analysis(prompt, pr_number)

            # Post results to GitHub
            self.post_analysis_results(pr_number, repo_name, analysis_result)

            return jsonify(
                {
                    "status": "success",
                    "pr_number": pr_number,
                    "analysis": analysis_result,
                }
            ), 200

        except Exception as e:
            logger.error(f"PR analysis failed: {e}")
            return jsonify({"error": str(e)}), 500

    def create_analysis_prompt(self, title: str, body: str) -> str:
        """Create analysis prompt from PR title and body"""
        prompt = f"""
# Pull Request Analysis Request

## PR Title: {title}

## PR Description:
{body}

## Analysis Tasks:
1. Review the proposed changes
2. Identify potential issues or improvements
3. Suggest code quality enhancements
4. Check for security concerns
5. Provide implementation recommendations

Please analyze this pull request and provide detailed feedback.
"""
        return prompt

    def run_codeconductor_analysis(self, prompt: str, pr_number: int) -> Dict[str, Any]:
        """Run CodeConductor pipeline analysis"""
        try:
            # Create temporary prompt file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
                f.write(prompt)
                prompt_path = f.name

            # Run pipeline with distributed execution
            cmd = [
                "python",
                "pipeline.py",
                "--prompt",
                prompt_path,
                "--iters",
                "1",
                "--distributed",
                "--mock",  # Use mock for webhook testing
            ]

            logger.info(f"Running CodeConductor analysis for PR #{pr_number}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            # Clean up
            Path(prompt_path).unlink(missing_ok=True)

            if result.returncode == 0:
                # Parse pipeline output for metrics
                metrics = self.parse_pipeline_output(result.stdout)
                return {
                    "status": "success",
                    "metrics": metrics,
                    "output": result.stdout[-1000:],  # Last 1000 chars
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                return {
                    "status": "failed",
                    "error": result.stderr,
                    "timestamp": datetime.now().isoformat(),
                }

        except Exception as e:
            logger.error(f"CodeConductor analysis failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def parse_pipeline_output(self, output: str) -> Dict[str, Any]:
        """Parse pipeline output for metrics"""
        metrics = {
            "tests_passed": False,
            "complexity_score": 0.0,
            "reward": 0.0,
            "execution_mode": "unknown",
        }

        try:
            lines = output.split("\n")
            for line in lines:
                if "Tests passed:" in line:
                    metrics["tests_passed"] = "True" in line
                elif "Complexity score:" in line:
                    try:
                        metrics["complexity_score"] = float(line.split(":")[1].strip())
                    except:
                        pass
                elif "Reward:" in line:
                    try:
                        metrics["reward"] = float(line.split(":")[1].split()[0])
                    except:
                        pass
                elif "execution_mode" in line:
                    if "distributed" in line:
                        metrics["execution_mode"] = "distributed"
                    elif "local" in line:
                        metrics["execution_mode"] = "local"
        except Exception as e:
            logger.warning(f"Failed to parse pipeline output: {e}")

        return metrics

    def post_analysis_results(
        self, pr_number: int, repo_name: str, analysis: Dict[str, Any]
    ):
        """Post analysis results as PR comment"""
        try:
            # Create comment content
            comment = self.create_analysis_comment(analysis)

            # Post to GitHub API
            url = (
                f"https://api.github.com/repos/{repo_name}/issues/{pr_number}/comments"
            )
            headers = {
                "Authorization": f"token {self.github_token}",
                "Accept": "application/vnd.github.v3+json",
            }

            response = requests.post(url, json={"body": comment}, headers=headers)

            if response.status_code == 201:
                logger.info(f"Posted analysis results to PR #{pr_number}")
            else:
                logger.error(
                    f"Failed to post comment: {response.status_code} - {response.text}"
                )

        except Exception as e:
            logger.error(f"Failed to post analysis results: {e}")

    def create_analysis_comment(self, analysis: Dict[str, Any]) -> str:
        """Create formatted comment from analysis results"""
        status = analysis.get("status", "unknown")

        if status == "success":
            metrics = analysis.get("metrics", {})
            return f"""
🤖 **CodeConductor Analysis Complete**

## 📊 Results
- **Status**: ✅ Analysis completed successfully
- **Tests Passed**: {"✅" if metrics.get("tests_passed") else "❌"}
- **Complexity Score**: {metrics.get("complexity_score", 0):.2f}
- **Reward**: {metrics.get("reward", 0):.2f}
- **Execution Mode**: {metrics.get("execution_mode", "unknown")}

## 🔍 Analysis Details
```
{analysis.get("output", "No detailed output available")}
```

---
*Analyzed by CodeConductor at {analysis.get("timestamp", "unknown time")}*
"""
        else:
            return f"""
🤖 **CodeConductor Analysis Failed**

## ❌ Error
{analysis.get("error", "Unknown error occurred")}

---
*Analysis failed at {analysis.get("timestamp", "unknown time")}*
"""

    def handle_push(self, payload: Dict[str, Any]) -> tuple:
        """Handle push events"""
        # Could trigger analysis on main branch pushes
        return jsonify({"status": "ignored"}), 200

    def handle_issue(self, payload: Dict[str, Any]) -> tuple:
        """Handle issue events"""
        # Could trigger analysis on issue creation
        return jsonify({"status": "ignored"}), 200

    def handle_merged_pr(self, pr: Dict[str, Any]) -> tuple:
        """Handle merged PR events"""
        # Could trigger post-merge analysis
        return jsonify({"status": "ignored"}), 200

    def run(self, host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
        """Run the webhook server"""
        logger.info(f"Starting GitHub webhook server on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)


def create_webhook_app(secret: str = None, github_token: str = None) -> Flask:
    """Create Flask app for webhook handling"""
    webhook = GitHubWebhook(secret or "", github_token or "")
    return webhook.app


if __name__ == "__main__":
    # Load configuration
    config = OmegaConf.load("config/base.yaml")

    # Get webhook configuration
    webhook_config = config.get("github_webhook", {})
    secret = webhook_config.get("secret", "")
    github_token = webhook_config.get("token", "")

    if not github_token:
        print(
            "⚠️ GitHub token not configured. Set github_webhook.token in config/base.yaml"
        )
        print("💡 Get token from: https://github.com/settings/tokens")

    # Create and run webhook
    webhook = GitHubWebhook(secret, github_token)
    webhook.run(debug=True)
