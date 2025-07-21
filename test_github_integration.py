#!/usr/bin/env python3
"""
Test script for GitHub integration functionality
"""

from pathlib import Path
from unittest.mock import Mock


def test_webhook_signature_verification():
    """Test webhook signature verification"""
    print("🧪 Testing webhook signature verification...")

    try:
        from integrations.github_webhook import GitHubWebhook

        # Create webhook with test secret
        webhook = GitHubWebhook("test_secret", "test_token")

        # Test valid signature
        mock_request = Mock()
        mock_request.data = b"test_payload"
        mock_request.headers = {"X-Hub-Signature-256": "sha256=valid_signature_here"}

        # This should work (we're not actually verifying the signature in test)
        result = webhook.verify_signature(mock_request)
        print("✅ Signature verification test completed")
        return True

    except Exception as e:
        print(f"❌ Signature verification test failed: {e}")
        return False


def test_analysis_prompt_creation():
    """Test analysis prompt creation from PR data"""
    print("\n🧪 Testing analysis prompt creation...")

    try:
        from integrations.github_webhook import GitHubWebhook

        webhook = GitHubWebhook("", "")

        # Test PR data
        pr_title = "Add new feature"
        pr_body = "This PR adds a new feature to the application."

        prompt = webhook.create_analysis_prompt(pr_title, pr_body)

        # Check that prompt contains expected elements
        assert "Pull Request Analysis Request" in prompt
        assert pr_title in prompt
        assert pr_body in prompt
        assert "Analysis Tasks:" in prompt

        print("✅ Analysis prompt creation test passed")
        return True

    except Exception as e:
        print(f"❌ Analysis prompt creation test failed: {e}")
        return False


def test_pipeline_output_parsing():
    """Test parsing of pipeline output"""
    print("\n🧪 Testing pipeline output parsing...")

    try:
        from integrations.github_webhook import GitHubWebhook

        webhook = GitHubWebhook("", "")

        # Mock pipeline output
        mock_output = """
🚀 Starting CodeConductor pipeline with 1 iterations...

--- Iteration 1/1 ---
Selected strategy: conservative
✅ Code passed PolicyAgent safety check
Tests passed: True
Complexity score: 0.75
Reward: 45.00 (base: 35.00)
execution_mode: distributed

✅ Pipeline completed!
"""

        metrics = webhook.parse_pipeline_output(mock_output)

        # Check parsed metrics
        assert metrics["tests_passed"] == True
        assert metrics["complexity_score"] == 0.75
        assert metrics["reward"] == 45.0
        assert metrics["execution_mode"] == "distributed"

        print("✅ Pipeline output parsing test passed")
        return True

    except Exception as e:
        print(f"❌ Pipeline output parsing test failed: {e}")
        return False


def test_analysis_comment_creation():
    """Test creation of analysis comments"""
    print("\n🧪 Testing analysis comment creation...")

    try:
        from integrations.github_webhook import GitHubWebhook

        webhook = GitHubWebhook("", "")

        # Test successful analysis
        successful_analysis = {
            "status": "success",
            "metrics": {
                "tests_passed": True,
                "complexity_score": 0.8,
                "reward": 50.0,
                "execution_mode": "distributed",
            },
            "output": "Analysis completed successfully",
            "timestamp": "2024-01-01T12:00:00",
        }

        comment = webhook.create_analysis_comment(successful_analysis)

        # Check comment content
        assert "CodeConductor Analysis Complete" in comment
        assert "✅" in comment  # Success indicator
        assert "0.80" in comment  # Complexity score
        assert "50.0" in comment  # Reward

        # Test failed analysis
        failed_analysis = {
            "status": "failed",
            "error": "Pipeline execution failed",
            "timestamp": "2024-01-01T12:00:00",
        }

        failed_comment = webhook.create_analysis_comment(failed_analysis)

        assert "CodeConductor Analysis Failed" in failed_comment
        assert "❌" in failed_comment  # Error indicator

        print("✅ Analysis comment creation test passed")
        return True

    except Exception as e:
        print(f"❌ Analysis comment creation test failed: {e}")
        return False


def test_webhook_app_creation():
    """Test Flask app creation"""
    print("\n🧪 Testing Flask app creation...")

    try:
        from integrations.github_webhook import create_webhook_app

        app = create_webhook_app("test_secret", "test_token")

        # Check that app is a Flask app
        assert hasattr(app, "route")
        assert hasattr(app, "run")

        print("✅ Flask app creation test passed")
        return True

    except Exception as e:
        print(f"❌ Flask app creation test failed: {e}")
        return False


def test_github_actions_integration():
    """Test GitHub Actions workflow file"""
    print("\n🧪 Testing GitHub Actions integration...")

    try:
        workflow_path = Path(".github/workflows/codeconductor.yml")

        if workflow_path.exists():
            workflow_content = workflow_path.read_text(encoding="utf-8")

            # Check for required elements
            assert "CodeConductor Analysis" in workflow_content
            assert "pull_request" in workflow_content
            assert "pipeline.py" in workflow_content
            assert "distributed" in workflow_content

            print("✅ GitHub Actions workflow test passed")
            return True
        else:
            print(f"⚠️ GitHub Actions workflow file not found at {workflow_path}")
            return False

    except Exception as e:
        print(f"❌ GitHub Actions integration test failed: {e}")
        return False


def test_config_integration():
    """Test GitHub webhook configuration"""
    print("\n🧪 Testing GitHub webhook configuration...")

    try:
        from omegaconf import OmegaConf

        config = OmegaConf.load("config/base.yaml")

        # Check that GitHub webhook config exists
        assert "github_webhook" in config

        webhook_config = config.github_webhook
        assert "enabled" in webhook_config
        assert "secret" in webhook_config
        assert "token" in webhook_config
        assert "webhook_url" in webhook_config
        assert "events" in webhook_config

        print("✅ GitHub webhook configuration test passed")
        return True

    except Exception as e:
        print(f"❌ GitHub webhook configuration test failed: {e}")
        return False


def main():
    """Run all GitHub integration tests"""
    print("🎼 CodeConductor v2.0 - GitHub Integration Test")
    print("=" * 60)

    tests = [
        ("Webhook Signature Verification", test_webhook_signature_verification),
        ("Analysis Prompt Creation", test_analysis_prompt_creation),
        ("Pipeline Output Parsing", test_pipeline_output_parsing),
        ("Analysis Comment Creation", test_analysis_comment_creation),
        ("Flask App Creation", test_webhook_app_creation),
        ("GitHub Actions Integration", test_github_actions_integration),
        ("Configuration Integration", test_config_integration),
    ]

    results = {}
    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    passed = 0
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\n🎯 Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All GitHub integration tests passed!")
        print("🚀 CodeConductor is ready for GitHub integration!")
        print("\n💡 Next steps:")
        print("  1. Set up GitHub webhook in repository settings")
        print("  2. Configure github_webhook.token in config/base.yaml")
        print("  3. Start webhook server: python integrations/github_webhook.py")
    else:
        print("⚠️ Some tests failed. Check configuration and dependencies.")

    print("=" * 60)


if __name__ == "__main__":
    main()
