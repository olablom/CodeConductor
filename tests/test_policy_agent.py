"""
Unit tests for PolicyAgent safety checking functionality
"""

import unittest
from unittest.mock import patch
import sys
import os

# Add the parent directory to the path to import the agents module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.policy_agent import PolicyAgent


class TestPolicyAgent(unittest.TestCase):
    """Test cases for PolicyAgent safety checking"""

    def setUp(self):
        """Set up test fixtures"""
        self.agent = PolicyAgent("TestPolicyAgent")

    def test_init(self):
        """Test PolicyAgent initialization"""
        agent = PolicyAgent("TestAgent")
        self.assertEqual(agent.name, "TestAgent")
        self.assertIsNotNone(agent.dangerous_patterns)
        self.assertIn("critical", agent.dangerous_patterns)
        self.assertIn("high", agent.dangerous_patterns)
        self.assertIn("medium", agent.dangerous_patterns)
        self.assertIn("low", agent.dangerous_patterns)

    def test_check_code_safety_safe_code(self):
        """Test safety check with safe code"""
        safe_code = """
def safe_function():
    return "Hello, World!"

if __name__ == "__main__":
    safe_function()
"""

        result = self.agent.check_code_safety(safe_code)

        self.assertTrue(result["safe"])
        self.assertEqual(result["decision"], "pass")
        self.assertEqual(result["risk_level"], "low")
        self.assertEqual(len(result["violations"]), 0)

    def test_check_code_safety_critical_violations(self):
        """Test safety check with critical violations"""
        dangerous_code = """
def dangerous_function():
    eval("print('dangerous')")
    os.system("rm -rf /")
    return "dangerous"
"""

        result = self.agent.check_code_safety(dangerous_code)

        self.assertFalse(result["safe"])
        self.assertEqual(result["decision"], "block")
        self.assertEqual(result["risk_level"], "critical")
        self.assertGreater(result["critical_violations"], 0)

    def test_check_code_safety_high_violations(self):
        """Test safety check with high violations"""
        risky_code = """
def risky_function():
    open("file.txt", "w")
    input("Enter something: ")
    return "risky"
"""

        result = self.agent.check_code_safety(risky_code)

        self.assertFalse(result["safe"])
        self.assertEqual(result["decision"], "block")
        self.assertEqual(result["risk_level"], "high")
        self.assertGreater(result["high_violations"], 0)

    def test_check_code_safety_medium_violations(self):
        """Test safety check with medium violations"""
        medium_risk_code = """
def medium_risk_function():
    print("Hello")
    assert True
    return "medium risk"
"""

        result = self.agent.check_code_safety(medium_risk_code)

        self.assertTrue(result["safe"])
        self.assertEqual(result["decision"], "warn")
        self.assertEqual(result["risk_level"], "medium")
        self.assertGreater(result["medium_violations"], 0)

    def test_check_code_safety_low_violations(self):
        """Test safety check with low violations"""
        low_risk_code = """
def low_risk_function():
    # TODO: Add more features
    # FIXME: Fix this later
    return "low risk"
"""

        result = self.agent.check_code_safety(low_risk_code)

        self.assertTrue(result["safe"])
        self.assertEqual(result["decision"], "pass")
        self.assertEqual(result["risk_level"], "low")
        self.assertGreater(result["low_violations"], 0)

    def test_check_for_secrets(self):
        """Test secret detection"""
        code_with_secrets = """
def function_with_secrets():
    password = "secret123"
    api_key = "sk-1234567890abcdef"
    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    return "secrets"
"""

        violations = self.agent._check_for_secrets(code_with_secrets)

        self.assertGreater(len(violations), 0)
        self.assertTrue(all(v["severity"] == "critical" for v in violations))
        self.assertTrue(any("password" in v["match"] for v in violations))
        self.assertTrue(any("api_key" in v["match"] for v in violations))

    def test_check_network_operations(self):
        """Test network operation detection"""
        code_with_network = """
def function_with_network():
    import requests
    response = requests.get("https://api.example.com")
    return response.json()
"""

        violations = self.agent._check_network_operations(code_with_network)

        self.assertGreater(len(violations), 0)
        self.assertTrue(all(v["severity"] == "high" for v in violations))
        self.assertTrue(any("requests.get" in v["match"] for v in violations))

    def test_get_line_number(self):
        """Test line number calculation"""
        code = "line1\nline2\nline3"

        # Test position at start of line 2
        line_num = self.agent._get_line_number(code, 6)  # After "line1\n"
        self.assertEqual(line_num, 2)

        # Test position at end
        line_num = self.agent._get_line_number(code, len(code))
        self.assertEqual(line_num, 3)

    def test_get_violation_description(self):
        """Test violation description generation"""
        descriptions = [
            (r"eval\s*\(", "Use of eval() function - potential code injection risk"),
            (
                r"os\.system\s*\(",
                "System command execution - potential command injection risk",
            ),
            (r"print\s*\(", "Print statement - consider using logging"),
            (r"TODO", "TODO comment - address before production"),
        ]

        for pattern, expected in descriptions:
            description = self.agent._get_violation_description(pattern, "critical")
            self.assertEqual(description, expected)

    def test_check_code_safety_with_error(self):
        """Test safety check when an error occurs"""
        with patch.object(self.agent, "_check_for_secrets", side_effect=Exception("Test error")):
            result = self.agent.check_code_safety("def test(): pass")

            self.assertFalse(result["safe"])
            self.assertEqual(result["decision"], "error")
            self.assertIn("error", result)

    def test_dangerous_patterns_coverage(self):
        """Test that all dangerous patterns are detected"""
        test_cases = [
            ("eval('dangerous')", "critical"),
            ("exec('dangerous')", "critical"),
            ("os.system('rm -rf /')", "critical"),
            ("subprocess.run(['ls'])", "critical"),
            ("open('file.txt')", "high"),
            ("input('Enter:')", "high"),
            ("print('Hello')", "medium"),
            ("assert True", "medium"),
            ("# TODO: Fix this", "low"),
            ("# FIXME: Address later", "low"),
        ]

        for code, expected_severity in test_cases:
            result = self.agent.check_code_safety(code)

            # Find the violation for this pattern
            found_violation = False
            for violation in result["violations"]:
                if violation["severity"] == expected_severity:
                    found_violation = True
                    break

            self.assertTrue(found_violation, f"Pattern not detected: {code}")

    def test_secret_patterns_coverage(self):
        """Test that all secret patterns are detected"""
        test_cases = [
            'password = "secret123"',
            'secret = "mysecret"',
            'api_key = "sk-1234567890abcdef"',
            'token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"',
            'key = "secretkey"',
            'private_key = "-----BEGIN PRIVATE KEY-----"',
            'secret_key = "mysecretkey"',
        ]

        for secret_code in test_cases:
            violations = self.agent._check_for_secrets(secret_code)
            self.assertGreater(len(violations), 0, f"Secret not detected: {secret_code}")
            self.assertTrue(all(v["severity"] == "critical" for v in violations))

    def test_network_patterns_coverage(self):
        """Test that all network patterns are detected"""
        test_cases = [
            'requests.get("https://api.example.com")',
            'requests.post("https://api.example.com", data={})',
            'urllib.request.urlopen("https://example.com")',
            "socket.socket()",
            'http.client.HTTPConnection("example.com")',
        ]

        for network_code in test_cases:
            violations = self.agent._check_network_operations(network_code)
            self.assertGreater(len(violations), 0, f"Network operation not detected: {network_code}")
            self.assertTrue(all(v["severity"] == "high" for v in violations))

    def test_analyze_method(self):
        """Test analyze method"""
        context = {"task_type": "api", "language": "python"}

        result = self.agent.analyze(context)

        self.assertIn("policy_compliance", result)
        self.assertIn("risk_assessment", result)
        self.assertIn("recommendations", result)

    def test_propose_method(self):
        """Test propose method"""
        analysis = {"policy_compliance": True}
        context = {"task_type": "api"}

        result = self.agent.propose(analysis, context)

        self.assertIn("policy_improvements", result)
        self.assertIn("security_enhancements", result)
        self.assertIn("compliance_measures", result)

    def test_review_method(self):
        """Test review method"""
        proposal = {"policy_improvements": []}
        context = {"task_type": "api"}

        result = self.agent.review(proposal, context)

        self.assertIn("policy_compliant", result)
        self.assertIn("risk_level", result)
        self.assertIn("recommendations", result)


if __name__ == "__main__":
    unittest.main()
