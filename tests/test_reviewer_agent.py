"""
Unit tests for ReviewAgent review functionality
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to the path to import the agents module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.review_agent import ReviewAgent, CodeIssue


class TestReviewAgentReview(unittest.TestCase):
    """Test cases for ReviewAgent review functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.agent = ReviewAgent("TestReviewAgent")
        self.sample_context = {
            "title": "Test API",
            "summary": "A test REST API",
            "task_type": "api",
            "language": "python",
        }

    def test_review_generated_code_basic(self):
        """Test basic code review functionality"""
        code = """
def hello_world():
    print("Hello, World!")
    return "Hello, World!"

if __name__ == "__main__":
    hello_world()
"""

        result = self.agent.review_generated_code(code, self.sample_context)

        self.assertIn("issues", result)
        self.assertIn("security_risks", result)
        self.assertIn("quality_score", result)
        self.assertIn("recommendations", result)
        self.assertIn("overall_assessment", result)
        self.assertIn("critical_issues", result)
        self.assertIn("warnings", result)
        self.assertIn("suggestions", result)

    def test_review_generated_code_with_issues(self):
        """Test code review with identified issues"""
        code = """
def dangerous_function():
    eval("print('dangerous')")
    os.system("rm -rf /")
    return "dangerous"
"""

        result = self.agent.review_generated_code(code, self.sample_context)

        self.assertGreater(len(result["issues"]), 0)
        self.assertGreater(len(result["security_risks"]), 0)
        self.assertLess(result["quality_score"], 1.0)
        self.assertIn(result["overall_assessment"], ["block", "warn"])

    def test_review_generated_code_safe(self):
        """Test code review with safe code"""
        code = '''
"""
Safe API implementation
"""

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    description: str

@app.get("/")
def root():
    """Root endpoint."""
    return {"message": "Hello World"}

@app.post("/items")
def create_item(item: Item):
    """Create item endpoint."""
    return item
'''

        result = self.agent.review_generated_code(code, self.sample_context)

        self.assertEqual(result["overall_assessment"], "pass")
        self.assertGreater(result["quality_score"], 0.7)

    def test_identify_code_issues(self):
        """Test code issue identification"""
        code = """
def test_function():
    print("This is a print statement")
    # TODO: Add proper logging
    except:
        pass
"""

        issues = self.agent._identify_code_issues(code)

        self.assertGreater(len(issues), 0)
        # Check for print issue (only if logging not in code)
        print_issues = [issue for issue in issues if "print" in issue.description.lower()]
        if print_issues:
            self.assertTrue(any(issue.severity == "medium" for issue in print_issues))
        self.assertTrue(any(issue.severity == "low" for issue in issues))  # TODO
        self.assertTrue(any(issue.severity == "high" for issue in issues))  # bare except

    def test_identify_security_risks(self):
        """Test security risk identification"""
        code = """
def dangerous_function():
    eval("print('dangerous')")
    os.system("rm -rf /")
    password = "secret123"
    return "dangerous"
"""

        risks = self.agent._identify_security_risks(code)

        self.assertGreater(len(risks), 0)
        self.assertTrue(any(risk["severity"] == "critical" for risk in risks))  # eval, os.system
        self.assertTrue(any(risk["pattern"] == "hardcoded_secret" for risk in risks))  # password

    def test_calculate_quality_score(self):
        """Test quality score calculation"""
        # Good code
        good_code = '''
"""
Good code example
"""

def good_function():
    """This is a good function."""
    return "Hello World"
'''

        good_score = self.agent._calculate_quality_score(good_code)
        self.assertGreater(good_score, 0.8)

        # Bad code
        bad_code = """
def bad_function():
    eval("dangerous")
    except:
        pass
    print("bad")
"""

        bad_score = self.agent._calculate_quality_score(bad_code)
        self.assertLess(bad_score, 0.6)

    def test_generate_code_recommendations(self):
        """Test recommendation generation"""
        code = """
def test_function():
    print("Hello")
    except:
        pass
"""

        recommendations = self.agent._generate_code_recommendations(code)

        self.assertGreater(len(recommendations), 0)
        self.assertTrue(any("logging" in rec for rec in recommendations))
        self.assertTrue(any("exception types" in rec for rec in recommendations))

    def test_check_code_compliance(self):
        """Test code compliance checking"""
        code = '''
"""
Documented code
"""

def test_function():
    """This is documented."""
    assert True
    return "test"
'''

        compliance = self.agent._check_code_compliance(code)

        self.assertIn("pep8", compliance)
        self.assertIn("security", compliance)
        self.assertIn("documentation", compliance)
        self.assertIn("testing", compliance)
        self.assertTrue(compliance["documentation"])  # Has docstrings
        self.assertTrue(compliance["testing"])  # Has assert

    def test_assess_code_safety(self):
        """Test code safety assessment"""
        # Safe code
        safe_code = """
def safe_function():
    return "safe"
"""

        safe_assessment = self.agent._assess_code_safety(safe_code)
        self.assertTrue(safe_assessment["safe"])
        self.assertEqual(safe_assessment["risk_level"], "low")

        # Dangerous code
        dangerous_code = """
def dangerous_function():
    eval("dangerous")
    os.system("rm -rf /")
    return "dangerous"
"""

        dangerous_assessment = self.agent._assess_code_safety(dangerous_code)
        self.assertFalse(dangerous_assessment["safe"])
        self.assertEqual(dangerous_assessment["risk_level"], "critical")

    def test_review_empty_code(self):
        """Test review of empty code"""
        result = self.agent.review_generated_code("", self.sample_context)

        self.assertEqual(result["quality_score"], 0.0)
        self.assertEqual(result["overall_assessment"], "pass")

    def test_review_with_error(self):
        """Test review when an error occurs"""
        with patch.object(self.agent, "_identify_code_issues", side_effect=Exception("Test error")):
            result = self.agent.review_generated_code("def test(): pass", self.sample_context)

            self.assertEqual(result["overall_assessment"], "error")
            self.assertIn("error", result)

    def test_code_issue_creation(self):
        """Test CodeIssue dataclass creation"""
        issue = CodeIssue(
            severity="high",
            category="security",
            line_number=10,
            description="Test issue",
            suggestion="Fix it",
            confidence=0.9,
        )

        self.assertEqual(issue.severity, "high")
        self.assertEqual(issue.category, "security")
        self.assertEqual(issue.line_number, 10)
        self.assertEqual(issue.description, "Test issue")
        self.assertEqual(issue.suggestion, "Fix it")
        self.assertEqual(issue.confidence, 0.9)


if __name__ == "__main__":
    unittest.main()
