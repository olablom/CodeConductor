"""
Unit tests for ReviewAgent
"""

import unittest
from unittest.mock import patch
import sys
import os

# Add the parent directory to the path to import the agents module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.review_agent import ReviewAgent, CodeIssue


class TestReviewAgent(unittest.TestCase):
    """Test cases for ReviewAgent"""

    def setUp(self):
        """Set up test fixtures"""
        self.agent = ReviewAgent("TestReviewAgent")
        self.sample_code = '''
def calculate_fibonacci(n):
    """Calculate fibonacci number."""
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

def process_data(data):
    result = eval(data)  # Security issue
    return result
'''
        self.sample_context = {
            "code": self.sample_code,
            "requirements": {"require_type_hints": True, "require_docstrings": True},
            "constraints": {"performance_critical": False, "security_critical": True},
        }

    def test_init(self):
        """Test ReviewAgent initialization"""
        agent = ReviewAgent("TestAgent")
        self.assertEqual(agent.name, "TestAgent")
        self.assertIsNotNone(agent.severity_weights)
        self.assertIsNotNone(agent.quality_thresholds)

    def test_init_with_config(self):
        """Test ReviewAgent initialization with config"""
        config = {"review": {"custom_threshold": 0.8}}
        agent = ReviewAgent("TestAgent", config)
        self.assertEqual(agent.review_config, {"custom_threshold": 0.8})

    def test_analyze_basic(self):
        """Test basic analysis functionality"""
        result = self.agent.analyze(self.sample_context)

        self.assertIn("code_quality_score", result)
        self.assertIn("security_analysis", result)
        self.assertIn("performance_analysis", result)
        self.assertIn("maintainability_analysis", result)
        self.assertIn("issues_found", result)
        self.assertIn("strengths", result)
        self.assertIn("recommendations", result)

        self.assertIsInstance(result["code_quality_score"], float)
        self.assertGreaterEqual(result["code_quality_score"], 0.0)
        self.assertLessEqual(result["code_quality_score"], 1.0)

    def test_analyze_with_error(self):
        """Test analysis with error handling"""
        with patch.object(self.agent, "_assess_code_quality", side_effect=Exception("Test error")):
            result = self.agent.analyze(self.sample_context)

            self.assertIn("error", result)
            self.assertEqual(result["code_quality_score"], 0.0)

    def test_propose_basic(self):
        """Test basic proposal functionality"""
        analysis = {
            "code_quality_score": 0.6,
            "security_analysis": {"overall_security_score": 0.5},
            "performance_analysis": {"performance_score": 0.7},
            "maintainability_analysis": {"maintainability_score": 0.8},
            "issues_found": [],
            "recommendations": ["Improve security"],
        }

        result = self.agent.propose(analysis, self.sample_context)

        self.assertIn("improvement_plan", result)
        self.assertIn("refactoring_suggestions", result)
        self.assertIn("security_enhancements", result)
        self.assertIn("performance_optimizations", result)
        self.assertIn("priority_order", result)
        self.assertIn("estimated_effort", result)
        self.assertIn("risk_assessment", result)

    def test_propose_with_error(self):
        """Test proposal with error handling"""
        with patch.object(self.agent, "_create_improvement_plan", side_effect=Exception("Test error")):
            analysis = {"issues_found": [], "recommendations": []}
            result = self.agent.propose(analysis, self.sample_context)

            self.assertIn("error", result)
            self.assertEqual(result["improvement_plan"], [])

    def test_review_basic(self):
        """Test basic review functionality"""
        proposal = {
            "improvement_plan": [],
            "priority_order": [],
            "risk_assessment": {"breaking_changes": "low"},
        }

        result = self.agent.review(proposal, self.sample_context)

        self.assertIn("feasibility_assessment", result)
        self.assertIn("impact_analysis", result)
        self.assertIn("alternative_suggestions", result)
        self.assertIn("implementation_guidance", result)
        self.assertIn("validation_criteria", result)
        self.assertIn("rollback_plan", result)
        self.assertIn("approval_recommendation", result)
        self.assertIn("final_score", result)

    def test_review_with_error(self):
        """Test review with error handling"""
        with patch.object(self.agent, "_assess_feasibility", side_effect=Exception("Test error")):
            proposal = {"improvement_plan": [], "priority_order": []}
            result = self.agent.review(proposal, self.sample_context)

            self.assertIn("error", result)
            self.assertEqual(result["approval_recommendation"], "reject")

    def test_assess_code_quality(self):
        """Test code quality assessment"""
        score = self.agent._assess_code_quality(self.sample_code)

        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_analyze_security(self):
        """Test security analysis"""
        result = self.agent._analyze_security(self.sample_code)

        self.assertIn("issues", result)
        self.assertIn("overall_security_score", result)
        self.assertIn("risk_level", result)

        # Should detect eval() as security issue
        self.assertGreater(len(result["issues"]), 0)
        self.assertLess(result["overall_security_score"], 1.0)

    def test_analyze_performance(self):
        """Test performance analysis"""
        result = self.agent._analyze_performance(self.sample_code)

        self.assertIn("issues", result)
        self.assertIn("performance_score", result)
        self.assertIn("optimization_opportunities", result)

        self.assertIsInstance(result["performance_score"], float)
        self.assertGreaterEqual(result["performance_score"], 0.0)
        self.assertLessEqual(result["performance_score"], 1.0)

    def test_analyze_maintainability(self):
        """Test maintainability analysis"""
        result = self.agent._analyze_maintainability(self.sample_code)

        self.assertIn("issues", result)
        self.assertIn("maintainability_score", result)
        self.assertIn("complexity_level", result)

        self.assertIsInstance(result["maintainability_score"], float)
        self.assertGreaterEqual(result["maintainability_score"], 0.0)
        self.assertLessEqual(result["maintainability_score"], 1.0)

    def test_check_compliance(self):
        """Test compliance checking"""
        requirements = {"require_type_hints": True, "require_docstrings": True}

        result = self.agent._check_compliance(self.sample_code, requirements)

        self.assertIn("issues", result)
        self.assertIn("compliance_score", result)
        self.assertIn("standards_met", result)

        # Should detect missing type hints
        self.assertGreater(len(result["issues"]), 0)
        self.assertLess(result["compliance_score"], 1.0)

    def test_calculate_complexity_metrics(self):
        """Test complexity metrics calculation"""
        result = self.agent._calculate_complexity_metrics(self.sample_code)

        self.assertIn("total_lines", result)
        self.assertIn("non_empty_lines", result)
        self.assertIn("comment_lines", result)
        self.assertIn("cyclomatic_complexity", result)
        self.assertIn("comment_ratio", result)
        self.assertIn("complexity_per_line", result)

        self.assertIsInstance(result["total_lines"], int)
        self.assertIsInstance(result["cyclomatic_complexity"], int)
        self.assertIsInstance(result["comment_ratio"], float)

    def test_analyze_test_coverage(self):
        """Test test coverage analysis"""
        result = self.agent._analyze_test_coverage(self.sample_code)

        self.assertIn("test_patterns_found", result)
        self.assertIn("has_testing_framework", result)
        self.assertIn("has_assertions", result)
        self.assertIn("test_coverage_estimate", result)

        self.assertIsInstance(result["test_patterns_found"], int)
        self.assertIsInstance(result["has_testing_framework"], bool)
        self.assertIsInstance(result["test_coverage_estimate"], float)

    def test_assess_documentation(self):
        """Test documentation assessment"""
        result = self.agent._assess_documentation(self.sample_code)

        self.assertIn("documentation_patterns", result)
        self.assertIn("has_docstrings", result)
        self.assertIn("has_comments", result)
        self.assertIn("documentation_score", result)

        self.assertIsInstance(result["documentation_patterns"], int)
        self.assertIsInstance(result["has_docstrings"], bool)
        self.assertIsInstance(result["documentation_score"], float)

    def test_identify_issues(self):
        """Test issue identification"""
        requirements = {"require_type_hints": True}
        issues = self.agent._identify_issues(self.sample_code, requirements)

        self.assertIsInstance(issues, list)
        for issue in issues:
            self.assertIsInstance(issue, CodeIssue)
            self.assertIn(issue.severity, ["critical", "high", "medium", "low"])
            self.assertIn(
                issue.category,
                ["security", "performance", "maintainability", "style", "bug"],
            )
            self.assertIsInstance(issue.confidence, float)
            self.assertGreaterEqual(issue.confidence, 0.0)
            self.assertLessEqual(issue.confidence, 1.0)

    def test_identify_strengths(self):
        """Test strength identification"""
        strengths = self.agent._identify_strengths(self.sample_code)

        self.assertIsInstance(strengths, list)
        # Should identify docstrings as strength
        self.assertIn("Includes docstrings", strengths)

    def test_generate_recommendations(self):
        """Test recommendation generation"""
        analysis = {
            "code_quality_score": 0.5,
            "security_analysis": {"overall_security_score": 0.6},
            "performance_analysis": {"performance_score": 0.7},
            "maintainability_analysis": {"maintainability_score": 0.8},
        }
        requirements = {"require_type_hints": True}
        constraints = {"security_critical": True}

        recommendations = self.agent._generate_recommendations(analysis, requirements, constraints)

        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)

    def test_create_improvement_plan(self):
        """Test improvement plan creation"""
        issues = [
            CodeIssue("high", "security", None, "Test issue", "Fix it", 0.8),
            CodeIssue("medium", "performance", None, "Test issue 2", "Fix it 2", 0.7),
        ]
        recommendations = ["Improve security", "Optimize performance"]

        plan = self.agent._create_improvement_plan(issues, recommendations)

        self.assertIsInstance(plan, list)
        for item in plan:
            self.assertIn("category", item)
            self.assertIn("priority", item)
            self.assertIn("issues", item)
            self.assertIn("action", item)
            self.assertIn("estimated_effort", item)

    def test_generate_refactoring_suggestions(self):
        """Test refactoring suggestions generation"""
        analysis = {
            "complexity_metrics": {"complexity_per_line": 0.8, "total_lines": 150},
            "maintainability_analysis": {"maintainability_score": 0.5},
        }

        suggestions = self.agent._generate_refactoring_suggestions(analysis)

        self.assertIsInstance(suggestions, list)
        self.assertGreater(len(suggestions), 0)

    def test_propose_security_enhancements(self):
        """Test security enhancements proposal"""
        analysis = {"security_analysis": {"overall_security_score": 0.6}}

        enhancements = self.agent._propose_security_enhancements(analysis)

        self.assertIsInstance(enhancements, list)
        self.assertGreater(len(enhancements), 0)

    def test_propose_performance_optimizations(self):
        """Test performance optimizations proposal"""
        analysis = {"performance_analysis": {"performance_score": 0.6}}

        optimizations = self.agent._propose_performance_optimizations(analysis)

        self.assertIsInstance(optimizations, list)
        self.assertGreater(len(optimizations), 0)

    def test_propose_style_improvements(self):
        """Test style improvements proposal"""
        analysis = {"compliance_check": {"standards_met": False}}

        improvements = self.agent._propose_style_improvements(analysis)

        self.assertIsInstance(improvements, list)
        self.assertGreater(len(improvements), 0)

    def test_propose_testing_improvements(self):
        """Test testing improvements proposal"""
        analysis = {"test_coverage_analysis": {"test_coverage_estimate": 0.5}}

        improvements = self.agent._propose_testing_improvements(analysis)

        self.assertIsInstance(improvements, list)
        self.assertGreater(len(improvements), 0)

    def test_propose_documentation_improvements(self):
        """Test documentation improvements proposal"""
        analysis = {"documentation_quality": {"documentation_score": 0.4}}

        improvements = self.agent._propose_documentation_improvements(analysis)

        self.assertIsInstance(improvements, list)
        self.assertGreater(len(improvements), 0)

    def test_prioritize_improvements(self):
        """Test improvement prioritization"""
        issues = [
            CodeIssue("critical", "security", None, "Critical issue", "Fix it", 0.9),
            CodeIssue("high", "performance", None, "High issue", "Fix it", 0.8),
            CodeIssue("low", "style", None, "Low issue", "Fix it", 0.6),
        ]
        requirements = {"security_critical": True}

        priorities = self.agent._prioritize_improvements(issues, requirements)

        self.assertIsInstance(priorities, list)
        self.assertGreater(len(priorities), 0)
        # Critical issues should come first
        self.assertIn("CRITICAL", priorities[0])

    def test_estimate_effort(self):
        """Test effort estimation"""
        issues = [
            CodeIssue("critical", "security", None, "Test", "Fix", 0.8),
            CodeIssue("high", "performance", None, "Test", "Fix", 0.7),
        ]

        effort = self.agent._estimate_effort(issues)

        self.assertIn(effort, ["low", "medium", "high", "unknown"])

    def test_assess_improvement_risks(self):
        """Test improvement risk assessment"""
        analysis = {
            "code_quality_score": 0.3,
            "performance_analysis": {"performance_score": 0.4},
        }

        risks = self.agent._assess_improvement_risks(analysis)

        self.assertIn("breaking_changes", risks)
        self.assertIn("performance_impact", risks)
        self.assertIn("security_risks", risks)
        self.assertIn("maintenance_overhead", risks)

    def test_assess_feasibility(self):
        """Test feasibility assessment"""
        proposal = {
            "improvement_plan": [],
            "risk_assessment": {"breaking_changes": "low"},
        }
        context = {"analysis": {"code_quality_score": 0.8}}

        feasibility = self.agent._assess_feasibility(proposal, context)

        self.assertIn(feasibility, ["high", "medium", "low", "unknown"])

    def test_analyze_impact(self):
        """Test impact analysis"""
        proposal = {"improvement_plan": []}
        context = {"requirements": {}}

        impact = self.agent._analyze_impact(proposal, context)

        self.assertIn("code_quality_improvement", impact)
        self.assertIn("maintenance_effort", impact)
        self.assertIn("learning_curve", impact)
        self.assertIn("time_to_implement", impact)

    def test_suggest_alternatives(self):
        """Test alternative suggestions"""
        proposal = {"improvement_plan": []}
        context = {"requirements": {}}

        alternatives = self.agent._suggest_alternatives(proposal, context)

        self.assertIsInstance(alternatives, list)
        self.assertGreater(len(alternatives), 0)

    def test_provide_implementation_guidance(self):
        """Test implementation guidance"""
        proposal = {"improvement_plan": []}

        guidance = self.agent._provide_implementation_guidance(proposal)

        self.assertIn("step_by_step_plan", guidance)
        self.assertIn("tools_recommended", guidance)
        self.assertIn("best_practices", guidance)

    def test_define_validation_criteria(self):
        """Test validation criteria definition"""
        proposal = {"improvement_plan": []}

        criteria = self.agent._define_validation_criteria(proposal)

        self.assertIsInstance(criteria, list)
        self.assertGreater(len(criteria), 0)

    def test_create_rollback_plan(self):
        """Test rollback plan creation"""
        proposal = {"improvement_plan": []}

        rollback_plan = self.agent._create_rollback_plan(proposal)

        self.assertIn("backup_strategy", rollback_plan)
        self.assertIn("rollback_triggers", rollback_plan)
        self.assertIn("rollback_steps", rollback_plan)

    def test_make_approval_recommendation(self):
        """Test approval recommendation"""
        proposal = {
            "feasibility_assessment": "high",
            "risk_assessment": {"breaking_changes": "low"},
        }
        context = {"requirements": {}}

        recommendation = self.agent._make_approval_recommendation(proposal, context)

        self.assertIn(recommendation, ["approve", "approve_with_caution", "reject"])

    def test_calculate_final_score(self):
        """Test final score calculation"""
        proposal = {
            "feasibility_assessment": "high",
            "improvement_plan": [{"category": "security"}],
        }
        context = {"analysis": {"code_quality_score": 0.7}}

        score = self.agent._calculate_final_score(proposal, context)

        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_analyze_complexity(self):
        """Test complexity analysis"""
        score = self.agent._analyze_complexity(self.sample_code)

        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_analyze_code_style(self):
        """Test code style analysis"""
        score = self.agent._analyze_code_style(self.sample_code)

        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_analyze_code_structure(self):
        """Test code structure analysis"""
        score = self.agent._analyze_code_structure(self.sample_code)

        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestCodeIssue(unittest.TestCase):
    """Test cases for CodeIssue dataclass"""

    def test_code_issue_creation(self):
        """Test CodeIssue creation"""
        issue = CodeIssue(
            severity="high",
            category="security",
            line_number=42,
            description="Test issue",
            suggestion="Fix it",
            confidence=0.8,
        )

        self.assertEqual(issue.severity, "high")
        self.assertEqual(issue.category, "security")
        self.assertEqual(issue.line_number, 42)
        self.assertEqual(issue.description, "Test issue")
        self.assertEqual(issue.suggestion, "Fix it")
        self.assertEqual(issue.confidence, 0.8)


if __name__ == "__main__":
    unittest.main()
