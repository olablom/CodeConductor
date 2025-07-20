"""
Unit tests for CodeGenAgent
"""

import unittest
from unittest.mock import patch
import sys
import os

# Add the parent directory to the path to import the agents module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.codegen_agent import CodeGenAgent, CodeTemplate, CodePattern


class TestCodeGenAgent(unittest.TestCase):
    """Test cases for CodeGenAgent"""

    def setUp(self):
        """Set up test fixtures"""
        self.agent = CodeGenAgent("TestCodeGenAgent")
        self.sample_context = {
            "requirements": {
                "functionality": "Create a REST API endpoint",
                "language": "Python",
                "framework": "FastAPI",
                "input_validation": True,
                "error_handling": True,
                "documentation": True,
            },
            "constraints": {
                "performance": "high",
                "security": "critical",
                "maintainability": "medium",
            },
            "existing_code": """
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class UserModel(BaseModel):
    name: str
    email: str
""",
        }

    def test_init(self):
        """Test CodeGenAgent initialization"""
        agent = CodeGenAgent("TestAgent")
        self.assertEqual(agent.name, "TestAgent")
        self.assertIsNotNone(agent.codegen_config)
        self.assertIsNotNone(agent.language_patterns)
        self.assertIsNotNone(agent.framework_patterns)

    def test_init_with_config(self):
        """Test CodeGenAgent initialization with config"""
        config = {"codegen": {"default_language": "Python", "code_style": "pep8"}}
        agent = CodeGenAgent("TestAgent", config)
        self.assertEqual(
            agent.codegen_config, {"default_language": "Python", "code_style": "pep8"}
        )

    def test_analyze_basic(self):
        """Test basic analysis functionality"""
        result = self.agent.analyze(self.sample_context)

        self.assertIn("requirements_analysis", result)
        self.assertIn("code_patterns", result)
        self.assertIn("complexity_assessment", result)
        self.assertIn("language_support", result)
        self.assertIn("framework_compatibility", result)
        self.assertIn("code_quality_metrics", result)
        self.assertIn("generation_feasibility", result)
        self.assertIn("estimated_effort", result)

    def test_analyze_with_error(self):
        """Test analysis with error handling"""
        with patch.object(
            self.agent, "_extract_requirements", side_effect=Exception("Test error")
        ):
            result = self.agent.analyze(self.sample_context)

            self.assertIn("error", result)
            self.assertEqual(result["generation_feasibility"], "unknown")

    def test_propose_basic(self):
        """Test basic proposal functionality"""
        analysis = {
            "requirements_analysis": {"functionality": "REST API"},
            "code_patterns": ["endpoint_pattern"],
            "complexity_assessment": "medium",
            "language_support": "Python",
            "framework_compatibility": "FastAPI",
            "generation_feasibility": "high",
        }

        result = self.agent.propose(analysis, self.sample_context)

        self.assertIn("code_templates", result)
        self.assertIn("implementation_plan", result)
        self.assertIn("code_structure", result)
        self.assertIn("dependencies", result)
        self.assertIn("testing_strategy", result)
        self.assertIn("deployment_considerations", result)
        self.assertIn("estimated_complexity", result)
        self.assertIn("risk_assessment", result)

    def test_propose_with_error(self):
        """Test proposal with error handling"""
        with patch.object(
            self.agent, "_generate_code_templates", side_effect=Exception("Test error")
        ):
            analysis = {"generation_feasibility": "high"}
            result = self.agent.propose(analysis, self.sample_context)

            self.assertIn("error", result)
            self.assertEqual(result["code_templates"], [])

    def test_review_basic(self):
        """Test basic review functionality"""
        proposal = {
            "code_templates": [{"name": "endpoint", "code": "def endpoint(): pass"}],
            "implementation_plan": ["Step 1", "Step 2"],
            "code_structure": {"modules": ["main.py"]},
            "dependencies": ["fastapi"],
            "testing_strategy": ["unit tests"],
            "estimated_complexity": "medium",
        }

        result = self.agent.review(proposal, self.sample_context)

        self.assertIn("code_quality_assessment", result)
        self.assertIn("security_analysis", result)
        self.assertIn("performance_evaluation", result)
        self.assertIn("maintainability_review", result)
        self.assertIn("testing_coverage", result)
        self.assertIn("documentation_quality", result)
        self.assertIn("compliance_check", result)
        self.assertIn("approval_recommendation", result)
        self.assertIn("final_score", result)

    def test_review_with_error(self):
        """Test review with error handling"""
        with patch.object(
            self.agent, "_assess_code_quality", side_effect=Exception("Test error")
        ):
            proposal = {"code_templates": []}
            result = self.agent.review(proposal, self.sample_context)

            self.assertIn("error", result)
            self.assertEqual(result["approval_recommendation"], "reject")

    def test_extract_requirements(self):
        """Test requirements extraction"""
        requirements = self.agent._extract_requirements(self.sample_context)

        self.assertIn("functionality", requirements)
        self.assertIn("language", requirements)
        self.assertIn("framework", requirements)
        self.assertIn("input_validation", requirements)
        self.assertIn("error_handling", requirements)
        self.assertIn("documentation", requirements)

    def test_identify_code_patterns(self):
        """Test code pattern identification"""
        requirements = {"functionality": "REST API endpoint", "framework": "FastAPI"}

        patterns = self.agent._identify_code_patterns(requirements)

        self.assertIsInstance(patterns, list)
        self.assertGreater(len(patterns), 0)
        for pattern in patterns:
            self.assertIsInstance(pattern, CodePattern)
            self.assertIn(
                pattern.category,
                ["endpoint", "validation", "error_handling", "documentation"],
            )

    def test_assess_complexity(self):
        """Test complexity assessment"""
        requirements = {
            "functionality": "REST API endpoint",
            "input_validation": True,
            "error_handling": True,
            "documentation": True,
        }

        complexity = self.agent._assess_complexity(requirements)

        self.assertIn(complexity, ["low", "medium", "high"])

    def test_check_language_support(self):
        """Test language support checking"""
        requirements = {"language": "Python"}

        support = self.agent._check_language_support(requirements)

        self.assertIsInstance(support, str)
        self.assertIn(support, ["Python", "JavaScript", "Java", "C#", "Go", "Rust"])

    def test_check_framework_compatibility(self):
        """Test framework compatibility checking"""
        requirements = {"language": "Python", "framework": "FastAPI"}

        compatibility = self.agent._check_framework_compatibility(requirements)

        self.assertIsInstance(compatibility, str)
        self.assertIn(
            compatibility, ["FastAPI", "Flask", "Django", "Express", "Spring"]
        )

    def test_calculate_code_quality_metrics(self):
        """Test code quality metrics calculation"""
        existing_code = '''
def test_function():
    """Test function with docstring."""
    return "test"
'''

        metrics = self.agent._calculate_code_quality_metrics(existing_code)

        self.assertIn("readability_score", metrics)
        self.assertIn("maintainability_score", metrics)
        self.assertIn("documentation_score", metrics)
        self.assertIn("complexity_score", metrics)

        for score in metrics.values():
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_assess_generation_feasibility(self):
        """Test generation feasibility assessment"""
        requirements = {
            "functionality": "REST API endpoint",
            "language": "Python",
            "framework": "FastAPI",
        }
        constraints = {"performance": "high"}

        feasibility = self.agent._assess_generation_feasibility(
            requirements, constraints
        )

        self.assertIn(feasibility, ["high", "medium", "low", "unknown"])

    def test_estimate_effort(self):
        """Test effort estimation"""
        requirements = {
            "functionality": "REST API endpoint",
            "input_validation": True,
            "error_handling": True,
            "documentation": True,
        }
        complexity = "medium"

        effort = self.agent._estimate_effort(requirements, complexity)

        self.assertIn(effort, ["low", "medium", "high"])

    def test_generate_code_templates(self):
        """Test code template generation"""
        requirements = {
            "functionality": "REST API endpoint",
            "language": "Python",
            "framework": "FastAPI",
            "input_validation": True,
            "error_handling": True,
        }
        patterns = [
            CodePattern("endpoint", "REST endpoint pattern", "high", "FastAPI"),
            CodePattern("validation", "Input validation pattern", "medium", "Pydantic"),
        ]

        templates = self.agent._generate_code_templates(requirements, patterns)

        self.assertIsInstance(templates, list)
        self.assertGreater(len(templates), 0)
        for template in templates:
            self.assertIsInstance(template, CodeTemplate)
            self.assertIn(template.pattern_name, ["endpoint", "validation"])
            self.assertIsInstance(template.code, str)
            self.assertGreater(len(template.code), 0)

    def test_create_implementation_plan(self):
        """Test implementation plan creation"""
        requirements = {"functionality": "REST API endpoint", "framework": "FastAPI"}
        templates = [
            CodeTemplate(
                "endpoint", "FastAPI endpoint", "high", "def endpoint(): pass"
            ),
            CodeTemplate(
                "validation", "Input validation", "medium", "class Model: pass"
            ),
        ]

        plan = self.agent._create_implementation_plan(requirements, templates)

        self.assertIsInstance(plan, list)
        self.assertGreater(len(plan), 0)
        for step in plan:
            self.assertIsInstance(step, str)
            self.assertGreater(len(step), 0)

    def test_design_code_structure(self):
        """Test code structure design"""
        requirements = {"functionality": "REST API endpoint", "framework": "FastAPI"}
        templates = [
            CodeTemplate("endpoint", "FastAPI endpoint", "high", "def endpoint(): pass")
        ]

        structure = self.agent._design_code_structure(requirements, templates)

        self.assertIn("modules", structure)
        self.assertIn("classes", structure)
        self.assertIn("functions", structure)
        self.assertIn("dependencies", structure)
        self.assertIn("file_organization", structure)

    def test_identify_dependencies(self):
        """Test dependency identification"""
        requirements = {
            "language": "Python",
            "framework": "FastAPI",
            "input_validation": True,
        }
        templates = [
            CodeTemplate(
                "endpoint", "FastAPI endpoint", "high", "from fastapi import FastAPI"
            )
        ]

        dependencies = self.agent._identify_dependencies(requirements, templates)

        self.assertIsInstance(dependencies, list)
        self.assertGreater(len(dependencies), 0)
        for dep in dependencies:
            self.assertIsInstance(dep, str)

    def test_create_testing_strategy(self):
        """Test testing strategy creation"""
        requirements = {"functionality": "REST API endpoint", "framework": "FastAPI"}
        templates = [
            CodeTemplate("endpoint", "FastAPI endpoint", "high", "def endpoint(): pass")
        ]

        strategy = self.agent._create_testing_strategy(requirements, templates)

        self.assertIsInstance(strategy, list)
        self.assertGreater(len(strategy), 0)
        for test_type in strategy:
            self.assertIsInstance(test_type, str)

    def test_plan_deployment_considerations(self):
        """Test deployment considerations planning"""
        requirements = {"framework": "FastAPI", "performance": "high"}
        constraints = {"performance": "high", "security": "critical"}

        considerations = self.agent._plan_deployment_considerations(
            requirements, constraints
        )

        self.assertIsInstance(considerations, list)
        self.assertGreater(len(considerations), 0)
        for consideration in considerations:
            self.assertIsInstance(consideration, str)

    def test_assess_implementation_complexity(self):
        """Test implementation complexity assessment"""
        requirements = {
            "functionality": "REST API endpoint",
            "input_validation": True,
            "error_handling": True,
        }
        templates = [
            CodeTemplate(
                "endpoint", "FastAPI endpoint", "high", "def endpoint(): pass"
            ),
            CodeTemplate(
                "validation", "Input validation", "medium", "class Model: pass"
            ),
        ]

        complexity = self.agent._assess_implementation_complexity(
            requirements, templates
        )

        self.assertIn(complexity, ["low", "medium", "high"])

    def test_assess_implementation_risks(self):
        """Test implementation risk assessment"""
        requirements = {"security": "critical", "performance": "high"}
        templates = [
            CodeTemplate("endpoint", "FastAPI endpoint", "high", "def endpoint(): pass")
        ]

        risks = self.agent._assess_implementation_risks(requirements, templates)

        self.assertIn("security_risks", risks)
        self.assertIn("performance_risks", risks)
        self.assertIn("maintenance_risks", risks)
        self.assertIn("integration_risks", risks)

    def test_assess_code_quality(self):
        """Test code quality assessment"""
        templates = [
            CodeTemplate(
                "endpoint",
                "FastAPI endpoint",
                "high",
                '''
def endpoint():
    """REST API endpoint."""
    return {"message": "Hello World"}
''',
            )
        ]

        quality = self.agent._assess_code_quality(templates)

        self.assertIn("readability_score", quality)
        self.assertIn("maintainability_score", quality)
        self.assertIn("documentation_score", quality)
        self.assertIn("style_compliance", quality)

        for score in quality.values():
            if isinstance(score, float):
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)

    def test_analyze_security(self):
        """Test security analysis"""
        templates = [
            CodeTemplate(
                "endpoint",
                "FastAPI endpoint",
                "high",
                """
def endpoint(data):
    result = eval(data)  # Security issue
    return result
""",
            )
        ]

        security = self.agent._analyze_security(templates)

        self.assertIn("vulnerabilities", security)
        self.assertIn("security_score", security)
        self.assertIn("recommendations", security)

        self.assertIsInstance(security["vulnerabilities"], list)
        self.assertIsInstance(security["security_score"], float)
        self.assertIsInstance(security["recommendations"], list)

    def test_evaluate_performance(self):
        """Test performance evaluation"""
        templates = [
            CodeTemplate(
                "endpoint",
                "FastAPI endpoint",
                "high",
                """
def endpoint():
    for i in range(1000000):  # Performance issue
        pass
    return "done"
""",
            )
        ]

        performance = self.agent._evaluate_performance(templates)

        self.assertIn("performance_score", performance)
        self.assertIn("bottlenecks", performance)
        self.assertIn("optimization_opportunities", performance)

        self.assertIsInstance(performance["performance_score"], float)
        self.assertIsInstance(performance["bottlenecks"], list)
        self.assertIsInstance(performance["optimization_opportunities"], list)

    def test_review_maintainability(self):
        """Test maintainability review"""
        templates = [
            CodeTemplate(
                "endpoint",
                "FastAPI endpoint",
                "high",
                '''
def very_long_function_with_many_parameters(param1, param2, param3, param4, param5, param6, param7, param8, param9, param10):
    """This function is too long and has too many parameters."""
    result = 0
    for i in range(100):
        for j in range(100):
            for k in range(100):
                result += i + j + k
    return result
''',
            )
        ]

        maintainability = self.agent._review_maintainability(templates)

        self.assertIn("maintainability_score", maintainability)
        self.assertIn("complexity_issues", maintainability)
        self.assertIn("refactoring_suggestions", maintainability)

        self.assertIsInstance(maintainability["maintainability_score"], float)
        self.assertIsInstance(maintainability["complexity_issues"], list)
        self.assertIsInstance(maintainability["refactoring_suggestions"], list)

    def test_assess_testing_coverage(self):
        """Test testing coverage assessment"""
        requirements = {"functionality": "REST API endpoint"}
        strategy = ["unit tests", "integration tests"]

        coverage = self.agent._assess_testing_coverage(requirements, strategy)

        self.assertIn("coverage_score", coverage)
        self.assertIn("test_types", coverage)
        self.assertIn("coverage_gaps", coverage)

        self.assertIsInstance(coverage["coverage_score"], float)
        self.assertIsInstance(coverage["test_types"], list)
        self.assertIsInstance(coverage["coverage_gaps"], list)

    def test_evaluate_documentation_quality(self):
        """Test documentation quality evaluation"""
        templates = [
            CodeTemplate(
                "endpoint",
                "FastAPI endpoint",
                "high",
                '''
def endpoint():
    """REST API endpoint that returns a greeting message.
    
    Returns:
        dict: A dictionary containing a greeting message.
    """
    return {"message": "Hello World"}
''',
            )
        ]

        documentation = self.agent._evaluate_documentation_quality(templates)

        self.assertIn("documentation_score", documentation)
        self.assertIn("coverage", documentation)
        self.assertIn("quality_issues", documentation)

        self.assertIsInstance(documentation["documentation_score"], float)
        self.assertIsInstance(documentation["coverage"], dict)
        self.assertIsInstance(documentation["quality_issues"], list)

    def test_check_compliance(self):
        """Test compliance checking"""
        requirements = {"language": "Python", "framework": "FastAPI"}
        templates = [
            CodeTemplate("endpoint", "FastAPI endpoint", "high", "def endpoint(): pass")
        ]

        compliance = self.agent._check_compliance(requirements, templates)

        self.assertIn("standards_compliance", compliance)
        self.assertIn("framework_guidelines", compliance)
        self.assertIn("best_practices", compliance)
        self.assertIn("violations", compliance)

        self.assertIsInstance(compliance["standards_compliance"], dict)
        self.assertIsInstance(compliance["framework_guidelines"], dict)
        self.assertIsInstance(compliance["best_practices"], list)
        self.assertIsInstance(compliance["violations"], list)

    def test_make_approval_recommendation(self):
        """Test approval recommendation"""
        quality_assessment = {"readability_score": 0.8, "maintainability_score": 0.7}
        security_analysis = {"security_score": 0.9, "vulnerabilities": []}
        performance_evaluation = {"performance_score": 0.8}
        maintainability_review = {"maintainability_score": 0.7}
        testing_coverage = {"coverage_score": 0.8}
        documentation_quality = {"documentation_score": 0.9}
        compliance_check = {"standards_compliance": {"pep8": True}}

        recommendation = self.agent._make_approval_recommendation(
            quality_assessment,
            security_analysis,
            performance_evaluation,
            maintainability_review,
            testing_coverage,
            documentation_quality,
            compliance_check,
        )

        self.assertIn(recommendation, ["approve", "approve_with_caution", "reject"])

    def test_calculate_final_score(self):
        """Test final score calculation"""
        quality_assessment = {"readability_score": 0.8, "maintainability_score": 0.7}
        security_analysis = {"security_score": 0.9}
        performance_evaluation = {"performance_score": 0.8}
        maintainability_review = {"maintainability_score": 0.7}
        testing_coverage = {"coverage_score": 0.8}
        documentation_quality = {"documentation_score": 0.9}
        compliance_check = {"standards_compliance": {"pep8": True}}

        score = self.agent._calculate_final_score(
            quality_assessment,
            security_analysis,
            performance_evaluation,
            maintainability_review,
            testing_coverage,
            documentation_quality,
            compliance_check,
        )

        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_generate_python_template(self):
        """Test Python template generation"""
        pattern = CodePattern("endpoint", "REST endpoint", "high", "FastAPI")
        requirements = {"framework": "FastAPI", "input_validation": True}

        template = self.agent._generate_python_template(pattern, requirements)

        self.assertIsInstance(template, CodeTemplate)
        self.assertEqual(template.pattern_name, "endpoint")
        self.assertIn("def", template.code)
        self.assertIn("FastAPI", template.code)

    def test_generate_javascript_template(self):
        """Test JavaScript template generation"""
        pattern = CodePattern("endpoint", "REST endpoint", "high", "Express")
        requirements = {"framework": "Express", "input_validation": True}

        template = self.agent._generate_javascript_template(pattern, requirements)

        self.assertIsInstance(template, CodeTemplate)
        self.assertEqual(template.pattern_name, "endpoint")
        self.assertIn("function", template.code)
        self.assertIn("Express", template.code)

    def test_apply_code_style(self):
        """Test code style application"""
        code = """
def test_function(  param1,param2  ):
    return param1+param2
"""
        style = "pep8"

        styled_code = self.agent._apply_code_style(code, style)

        self.assertIsInstance(styled_code, str)
        self.assertNotEqual(code, styled_code)  # Should be different after styling

    def test_validate_code_syntax(self):
        """Test code syntax validation"""
        valid_code = """
def test_function():
    return "test"
"""

        is_valid = self.agent._validate_code_syntax(valid_code, "Python")
        self.assertTrue(is_valid)

        invalid_code = """
def test_function(:
    return "test"
"""

        is_valid = self.agent._validate_code_syntax(invalid_code, "Python")
        self.assertFalse(is_valid)


class TestCodeTemplate(unittest.TestCase):
    """Test cases for CodeTemplate dataclass"""

    def test_code_template_creation(self):
        """Test CodeTemplate creation"""
        template = CodeTemplate(
            pattern_name="endpoint",
            description="REST API endpoint",
            priority="high",
            code='def endpoint(): return {"message": "Hello"}',
        )

        self.assertEqual(template.pattern_name, "endpoint")
        self.assertEqual(template.description, "REST API endpoint")
        self.assertEqual(template.priority, "high")
        self.assertEqual(template.code, 'def endpoint(): return {"message": "Hello"}')


class TestCodePattern(unittest.TestCase):
    """Test cases for CodePattern dataclass"""

    def test_code_pattern_creation(self):
        """Test CodePattern creation"""
        pattern = CodePattern(
            name="endpoint",
            description="REST endpoint pattern",
            priority="high",
            framework="FastAPI",
        )

        self.assertEqual(pattern.name, "endpoint")
        self.assertEqual(pattern.description, "REST endpoint pattern")
        self.assertEqual(pattern.priority, "high")
        self.assertEqual(pattern.framework, "FastAPI")


if __name__ == "__main__":
    unittest.main()
