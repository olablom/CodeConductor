"""
CodeGenAgent - Specialized agent for code generation

This module implements a specialized agent that focuses on code generation,
analysis, and optimization. It inherits from BaseAgent and provides
expertise in programming patterns, best practices, and code quality.
"""

import logging
from typing import Dict, Any, List, Optional
from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class CodeGenAgent(BaseAgent):
    """
    Specialized agent for code generation and analysis.

    This agent focuses on:
    - Code generation strategies
    - Programming patterns and best practices
    - Code quality analysis
    - Performance optimization
    - Language-specific recommendations
    """

    def __init__(
        self, name: str = "codegen_agent", config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the code generation agent."""
        default_config = {
            "preferred_languages": ["python", "javascript", "typescript", "java", "go"],
            "code_style": "clean",  # "clean", "verbose", "minimal"
            "include_tests": True,
            "include_docs": True,
            "performance_focus": "balanced",  # "balanced", "speed", "memory"
            "security_level": "standard",  # "standard", "high", "paranoid"
            "complexity_preference": "moderate",  # "simple", "moderate", "advanced"
        }

        if config:
            default_config.update(config)

        super().__init__(name, default_config)

        logger.info(f"Initialized CodeGenAgent with config: {self.config}")

    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the task context for code generation.

        Args:
            context: Dictionary containing task requirements and context

        Returns:
            Analysis of the code generation task
        """
        task_type = context.get("task_type", "unknown")
        requirements = context.get("requirements", "")
        language = context.get("language", "python")
        complexity = context.get("complexity", "moderate")

        analysis = {
            "task_type": task_type,
            "language": language,
            "complexity_level": complexity,
            "key_requirements": self._extract_requirements(requirements),
            "recommended_patterns": self._identify_patterns(task_type, language),
            "potential_challenges": self._identify_challenges(requirements, language),
            "estimated_complexity": self._estimate_complexity(requirements, language),
            "recommended_approach": self._recommend_approach(
                task_type, language, complexity
            ),
            "code_quality_focus": self._determine_quality_focus(context),
            "performance_considerations": self._analyze_performance_needs(context),
            "security_considerations": self._analyze_security_needs(context),
            "testing_strategy": self._recommend_testing_strategy(task_type, language),
            "documentation_needs": self._assess_documentation_needs(context),
        }

        logger.debug(f"CodeGenAgent analysis completed for {task_type}")
        return analysis

    def propose(
        self, analysis: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Propose a code generation strategy based on analysis and context.

        Args:
            analysis: Analysis results from analyze()
            context: Original context information

        Returns:
            Proposed code generation approach
        """
        task_type = analysis.get("task_type", "unknown")
        language = analysis.get("language", "python")
        complexity = analysis.get("complexity_level", "moderate")

        proposal = {
            "approach": self._generate_approach(analysis),
            "code_structure": self._design_code_structure(analysis),
            "implementation_plan": self._create_implementation_plan(analysis),
            "code_template": self._generate_code_template(analysis),
            "patterns_to_use": analysis.get("recommended_patterns", []),
            "quality_guidelines": self._define_quality_guidelines(analysis),
            "testing_approach": analysis.get("testing_strategy", "unit_tests"),
            "documentation_plan": self._create_documentation_plan(analysis),
            "performance_optimizations": analysis.get("performance_considerations", []),
            "security_measures": analysis.get("security_considerations", []),
            "estimated_lines": self._estimate_code_size(analysis),
            "confidence": self._calculate_confidence(analysis),
            "reasoning": self._generate_reasoning(analysis),
            "suggestions": self._generate_suggestions(analysis),
        }

        logger.debug(
            f"CodeGenAgent proposal completed with confidence {proposal['confidence']:.2f}"
        )
        return proposal

    def review(
        self, proposal: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Review a proposal and provide feedback on quality and improvements.

        Args:
            proposal: Proposal to review
            context: Original context information

        Returns:
            Review results with feedback and recommendations
        """
        # Extract code from proposal if available
        code = proposal.get("code_template", "")

        review = {
            "quality_score": self._assess_code_quality(code),
            "issues": self._identify_issues(code),
            "suggestions": self._generate_improvement_suggestions(code),
            "best_practices": self._check_best_practices(code),
            "performance_analysis": self._analyze_performance(code),
            "security_analysis": self._analyze_security(code),
            "readability_score": self._assess_readability(code),
            "maintainability_score": self._assess_maintainability(code),
            "test_coverage_recommendations": self._recommend_test_coverage(code),
            "documentation_gaps": self._identify_documentation_gaps(code),
            "overall_assessment": self._provide_overall_assessment(code),
            "proposal_assessment": self._assess_proposal_quality(proposal),
        }

        logger.debug(
            f"CodeGenAgent review completed with quality score {review['quality_score']:.2f}"
        )
        return review

    def _extract_requirements(self, requirements) -> List[str]:
        """Extract key requirements from the requirements text or dict."""
        if not requirements:
            return ["Implement basic functionality"]

        # Handle both string and dictionary inputs
        if isinstance(requirements, dict):
            # Extract requirements from dictionary
            req_list = []
            for key, value in requirements.items():
                if isinstance(value, str):
                    req_list.append(f"{key}: {value}")
                else:
                    req_list.append(f"{key}: {str(value)}")
            requirements_text = "\n".join(req_list)
        else:
            requirements_text = str(requirements)

        # Simple requirement extraction - can be enhanced with NLP
        lines = requirements_text.split("\n")
        key_requirements = []

        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                if any(
                    keyword in line.lower()
                    for keyword in ["must", "should", "need", "require", "implement"]
                ):
                    key_requirements.append(line)

        return (
            key_requirements if key_requirements else ["Implement basic functionality"]
        )

    def _identify_patterns(self, task_type: str, language: str) -> List[str]:
        """Identify recommended design patterns for the task."""
        patterns = []

        # Language-specific patterns
        if language == "python":
            patterns.extend(["context managers", "decorators", "generators"])
        elif language == "javascript":
            patterns.extend(["async/await", "modules", "closures"])
        elif language == "java":
            patterns.extend(["builder pattern", "factory pattern", "singleton"])

        # Task-specific patterns
        if "api" in task_type.lower():
            patterns.extend(["RESTful design", "error handling", "validation"])
        elif "database" in task_type.lower():
            patterns.extend(
                ["repository pattern", "data access objects", "transactions"]
            )
        elif "web" in task_type.lower():
            patterns.extend(["MVC pattern", "routing", "middleware"])

        return patterns[:5]  # Return top 5 patterns

    def _identify_challenges(self, requirements, language: str) -> List[str]:
        """Identify potential challenges in the implementation."""
        challenges = []

        # Convert requirements to string if it's a dict
        if isinstance(requirements, dict):
            requirements_text = str(requirements)
        else:
            requirements_text = str(requirements)

        if (
            "async" in requirements_text.lower()
            or "concurrent" in requirements_text.lower()
        ):
            challenges.append("Concurrency management")

        if (
            "performance" in requirements_text.lower()
            or "optimization" in requirements_text.lower()
        ):
            challenges.append("Performance optimization")

        if "security" in requirements_text.lower():
            challenges.append("Security implementation")

        if "scalable" in requirements_text.lower():
            challenges.append("Scalability considerations")

        # Language-specific challenges
        if language == "python" and "memory" in requirements_text.lower():
            challenges.append("Memory management in Python")
        elif language == "javascript" and "browser" in requirements_text.lower():
            challenges.append("Browser compatibility")

        return challenges

    def _estimate_complexity(self, requirements, language: str) -> str:
        """Estimate the complexity of the implementation."""
        complexity_indicators = {
            "simple": ["basic", "simple", "straightforward"],
            "moderate": ["standard", "typical", "common"],
            "complex": ["advanced", "complex", "sophisticated", "optimized"],
        }

        # Convert requirements to string if it's a dict
        if isinstance(requirements, dict):
            requirements_text = str(requirements)
        else:
            requirements_text = str(requirements)

        requirements_lower = requirements_text.lower()

        for level, indicators in complexity_indicators.items():
            if any(indicator in requirements_lower for indicator in indicators):
                return level

        return "moderate"  # Default complexity

    def _recommend_approach(
        self, task_type: str, language: str, complexity: str
    ) -> str:
        """Recommend the overall approach for implementation."""
        if "api" in task_type.lower():
            return "RESTful API with proper error handling and validation"
        elif "database" in task_type.lower():
            return "Data access layer with repository pattern and transactions"
        elif "web" in task_type.lower():
            return "Web application with MVC architecture and routing"
        elif "utility" in task_type.lower():
            return "Utility functions with comprehensive error handling"
        else:
            return f"Standard {language} implementation with best practices"

    def _determine_quality_focus(self, context: Dict[str, Any]) -> List[str]:
        """Determine the focus areas for code quality."""
        focus_areas = ["readability", "maintainability"]

        if context.get("performance_critical", False):
            focus_areas.append("performance")

        if context.get("security_critical", False):
            focus_areas.append("security")

        if context.get("test_coverage", False):
            focus_areas.append("testability")

        return focus_areas

    def _analyze_performance_needs(self, context: Dict[str, Any]) -> List[str]:
        """Analyze performance requirements and considerations."""
        considerations = []

        if context.get("high_performance", False):
            considerations.extend(
                [
                    "Algorithm optimization",
                    "Memory usage optimization",
                    "Caching strategies",
                    "Async processing where applicable",
                ]
            )

        if context.get("scalable", False):
            considerations.extend(
                [
                    "Horizontal scaling considerations",
                    "Database query optimization",
                    "Load balancing preparation",
                ]
            )

        return considerations

    def _analyze_security_needs(self, context: Dict[str, Any]) -> List[str]:
        """Analyze security requirements and considerations."""
        considerations = []
        security_level = self.config["security_level"]

        if security_level in ["high", "paranoid"] or context.get(
            "security_critical", False
        ):
            considerations.extend(
                [
                    "Input validation and sanitization",
                    "Authentication and authorization",
                    "Data encryption",
                    "SQL injection prevention",
                    "XSS protection",
                ]
            )

        return considerations

    def _recommend_testing_strategy(self, task_type: str, language: str) -> str:
        """Recommend testing strategy based on task type and language."""
        if "api" in task_type.lower():
            return "Integration tests with API endpoints"
        elif "database" in task_type.lower():
            return "Unit tests with database mocking"
        elif "web" in task_type.lower():
            return "Unit tests + integration tests"
        else:
            return "Unit tests with comprehensive coverage"

    def _assess_documentation_needs(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess documentation requirements."""
        return {
            "api_docs": "api" in context.get("task_type", "").lower(),
            "code_comments": True,
            "readme": True,
            "examples": context.get("include_examples", True),
        }

    def _generate_approach(self, analysis: Dict[str, Any]) -> str:
        """Generate the overall approach for code generation."""
        task_type = analysis.get("task_type", "unknown")
        language = analysis.get("language", "python")
        complexity = analysis.get("complexity_level", "moderate")

        approach = f"Implement {task_type} in {language} using {complexity} complexity approach. "
        approach += f"Focus on {', '.join(analysis.get('code_quality_focus', ['readability']))}. "
        approach += (
            f"Use patterns: {', '.join(analysis.get('recommended_patterns', [])[:3])}."
        )

        return approach

    def _design_code_structure(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Design the structure of the code."""
        task_type = analysis.get("task_type", "unknown")

        if "api" in task_type.lower():
            return {
                "main_file": "main.py",
                "routes": "routes/",
                "models": "models/",
                "services": "services/",
                "tests": "tests/",
                "config": "config.py",
            }
        elif "utility" in task_type.lower():
            return {"main_file": "main.py", "utils": "utils/", "tests": "tests/"}
        else:
            return {"main_file": "main.py", "modules": "modules/", "tests": "tests/"}

    def _create_implementation_plan(self, analysis: Dict[str, Any]) -> List[str]:
        """Create a step-by-step implementation plan."""
        return [
            "1. Set up project structure",
            "2. Implement core functionality",
            "3. Add error handling and validation",
            "4. Write unit tests",
            "5. Add documentation",
            "6. Performance optimization",
            "7. Security review",
            "8. Final testing and validation",
        ]

    def _generate_code_template(self, analysis: Dict[str, Any]) -> str:
        """Generate a basic code template."""
        language = analysis.get("language", "python")
        task_type = analysis.get("task_type", "unknown")

        if language == "python":
            if "api" in task_type.lower():
                return self._generate_python_api_template()
            else:
                return self._generate_python_basic_template()
        else:
            return f"# {language} template for {task_type}\n# Implementation needed"

    def _generate_python_api_template(self) -> str:
        """Generate a Python API template."""
        return '''from flask import Flask, request, jsonify
from typing import Dict, Any
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route('/api/endpoint', methods=['GET'])
def get_data():
    """Get data endpoint."""
    try:
        # Implementation here
        return jsonify({"status": "success", "data": {}})
    except Exception as e:
        logging.error(f"Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)'''

    def _generate_python_basic_template(self) -> str:
        """Generate a basic Python template."""
        return '''#!/usr/bin/env python3
"""
Module description.
"""

from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

def main_function():
    """Main function implementation."""
    try:
        # Implementation here
        pass
    except Exception as e:
        logger.error(f"Error: {e}")
        raise

if __name__ == "__main__":
    main_function()'''

    def _define_quality_guidelines(self, analysis: Dict[str, Any]) -> List[str]:
        """Define quality guidelines for the implementation."""
        guidelines = [
            "Follow PEP 8 style guidelines",
            "Use type hints",
            "Add docstrings to all functions",
            "Handle exceptions properly",
            "Write meaningful variable names",
        ]

        if analysis.get("performance_considerations"):
            guidelines.append("Optimize for performance where needed")

        if analysis.get("security_considerations"):
            guidelines.append("Implement security best practices")

        return guidelines

    def _create_documentation_plan(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create a documentation plan."""
        return {
            "readme": "Comprehensive README with setup and usage instructions",
            "api_docs": "API documentation if applicable",
            "code_comments": "Inline comments for complex logic",
            "examples": "Usage examples and code samples",
            "architecture": "Architecture documentation for complex systems",
        }

    def _estimate_code_size(self, analysis: Dict[str, Any]) -> Dict[str, int]:
        """Estimate the size of the code implementation."""
        complexity = analysis.get("complexity_level", "moderate")

        size_estimates = {
            "simple": {"lines": 50, "functions": 3, "classes": 1},
            "moderate": {"lines": 150, "functions": 8, "classes": 3},
            "complex": {"lines": 300, "functions": 15, "classes": 6},
        }

        return size_estimates.get(complexity, size_estimates["moderate"])

    def _calculate_confidence(self, analysis: Dict[str, Any]) -> float:
        """Calculate confidence in the proposal."""
        # Base confidence
        confidence = 0.7

        # Adjust based on task familiarity
        task_type = analysis.get("task_type", "").lower()
        if any(familiar in task_type for familiar in ["api", "web", "utility"]):
            confidence += 0.1

        # Adjust based on language familiarity
        language = analysis.get("language", "python")
        if language in self.config["preferred_languages"]:
            confidence += 0.1

        # Adjust based on complexity
        complexity = analysis.get("complexity_level", "moderate")
        if complexity == "moderate":
            confidence += 0.05
        elif complexity == "simple":
            confidence += 0.1

        return min(1.0, confidence)

    def _generate_reasoning(self, analysis: Dict[str, Any]) -> str:
        """Generate reasoning for the proposal."""
        task_type = analysis.get("task_type", "unknown")
        language = analysis.get("language", "python")
        patterns = analysis.get("recommended_patterns", [])

        reasoning = f"Based on the {task_type} task in {language}, "
        reasoning += f"I recommend using {', '.join(patterns[:3])} patterns. "
        reasoning += f"The approach focuses on {analysis.get('recommended_approach', 'best practices')} "
        reasoning += f"with emphasis on {', '.join(analysis.get('code_quality_focus', ['quality']))}."

        return reasoning

    def _generate_suggestions(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate additional suggestions for the implementation."""
        suggestions = []

        if analysis.get("performance_considerations"):
            suggestions.append(
                "Consider implementing caching for frequently accessed data"
            )

        if analysis.get("security_considerations"):
            suggestions.append("Implement input validation and sanitization")

        if analysis.get("testing_strategy"):
            suggestions.append("Aim for at least 80% test coverage")

        suggestions.append("Use environment variables for configuration")
        suggestions.append("Add logging for debugging and monitoring")

        return suggestions

    def _assess_code_quality(self, code: str) -> float:
        """Assess the overall quality of the code."""
        # Simple quality assessment - can be enhanced with static analysis
        score = 0.5  # Base score

        # Check for basic quality indicators
        if "def " in code or "class " in code:
            score += 0.1

        if '"""' in code or "'''" in code:
            score += 0.1  # Documentation

        if "try:" in code and "except" in code:
            score += 0.1  # Error handling

        if "import" in code:
            score += 0.1  # Proper imports

        if "if __name__" in code:
            score += 0.1  # Proper main guard

        return min(1.0, score)

    def _identify_issues(self, code: str) -> List[str]:
        """Identify issues in the code."""
        issues = []

        # Basic issue detection
        if "TODO" in code or "FIXME" in code:
            issues.append("Contains TODO/FIXME comments")

        if code.count("print(") > 3:
            issues.append("Excessive print statements - consider logging")

        if "pass" in code and code.count("pass") > 2:
            issues.append("Multiple pass statements - incomplete implementation")

        return issues

    def _generate_improvement_suggestions(self, code: str) -> List[str]:
        """Generate suggestions for code improvement."""
        suggestions = []

        if "print(" in code:
            suggestions.append("Replace print statements with proper logging")

        if "except:" in code:
            suggestions.append("Use specific exception types instead of bare except")

        if len(code.split("\n")) > 100:
            suggestions.append("Consider breaking large functions into smaller ones")

        return suggestions

    def _check_best_practices(self, code: str) -> Dict[str, bool]:
        """Check if code follows best practices."""
        return {
            "has_docstrings": '"""' in code or "'''" in code,
            "has_error_handling": "try:" in code and "except" in code,
            "has_type_hints": "->" in code or ": " in code,
            "has_main_guard": "if __name__" in code,
            "has_logging": "logging" in code or "logger" in code,
        }

    def _analyze_performance(self, code: str) -> List[str]:
        """Analyze performance aspects of the code."""
        analysis = []

        if "for " in code and " in " in code:
            analysis.append("Consider list comprehensions for better performance")

        if "import *" in code:
            analysis.append("Avoid wildcard imports for better performance")

        return analysis

    def _analyze_security(self, code: str) -> List[str]:
        """Analyze security aspects of the code."""
        analysis = []

        if "eval(" in code:
            analysis.append("WARNING: eval() usage detected - security risk")

        if "input(" in code and "int(" not in code:
            analysis.append("Consider input validation for security")

        return analysis

    def _assess_readability(self, code: str) -> float:
        """Assess code readability."""
        # Simple readability assessment
        lines = code.split("\n")
        avg_line_length = sum(len(line) for line in lines) / len(lines) if lines else 0

        if avg_line_length < 80:
            return 0.9
        elif avg_line_length < 120:
            return 0.7
        else:
            return 0.5

    def _assess_maintainability(self, code: str) -> float:
        """Assess code maintainability."""
        # Simple maintainability assessment
        score = 0.5

        if "def " in code:
            score += 0.2  # Functions present

        if "class " in code:
            score += 0.1  # Classes present

        if '"""' in code:
            score += 0.2  # Documentation present

        return min(1.0, score)

    def _recommend_test_coverage(self, code: str) -> List[str]:
        """Recommend test coverage areas."""
        recommendations = []

        if "def " in code:
            recommendations.append("Unit tests for all functions")

        if "class " in code:
            recommendations.append("Unit tests for all classes")

        if "if " in code:
            recommendations.append("Test all conditional branches")

        return recommendations

    def _identify_documentation_gaps(self, code: str) -> List[str]:
        """Identify gaps in documentation."""
        gaps = []

        if "def " in code and '"""' not in code:
            gaps.append("Missing function docstrings")

        if "class " in code and '"""' not in code:
            gaps.append("Missing class docstrings")

        if "import " in code and "#" not in code:
            gaps.append("Consider adding import comments for clarity")

        return gaps

    def _provide_overall_assessment(self, code: str) -> str:
        """Provide overall assessment of the code."""
        quality_score = self._assess_code_quality(code)

        if quality_score >= 0.8:
            return "Excellent code quality with good practices"
        elif quality_score >= 0.6:
            return "Good code quality with room for improvement"
        elif quality_score >= 0.4:
            return "Acceptable code quality, needs improvements"
        else:
            return "Code quality needs significant improvement"

    def _assess_proposal_quality(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the quality of a proposal."""
        assessment = {
            "completeness": self._assess_proposal_completeness(proposal),
            "feasibility": self._assess_proposal_feasibility(proposal),
            "innovation": self._assess_proposal_innovation(proposal),
            "overall_score": 0.0,
        }

        # Calculate overall score
        scores = [
            assessment["completeness"],
            assessment["feasibility"],
            assessment["innovation"],
        ]
        assessment["overall_score"] = sum(scores) / len(scores)

        return assessment

    def _assess_proposal_completeness(self, proposal: Dict[str, Any]) -> float:
        """Assess how complete the proposal is."""
        required_fields = ["approach", "implementation_plan", "code_template"]
        present_fields = sum(1 for field in required_fields if field in proposal)
        return present_fields / len(required_fields)

    def _assess_proposal_feasibility(self, proposal: Dict[str, Any]) -> float:
        """Assess how feasible the proposal is."""
        # Simple feasibility check based on confidence
        confidence = proposal.get("confidence", 0.5)
        return min(confidence, 1.0)

    def _assess_proposal_innovation(self, proposal: Dict[str, Any]) -> float:
        """Assess how innovative the proposal is."""
        # Simple innovation score based on patterns and approach
        patterns = proposal.get("patterns_to_use", [])
        approach = proposal.get("approach", "")

        innovation_score = 0.5  # Base score

        if len(patterns) > 3:
            innovation_score += 0.2
        if "modern" in approach.lower() or "advanced" in approach.lower():
            innovation_score += 0.3

        return min(innovation_score, 1.0)
