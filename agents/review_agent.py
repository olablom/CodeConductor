"""
Review Agent for CodeConductor

This agent specializes in code review, quality assessment, and improvement suggestions.
It analyzes code for issues, proposes improvements, and reviews changes.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from agents.base_agent import BaseAgent


@dataclass
class CodeIssue:
    """Represents a code quality issue found during review."""

    severity: str  # 'critical', 'high', 'medium', 'low'
    category: str  # 'security', 'performance', 'maintainability', 'style', 'bug'
    line_number: Optional[int]
    description: str
    suggestion: str
    confidence: float  # 0.0 to 1.0


@dataclass
class ReviewReport:
    """Comprehensive review report with findings and recommendations."""

    overall_score: float  # 0.0 to 1.0
    issues: List[CodeIssue]
    strengths: List[str]
    recommendations: List[str]
    estimated_effort: str  # 'low', 'medium', 'high'
    priority_fixes: List[str]


class ReviewAgent(BaseAgent):
    """
    Specialized agent for code review and quality assessment.

    This agent focuses on:
    - Code quality analysis
    - Security vulnerability detection
    - Performance optimization opportunities
    - Maintainability assessment
    - Best practices compliance
    """

    def __init__(
        self, name: str = "ReviewAgent", config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name, config)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Review-specific configuration
        self.review_config = config.get("review", {}) if config else {}
        self.severity_weights = {
            "critical": 1.0,
            "high": 0.8,
            "medium": 0.5,
            "low": 0.2,
        }

        # Quality thresholds
        self.quality_thresholds = {
            "excellent": 0.9,
            "good": 0.7,
            "acceptable": 0.5,
            "needs_improvement": 0.3,
        }

        self.logger.info(
            f"ReviewAgent '{name}' initialized with config: {self.review_config}"
        )

    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze code for quality issues, security vulnerabilities, and improvement opportunities.

        Args:
            context: Contains 'code', 'requirements', 'constraints', etc.

        Returns:
            Analysis results with detailed findings
        """
        self.logger.info(f"Starting code analysis for {self.name}")

        try:
            code = context.get("code", "")
            requirements = context.get("requirements", {})
            constraints = context.get("constraints", {})

            # Perform comprehensive code analysis
            analysis_result = {
                "code_quality_score": self._assess_code_quality(code),
                "security_analysis": self._analyze_security(code),
                "performance_analysis": self._analyze_performance(code),
                "maintainability_analysis": self._analyze_maintainability(code),
                "compliance_check": self._check_compliance(code, requirements),
                "complexity_metrics": self._calculate_complexity_metrics(code),
                "test_coverage_analysis": self._analyze_test_coverage(code),
                "documentation_quality": self._assess_documentation(code),
                "issues_found": [],
                "strengths": [],
                "recommendations": [],
            }

            # Identify specific issues
            analysis_result["issues_found"] = self._identify_issues(code, requirements)
            analysis_result["strengths"] = self._identify_strengths(code)
            analysis_result["recommendations"] = self._generate_recommendations(
                analysis_result, requirements, constraints
            )

            self.logger.info(
                f"Analysis completed. Found {len(analysis_result['issues_found'])} issues"
            )
            return analysis_result

        except Exception as e:
            self.logger.error(f"Error during analysis: {e}")
            return {
                "error": str(e),
                "code_quality_score": 0.0,
                "issues_found": [],
                "strengths": [],
                "recommendations": [],
            }

    def propose(
        self, analysis: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Propose specific improvements and refactoring suggestions based on analysis.

        Args:
            analysis: Results from the analyze method
            context: Original context with requirements and constraints

        Returns:
            Proposal with specific improvements and implementation plan
        """
        self.logger.info(f"Generating improvement proposals for {self.name}")

        try:
            issues = analysis.get("issues_found", [])
            recommendations = analysis.get("recommendations", [])
            requirements = context.get("requirements", {})

            proposal = {
                "improvement_plan": self._create_improvement_plan(
                    issues, recommendations
                ),
                "refactoring_suggestions": self._generate_refactoring_suggestions(
                    analysis
                ),
                "security_enhancements": self._propose_security_enhancements(analysis),
                "performance_optimizations": self._propose_performance_optimizations(
                    analysis
                ),
                "code_style_improvements": self._propose_style_improvements(analysis),
                "testing_improvements": self._propose_testing_improvements(analysis),
                "documentation_improvements": self._propose_documentation_improvements(
                    analysis
                ),
                "priority_order": self._prioritize_improvements(issues, requirements),
                "estimated_effort": self._estimate_effort(issues),
                "risk_assessment": self._assess_improvement_risks(analysis),
            }

            self.logger.info(
                f"Generated proposal with {len(proposal['improvement_plan'])} improvements"
            )
            return proposal

        except Exception as e:
            self.logger.error(f"Error generating proposal: {e}")
            return {
                "error": str(e),
                "improvement_plan": [],
                "refactoring_suggestions": [],
                "priority_order": [],
            }

    def review(
        self, proposal: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Review the proposed improvements and provide feedback on feasibility and impact.

        Args:
            proposal: Results from the propose method
            context: Original context and analysis

        Returns:
            Review feedback with feasibility assessment and recommendations
        """
        self.logger.info(f"Reviewing proposal for {self.name}")

        try:
            review_result = {
                "feasibility_assessment": self._assess_feasibility(proposal, context),
                "impact_analysis": self._analyze_impact(proposal, context),
                "alternative_suggestions": self._suggest_alternatives(
                    proposal, context
                ),
                "implementation_guidance": self._provide_implementation_guidance(
                    proposal
                ),
                "validation_criteria": self._define_validation_criteria(proposal),
                "rollback_plan": self._create_rollback_plan(proposal),
                "approval_recommendation": self._make_approval_recommendation(
                    proposal, context
                ),
                "final_score": self._calculate_final_score(proposal, context),
            }

            self.logger.info(
                f"Review completed. Final score: {review_result['final_score']}"
            )
            return review_result

        except Exception as e:
            self.logger.error(f"Error during review: {e}")
            return {
                "error": str(e),
                "feasibility_assessment": "unknown",
                "approval_recommendation": "reject",
                "final_score": 0.0,
            }

    def _assess_code_quality(self, code: str) -> float:
        """Assess overall code quality score."""
        try:
            # Analyze various quality aspects
            complexity_score = self._analyze_complexity(code)
            style_score = self._analyze_code_style(code)
            structure_score = self._analyze_code_structure(code)

            # Weighted average
            quality_score = (
                complexity_score * 0.4 + style_score * 0.3 + structure_score * 0.3
            )

            return min(1.0, max(0.0, quality_score))

        except Exception as e:
            self.logger.error(f"Error assessing code quality: {e}")
            return 0.0

    def _analyze_security(self, code: str) -> Dict[str, Any]:
        """Analyze code for security vulnerabilities."""
        try:
            security_issues = []

            # Check for common security patterns
            if "eval(" in code:
                security_issues.append(
                    {
                        "type": "dangerous_function",
                        "description": "Use of eval() function detected",
                        "severity": "critical",
                        "recommendation": "Replace with safer alternatives",
                    }
                )

            if "exec(" in code:
                security_issues.append(
                    {
                        "type": "dangerous_function",
                        "description": "Use of exec() function detected",
                        "severity": "critical",
                        "recommendation": "Replace with safer alternatives",
                    }
                )

            # Check for SQL injection patterns
            if 'f"SELECT' in code or 'f"INSERT' in code or 'f"UPDATE' in code:
                security_issues.append(
                    {
                        "type": "sql_injection_risk",
                        "description": "Potential SQL injection vulnerability",
                        "severity": "high",
                        "recommendation": "Use parameterized queries",
                    }
                )

            return {
                "issues": security_issues,
                "overall_security_score": max(0.0, 1.0 - len(security_issues) * 0.2),
                "risk_level": "high"
                if any(i["severity"] == "critical" for i in security_issues)
                else "medium",
            }

        except Exception as e:
            self.logger.error(f"Error analyzing security: {e}")
            return {
                "issues": [],
                "overall_security_score": 0.0,
                "risk_level": "unknown",
            }

    def _analyze_performance(self, code: str) -> Dict[str, Any]:
        """Analyze code for performance issues."""
        try:
            performance_issues = []

            # Check for inefficient patterns
            if "for i in range(len(" in code:
                performance_issues.append(
                    {
                        "type": "inefficient_iteration",
                        "description": "Inefficient list iteration pattern",
                        "severity": "medium",
                        "recommendation": "Use enumerate() or direct iteration",
                    }
                )

            if ".keys()" in code and "for" in code:
                performance_issues.append(
                    {
                        "type": "unnecessary_keys",
                        "description": "Unnecessary .keys() call in iteration",
                        "severity": "low",
                        "recommendation": "Iterate directly over dictionary",
                    }
                )

            return {
                "issues": performance_issues,
                "performance_score": max(0.0, 1.0 - len(performance_issues) * 0.1),
                "optimization_opportunities": len(performance_issues),
            }

        except Exception as e:
            self.logger.error(f"Error analyzing performance: {e}")
            return {
                "issues": [],
                "performance_score": 0.0,
                "optimization_opportunities": 0,
            }

    def _analyze_maintainability(self, code: str) -> Dict[str, Any]:
        """Analyze code maintainability."""
        try:
            maintainability_issues = []

            # Check function length
            lines = code.split("\n")
            if len(lines) > 50:
                maintainability_issues.append(
                    {
                        "type": "long_function",
                        "description": "Function is too long",
                        "severity": "medium",
                        "recommendation": "Break into smaller functions",
                    }
                )

            # Check for magic numbers
            import re

            magic_numbers = re.findall(r"\b\d{3,}\b", code)
            if magic_numbers:
                maintainability_issues.append(
                    {
                        "type": "magic_numbers",
                        "description": f"Found {len(magic_numbers)} magic numbers",
                        "severity": "low",
                        "recommendation": "Define constants for magic numbers",
                    }
                )

            return {
                "issues": maintainability_issues,
                "maintainability_score": max(
                    0.0, 1.0 - len(maintainability_issues) * 0.15
                ),
                "complexity_level": "high"
                if len(maintainability_issues) > 3
                else "medium",
            }

        except Exception as e:
            self.logger.error(f"Error analyzing maintainability: {e}")
            return {
                "issues": [],
                "maintainability_score": 0.0,
                "complexity_level": "unknown",
            }

    def _check_compliance(
        self, code: str, requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check compliance with coding standards and requirements."""
        try:
            compliance_issues = []

            # Check for required patterns
            if requirements.get("require_type_hints", False):
                if "def " in code and "->" not in code:
                    compliance_issues.append(
                        {
                            "type": "missing_type_hints",
                            "description": "Type hints required but not found",
                            "severity": "medium",
                            "recommendation": "Add type hints to function signatures",
                        }
                    )

            # Check for required documentation
            if requirements.get("require_docstrings", False):
                if '"""' not in code and "'''" not in code:
                    compliance_issues.append(
                        {
                            "type": "missing_docstrings",
                            "description": "Docstrings required but not found",
                            "severity": "medium",
                            "recommendation": "Add docstrings to functions and classes",
                        }
                    )

            return {
                "issues": compliance_issues,
                "compliance_score": max(0.0, 1.0 - len(compliance_issues) * 0.2),
                "standards_met": len(compliance_issues) == 0,
            }

        except Exception as e:
            self.logger.error(f"Error checking compliance: {e}")
            return {"issues": [], "compliance_score": 0.0, "standards_met": False}

    def _calculate_complexity_metrics(self, code: str) -> Dict[str, Any]:
        """Calculate code complexity metrics."""
        try:
            lines = code.split("\n")

            # Basic metrics
            total_lines = len(lines)
            non_empty_lines = len([line for line in lines if line.strip()])
            comment_lines = len(
                [line for line in lines if line.strip().startswith("#")]
            )

            # Cyclomatic complexity approximation
            complexity_keywords = [
                "if",
                "elif",
                "else",
                "for",
                "while",
                "except",
                "and",
                "or",
            ]
            complexity = sum(
                1
                for line in lines
                for keyword in complexity_keywords
                if keyword in line
            )

            return {
                "total_lines": total_lines,
                "non_empty_lines": non_empty_lines,
                "comment_lines": comment_lines,
                "cyclomatic_complexity": complexity,
                "comment_ratio": comment_lines / max(1, non_empty_lines),
                "complexity_per_line": complexity / max(1, non_empty_lines),
            }

        except Exception as e:
            self.logger.error(f"Error calculating complexity metrics: {e}")
            return {
                "total_lines": 0,
                "non_empty_lines": 0,
                "comment_lines": 0,
                "cyclomatic_complexity": 0,
                "comment_ratio": 0.0,
                "complexity_per_line": 0.0,
            }

    def _analyze_test_coverage(self, code: str) -> Dict[str, Any]:
        """Analyze test coverage and testing patterns."""
        try:
            test_indicators = ["test_", "Test", "assert", "unittest", "pytest"]
            test_patterns = sum(1 for indicator in test_indicators if indicator in code)

            return {
                "test_patterns_found": test_patterns,
                "has_testing_framework": any(
                    framework in code for framework in ["unittest", "pytest"]
                ),
                "has_assertions": "assert" in code,
                "test_coverage_estimate": min(1.0, test_patterns * 0.2),
            }

        except Exception as e:
            self.logger.error(f"Error analyzing test coverage: {e}")
            return {
                "test_patterns_found": 0,
                "has_testing_framework": False,
                "has_assertions": False,
                "test_coverage_estimate": 0.0,
            }

    def _assess_documentation(self, code: str) -> Dict[str, Any]:
        """Assess documentation quality."""
        try:
            doc_indicators = ['"""', "'''", "#", "docstring", "README"]
            doc_patterns = sum(1 for indicator in doc_indicators if indicator in code)

            return {
                "documentation_patterns": doc_patterns,
                "has_docstrings": '"""' in code or "'''" in code,
                "has_comments": "#" in code,
                "documentation_score": min(1.0, doc_patterns * 0.15),
            }

        except Exception as e:
            self.logger.error(f"Error assessing documentation: {e}")
            return {
                "documentation_patterns": 0,
                "has_docstrings": False,
                "has_comments": False,
                "documentation_score": 0.0,
            }

    def _identify_issues(
        self, code: str, requirements: Dict[str, Any]
    ) -> List[CodeIssue]:
        """Identify specific code issues."""
        try:
            issues = []

            # Security issues
            security_analysis = self._analyze_security(code)
            for issue in security_analysis["issues"]:
                issues.append(
                    CodeIssue(
                        severity=issue["severity"],
                        category="security",
                        line_number=None,
                        description=issue["description"],
                        suggestion=issue["recommendation"],
                        confidence=0.9,
                    )
                )

            # Performance issues
            performance_analysis = self._analyze_performance(code)
            for issue in performance_analysis["issues"]:
                issues.append(
                    CodeIssue(
                        severity=issue["severity"],
                        category="performance",
                        line_number=None,
                        description=issue["description"],
                        suggestion=issue["recommendation"],
                        confidence=0.7,
                    )
                )

            # Maintainability issues
            maintainability_analysis = self._analyze_maintainability(code)
            for issue in maintainability_analysis["issues"]:
                issues.append(
                    CodeIssue(
                        severity=issue["severity"],
                        category="maintainability",
                        line_number=None,
                        description=issue["description"],
                        suggestion=issue["recommendation"],
                        confidence=0.8,
                    )
                )

            return issues

        except Exception as e:
            self.logger.error(f"Error identifying issues: {e}")
            return []

    def _identify_strengths(self, code: str) -> List[str]:
        """Identify code strengths and good practices."""
        try:
            strengths = []

            # Check for good practices
            if "def " in code and "->" in code:
                strengths.append("Uses type hints")

            if '"""' in code or "'''" in code:
                strengths.append("Includes docstrings")

            if "assert" in code:
                strengths.append("Includes assertions")

            if "try:" in code and "except:" in code:
                strengths.append("Includes error handling")

            if "class " in code:
                strengths.append("Uses object-oriented design")

            return strengths

        except Exception as e:
            self.logger.error(f"Error identifying strengths: {e}")
            return []

    def _generate_recommendations(
        self,
        analysis: Dict[str, Any],
        requirements: Dict[str, Any],
        constraints: Dict[str, Any],
    ) -> List[str]:
        """Generate improvement recommendations."""
        try:
            recommendations = []

            # Based on quality score
            quality_score = analysis.get("code_quality_score", 0.0)
            if quality_score < 0.7:
                recommendations.append(
                    "Improve overall code quality through refactoring"
                )

            # Based on security analysis
            security_score = analysis.get("security_analysis", {}).get(
                "overall_security_score", 0.0
            )
            if security_score < 0.8:
                recommendations.append("Address security vulnerabilities")

            # Based on performance analysis
            performance_score = analysis.get("performance_analysis", {}).get(
                "performance_score", 0.0
            )
            if performance_score < 0.8:
                recommendations.append("Optimize performance bottlenecks")

            # Based on maintainability
            maintainability_score = analysis.get("maintainability_analysis", {}).get(
                "maintainability_score", 0.0
            )
            if maintainability_score < 0.7:
                recommendations.append("Improve code maintainability")

            return recommendations

        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return []

    def _create_improvement_plan(
        self, issues: List[CodeIssue], recommendations: List[str]
    ) -> List[Dict[str, Any]]:
        """Create a detailed improvement plan."""
        try:
            plan = []

            # Group issues by category
            categories = {}
            for issue in issues:
                if issue.category not in categories:
                    categories[issue.category] = []
                categories[issue.category].append(issue)

            # Create improvement items
            for category, category_issues in categories.items():
                plan.append(
                    {
                        "category": category,
                        "priority": "high"
                        if any(
                            i.severity in ["critical", "high"] for i in category_issues
                        )
                        else "medium",
                        "issues": category_issues,
                        "action": f"Address {category} issues",
                        "estimated_effort": "high"
                        if len(category_issues) > 5
                        else "medium",
                    }
                )

            return plan

        except Exception as e:
            self.logger.error(f"Error creating improvement plan: {e}")
            return []

    def _generate_refactoring_suggestions(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate specific refactoring suggestions."""
        try:
            suggestions = []

            # Based on complexity metrics
            complexity_metrics = analysis.get("complexity_metrics", {})
            if complexity_metrics.get("complexity_per_line", 0) > 0.5:
                suggestions.append(
                    "Break down complex functions into smaller, focused functions"
                )

            if complexity_metrics.get("total_lines", 0) > 100:
                suggestions.append(
                    "Consider splitting large files into smaller modules"
                )

            # Based on maintainability
            maintainability_analysis = analysis.get("maintainability_analysis", {})
            if maintainability_analysis.get("maintainability_score", 0) < 0.7:
                suggestions.append("Extract magic numbers into named constants")

            return suggestions

        except Exception as e:
            self.logger.error(f"Error generating refactoring suggestions: {e}")
            return []

    def _propose_security_enhancements(self, analysis: Dict[str, Any]) -> List[str]:
        """Propose security enhancements."""
        try:
            enhancements = []

            security_analysis = analysis.get("security_analysis", {})
            if security_analysis.get("overall_security_score", 0) < 0.9:
                enhancements.append("Implement input validation for all user inputs")
                enhancements.append("Use parameterized queries for database operations")
                enhancements.append("Add authentication and authorization checks")

            return enhancements

        except Exception as e:
            self.logger.error(f"Error proposing security enhancements: {e}")
            return []

    def _propose_performance_optimizations(self, analysis: Dict[str, Any]) -> List[str]:
        """Propose performance optimizations."""
        try:
            optimizations = []

            performance_analysis = analysis.get("performance_analysis", {})
            if performance_analysis.get("performance_score", 0) < 0.8:
                optimizations.append("Optimize database queries with proper indexing")
                optimizations.append("Implement caching for frequently accessed data")
                optimizations.append("Use more efficient data structures")

            return optimizations

        except Exception as e:
            self.logger.error(f"Error proposing performance optimizations: {e}")
            return []

    def _propose_style_improvements(self, analysis: Dict[str, Any]) -> List[str]:
        """Propose code style improvements."""
        try:
            improvements = []

            # Based on compliance check
            compliance_check = analysis.get("compliance_check", {})
            if not compliance_check.get("standards_met", True):
                improvements.append("Follow PEP 8 style guidelines")
                improvements.append("Add consistent indentation and spacing")
                improvements.append("Use meaningful variable and function names")

            return improvements

        except Exception as e:
            self.logger.error(f"Error proposing style improvements: {e}")
            return []

    def _propose_testing_improvements(self, analysis: Dict[str, Any]) -> List[str]:
        """Propose testing improvements."""
        try:
            improvements = []

            test_coverage = analysis.get("test_coverage_analysis", {})
            if test_coverage.get("test_coverage_estimate", 0) < 0.7:
                improvements.append("Increase test coverage to at least 80%")
                improvements.append("Add unit tests for all public methods")
                improvements.append("Implement integration tests for critical paths")

            return improvements

        except Exception as e:
            self.logger.error(f"Error proposing testing improvements: {e}")
            return []

    def _propose_documentation_improvements(
        self, analysis: Dict[str, Any]
    ) -> List[str]:
        """Propose documentation improvements."""
        try:
            improvements = []

            doc_quality = analysis.get("documentation_quality", {})
            if doc_quality.get("documentation_score", 0) < 0.6:
                improvements.append(
                    "Add comprehensive docstrings to all functions and classes"
                )
                improvements.append("Create README with setup and usage instructions")
                improvements.append("Add inline comments for complex logic")

            return improvements

        except Exception as e:
            self.logger.error(f"Error proposing documentation improvements: {e}")
            return []

    def _prioritize_improvements(
        self, issues: List[CodeIssue], requirements: Dict[str, Any]
    ) -> List[str]:
        """Prioritize improvements based on severity and requirements."""
        try:
            # Sort issues by severity weight
            sorted_issues = sorted(
                issues,
                key=lambda x: self.severity_weights.get(x.severity, 0),
                reverse=True,
            )

            priorities = []
            for issue in sorted_issues[:10]:  # Top 10 issues
                priorities.append(f"{issue.severity.upper()}: {issue.description}")

            return priorities

        except Exception as e:
            self.logger.error(f"Error prioritizing improvements: {e}")
            return []

    def _estimate_effort(self, issues: List[CodeIssue]) -> str:
        """Estimate effort required for improvements."""
        try:
            total_severity = sum(
                self.severity_weights.get(issue.severity, 0) for issue in issues
            )

            if total_severity > 5.0:
                return "high"
            elif total_severity > 2.0:
                return "medium"
            else:
                return "low"

        except Exception as e:
            self.logger.error(f"Error estimating effort: {e}")
            return "unknown"

    def _assess_improvement_risks(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risks associated with proposed improvements."""
        try:
            risks = {
                "breaking_changes": "low",
                "performance_impact": "low",
                "security_risks": "low",
                "maintenance_overhead": "low",
            }

            # Assess based on analysis results
            if analysis.get("code_quality_score", 0) < 0.5:
                risks["breaking_changes"] = "medium"

            if (
                analysis.get("performance_analysis", {}).get("performance_score", 0)
                < 0.6
            ):
                risks["performance_impact"] = "medium"

            return risks

        except Exception as e:
            self.logger.error(f"Error assessing improvement risks: {e}")
            return {"breaking_changes": "unknown", "performance_impact": "unknown"}

    def _assess_feasibility(
        self, proposal: Dict[str, Any], context: Dict[str, Any]
    ) -> str:
        """Assess feasibility of proposed improvements."""
        try:
            improvement_plan = proposal.get("improvement_plan", [])
            risk_assessment = proposal.get("risk_assessment", {})

            # Check if improvements are feasible
            if len(improvement_plan) == 0:
                return "high"  # No improvements needed

            if risk_assessment.get("breaking_changes") == "high":
                return "low"

            return "medium"

        except Exception as e:
            self.logger.error(f"Error assessing feasibility: {e}")
            return "unknown"

    def _analyze_impact(
        self, proposal: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze impact of proposed improvements."""
        try:
            return {
                "code_quality_improvement": "significant",
                "maintenance_effort": "moderate",
                "learning_curve": "low",
                "time_to_implement": "medium",
            }

        except Exception as e:
            self.logger.error(f"Error analyzing impact: {e}")
            return {}

    def _suggest_alternatives(
        self, proposal: Dict[str, Any], context: Dict[str, Any]
    ) -> List[str]:
        """Suggest alternative approaches."""
        try:
            alternatives = []

            # Suggest incremental improvements
            alternatives.append("Implement improvements incrementally to reduce risk")
            alternatives.append("Focus on high-priority issues first")
            alternatives.append("Consider automated refactoring tools")

            return alternatives

        except Exception as e:
            self.logger.error(f"Error suggesting alternatives: {e}")
            return []

    def _provide_implementation_guidance(
        self, proposal: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Provide implementation guidance for improvements."""
        try:
            return {
                "step_by_step_plan": [
                    "1. Address critical security issues first",
                    "2. Implement performance optimizations",
                    "3. Improve code maintainability",
                    "4. Add comprehensive testing",
                    "5. Enhance documentation",
                ],
                "tools_recommended": [
                    "Static analysis tools (e.g., pylint, flake8)",
                    "Security scanning tools",
                    "Performance profiling tools",
                    "Code coverage tools",
                ],
                "best_practices": [
                    "Write tests before implementing changes",
                    "Review changes with team members",
                    "Monitor performance after changes",
                    "Document all changes",
                ],
            }

        except Exception as e:
            self.logger.error(f"Error providing implementation guidance: {e}")
            return {}

    def _define_validation_criteria(self, proposal: Dict[str, Any]) -> List[str]:
        """Define validation criteria for improvements."""
        try:
            return [
                "All existing tests pass",
                "Code coverage remains above 80%",
                "No new security vulnerabilities introduced",
                "Performance benchmarks show improvement or no regression",
                "Code follows style guidelines",
                "Documentation is updated",
            ]

        except Exception as e:
            self.logger.error(f"Error defining validation criteria: {e}")
            return []

    def _create_rollback_plan(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Create a rollback plan for proposed changes."""
        try:
            return {
                "backup_strategy": "Create git branches for each major change",
                "rollback_triggers": [
                    "Critical bugs introduced",
                    "Performance regression > 10%",
                    "Security vulnerabilities found",
                    "Test failures in production",
                ],
                "rollback_steps": [
                    "1. Identify the problematic change",
                    "2. Revert to previous stable version",
                    "3. Run full test suite",
                    "4. Deploy rollback",
                    "5. Investigate root cause",
                ],
            }

        except Exception as e:
            self.logger.error(f"Error creating rollback plan: {e}")
            return {}

    def _make_approval_recommendation(
        self, proposal: Dict[str, Any], context: Dict[str, Any]
    ) -> str:
        """Make approval recommendation based on analysis."""
        try:
            feasibility = proposal.get("feasibility_assessment", "unknown")
            risk_assessment = proposal.get("risk_assessment", {})

            if (
                feasibility == "high"
                and risk_assessment.get("breaking_changes") == "low"
            ):
                return "approve"
            elif (
                feasibility == "medium"
                and risk_assessment.get("breaking_changes") == "low"
            ):
                return "approve_with_caution"
            else:
                return "reject"

        except Exception as e:
            self.logger.error(f"Error making approval recommendation: {e}")
            return "reject"

    def _calculate_final_score(
        self, proposal: Dict[str, Any], context: Dict[str, Any]
    ) -> float:
        """Calculate final review score."""
        try:
            # Base score from original analysis
            original_analysis = context.get("analysis", {})
            base_score = original_analysis.get("code_quality_score", 0.0)

            # Adjust based on proposal quality
            improvement_plan = proposal.get("improvement_plan", [])
            if len(improvement_plan) > 0:
                base_score += 0.1  # Bonus for having improvement plan

            # Adjust based on feasibility
            feasibility = proposal.get("feasibility_assessment", "unknown")
            if feasibility == "high":
                base_score += 0.1
            elif feasibility == "low":
                base_score -= 0.1

            return min(1.0, max(0.0, base_score))

        except Exception as e:
            self.logger.error(f"Error calculating final score: {e}")
            return 0.0

    def _analyze_complexity(self, code: str) -> float:
        """Analyze code complexity."""
        try:
            lines = code.split("\n")
            complexity_keywords = [
                "if",
                "elif",
                "else",
                "for",
                "while",
                "except",
                "and",
                "or",
            ]
            complexity = sum(
                1
                for line in lines
                for keyword in complexity_keywords
                if keyword in line
            )

            # Normalize complexity score (lower is better)
            complexity_score = max(0.0, 1.0 - (complexity / max(1, len(lines))) * 2)
            return complexity_score

        except Exception as e:
            self.logger.error(f"Error analyzing complexity: {e}")
            return 0.0

    def _analyze_code_style(self, code: str) -> float:
        """Analyze code style."""
        try:
            lines = code.split("\n")
            style_issues = 0

            for line in lines:
                if line.strip() and len(line) > 120:  # Line too long
                    style_issues += 1
                if (
                    line.strip() and not line.startswith(" ") and "def " in line
                ):  # Missing indentation
                    style_issues += 1

            # Normalize style score (lower issues is better)
            style_score = max(0.0, 1.0 - (style_issues / max(1, len(lines))))
            return style_score

        except Exception as e:
            self.logger.error(f"Error analyzing code style: {e}")
            return 0.0

    def _analyze_code_structure(self, code: str) -> float:
        """Analyze code structure."""
        try:
            structure_score = 0.5  # Base score

            # Bonus for good structure indicators
            if "class " in code:
                structure_score += 0.2
            if "def " in code:
                structure_score += 0.2
            if "import " in code:
                structure_score += 0.1

            return min(1.0, structure_score)

        except Exception as e:
            self.logger.error(f"Error analyzing code structure: {e}")
            return 0.0
