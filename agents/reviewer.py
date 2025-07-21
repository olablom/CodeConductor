"""
ReviewerAgent - Analyzes code quality and potential issues.

Part of the multi-agent discussion system for CodeConductor.
"""

from typing import Dict, Any, List
from pathlib import Path

from integrations.lm_studio import generate_code


class ReviewerAgent:
    """Analyserar kodkvalitet och potentiella problem."""

    def __init__(self):
        self.name = "ReviewerAgent"
        self.role = "Code Quality & Security Expert"

    def analyze(self, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyserar kodkvalitet och säkerhet för en given prompt.

        Args:
            prompt: Kodkravet att analysera
            context: Ytterligare kontext (valfritt)

        Returns:
            Dictionary med kvalitetsanalys
        """
        if context is None:
            context = {}

        # Skapa en kvalitets-prompt
        quality_prompt = f"""
As a code quality and security expert, analyze this implementation request:

{prompt}

Consider:
- What are the potential security risks?
- What quality issues might arise?
- What testing considerations apply?
- What are the maintainability concerns?

Provide a quality and security analysis.
"""

        # Skapa temporär prompt-fil för analys
        temp_prompt_path = Path("data/temp_quality.md")
        temp_prompt_path.parent.mkdir(parents=True, exist_ok=True)
        temp_prompt_path.write_text(quality_prompt)

        try:
            # Använd LM Studio för analys
            analysis = generate_code(temp_prompt_path, "conservative")

            if analysis:
                return {
                    "agent": self.name,
                    "role": self.role,
                    "security_risks": self._extract_risks(analysis),
                    "quality_issues": self._extract_quality_issues(analysis),
                    "testing_needs": self._extract_testing_needs(analysis),
                    "maintainability": "medium",
                    "recommendation": "defensive_programming",
                }
            else:
                # Fallback till fördefinierad analys
                return self._fallback_analysis(prompt)

        except Exception as e:
            print(f"[{self.name}] Analysis failed: {e}")
            return self._fallback_analysis(prompt)
        finally:
            # Cleanup
            if temp_prompt_path.exists():
                temp_prompt_path.unlink()

    def _extract_risks(self, analysis: str) -> List[str]:
        """Extraherar säkerhetsrisker från analysen."""
        risks = []

        # Enkel risk-extraktion
        risk_keywords = [
            "injection",
            "overflow",
            "race_condition",
            "memory_leak",
            "input_validation",
            "authentication",
            "authorization",
        ]

        analysis_lower = analysis.lower()
        for risk in risk_keywords:
            if risk in analysis_lower:
                risks.append(risk)

        # Om inga risks hittades, returnera defaults
        if not risks:
            risks = ["input_validation"]

        return risks

    def _extract_quality_issues(self, analysis: str) -> List[str]:
        """Extraherar kvalitetsproblem från analysen."""
        issues = []

        # Enkel issue-extraktion
        issue_keywords = [
            "complexity",
            "readability",
            "performance",
            "memory",
            "error_handling",
            "documentation",
            "naming",
        ]

        analysis_lower = analysis.lower()
        for issue in issue_keywords:
            if issue in analysis_lower:
                issues.append(issue)

        # Om inga issues hittades, returnera defaults
        if not issues:
            issues = ["readability"]

        return issues

    def _extract_testing_needs(self, analysis: str) -> List[str]:
        """Extraherar testbehov från analysen."""
        testing_needs = []

        # Enkel testing-extraktion
        testing_keywords = [
            "unit_test",
            "integration_test",
            "edge_case",
            "boundary",
            "error_test",
            "performance_test",
            "security_test",
        ]

        analysis_lower = analysis.lower()
        for need in testing_keywords:
            if need in analysis_lower:
                testing_needs.append(need)

        # Om inga testing needs hittades, returnera defaults
        if not testing_needs:
            testing_needs = ["unit_test", "edge_case"]

        return testing_needs

    def _fallback_analysis(self, prompt: str) -> Dict[str, Any]:
        """Fallback analys om LM Studio misslyckas."""
        return {
            "agent": self.name,
            "role": self.role,
            "security_risks": ["input_validation"],
            "quality_issues": ["readability"],
            "testing_needs": ["unit_test", "edge_case"],
            "maintainability": "low",
            "recommendation": "add_validation",
        }

    def suggest_improvements(self, current_quality: str = "medium") -> Dict[str, Any]:
        """Föreslår kvalitetsförbättringar."""
        improvement_suggestions = {
            "low": [
                "Add input validation",
                "Improve error handling",
                "Add documentation",
            ],
            "medium": ["Add unit tests", "Improve naming", "Add type hints"],
            "high": [
                "Add integration tests",
                "Performance optimization",
                "Security audit",
            ],
        }

        return {
            "agent": self.name,
            "suggestions": improvement_suggestions.get(current_quality, ["Add validation"]),
            "priority": "medium",
        }
