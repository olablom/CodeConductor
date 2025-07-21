"""
CodeGenAgent - Analyzes implementation strategies and approaches.

Part of the multi-agent discussion system for CodeConductor.
"""

from typing import Dict, Any
from pathlib import Path

from integrations.lm_studio import generate_code


class CodeGenAgent:
    """Föreslår implementation strategier och analyserar kodkrav."""

    def __init__(self):
        self.name = "CodeGenAgent"
        self.role = "Implementation Strategy Expert"

    def analyze(self, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyserar en prompt och föreslår implementation strategier.

        Args:
            prompt: Kodkravet att analysera
            context: Ytterligare kontext (valfritt)

        Returns:
            Dictionary med analys och rekommendationer
        """
        if context is None:
            context = {}

        # Skapa en analys-prompt
        analysis_prompt = f"""
As a code generation expert, analyze this implementation request:

{prompt}

Consider:
- What programming patterns would work best?
- What are the key implementation challenges?
- What approach would you recommend?
- What are the risks and trade-offs?

Provide a structured analysis.
"""

        # Skapa temporär prompt-fil för analys
        temp_prompt_path = Path("data/temp_analysis.md")
        temp_prompt_path.parent.mkdir(parents=True, exist_ok=True)
        temp_prompt_path.write_text(analysis_prompt)

        try:
            # Använd LM Studio för analys
            analysis = generate_code(temp_prompt_path, "balanced")

            if analysis:
                return {
                    "agent": self.name,
                    "role": self.role,
                    "approach": analysis,
                    "confidence": 0.85,
                    "alternatives": ["functional", "oop", "procedural"],
                    "risks": ["complexity", "performance"],
                    "recommendation": "structured_approach",
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

    def _fallback_analysis(self, prompt: str) -> Dict[str, Any]:
        """Fallback analys om LM Studio misslyckas."""
        return {
            "agent": self.name,
            "role": self.role,
            "approach": "Standard implementation with error handling",
            "confidence": 0.7,
            "alternatives": ["simple", "robust"],
            "risks": ["edge_cases"],
            "recommendation": "defensive_programming",
        }

    def suggest_improvements(self, current_approach: str) -> Dict[str, Any]:
        """Föreslår förbättringar för en given approach."""
        return {
            "agent": self.name,
            "suggestions": [
                "Add input validation",
                "Implement error handling",
                "Consider edge cases",
                "Add documentation",
            ],
            "priority": "medium",
        }
