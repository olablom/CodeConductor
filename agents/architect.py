"""
ArchitectAgent - Analyzes architecture and design patterns.

Part of the multi-agent discussion system for CodeConductor.
"""

from typing import Dict, Any, List
from pathlib import Path
import json

from integrations.lm_studio import generate_code


class ArchitectAgent:
    """Analyserar arkitektur och design patterns."""

    def __init__(self):
        self.name = "ArchitectAgent"
        self.role = "Software Architecture Expert"

    def analyze(self, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyserar arkitektur och design patterns för en given prompt.

        Args:
            prompt: Kodkravet att analysera
            context: Ytterligare kontext (valfritt)

        Returns:
            Dictionary med arkitekturanalys
        """
        if context is None:
            context = {}

        # Skapa en arkitektur-prompt
        architecture_prompt = f"""
As a software architect, analyze the architecture needs for:

{prompt}

Consider:
- What design patterns would be most appropriate?
- What is the optimal code structure?
- What are the architectural trade-offs?
- What scalability considerations apply?

Provide architectural recommendations.
"""

        # Skapa temporär prompt-fil för analys
        temp_prompt_path = Path("data/temp_architecture.md")
        temp_prompt_path.parent.mkdir(parents=True, exist_ok=True)
        temp_prompt_path.write_text(architecture_prompt)

        try:
            # Använd LM Studio för analys
            analysis = generate_code(temp_prompt_path, "conservative")

            if analysis:
                return {
                    "agent": self.name,
                    "role": self.role,
                    "patterns": self._extract_patterns(analysis),
                    "structure": analysis,
                    "risks": ["complexity", "maintenance"],
                    "scalability": "medium",
                    "recommendation": "modular_design",
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

    def _extract_patterns(self, analysis: str) -> List[str]:
        """Extraherar design patterns från analysen."""
        patterns = []

        # Enkel pattern-extraktion
        pattern_keywords = [
            "factory",
            "observer",
            "singleton",
            "strategy",
            "command",
            "adapter",
            "decorator",
            "template",
        ]

        analysis_lower = analysis.lower()
        for pattern in pattern_keywords:
            if pattern in analysis_lower:
                patterns.append(pattern)

        # Om inga patterns hittades, returnera defaults
        if not patterns:
            patterns = ["simple", "modular"]

        return patterns

    def _fallback_analysis(self, prompt: str) -> Dict[str, Any]:
        """Fallback analys om LM Studio misslyckas."""
        return {
            "agent": self.name,
            "role": self.role,
            "patterns": ["simple", "modular"],
            "structure": "Single responsibility with clear separation",
            "risks": ["tight_coupling"],
            "scalability": "low",
            "recommendation": "simple_structure",
        }

    def suggest_patterns(self, complexity: str = "medium") -> Dict[str, Any]:
        """Föreslår design patterns baserat på komplexitet."""
        pattern_suggestions = {
            "low": ["simple", "procedural"],
            "medium": ["factory", "strategy", "observer"],
            "high": ["command", "adapter", "decorator", "template"],
        }

        return {
            "agent": self.name,
            "suggested_patterns": pattern_suggestions.get(complexity, ["simple"]),
            "reasoning": f"Based on {complexity} complexity level",
        }
