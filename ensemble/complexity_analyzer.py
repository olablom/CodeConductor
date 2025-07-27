#!/usr/bin/env python3
"""
Complexity Analyzer for CodeConductor MVP
Determines when to escalate from local LLMs to cloud APIs for complex tasks.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ComplexityLevel(Enum):
    """Complexity levels for tasks."""

    SIMPLE = "simple"  # Local LLMs can handle
    MODERATE = "moderate"  # Local LLMs with high confidence
    COMPLEX = "complex"  # Needs cloud escalation
    EXPERT = "expert"  # Requires human review + cloud


@dataclass
class ComplexityResult:
    """Result of complexity analysis."""

    level: ComplexityLevel
    confidence: float
    reasons: List[str]
    estimated_tokens: int
    requires_cloud: bool
    suggested_models: List[str]


class ComplexityAnalyzer:
    """Analyzes task complexity to determine escalation strategy."""

    def __init__(self):
        # Complexity indicators
        self.complex_keywords = [
            "algorithm",
            "optimization",
            "performance",
            "scalability",
            "security",
            "vulnerability",
            "encryption",
            "authentication",
            "architecture",
            "design pattern",
            "microservices",
            "distributed",
            "concurrent",
            "threading",
            "async",
            "parallel",
            "machine learning",
            "AI",
            "neural network",
            "data science",
            "database design",
            "ORM",
            "migration",
            "schema",
            "API design",
            "REST",
            "GraphQL",
            "websocket",
            "testing strategy",
            "TDD",
            "BDD",
            "integration test",
            "deployment",
            "CI/CD",
            "docker",
            "kubernetes",
            "monitoring",
            "logging",
            "metrics",
            "observability",
        ]

        self.expert_keywords = [
            "zero-day",
            "exploit",
            "penetration test",
            "security audit",
            "performance tuning",
            "profiling",
            "memory leak",
            "race condition",
            "distributed system",
            "consistency",
            "availability",
            "partition tolerance",
            "machine learning model",
            "training",
            "inference",
            "model serving",
            "real-time",
            "streaming",
            "event-driven",
            "message queue",
        ]

        self.simple_patterns = [
            r"create.*function",
            r"add.*method",
            r"fix.*bug",
            r"update.*test",
            r"rename.*variable",
            r"add.*comment",
            r"format.*code",
        ]

    def analyze_complexity(
        self, task: str, context: Optional[Dict] = None
    ) -> ComplexityResult:
        """
        Analyze task complexity and recommend escalation strategy.

        Args:
            task: The development task description
            context: Optional context (file size, project structure, etc.)

        Returns:
            ComplexityResult with analysis and recommendations
        """
        logger.info(f"🔍 Analyzing complexity for task: {task[:100]}...")

        # Initialize analysis
        reasons = []
        complexity_score = 0.0
        estimated_tokens = self._estimate_tokens(task)

        # Check for expert-level keywords
        expert_matches = self._count_keyword_matches(task, self.expert_keywords)
        if expert_matches > 0:
            complexity_score += expert_matches * 0.4
            reasons.append(f"Contains {expert_matches} expert-level concepts")

        # Check for complex keywords
        complex_matches = self._count_keyword_matches(task, self.complex_keywords)
        if complex_matches > 0:
            complexity_score += complex_matches * 0.2
            reasons.append(f"Contains {complex_matches} complex concepts")

        # Check for simple patterns
        simple_matches = self._count_pattern_matches(task, self.simple_patterns)
        if simple_matches > 0:
            complexity_score -= simple_matches * 0.1
            reasons.append(f"Contains {simple_matches} simple patterns")

        # Context-based adjustments
        if context:
            complexity_score = self._adjust_for_context(
                complexity_score, context, reasons
            )

        # Determine complexity level
        level, confidence = self._determine_level(complexity_score, estimated_tokens)

        # Determine if cloud escalation is needed
        requires_cloud = level in [ComplexityLevel.COMPLEX, ComplexityLevel.EXPERT]

        # Suggest appropriate models
        suggested_models = self._suggest_models(level, requires_cloud)

        result = ComplexityResult(
            level=level,
            confidence=confidence,
            reasons=reasons,
            estimated_tokens=estimated_tokens,
            requires_cloud=requires_cloud,
            suggested_models=suggested_models,
        )

        logger.info(
            f"📊 Complexity analysis: {level.value} (confidence: {confidence:.2f})"
        )
        return result

    def _count_keyword_matches(self, text: str, keywords: List[str]) -> int:
        """Count how many keywords appear in the text."""
        text_lower = text.lower()
        matches = 0
        for keyword in keywords:
            if keyword.lower() in text_lower:
                matches += 1
        return matches

    def _count_pattern_matches(self, text: str, patterns: List[str]) -> int:
        """Count how many patterns match in the text."""
        matches = 0
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                matches += 1
        return matches

    def _estimate_tokens(self, text: str) -> int:
        """Rough estimation of token count."""
        # Simple estimation: 1 token ≈ 4 characters
        return len(text) // 4

    def _adjust_for_context(
        self, score: float, context: Dict, reasons: List[str]
    ) -> float:
        """Adjust complexity score based on context."""
        # File size adjustment
        if "file_size" in context:
            file_size = context["file_size"]
            if file_size > 1000:  # Large files are more complex
                score += 0.2
                reasons.append("Large file (>1000 lines)")
            elif file_size < 100:  # Small files are simpler
                score -= 0.1
                reasons.append("Small file (<100 lines)")

        # Project structure adjustment
        if "project_complexity" in context:
            project_complexity = context["project_complexity"]
            if project_complexity == "large":
                score += 0.3
                reasons.append("Large project context")
            elif project_complexity == "small":
                score -= 0.1
                reasons.append("Small project context")

        return max(0.0, min(1.0, score))  # Clamp between 0 and 1

    def _determine_level(
        self, score: float, tokens: int
    ) -> Tuple[ComplexityLevel, float]:
        """Determine complexity level and confidence."""
        # Base level on score
        if score < 0.2:
            level = ComplexityLevel.SIMPLE
            confidence = 0.9
        elif score < 0.5:
            level = ComplexityLevel.MODERATE
            confidence = 0.8
        elif score < 0.8:
            level = ComplexityLevel.COMPLEX
            confidence = 0.7
        else:
            level = ComplexityLevel.EXPERT
            confidence = 0.9

        # Adjust for token count
        if tokens > 2000:
            level = ComplexityLevel.COMPLEX
            confidence = min(confidence, 0.8)
        elif tokens > 5000:
            level = ComplexityLevel.EXPERT
            confidence = min(confidence, 0.9)

        return level, confidence

    def _suggest_models(
        self, level: ComplexityLevel, requires_cloud: bool
    ) -> List[str]:
        """Suggest appropriate models for the complexity level."""
        if level == ComplexityLevel.SIMPLE:
            return ["codellama-7b-instruct", "mistral-7b-instruct-v0.1", "phi3:mini"]
        elif level == ComplexityLevel.MODERATE:
            return [
                "meta-llama-3.1-8b-instruct",
                "google/gemma-3-12b",
                "deepseek-r1-distill-qwen-7b",
            ]
        elif level == ComplexityLevel.COMPLEX:
            if requires_cloud:
                return ["gpt-4", "claude-3-sonnet", "gpt-3.5-turbo"]
            else:
                return ["google/gemma-3-12b", "meta-llama-3.1-8b-instruct"]
        else:  # EXPERT
            return ["gpt-4", "claude-3-opus", "gpt-4-turbo"]

    def should_escalate(self, task: str, local_confidence: float = 0.0) -> bool:
        """
        Quick check if task should be escalated to cloud.

        Args:
            task: The development task
            local_confidence: Confidence from local ensemble (0.0-1.0)

        Returns:
            True if should escalate to cloud
        """
        analysis = self.analyze_complexity(task)

        # Escalate if:
        # 1. Task is complex/expert level
        # 2. Local confidence is low (< 0.7)
        # 3. Task requires cloud models

        should_escalate = (
            analysis.requires_cloud
            or local_confidence < 0.7
            or analysis.level in [ComplexityLevel.COMPLEX, ComplexityLevel.EXPERT]
        )

        logger.info(
            f"🚀 Escalation decision: {should_escalate} (local_confidence: {local_confidence:.2f})"
        )
        return should_escalate


# Convenience functions
def analyze_task_complexity(
    task: str, context: Optional[Dict] = None
) -> ComplexityResult:
    """Analyze task complexity."""
    analyzer = ComplexityAnalyzer()
    return analyzer.analyze_complexity(task, context)


def should_escalate_to_cloud(task: str, local_confidence: float = 0.0) -> bool:
    """Quick check if task should be escalated."""
    analyzer = ComplexityAnalyzer()
    return analyzer.should_escalate(task, local_confidence)


if __name__ == "__main__":
    # Demo
    analyzer = ComplexityAnalyzer()

    # Test cases
    test_tasks = [
        "Create a simple calculator class with basic operations",
        "Implement a secure authentication system with JWT tokens",
        "Design a distributed microservices architecture for e-commerce",
        "Fix the bug in the login function",
        "Optimize the database queries for better performance",
    ]

    print("🎯 CodeConductor Complexity Analyzer Demo")
    print("=" * 50)

    for task in test_tasks:
        result = analyzer.analyze_complexity(task)
        print(f"\n📝 Task: {task}")
        print(f"🏷️  Level: {result.level.value}")
        print(f"📊 Confidence: {result.confidence:.2f}")
        print(f"☁️  Requires Cloud: {result.requires_cloud}")
        print(f"🤖 Suggested Models: {', '.join(result.suggested_models)}")
        print(f"📋 Reasons: {', '.join(result.reasons)}")
