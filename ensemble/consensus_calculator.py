"""
Consensus Calculator for LLM Ensemble

Analyzes and compares responses from multiple LLMs to generate consensus.
"""

import json
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
from difflib import SequenceMatcher
from typing import Dict, Any
import ast

logger = logging.getLogger(__name__)


@dataclass
class ConsensusResult:
    """Result of consensus calculation"""

    consensus: str
    confidence: float
    model_scores: Dict[str, float]
    reasoning: str
    code_quality_score: float
    syntax_valid: bool


class ConsensusCalculator:
    """Enhanced consensus calculator with intelligent scoring"""

    def __init__(self):
        self.quality_weights = {
            "syntax_valid": 0.3,
            "code_quality": 0.3,
            "model_confidence": 0.2,
            "response_length": 0.1,
            "consistency": 0.1,
        }

    def calculate_consensus(self, responses: List[Dict[str, Any]]) -> ConsensusResult:
        """
        Calculate consensus from multiple model responses with intelligent scoring

        Args:
            responses: List of responses from different models

        Returns:
            ConsensusResult with best response and scoring details
        """
        if not responses:
            return ConsensusResult(
                consensus="",
                confidence=0.0,
                model_scores={},
                reasoning="No responses provided",
                code_quality_score=0.0,
                syntax_valid=False,
            )

        # Score each response
        scored_responses = []
        for response in responses:
            score = self._score_response(response)
            scored_responses.append(
                {
                    "response": response,
                    "score": score,
                    "model": response.get("model", "unknown"),
                }
            )

        # Sort by score (highest first)
        scored_responses.sort(key=lambda x: x["score"], reverse=True)

        # Get best response
        best_response = scored_responses[0]

        # Calculate overall confidence
        total_score = sum(r["score"] for r in scored_responses)
        avg_score = total_score / len(scored_responses)

        # Check consistency between top responses
        consistency_score = self._calculate_consistency(scored_responses[:3])

        # Extract code from best response
        content = best_response["response"].get("content", "")
        if not content and "choices" in best_response["response"]:
            # Handle OpenAI-style response format
            choices = best_response["response"]["choices"]
            if choices and "message" in choices[0]:
                content = choices[0]["message"].get("content", "")

        code = self._extract_code(content)

        # Validate syntax
        syntax_valid = self._validate_syntax(code)

        # Calculate code quality
        code_quality = self._calculate_code_quality(code)

        # Create model scores dict
        model_scores = {r["model"]: r["score"] for r in scored_responses}

        # Generate reasoning
        reasoning = self._generate_reasoning(
            scored_responses, consistency_score, syntax_valid
        )

        # TEMPORARY FIX: Better confidence calculation for single model
        if len(scored_responses) == 1:
            # For single model, use avg_score directly (don't multiply by consistency)
            final_confidence = avg_score
        else:
            # For multiple models, use consistency
            final_confidence = avg_score * consistency_score

        return ConsensusResult(
            consensus=code,
            confidence=final_confidence,
            model_scores=model_scores,
            reasoning=reasoning,
            code_quality_score=code_quality,
            syntax_valid=syntax_valid,
        )

    def _score_response(self, response: Dict[str, Any]) -> float:
        """Score a single response based on multiple criteria"""
        content = response.get("content", "")
        if not content and "choices" in response:
            # Handle OpenAI-style response format
            choices = response["choices"]
            if choices and "message" in choices[0]:
                content = choices[0]["message"].get("content", "")

        model_confidence = response.get("confidence", 0.5)

        # Extract code
        code = self._extract_code(content)

        # 1. Syntax validation (30% weight)
        syntax_score = 1.0 if self._validate_syntax(code) else 0.0

        # 2. Code quality (30% weight)
        quality_score = self._calculate_code_quality(code)

        # 3. Model confidence (20% weight)
        confidence_score = model_confidence

        # 4. Response length (10% weight)
        length_score = min(len(content) / 1000, 1.0)  # Normalize to 0-1

        # 5. Consistency with other responses (10% weight)
        # This will be calculated separately

        # Calculate weighted score
        score = (
            syntax_score * self.quality_weights["syntax_valid"]
            + quality_score * self.quality_weights["code_quality"]
            + confidence_score * self.quality_weights["model_confidence"]
            + length_score * self.quality_weights["response_length"]
        )

        return score

    def _extract_code(self, content: str) -> str:
        """Extract code blocks from response content"""
        # Look for code blocks
        code_blocks = re.findall(r"```(?:python)?\n(.*?)\n```", content, re.DOTALL)
        if code_blocks:
            return "\n\n".join(code_blocks)

        # Look for inline code
        inline_code = re.findall(r"`([^`]+)`", content)
        if inline_code:
            return "\n".join(inline_code)

        # If no code blocks found, return the whole content
        return content

    def _validate_syntax(self, code: str) -> bool:
        """Validate Python syntax"""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def _calculate_code_quality(self, code: str) -> float:
        """Calculate code quality score based on various metrics"""
        if not code.strip():
            return 0.0

        score = 0.0
        lines = code.split("\n")

        # 1. Check for proper indentation
        try:
            ast.parse(code)
            score += 0.2
        except:
            pass

        # 2. Check for docstrings
        if '"""' in code or "'''" in code:
            score += 0.15

        # 3. Check for type hints
        if ":" in code and any(
            hint in code for hint in ["str", "int", "bool", "List", "Dict"]
        ):
            score += 0.15

        # 4. Check for error handling
        if any(keyword in code for keyword in ["try:", "except:", "finally:"]):
            score += 0.15

        # 5. Check for meaningful variable names
        meaningful_names = len(re.findall(r"\b[a-z_][a-z0-9_]*\b", code))
        total_words = len(code.split())
        if total_words > 0:
            name_ratio = meaningful_names / total_words
            score += min(name_ratio * 0.2, 0.2)

        # 6. Check for reasonable line length
        long_lines = sum(1 for line in lines if len(line) > 80)
        if lines:
            line_quality = 1.0 - (long_lines / len(lines))
            score += line_quality * 0.15

        return min(score, 1.0)

    def _calculate_consistency(self, top_responses: List[Dict]) -> float:
        """Calculate consistency between top responses"""
        if len(top_responses) < 2:
            return 1.0

        # Extract code from responses
        codes = []
        for response in top_responses:
            content = response["response"].get("content", "")
            if not content and "choices" in response["response"]:
                # Handle OpenAI-style response format
                choices = response["response"]["choices"]
                if choices and "message" in choices[0]:
                    content = choices[0]["message"].get("content", "")

            code = self._extract_code(content)
            codes.append(code)

        # Simple similarity check (can be enhanced with more sophisticated methods)
        similarities = []
        for i in range(len(codes)):
            for j in range(i + 1, len(codes)):
                similarity = self._calculate_similarity(codes[i], codes[j])
                similarities.append(similarity)

        if similarities:
            return sum(similarities) / len(similarities)
        return 1.0

    def _calculate_similarity(self, code1: str, code2: str) -> float:
        """Calculate similarity between two code snippets"""
        # Simple token-based similarity
        tokens1 = set(re.findall(r"\b\w+\b", code1.lower()))
        tokens2 = set(re.findall(r"\b\w+\b", code2.lower()))

        if not tokens1 or not tokens2:
            return 0.0

        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)

        return len(intersection) / len(union)

    def _generate_reasoning(
        self, scored_responses: List[Dict], consistency: float, syntax_valid: bool
    ) -> str:
        """Generate reasoning for consensus decision"""
        best_model = scored_responses[0]["model"]
        best_score = scored_responses[0]["score"]

        reasoning_parts = []

        # Model selection reasoning
        reasoning_parts.append(
            f"Selected {best_model} as primary model (score: {best_score:.2f})"
        )

        # Syntax validation
        if syntax_valid:
            reasoning_parts.append("Code passes syntax validation")
        else:
            reasoning_parts.append("⚠️ Code has syntax issues")

        # Consistency check
        if consistency > 0.8:
            reasoning_parts.append("High consistency between model responses")
        elif consistency > 0.5:
            reasoning_parts.append("Moderate consistency between model responses")
        else:
            reasoning_parts.append("⚠️ Low consistency between model responses")

        # Quality assessment
        if best_score > 0.8:
            reasoning_parts.append("High quality response selected")
        elif best_score > 0.6:
            reasoning_parts.append("Good quality response selected")
        else:
            reasoning_parts.append("⚠️ Lower quality response selected")

        return ". ".join(reasoning_parts)
