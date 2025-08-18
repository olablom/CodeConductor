"""
Consensus Calculator for LLM Ensemble

Analyzes and compares responses from multiple LLMs to generate consensus.
"""

import ast
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ConsensusResult:
    """Result of consensus calculation"""

    consensus: str
    confidence: float
    model_scores: dict[str, float]
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

    def calculate_consensus(self, responses: list[dict[str, Any]]) -> ConsensusResult:
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
            # Guard against non-dict or boolean entries
            if not isinstance(response, dict):
                continue
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

    def _score_response(self, response: dict[str, Any]) -> float:
        """Score a single response based on multiple criteria"""
        # Robust extraction with guards
        if not isinstance(response, dict):
            return 0.0
        content = response.get("content", "")
        if not content and "choices" in response:
            # Handle OpenAI-style response format
            try:
                choices = response["choices"]
                if choices and isinstance(choices, list) and len(choices) > 0:
                    first = choices[0]
                    if isinstance(first, dict) and "message" in first:
                        content = first["message"].get("content", "")
            except Exception:
                content = ""

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
        except Exception:
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

    def _calculate_consistency(self, top_responses: list[dict]) -> float:
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

        # Similarity check using CodeBLEU-inspired fast heuristic
        similarities = []
        for i in range(len(codes)):
            for j in range(i + 1, len(codes)):
                similarity = self._calculate_similarity(codes[i], codes[j])
                similarities.append(similarity)

        if similarities:
            return sum(similarities) / len(similarities)
        return 1.0

    def _calculate_similarity(self, code1: str, code2: str) -> float:
        """Calculate similarity between two code snippets using a fast CodeBLEU-inspired heuristic.

        This avoids heavy dependencies by combining:
        - Unigram/bigram precision (BLEU-1/2 style)
        - AST node-type Jaccard
        - Token Jaccard

        Returns a score in [0, 1].
        """
        try:
            return self._codebleu_fast_similarity(code1, code2)
        except Exception:
            # Fallback to simple token Jaccard
            tokens1 = set(re.findall(r"\b\w+\b", code1.lower()))
            tokens2 = set(re.findall(r"\b\w+\b", code2.lower()))
            if not tokens1 or not tokens2:
                return 0.0
            intersection = tokens1.intersection(tokens2)
            union = tokens1.union(tokens2)
            return len(intersection) / len(union)

    def _codebleu_fast_similarity(self, code1: str, code2: str) -> float:
        """Lightweight, dependency-free approximation of CodeBLEU.

        Weighted combination (env-tunable via CODEBLEU_WEIGHTS="ngram,ast,token"):
        - default 50%: n-gram precision (unigram + bigram average)
        - default 30%: AST node-type Jaccard
        - default 20%: token Jaccard
        """
        # Normalize input
        s1 = code1 or ""
        s2 = code2 or ""
        lang = (os.getenv("CODEBLEU_LANG") or "").strip().lower()
        do_norm = os.getenv("CODEBLEU_NORMALIZE", "0") == "1"
        strip_comments = os.getenv("CODEBLEU_STRIP_COMMENTS", "0") == "1"
        strip_docstrings = os.getenv("CODEBLEU_STRIP_DOCSTRINGS", "0") == "1"

        if do_norm:
            s1 = s1.strip().lower()
            s2 = s2.strip().lower()
        if lang in ("py", "python"):
            if strip_comments:
                s1 = re.sub(r"(?m)#.*$", "", s1)
                s2 = re.sub(r"(?m)#.*$", "", s2)
            if strip_docstrings:
                s1 = re.sub(r"\"\"\"[\s\S]*?\"\"\"|\'\'\'[\s\S]*?\'\'\'", "", s1)
                s2 = re.sub(r"\"\"\"[\s\S]*?\"\"\"|\'\'\'[\s\S]*?\'\'\'", "", s2)
        if not s1.strip() or not s2.strip():
            return 0.0

        # Tokens
        toks1 = re.findall(r"\b\w+\b", s1.lower())
        toks2 = re.findall(r"\b\w+\b", s2.lower())

        # Unigram precision
        def precision_ngram(t1: list[str], t2: list[str], n: int) -> float:
            def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
                return [
                    tuple(tokens[i : i + n])
                    for i in range(0, max(len(tokens) - n + 1, 0))
                ]

            n1 = Counter(ngrams(t1, n))
            n2 = Counter(ngrams(t2, n))
            if not n1 or not n2:
                return 0.0
            overlap = sum((n1 & n2).values())
            total = sum(
                n2.values()
            )  # precision: how much of candidate (t2) appears in reference (t1)
            if total == 0:
                return 0.0
            return min(overlap / total, 1.0)

        p1 = precision_ngram(toks1, toks2, 1)
        p2 = precision_ngram(toks1, toks2, 2)
        ngram_score = 0.5 * p1 + 0.5 * p2

        # AST node-type Jaccard
        def ast_types(s: str) -> list[str]:
            try:
                tree = ast.parse(s)
                types: list[str] = []
                for node in ast.walk(tree):
                    types.append(type(node).__name__)
                return types
            except Exception:
                return []

        types1 = set(ast_types(s1))
        types2 = set(ast_types(s2))
        if types1 or types2:
            ast_jaccard = (
                len(types1 & types2) / len(types1 | types2)
                if (types1 | types2)
                else 0.0
            )
        else:
            ast_jaccard = 0.0

        # Operator mismatch penalty (e.g., Add vs Mult) to reduce false-high similarity
        def op_types(s: str) -> list[str]:
            try:
                tree = ast.parse(s)
                ops: list[str] = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.BinOp):
                        ops.append(type(node.op).__name__)
                return ops
            except Exception:
                return []

        ops1 = set(op_types(s1))
        ops2 = set(op_types(s2))
        op_penalty = 0.0
        if ops1 and ops2 and len(ops1 & ops2) == 0:
            # No shared operators between snippets → apply penalty
            op_penalty = 0.2

        # Token Jaccard
        tok_set1 = set(toks1)
        tok_set2 = set(toks2)
        tok_jaccard = (
            len(tok_set1 & tok_set2) / len(tok_set1 | tok_set2)
            if (tok_set1 | tok_set2)
            else 0.0
        )

        # Weighted combination (env override)
        w_str = os.getenv("CODEBLEU_WEIGHTS") or ""
        if w_str:
            try:
                parts = [float(x) for x in w_str.split(",")]
                if len(parts) == 3 and sum(parts) > 0:
                    w_ng, w_ast, w_tok = parts
                    s = w_ng + w_ast + w_tok
                    w_ng, w_ast, w_tok = w_ng / s, w_ast / s, w_tok / s
                else:
                    w_ng, w_ast, w_tok = 0.5, 0.3, 0.2
            except Exception:
                w_ng, w_ast, w_tok = 0.5, 0.3, 0.2
        else:
            w_ng, w_ast, w_tok = 0.5, 0.3, 0.2
        score = w_ng * ngram_score + w_ast * ast_jaccard + w_tok * tok_jaccard
        score = max(0.0, score - op_penalty)
        # Clamp
        return max(0.0, min(1.0, score))

    def _generate_reasoning(
        self, scored_responses: list[dict], consistency: float, syntax_valid: bool
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
