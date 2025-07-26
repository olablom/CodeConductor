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

logger = logging.getLogger(__name__)


@dataclass
class ConsensusResult:
    consensus: Dict[str, Any]
    confidence: float
    disagreements: List[str]
    model_agreement: Dict[str, float]
    raw_responses: List[Dict[str, Any]]


class ConsensusCalculator:
    """Calculates consensus from multiple LLM responses."""

    def __init__(self, min_confidence: float = 0.7):
        self.min_confidence = min_confidence

    def calculate_consensus(self, results: List[Dict[str, Any]]) -> ConsensusResult:
        """Calculate consensus from multiple model responses."""
        # Filter successful responses
        successful_results = [r for r in results if r.success]

        if not successful_results:
            return ConsensusResult(
                consensus={},
                confidence=0.0,
                disagreements=["No successful responses"],
                model_agreement={},
                raw_responses=results,
            )

        # Parse JSON responses
        parsed_responses = []
        for result in successful_results:
            try:
                parsed = self._parse_json_response(result.response)
                if parsed:
                    parsed_responses.append((result.model_id, parsed))
            except Exception as e:
                logger.warning(f"Failed to parse response from {result.model_id}: {e}")

        if not parsed_responses:
            return ConsensusResult(
                consensus={},
                confidence=0.0,
                disagreements=["No valid JSON responses"],
                model_agreement={},
                raw_responses=results,
            )

        # Calculate consensus for each field
        consensus = {}
        disagreements = []
        model_agreement = {}

        # Get all unique fields
        all_fields = set()
        for _, response in parsed_responses:
            all_fields.update(response.keys())

        for field in all_fields:
            field_values = [response.get(field) for _, response in parsed_responses]
            field_consensus, field_confidence, field_disagreements = (
                self._consensus_for_field(field, field_values, parsed_responses)
            )

            consensus[field] = field_consensus
            disagreements.extend(field_disagreements)

            # Track model agreement per field
            for model_id, _ in parsed_responses:
                if model_id not in model_agreement:
                    model_agreement[model_id] = {}
                model_agreement[model_id][field] = field_confidence

        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(
            consensus, model_agreement, len(parsed_responses)
        )

        return ConsensusResult(
            consensus=consensus,
            confidence=overall_confidence,
            disagreements=disagreements,
            model_agreement=model_agreement,
            raw_responses=results,
        )

    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON response, handling common formatting issues."""
        # Try to extract JSON from response
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # Try parsing the entire response as JSON
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            logger.warning(f"Could not parse JSON from response: {response[:100]}...")
            return None

    def _consensus_for_field(
        self,
        field: str,
        values: List[Any],
        parsed_responses: List[Tuple[str, Dict[str, Any]]],
    ) -> Tuple[Any, float, List[str]]:
        """Calculate consensus for a specific field."""
        # Remove None values
        valid_values = [v for v in values if v is not None]

        if not valid_values:
            return None, 0.0, [f"No valid values for field '{field}'"]

        if len(valid_values) == 1:
            return valid_values[0], 1.0, []

        # Handle different field types
        if field in ["complexity"]:
            return self._consensus_for_enum_field(field, valid_values, parsed_responses)
        elif field in ["files_needed", "dependencies"]:
            return self._consensus_for_list_field(field, valid_values, parsed_responses)
        elif field in ["approach", "reasoning"]:
            return self._consensus_for_text_field(field, valid_values, parsed_responses)
        else:
            return self._consensus_for_generic_field(
                field, valid_values, parsed_responses
            )

    def _consensus_for_enum_field(
        self,
        field: str,
        values: List[Any],
        parsed_responses: List[Tuple[str, Dict[str, Any]]],
    ) -> Tuple[Any, float, List[str]]:
        """Calculate consensus for enum-like fields (e.g., complexity)."""
        # Count occurrences
        value_counts = {}
        for value in values:
            value_str = str(value).lower().strip()
            value_counts[value_str] = value_counts.get(value_str, 0) + 1

        # Find most common value
        most_common = max(value_counts.items(), key=lambda x: x[1])
        consensus_value = most_common[0]
        confidence = most_common[1] / len(values)

        disagreements = []
        if confidence < 1.0:
            disagreements.append(f"Models disagree on {field}: {value_counts}")

        return consensus_value, confidence, disagreements

    def _consensus_for_list_field(
        self,
        field: str,
        values: List[Any],
        parsed_responses: List[Tuple[str, Dict[str, Any]]],
    ) -> Tuple[Any, float, List[str]]:
        """Calculate consensus for list fields (e.g., files_needed, dependencies)."""
        # Flatten and normalize list items
        all_items = []
        for value in values:
            if isinstance(value, list):
                all_items.extend([str(item).lower().strip() for item in value])
            elif isinstance(value, str):
                # Try to parse as comma-separated
                items = [item.strip() for item in value.split(",")]
                all_items.extend(items)

        # Count item occurrences
        item_counts = {}
        for item in all_items:
            if item:  # Skip empty items
                item_counts[item] = item_counts.get(item, 0) + 1

        # Select items that appear in majority of responses
        threshold = len(values) * 0.5  # 50% threshold
        consensus_items = [
            item for item, count in item_counts.items() if count >= threshold
        ]

        confidence = len(consensus_items) / max(len(all_items), 1)

        disagreements = []
        if len(consensus_items) != len(all_items):
            disagreements.append(f"Models disagree on {field} items: {item_counts}")

        return consensus_items, confidence, disagreements

    def _consensus_for_text_field(
        self,
        field: str,
        values: List[Any],
        parsed_responses: List[Tuple[str, Dict[str, Any]]],
    ) -> Tuple[Any, float, List[str]]:
        """Calculate consensus for text fields (e.g., approach, reasoning)."""
        # For text fields, we'll use the most detailed response
        # or combine multiple responses if they're similar

        # Calculate similarity between responses
        similarities = []
        for i, val1 in enumerate(values):
            for j, val2 in enumerate(values[i + 1 :], i + 1):
                similarity = SequenceMatcher(None, str(val1), str(val2)).ratio()
                similarities.append(similarity)

        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0

        # If responses are very similar, use the longest one
        if avg_similarity > 0.8:
            consensus_value = max(values, key=len)
            confidence = avg_similarity
            disagreements = []
        else:
            # Use the most detailed response
            consensus_value = max(values, key=len)
            confidence = 0.5  # Lower confidence for dissimilar responses
            disagreements = [f"Models have different {field} approaches"]

        return consensus_value, confidence, disagreements

    def _consensus_for_generic_field(
        self,
        field: str,
        values: List[Any],
        parsed_responses: List[Tuple[str, Dict[str, Any]]],
    ) -> Tuple[Any, float, List[str]]:
        """Calculate consensus for generic fields."""
        # For generic fields, use majority voting
        value_counts = {}
        for value in values:
            value_str = str(value)
            value_counts[value_str] = value_counts.get(value_str, 0) + 1

        most_common = max(value_counts.items(), key=lambda x: x[1])
        consensus_value = most_common[0]
        confidence = most_common[1] / len(values)

        disagreements = []
        if confidence < 1.0:
            disagreements.append(f"Models disagree on {field}: {value_counts}")

        return consensus_value, confidence, disagreements

    def _calculate_overall_confidence(
        self,
        consensus: Dict[str, Any],
        model_agreement: Dict[str, Dict[str, float]],
        num_models: int,
    ) -> float:
        """Calculate overall confidence score."""
        if not consensus:
            return 0.0

        # Weight different fields
        field_weights = {
            "approach": 0.3,
            "files_needed": 0.25,
            "dependencies": 0.2,
            "complexity": 0.15,
            "reasoning": 0.1,
        }

        weighted_confidence = 0.0
        total_weight = 0.0

        for field, weight in field_weights.items():
            if field in consensus:
                # Calculate average confidence for this field across models
                field_confidences = []
                for model_agreements in model_agreement.values():
                    if field in model_agreements:
                        field_confidences.append(model_agreements[field])

                if field_confidences:
                    avg_field_confidence = sum(field_confidences) / len(
                        field_confidences
                    )
                    weighted_confidence += avg_field_confidence * weight
                    total_weight += weight

        # Normalize by total weight
        if total_weight > 0:
            overall_confidence = weighted_confidence / total_weight
        else:
            overall_confidence = 0.0

        # Boost confidence if we have more models agreeing
        model_boost = min(num_models / 3.0, 1.0)  # Cap at 3 models
        overall_confidence = min(1.0, overall_confidence + (model_boost * 0.1))

        return overall_confidence
