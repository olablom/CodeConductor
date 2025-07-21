"""
Unit tests for agent components.

Tests CodeGenAgent, ArchitectAgent, ReviewerAgent with proper mocking.
"""

from unittest.mock import patch

from agents.code_gen import CodeGenAgent
from agents.architect import ArchitectAgent
from agents.reviewer import ReviewerAgent


class TestCodeGenAgent:
    """Test CodeGenAgent functionality."""

    def setup_method(self):
        """Setup for each test."""
        self.agent = CodeGenAgent()

    def test_analyze_returns_dict(self):
        """Test that analyze returns proper dictionary structure."""
        result = self.agent.analyze("Create a function", {})

        assert isinstance(result, dict)
        assert "agent" in result
        assert "role" in result
        assert "approach" in result
        assert "confidence" in result
        assert "alternatives" in result
        assert "risks" in result
        assert "recommendation" in result

    def test_confidence_in_valid_range(self):
        """Test that confidence is always between 0 and 1."""
        result = self.agent.analyze("Test prompt", {})
        assert 0 <= result["confidence"] <= 1

    def test_empty_prompt_handling(self):
        """Test handling of empty prompts."""
        result = self.agent.analyze("", {})
        # Empty prompts should still return valid structure
        assert isinstance(result, dict)
        assert "confidence" in result
        assert 0 <= result["confidence"] <= 1

    def test_none_context_handling(self):
        """Test handling of None context."""
        result = self.agent.analyze("Test prompt", None)
        assert isinstance(result, dict)
        assert "confidence" in result

    @patch("agents.code_gen.generate_code")
    def test_lm_studio_integration(self, mock_generate):
        """Test LM Studio integration."""
        mock_generate.return_value = "def test(): return 42"

        result = self.agent.analyze("Test prompt", {})

        assert mock_generate.called
        assert result["approach"] == "def test(): return 42"

    @patch("agents.code_gen.generate_code")
    def test_lm_studio_fallback(self, mock_generate):
        """Test fallback when LM Studio fails."""
        mock_generate.return_value = None

        result = self.agent.analyze("Test prompt", {})

        assert result["approach"] == "Standard implementation with error handling"
        assert result["confidence"] == 0.7

    def test_suggest_improvements(self):
        """Test improvement suggestions."""
        suggestions = self.agent.suggest_improvements("current approach")

        assert isinstance(suggestions, dict)
        assert "agent" in suggestions
        assert "suggestions" in suggestions
        assert "priority" in suggestions
        assert isinstance(suggestions["suggestions"], list)
        assert len(suggestions["suggestions"]) > 0


class TestArchitectAgent:
    """Test ArchitectAgent functionality."""

    def setup_method(self):
        """Setup for each test."""
        self.agent = ArchitectAgent()

    def test_analyze_returns_dict(self):
        """Test that analyze returns proper dictionary structure."""
        result = self.agent.analyze("Create a factory pattern", {})

        assert isinstance(result, dict)
        assert "agent" in result
        assert "role" in result
        assert "patterns" in result
        assert "structure" in result
        assert "risks" in result
        assert "scalability" in result
        assert "recommendation" in result

    def test_patterns_is_list(self):
        """Test that patterns is always a list."""
        result = self.agent.analyze("Test prompt", {})
        assert isinstance(result["patterns"], list)

    def test_risks_is_list(self):
        """Test that risks is always a list."""
        result = self.agent.analyze("Test prompt", {})
        assert isinstance(result["risks"], list)

    def test_scalability_values(self):
        """Test that scalability has valid values."""
        result = self.agent.analyze("Test prompt", {})
        assert result["scalability"] in ["low", "medium", "high"]

    @patch("agents.architect.generate_code")
    def test_pattern_extraction(self, mock_generate):
        """Test pattern extraction from LM Studio response."""
        mock_generate.return_value = "Use factory pattern and observer pattern for this"

        result = self.agent.analyze("Test prompt", {})

        assert "factory" in result["patterns"]
        assert "observer" in result["patterns"]

    def test_suggest_patterns_by_complexity(self):
        """Test pattern suggestions based on complexity."""
        for complexity in ["low", "medium", "high"]:
            result = self.agent.suggest_patterns(complexity)

            assert isinstance(result, dict)
            assert "agent" in result
            assert "suggested_patterns" in result
            assert "reasoning" in result
            assert isinstance(result["suggested_patterns"], list)
            assert len(result["suggested_patterns"]) > 0

    def test_fallback_analysis(self):
        """Test fallback analysis when LM Studio fails."""
        result = self.agent._fallback_analysis("Test prompt")

        assert result["patterns"] == ["simple", "modular"]
        assert result["scalability"] == "low"
        assert result["recommendation"] == "simple_structure"


class TestReviewerAgent:
    """Test ReviewerAgent functionality."""

    def setup_method(self):
        """Setup for each test."""
        self.agent = ReviewerAgent()

    def test_analyze_returns_dict(self):
        """Test that analyze returns proper dictionary structure."""
        result = self.agent.analyze("Create a secure function", {})

        assert isinstance(result, dict)
        assert "agent" in result
        assert "role" in result
        assert "security_risks" in result
        assert "quality_issues" in result
        assert "testing_needs" in result
        assert "maintainability" in result
        assert "recommendation" in result

    def test_security_risks_is_list(self):
        """Test that security_risks is always a list."""
        result = self.agent.analyze("Test prompt", {})
        assert isinstance(result["security_risks"], list)

    def test_quality_issues_is_list(self):
        """Test that quality_issues is always a list."""
        result = self.agent.analyze("Test prompt", {})
        assert isinstance(result["quality_issues"], list)

    def test_testing_needs_is_list(self):
        """Test that testing_needs is always a list."""
        result = self.agent.analyze("Test prompt", {})
        assert isinstance(result["testing_needs"], list)

    def test_maintainability_values(self):
        """Test that maintainability has valid values."""
        result = self.agent.analyze("Test prompt", {})
        assert result["maintainability"] in ["low", "medium", "high"]

    @patch("agents.reviewer.generate_code")
    def test_risk_extraction(self, mock_generate):
        """Test risk extraction from LM Studio response."""
        mock_generate.return_value = "This code has injection risks and overflow issues"

        result = self.agent.analyze("Test prompt", {})

        assert "injection" in result["security_risks"]
        assert "overflow" in result["security_risks"]

    def test_suggest_improvements(self):
        """Test improvement suggestions."""
        for quality in ["low", "medium", "high"]:
            result = self.agent.suggest_improvements(quality)

            assert isinstance(result, dict)
            assert "agent" in result
            assert "suggestions" in result
            assert "priority" in result
            assert isinstance(result["suggestions"], list)
            assert len(result["suggestions"]) > 0

    def test_fallback_analysis(self):
        """Test fallback analysis when LM Studio fails."""
        result = self.agent._fallback_analysis("Test prompt")

        assert result["security_risks"] == ["input_validation"]
        assert result["quality_issues"] == ["readability"]
        assert result["testing_needs"] == ["unit_test", "edge_case"]
        assert result["maintainability"] == "low"


class TestAgentIntegration:
    """Test integration between agents."""

    def test_all_agents_same_interface(self):
        """Test that all agents have consistent interfaces."""
        agents = [CodeGenAgent(), ArchitectAgent(), ReviewerAgent()]

        for agent in agents:
            result = agent.analyze("Test prompt", {})

            # All should return dicts
            assert isinstance(result, dict)

            # All should have agent name
            assert "agent" in result
            assert isinstance(result["agent"], str)

            # All should have role
            assert "role" in result
            assert isinstance(result["role"], str)

    def test_agent_names_unique(self):
        """Test that agent names are unique."""
        agents = [CodeGenAgent(), ArchitectAgent(), ReviewerAgent()]

        names = [agent.name for agent in agents]
        assert len(names) == len(set(names)), "Agent names should be unique"

    def test_agent_roles_unique(self):
        """Test that agent roles are unique."""
        agents = [CodeGenAgent(), ArchitectAgent(), ReviewerAgent()]

        roles = [agent.role for agent in agents]
        assert len(roles) == len(set(roles)), "Agent roles should be unique"
