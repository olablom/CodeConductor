"""
Tests for BaseAgent class.

This module tests the abstract BaseAgent class and ensures it properly
requires implementation of abstract methods.
"""

import pytest
from typing import Dict, Any
from agents.base_agent import BaseAgent


class MockAgent(BaseAgent):
    """Mock implementation of BaseAgent for testing."""

    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Mock analyze implementation."""
        return {
            "insights": ["Mock insight 1", "Mock insight 2"],
            "complexity": "low",
            "estimated_time": "1 hour",
        }

    def propose(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Mock propose implementation."""
        return {
            "solution": "Mock solution",
            "approach": "Mock approach",
            "confidence": 0.8,
        }

    def review(self, code: str) -> Dict[str, Any]:
        """Mock review implementation."""
        return {
            "quality_score": 0.9,
            "issues": [],
            "recommendations": ["Mock recommendation"],
        }


class TestBaseAgent:
    """Test cases for BaseAgent class."""

    def test_base_agent_abstract(self):
        """Test that BaseAgent cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseAgent("test_agent")

    def test_mock_agent_instantiation(self):
        """Test that a concrete implementation can be instantiated."""
        agent = MockAgent("test_agent")
        assert agent.name == "test_agent"
        assert agent.config == {}
        assert agent.message_bus is None
        assert agent.llm_client is None

    def test_mock_agent_with_config(self):
        """Test agent instantiation with configuration."""
        config = {"temperature": 0.7, "max_tokens": 2048}
        agent = MockAgent("test_agent", config)
        assert agent.config == config

    def test_analyze_method(self):
        """Test the analyze method."""
        agent = MockAgent("test_agent")
        context = {"requirements": "Create a simple API", "language": "Python"}

        result = agent.analyze(context)

        assert isinstance(result, dict)
        assert "insights" in result
        assert "complexity" in result
        assert "estimated_time" in result
        assert len(result["insights"]) == 2

    def test_propose_method(self):
        """Test the propose method."""
        agent = MockAgent("test_agent")
        analysis = {
            "insights": ["Simple API needed"],
            "complexity": "low",
            "estimated_time": "1 hour",
        }

        result = agent.propose(analysis)

        assert isinstance(result, dict)
        assert "solution" in result
        assert "approach" in result
        assert "confidence" in result
        assert result["confidence"] == 0.8

    def test_review_method(self):
        """Test the review method."""
        agent = MockAgent("test_agent")
        code = "def hello_world():\n    print('Hello, World!')"

        result = agent.review(code)

        assert isinstance(result, dict)
        assert "quality_score" in result
        assert "issues" in result
        assert "recommendations" in result
        assert result["quality_score"] == 0.9
        assert isinstance(result["issues"], list)

    def test_set_message_bus(self):
        """Test setting message bus."""
        agent = MockAgent("test_agent")
        mock_bus = object()

        agent.set_message_bus(mock_bus)
        assert agent.message_bus is mock_bus

    def test_set_llm_client(self):
        """Test setting LLM client."""
        agent = MockAgent("test_agent")
        mock_client = object()

        agent.set_llm_client(mock_client)
        assert agent.llm_client is mock_client

    def test_get_status(self):
        """Test getting agent status."""
        agent = MockAgent("test_agent", {"test": "config"})

        status = agent.get_status()

        assert status["name"] == "test_agent"
        assert status["config"] == {"test": "config"}
        assert status["has_message_bus"] is False
        assert status["has_llm_client"] is False

    def test_get_status_with_connections(self):
        """Test status when message bus and LLM client are set."""
        agent = MockAgent("test_agent")
        mock_bus = object()
        mock_client = object()

        agent.set_message_bus(mock_bus)
        agent.set_llm_client(mock_client)

        status = agent.get_status()
        assert status["has_message_bus"] is True
        assert status["has_llm_client"] is True

    def test_string_representation(self):
        """Test string representation of agent."""
        agent = MockAgent("test_agent")

        assert str(agent) == "MockAgent(name='test_agent')"

    def test_repr_representation(self):
        """Test detailed string representation of agent."""
        config = {"temperature": 0.7}
        agent = MockAgent("test_agent", config)

        expected = "MockAgent(name='test_agent', config={'temperature': 0.7})"
        assert repr(agent) == expected


class TestBaseAgentIntegration:
    """Integration tests for BaseAgent."""

    def test_full_workflow(self):
        """Test a complete agent workflow."""
        agent = MockAgent("workflow_agent")

        # Step 1: Analyze context
        context = {
            "task": "Create a REST API",
            "language": "Python",
            "framework": "FastAPI",
        }
        analysis = agent.analyze(context)

        # Step 2: Propose solution
        proposal = agent.propose(analysis)

        # Step 3: Review generated code
        code = """
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}
        """
        review = agent.review(code)

        # Verify all steps completed successfully
        assert analysis is not None
        assert proposal is not None
        assert review is not None
        assert review["quality_score"] > 0

    def test_agent_with_llm_client(self):
        """Test agent integration with LLM client."""
        from integrations.llm_client import LLMClient

        agent = MockAgent("llm_agent")
        llm_client = LLMClient("http://localhost:1234")

        agent.set_llm_client(llm_client)

        status = agent.get_status()
        assert status["has_llm_client"] is True
        assert agent.llm_client is llm_client
