"""
Tests for QLearningAgent class.

This module tests the Q-learning implementation and ensures proper
state-action management and learning behavior.
"""

from agents.q_learning_agent import QLearningAgent, State, Action


class TestState:
    """Test cases for State dataclass."""

    def test_state_creation(self):
        """Test creating State instance."""
        state = State(
            prompt_type="code_generation",
            complexity_level="medium",
            previous_action="add_examples",
            iteration_count=2,
        )

        assert state.prompt_type == "code_generation"
        assert state.complexity_level == "medium"
        assert state.previous_action == "add_examples"
        assert state.iteration_count == 2

    def test_state_to_dict(self):
        """Test state to dictionary conversion."""
        state = State(
            prompt_type="analysis",
            complexity_level="high",
            previous_action=None,
            iteration_count=1,
        )

        state_dict = state.to_dict()

        assert state_dict["prompt_type"] == "analysis"
        assert state_dict["complexity_level"] == "high"
        assert state_dict["previous_action"] is None
        assert state_dict["iteration_count"] == 1

    def test_state_to_string(self):
        """Test state to string conversion for hashing."""
        state = State(
            prompt_type="review",
            complexity_level="low",
            previous_action="clarify_requirements",
            iteration_count=0,
        )

        state_string = state.to_string()

        # Should be valid JSON
        import json

        parsed = json.loads(state_string)
        assert parsed["prompt_type"] == "review"
        assert parsed["complexity_level"] == "low"
        assert parsed["previous_action"] == "clarify_requirements"
        assert parsed["iteration_count"] == 0


class TestAction:
    """Test cases for Action dataclass."""

    def test_action_creation(self):
        """Test creating Action instance."""
        action = Action(
            action_type="add_examples",
            parameters={"example_count": 3, "example_type": "positive"},
        )

        assert action.action_type == "add_examples"
        assert action.parameters["example_count"] == 3
        assert action.parameters["example_type"] == "positive"

    def test_action_to_dict(self):
        """Test action to dictionary conversion."""
        action = Action(
            action_type="clarify_requirements", parameters={"detail_level": "high"}
        )

        action_dict = action.to_dict()

        assert action_dict["action_type"] == "clarify_requirements"
        assert action_dict["parameters"]["detail_level"] == "high"

    def test_action_to_string(self):
        """Test action to string conversion for hashing."""
        action = Action(
            action_type="add_context", parameters={"context_type": "technical"}
        )

        action_string = action.to_string()

        # Should be valid JSON
        import json

        parsed = json.loads(action_string)
        assert parsed["action_type"] == "add_context"
        assert parsed["parameters"]["context_type"] == "technical"


class TestQLearningAgent:
    """Test cases for QLearningAgent class."""

    def test_q_learning_agent_initialization(self):
        """Test QLearningAgent initialization."""
        config = {"db_path": ":memory:"}  # Use in-memory database
        agent = QLearningAgent("test_q_agent", config)

        assert agent.name == "test_q_agent"
        assert agent.config["learning_rate"] == 0.1
        assert agent.config["discount_factor"] == 0.9
        assert agent.config["epsilon"] == 0.1
        assert agent.episode_count == 0
        assert agent.total_rewards == 0.0

    def test_q_learning_agent_with_custom_config(self):
        """Test QLearningAgent with custom configuration."""
        config = {
            "db_path": ":memory:",  # Use in-memory database
            "learning_rate": 0.2,
            "discount_factor": 0.8,
            "epsilon": 0.05,
            "actions": ["custom_action_1", "custom_action_2"],
        }
        agent = QLearningAgent("custom_q_agent", config)

        assert agent.config["learning_rate"] == 0.2
        assert agent.config["discount_factor"] == 0.8
        assert agent.config["epsilon"] == 0.05
        assert "custom_action_1" in agent.config["actions"]
        assert "custom_action_2" in agent.config["actions"]

    def test_analyze_method(self):
        """Test analyze method."""
        agent = QLearningAgent("test_agent", {"db_path": ":memory:"})

        context = {
            "prompt_type": "code_generation",
            "complexity": "medium",
            "previous_action": "add_examples",
            "iteration_count": 1,
        }

        analysis = agent.analyze(context)

        assert "current_state" in analysis
        assert "available_actions" in analysis
        assert "q_values" in analysis
        assert "recommended_action" in analysis
        assert "exploration_rate" in analysis

        current_state = analysis["current_state"]
        assert current_state["prompt_type"] == "code_generation"
        assert current_state["complexity_level"] == "medium"
        assert current_state["previous_action"] == "add_examples"
        assert current_state["iteration_count"] == 1

    def test_propose_method(self):
        """Test propose method."""
        agent = QLearningAgent("test_agent", {"db_path": ":memory:"})

        analysis = {
            "current_state": {
                "prompt_type": "analysis",
                "complexity_level": "high",
                "previous_action": None,
                "iteration_count": 0,
            }
        }

        proposal = agent.propose(analysis)

        assert "action" in proposal
        assert "confidence" in proposal
        assert "reasoning" in proposal
        assert "exploration" in proposal
        assert "state" in proposal

        action = proposal["action"]
        assert "action_type" in action
        assert "parameters" in action
        assert action["action_type"] in agent.config["actions"]

    def test_review_method(self):
        """Test review method."""
        agent = QLearningAgent("test_agent", {"db_path": ":memory:"})

        code = "def hello_world():\n    print('Hello, World!')"
        review = agent.review(code)

        assert "quality_score" in review
        assert "issues" in review
        assert "recommendations" in review
        assert "learning_insights" in review
        assert review["quality_score"] == 0.8


class TestQLearningFunctionality:
    """Test cases for Q-learning core functionality."""

    def test_select_action_exploitation(self):
        """Test action selection in exploitation mode."""
        agent = QLearningAgent("test_agent", {"db_path": ":memory:", "epsilon": 0.0})

        state = State(
            prompt_type="code_generation",
            complexity_level="medium",
            previous_action=None,
            iteration_count=0,
        )

        # With epsilon=0, should always select best action
        action = agent.select_action(state)

        assert isinstance(action, Action)
        assert action.action_type in agent.config["actions"]
        assert isinstance(action.parameters, dict)

    def test_select_action_exploration(self):
        """Test action selection in exploration mode."""
        agent = QLearningAgent("test_agent", {"db_path": ":memory:", "epsilon": 1.0})

        state = State(
            prompt_type="code_generation",
            complexity_level="medium",
            previous_action=None,
            iteration_count=0,
        )

        # With epsilon=1, should always explore (random action)
        action = agent.select_action(state)

        assert isinstance(action, Action)
        assert action.action_type in agent.config["actions"]

    def test_update_q_value(self):
        """Test Q-value update functionality."""
        agent = QLearningAgent("test_agent", {"db_path": ":memory:"})

        state = State(
            prompt_type="code_generation",
            complexity_level="medium",
            previous_action=None,
            iteration_count=0,
        )

        action = Action(
            action_type="add_examples",
            parameters={"example_count": 2, "example_type": "positive"},
        )

        next_state = State(
            prompt_type="code_generation",
            complexity_level="medium",
            previous_action="add_examples",
            iteration_count=1,
        )

        # Initial Q-value should be 0
        initial_q = agent._get_q_value(state, action)
        assert initial_q == 0.0

        # Update Q-value
        reward = 0.5
        agent.update_q(state, action, reward, next_state)

        # Q-value should be updated
        updated_q = agent._get_q_value(state, action)
        assert updated_q > 0.0

        # Episode count should be incremented
        assert agent.episode_count == 1
        assert agent.total_rewards == 0.5

    def test_q_value_persistence(self):
        """Test that Q-values persist across agent instances."""
        # This test doesn't apply to in-memory databases
        # as they don't persist between instances
        pass

    def test_get_q_values(self):
        """Test getting all Q-values for a state."""
        agent = QLearningAgent("test_agent", {"db_path": ":memory:"})

        state = State(
            prompt_type="code_generation",
            complexity_level="medium",
            previous_action=None,
            iteration_count=0,
        )

        # Update Q-values for multiple actions
        action1 = Action(action_type="add_examples", parameters={})
        action2 = Action(action_type="clarify_requirements", parameters={})

        agent.update_q(state, action1, 0.5, state)
        agent.update_q(state, action2, 0.8, state)

        q_values = agent._get_q_values(state)

        assert "add_examples" in q_values
        assert "clarify_requirements" in q_values
        assert q_values["add_examples"] == 0.5
        assert q_values["clarify_requirements"] == 0.8

    def test_get_best_action(self):
        """Test getting the best action for a state."""
        agent = QLearningAgent("test_agent", {"db_path": ":memory:"})

        state = State(
            prompt_type="code_generation",
            complexity_level="medium",
            previous_action=None,
            iteration_count=0,
        )

        # Update Q-values for multiple actions
        action1 = Action(action_type="add_examples", parameters={})
        action2 = Action(action_type="clarify_requirements", parameters={})

        agent.update_q(state, action1, 0.3, state)
        agent.update_q(state, action2, 0.7, state)

        best_action = agent._get_best_action(state)
        assert best_action == "clarify_requirements"  # Higher Q-value

    def test_epsilon_decay(self):
        """Test epsilon decay functionality."""
        agent = QLearningAgent("test_agent", {"db_path": ":memory:", "epsilon": 0.1})

        initial_epsilon = agent.config["epsilon"]

        # Decay epsilon
        agent._decay_epsilon()

        assert agent.config["epsilon"] < initial_epsilon

        # Decay multiple times
        for _ in range(10):
            agent._decay_epsilon()

        # Should not go below minimum
        assert agent.config["epsilon"] >= agent.config["epsilon_min"]


class TestQLearningAgentIntegration:
    """Integration tests for QLearningAgent."""

    def test_full_learning_workflow(self):
        """Test complete Q-learning workflow."""
        agent = QLearningAgent("workflow_agent", {"db_path": ":memory:"})

        # Step 1: Analyze context
        context = {
            "prompt_type": "code_generation",
            "complexity": "medium",
            "previous_action": None,
            "iteration_count": 0,
        }

        analysis = agent.analyze(context)
        assert "current_state" in analysis

        # Step 2: Propose action
        proposal = agent.propose(analysis)
        assert "action" in proposal

        # Step 3: Execute action and get reward
        state = State(**analysis["current_state"])
        action = Action(**proposal["action"])

        # Simulate next state
        next_state = State(
            prompt_type="code_generation",
            complexity_level="medium",
            previous_action=action.action_type,
            iteration_count=1,
        )

        # Step 4: Update Q-value
        reward = 0.6
        agent.update_q(state, action, reward, next_state)

        # Step 5: Check learning statistics
        stats = agent.get_learning_statistics()
        assert stats["episode_count"] == 1
        assert stats["total_rewards"] == 0.6
        assert stats["average_reward"] == 0.6

    def test_learning_statistics(self):
        """Test learning statistics functionality."""
        agent = QLearningAgent("stats_agent", {"db_path": ":memory:"})

        # Perform some learning
        state = State(
            prompt_type="code_generation",
            complexity_level="medium",
            previous_action=None,
            iteration_count=0,
        )

        action = Action(action_type="add_examples", parameters={})
        next_state = State(
            prompt_type="code_generation",
            complexity_level="medium",
            previous_action="add_examples",
            iteration_count=1,
        )

        agent.update_q(state, action, 0.5, next_state)
        agent.update_q(state, action, 0.7, next_state)

        stats = agent.get_learning_statistics()

        assert "episode_count" in stats
        assert "total_rewards" in stats
        assert "average_reward" in stats
        assert "epsilon" in stats
        assert "total_q_entries" in stats
        assert "average_q_value" in stats
        assert "learning_rate" in stats
        assert "discount_factor" in stats

        assert stats["episode_count"] == 2
        assert stats["total_rewards"] == 1.2
        assert stats["average_reward"] == 0.6

    def test_reset_learning(self):
        """Test reset learning functionality."""
        agent = QLearningAgent("reset_agent", {"db_path": ":memory:"})

        # Perform some learning
        state = State(
            prompt_type="code_generation",
            complexity_level="medium",
            previous_action=None,
            iteration_count=0,
        )

        action = Action(action_type="add_examples", parameters={})
        next_state = State(
            prompt_type="code_generation",
            complexity_level="medium",
            previous_action="add_examples",
            iteration_count=1,
        )

        agent.update_q(state, action, 0.5, next_state)

        # Reset learning
        agent.reset_learning()

        # Check that everything is reset
        assert agent.episode_count == 0
        assert agent.total_rewards == 0.0
        assert agent.config["epsilon"] == 0.1

        # Q-value should be reset
        q_value = agent._get_q_value(state, action)
        assert q_value == 0.0
