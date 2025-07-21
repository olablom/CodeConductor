"""
Unit tests for CodeGenAgent
"""

import unittest
from unittest.mock import MagicMock
import sys
import os

# Add the parent directory to the path to import the agents module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.codegen_agent import CodeGenAgent, FastAPIGenerator


class TestCodeGenAgent(unittest.TestCase):
    """Test cases for CodeGenAgent"""

    def setUp(self):
        """Set up test fixtures"""
        self.agent = CodeGenAgent("TestCodeGenAgent")
        self.sample_context = {
            "title": "Test API",
            "summary": "A test REST API",
            "task_type": "api",
            "language": "python",
            "requirements": "Create a REST API endpoint",
            "complexity": "moderate",
        }

    def test_init(self):
        """Test CodeGenAgent initialization"""
        agent = CodeGenAgent("TestAgent")
        self.assertEqual(agent.name, "TestAgent")
        self.assertIsNotNone(agent.config)
        self.assertIsNone(agent.llm)
        self.assertIsNotNone(agent.fallback_generator)

    def test_init_with_config(self):
        """Test CodeGenAgent initialization with config"""
        config = {"preferred_languages": ["python"], "code_style": "clean"}
        agent = CodeGenAgent("TestAgent", config)
        self.assertIn("python", agent.config["preferred_languages"])
        self.assertEqual(agent.config["code_style"], "clean")

    def test_set_llm_client(self):
        """Test setting LLM client"""
        mock_llm = MagicMock()
        self.agent.set_llm_client(mock_llm)
        self.assertEqual(self.agent.llm, mock_llm)

    def test_analyze_basic(self):
        """Test basic analysis functionality"""
        result = self.agent.analyze(self.sample_context)

        self.assertIn("task_type", result)
        self.assertIn("language", result)
        self.assertIn("complexity_level", result)
        self.assertIn("key_requirements", result)
        self.assertIn("recommended_patterns", result)
        self.assertIn("potential_challenges", result)
        self.assertIn("estimated_complexity", result)
        self.assertIn("recommended_approach", result)

    def test_propose_with_llm_success(self):
        """Test propose with successful LLM generation"""
        # Mock LLM client
        mock_llm = MagicMock()
        mock_llm.complete.return_value = """
def test_function():
    \"\"\"Test function for API.\"\"\"
    return {"message": "test", "status": "success"}

def api_endpoint():
    \"\"\"API endpoint for testing.\"\"\"
    return test_function()
"""
        self.agent.set_llm_client(mock_llm)

        analysis = {
            "task_type": "api",
            "language": "python",
            "complexity_level": "moderate",
            "key_requirements": ["Create API endpoint"],
        }

        result = self.agent.propose(analysis, self.sample_context)

        self.assertIn("agent", result)
        self.assertIn("code", result)
        self.assertIn("confidence", result)
        self.assertIn("language", result)
        self.assertIn("title", result)
        self.assertIn("description", result)
        self.assertEqual(result["agent"], "TestCodeGenAgent")
        # Should use LLM code since it's longer than 100 chars
        self.assertIn("def test_function", result["code"])
        self.assertGreater(result["confidence"], 0.8)

    def test_propose_with_llm_empty_response(self):
        """Test propose when LLM returns empty response"""
        # Mock LLM client returning empty response
        mock_llm = MagicMock()
        mock_llm.complete.return_value = ""
        self.agent.set_llm_client(mock_llm)

        analysis = {
            "task_type": "api",
            "language": "python",
            "complexity_level": "moderate",
            "key_requirements": ["Create API endpoint"],
        }

        result = self.agent.propose(analysis, self.sample_context)

        self.assertIn("code", result)
        self.assertIn("confidence", result)
        # Should use fallback generator
        self.assertIn("FastAPI", result["code"])
        self.assertGreater(result["confidence"], 0.6)

    def test_propose_with_llm_error(self):
        """Test propose when LLM throws error"""
        # Mock LLM client throwing error
        mock_llm = MagicMock()
        mock_llm.complete.side_effect = Exception("LLM error")
        self.agent.set_llm_client(mock_llm)

        analysis = {
            "task_type": "api",
            "language": "python",
            "complexity_level": "moderate",
            "key_requirements": ["Create API endpoint"],
        }

        result = self.agent.propose(analysis, self.sample_context)

        self.assertIn("code", result)
        self.assertIn("confidence", result)
        # Should use fallback generator
        self.assertIn("FastAPI", result["code"])
        self.assertGreater(result["confidence"], 0.6)

    def test_propose_without_llm(self):
        """Test propose without LLM client (fallback only)"""
        analysis = {
            "task_type": "api",
            "language": "python",
            "complexity_level": "moderate",
            "key_requirements": ["Create API endpoint"],
        }

        result = self.agent.propose(analysis, self.sample_context)

        self.assertIn("code", result)
        self.assertIn("confidence", result)
        # Should use fallback generator
        self.assertIn("FastAPI", result["code"])
        self.assertGreater(result["confidence"], 0.6)

    def test_propose_basic_python_template(self):
        """Test propose for non-API application (basic Python template)"""
        context = {
            "title": "Data Processor",
            "summary": "A data processing application",
            "task_type": "data_processing",
            "language": "python",
            "requirements": "Process data",
            "complexity": "simple",
        }

        analysis = {
            "task_type": "data_processing",
            "language": "python",
            "complexity_level": "simple",
            "key_requirements": ["Process data"],
        }

        result = self.agent.propose(analysis, context)

        self.assertIn("code", result)
        self.assertIn("confidence", result)
        # Should use basic Python template
        self.assertIn("def main():", result["code"])
        self.assertIn("Data Processor", result["code"])
        self.assertGreater(result["confidence"], 0.5)

    def test_build_code_generation_prompt(self):
        """Test prompt building"""
        analysis = {
            "language": "python",
            "complexity_level": "moderate",
            "key_requirements": ["Create API", "Handle errors"],
        }

        prompt = self.agent._build_code_generation_prompt(analysis, self.sample_context)

        self.assertIn("Generate a complete, working python application", prompt)
        self.assertIn("Test API", prompt)
        self.assertIn("A test REST API", prompt)
        self.assertIn("- Create API", prompt)
        self.assertIn("- Handle errors", prompt)

    def test_generate_basic_python_template(self):
        """Test basic Python template generation"""
        title = "Test App"
        description = "A test application"

        code = self.agent._generate_basic_python_template(title, description)

        self.assertIn("Test App", code)
        self.assertIn("A test application", code)
        self.assertIn("def main():", code)
        self.assertIn("def process_data():", code)
        self.assertIn('if __name__ == "__main__":', code)

    def test_review_basic(self):
        """Test basic review functionality"""
        proposal = {
            "code": "def test_function(): return 'test'",
            "confidence": 0.8,
            "language": "python",
        }

        result = self.agent.review(proposal, self.sample_context)

        self.assertIn("quality_score", result)
        self.assertIn("issues", result)
        self.assertIn("suggestions", result)
        self.assertIn("best_practices", result)
        self.assertIn("performance_analysis", result)
        self.assertIn("security_analysis", result)
        self.assertIn("readability_score", result)
        self.assertIn("maintainability_score", result)

    def test_extract_requirements(self):
        """Test requirements extraction"""
        requirements = "Create a REST API with authentication"
        result = self.agent._extract_requirements(requirements)

        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test_extract_requirements_dict(self):
        """Test requirements extraction from dict"""
        requirements = {"functionality": "REST API", "security": "authentication"}
        result = self.agent._extract_requirements(requirements)

        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test_identify_patterns(self):
        """Test pattern identification"""
        patterns = self.agent._identify_patterns("api", "python")

        self.assertIsInstance(patterns, list)
        self.assertGreater(len(patterns), 0)

    def test_identify_challenges(self):
        """Test challenge identification"""
        challenges = self.agent._identify_challenges("Create async API", "python")

        self.assertIsInstance(challenges, list)

    def test_estimate_complexity(self):
        """Test complexity estimation"""
        complexity = self.agent._estimate_complexity("Simple function", "python")

        self.assertIn(complexity, ["low", "medium", "high", "simple"])

    def test_recommend_approach(self):
        """Test approach recommendation"""
        approach = self.agent._recommend_approach("api", "python", "moderate")

        self.assertIsInstance(approach, str)
        self.assertGreater(len(approach), 0)


class TestFastAPIGenerator(unittest.TestCase):
    """Test cases for FastAPIGenerator"""

    def setUp(self):
        """Set up test fixtures"""
        self.generator = FastAPIGenerator()

    def test_generate_basic(self):
        """Test basic FastAPI generation"""
        title = "Test API"
        description = "A test REST API"

        code = self.generator.generate(title, description)

        self.assertIn("Test API", code)
        self.assertIn("A test REST API", code)
        self.assertIn("from fastapi import FastAPI", code)
        self.assertIn("app = FastAPI", code)
        self.assertIn("class Item(BaseModel):", code)
        self.assertIn('@app.get("/")', code)
        self.assertIn('@app.post("/items", response_model=Item)', code)
        self.assertIn("uvicorn.run", code)

    def test_generate_with_special_characters(self):
        """Test generation with special characters in title/description"""
        title = "API with 'quotes' & symbols"
        description = 'Description with "quotes" and special chars: @#$%'

        code = self.generator.generate(title, description)

        self.assertIn("API with 'quotes' & symbols", code)
        self.assertIn('Description with "quotes" and special chars: @#$%', code)


if __name__ == "__main__":
    unittest.main()
