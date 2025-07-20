"""
Agent Service - Core agent implementations

This module contains the core agent classes migrated from the main CodeConductor
codebase to the microservices architecture.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import json

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the CodeConductor system.

    All agents must implement the core methods: analyze(), propose(), and review().
    This ensures consistent interface across different agent types.
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base agent.

        Args:
            name: Unique identifier for this agent
            config: Configuration dictionary for agent behavior
        """
        self.name = name
        self.config = config or {}
        self.message_bus = None
        self.llm_client = None

        logger.info(f"Initialized agent: {self.name}")

    @abstractmethod
    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze context and return structured insights.

        This method should examine the input context and extract relevant
        information, patterns, and insights that will inform the proposal.

        Args:
            context: Dictionary containing context information
                     (e.g., requirements, existing code, constraints)

        Returns:
            Dictionary containing analysis results and insights
        """
        pass

    @abstractmethod
    def propose(
        self, analysis: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a proposal based on the analysis and context.

        This method should use the analysis results and original context to create a concrete
        proposal for how to proceed (e.g., code structure, architecture).

        Args:
            analysis: Dictionary containing analysis results from analyze()
            context: Dictionary containing original context information

        Returns:
            Dictionary containing the proposal and supporting information
        """
        pass

    @abstractmethod
    def review(
        self, proposal: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Review a proposal and provide feedback on quality and safety.

        This method should examine the proposal and provide feedback
        on various aspects like quality, security, performance, etc.

        Args:
            proposal: Dictionary containing the proposal to review
            context: Dictionary containing original context information

        Returns:
            Dictionary containing review results and recommendations
        """
        pass

    def set_message_bus(self, message_bus):
        """Set the message bus for inter-agent communication."""
        self.message_bus = message_bus
        logger.debug(f"Agent {self.name} connected to message bus")

    def set_llm_client(self, llm_client):
        """Set the LLM client for this agent."""
        self.llm_client = llm_client
        logger.debug(f"Agent {self.name} connected to LLM client")

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the agent."""
        return {
            "name": self.name,
            "config": self.config,
            "has_message_bus": self.message_bus is not None,
            "has_llm_client": self.llm_client is not None,
        }

    def __str__(self) -> str:
        """String representation of the agent."""
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __repr__(self) -> str:
        """Detailed string representation of the agent."""
        return f"{self.__class__.__name__}(name='{self.name}', config={self.config})"


class CodeGenAgent(BaseAgent):
    """
    Specialized agent for code generation, analysis, and optimization.

    This agent provides expertise in programming patterns, best practices,
    and code quality assessment.
    """

    def __init__(
        self, name: str = "codegen_agent", config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the CodeGenAgent."""
        super().__init__(name, config)

        # Default configuration
        self.default_config = {
            "max_code_length": 1000,
            "quality_threshold": 0.7,
            "supported_languages": ["python", "javascript", "typescript", "java"],
            "code_style": "pep8",
            "include_tests": True,
            "include_docs": True,
        }

        # Merge with provided config
        self.config = {**self.default_config, **(config or {})}

        logger.info(f"CodeGenAgent initialized with config: {self.config}")

    def set_llm_client(self, llm_client):
        """Set the LLM client for this agent."""
        super().set_llm_client(llm_client)
        logger.info(f"CodeGenAgent {self.name} connected to LLM client")

    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the task context and extract key insights.

        Args:
            context: Dictionary containing task information

        Returns:
            Dictionary with analysis results
        """
        logger.info(f"CodeGenAgent {self.name} analyzing context")

        task = context.get("task", "")
        requirements = context.get("requirements", [])
        language = context.get("language", "python")

        # Extract requirements
        extracted_requirements = self._extract_requirements(requirements)

        # Identify patterns
        patterns = self._identify_patterns(task, language)

        # Identify challenges
        challenges = self._identify_challenges(requirements, language)

        # Estimate complexity
        complexity = self._estimate_complexity(requirements, language)

        # Recommend approach
        approach = self._recommend_approach(task, language, complexity)

        analysis = {
            "task": task,
            "language": language,
            "requirements": extracted_requirements,
            "patterns": patterns,
            "challenges": challenges,
            "complexity": complexity,
            "recommended_approach": approach,
            "quality_focus": self._determine_quality_focus(context),
            "performance_needs": self._analyze_performance_needs(context),
            "security_needs": self._analyze_security_needs(context),
            "testing_strategy": self._recommend_testing_strategy(task, language),
            "documentation_needs": self._assess_documentation_needs(context),
        }

        logger.info(f"Analysis complete: {len(analysis)} insights extracted")
        return analysis

    def propose(
        self, analysis: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a code generation proposal based on analysis.

        Args:
            analysis: Results from analyze() method
            context: Original context

        Returns:
            Dictionary with proposal details
        """
        logger.info(f"CodeGenAgent {self.name} generating proposal")

        # Generate approach
        approach = self._generate_approach(analysis)

        # Design code structure
        structure = self._design_code_structure(analysis)

        # Create implementation plan
        implementation_plan = self._create_implementation_plan(analysis)

        # Generate code template
        code_template = self._generate_code_template(analysis)

        # Define quality guidelines
        quality_guidelines = self._define_quality_guidelines(analysis)

        # Create documentation plan
        documentation_plan = self._create_documentation_plan(analysis)

        # Estimate code size
        size_estimate = self._estimate_code_size(analysis)

        # Calculate confidence
        confidence = self._calculate_confidence(analysis)

        # Generate reasoning
        reasoning = self._generate_reasoning(analysis)

        # Generate suggestions
        suggestions = self._generate_suggestions(analysis)

        proposal = {
            "approach": approach,
            "structure": structure,
            "implementation_plan": implementation_plan,
            "code_template": code_template,
            "quality_guidelines": quality_guidelines,
            "documentation_plan": documentation_plan,
            "size_estimate": size_estimate,
            "confidence": confidence,
            "reasoning": reasoning,
            "suggestions": suggestions,
            "agent_name": self.name,
            "analysis_summary": {
                "complexity": analysis.get("complexity"),
                "language": analysis.get("language"),
                "requirements_count": len(analysis.get("requirements", [])),
            },
        }

        logger.info(f"Proposal generated with confidence: {confidence:.2f}")
        return proposal

    def review(
        self, proposal: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Review a code proposal and provide feedback.

        Args:
            proposal: The proposal to review
            context: Original context

        Returns:
            Dictionary with review results
        """
        logger.info(f"CodeGenAgent {self.name} reviewing proposal")

        code = proposal.get("code_template", "")

        # Assess code quality
        quality_score = self._assess_code_quality(code)

        # Identify issues
        issues = self._identify_issues(code)

        # Generate improvement suggestions
        improvements = self._generate_improvement_suggestions(code)

        # Check best practices
        best_practices = self._check_best_practices(code)

        # Analyze performance
        performance_notes = self._analyze_performance(code)

        # Analyze security
        security_notes = self._analyze_security(code)

        # Assess readability
        readability_score = self._assess_readability(code)

        # Assess maintainability
        maintainability_score = self._assess_maintainability(code)

        # Recommend test coverage
        test_recommendations = self._recommend_test_coverage(code)

        # Identify documentation gaps
        doc_gaps = self._identify_documentation_gaps(code)

        # Provide overall assessment
        overall_assessment = self._provide_overall_assessment(code)

        # Assess proposal quality
        proposal_quality = self._assess_proposal_quality(proposal)

        review = {
            "quality_score": quality_score,
            "issues": issues,
            "improvements": improvements,
            "best_practices": best_practices,
            "performance_notes": performance_notes,
            "security_notes": security_notes,
            "readability_score": readability_score,
            "maintainability_score": maintainability_score,
            "test_recommendations": test_recommendations,
            "documentation_gaps": doc_gaps,
            "overall_assessment": overall_assessment,
            "proposal_quality": proposal_quality,
            "agent_name": self.name,
            "review_summary": {
                "overall_score": (
                    quality_score + readability_score + maintainability_score
                )
                / 3,
                "issues_count": len(issues),
                "improvements_count": len(improvements),
            },
        }

        logger.info(
            f"Review complete: overall score {review['review_summary']['overall_score']:.2f}"
        )
        return review

    # Helper methods for analysis
    def _extract_requirements(self, requirements) -> List[str]:
        """Extract and normalize requirements."""
        if isinstance(requirements, str):
            return [req.strip() for req in requirements.split(",") if req.strip()]
        elif isinstance(requirements, list):
            return [str(req).strip() for req in requirements if req]
        return []

    def _identify_patterns(self, task_type: str, language: str) -> List[str]:
        """Identify relevant design patterns."""
        patterns = []

        if "api" in task_type.lower():
            patterns.extend(["REST API", "CRUD operations", "HTTP methods"])
        if "web" in task_type.lower():
            patterns.extend(["MVC", "Routing", "Middleware"])
        if "data" in task_type.lower():
            patterns.extend(["Data validation", "Error handling", "Logging"])

        if language == "python":
            patterns.extend(["Type hints", "Docstrings", "PEP 8"])
        elif language == "javascript":
            patterns.extend(["ES6+", "Async/await", "Error handling"])

        return patterns

    def _identify_challenges(self, requirements, language: str) -> List[str]:
        """Identify potential challenges."""
        challenges = []

        if any("validation" in str(req).lower() for req in requirements):
            challenges.append("Input validation and sanitization")
        if any("error" in str(req).lower() for req in requirements):
            challenges.append("Comprehensive error handling")
        if any("test" in str(req).lower() for req in requirements):
            challenges.append("Test coverage and quality")
        if any("security" in str(req).lower() for req in requirements):
            challenges.append("Security best practices")

        return challenges

    def _estimate_complexity(self, requirements, language: str) -> str:
        """Estimate task complexity."""
        req_count = len(self._extract_requirements(requirements))

        if req_count <= 3:
            return "low"
        elif req_count <= 7:
            return "medium"
        else:
            return "high"

    def _recommend_approach(
        self, task_type: str, language: str, complexity: str
    ) -> str:
        """Recommend development approach."""
        if complexity == "low":
            return "Direct implementation with basic structure"
        elif complexity == "medium":
            return "Modular design with clear separation of concerns"
        else:
            return "Architecture-first approach with detailed planning"

    def _determine_quality_focus(self, context: Dict[str, Any]) -> List[str]:
        """Determine quality focus areas."""
        focus = ["Code readability", "Error handling"]

        if context.get("performance_requirements"):
            focus.append("Performance optimization")
        if context.get("security_requirements"):
            focus.append("Security best practices")
        if context.get("test_requirements"):
            focus.append("Test coverage")

        return focus

    def _analyze_performance_needs(self, context: Dict[str, Any]) -> List[str]:
        """Analyze performance requirements."""
        needs = []

        if context.get("high_traffic"):
            needs.append("Caching strategy")
        if context.get("large_data"):
            needs.append("Efficient data processing")
        if context.get("real_time"):
            needs.append("Async processing")

        return needs

    def _analyze_security_needs(self, context: Dict[str, Any]) -> List[str]:
        """Analyze security requirements."""
        needs = []

        if context.get("user_input"):
            needs.append("Input validation")
        if context.get("authentication"):
            needs.append("Authentication/authorization")
        if context.get("sensitive_data"):
            needs.append("Data encryption")

        return needs

    def _recommend_testing_strategy(self, task_type: str, language: str) -> str:
        """Recommend testing strategy."""
        if "api" in task_type.lower():
            return "Unit tests + Integration tests + API tests"
        elif "web" in task_type.lower():
            return "Unit tests + E2E tests"
        else:
            return "Unit tests + Functional tests"

    def _assess_documentation_needs(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess documentation requirements."""
        return {
            "api_docs": "api" in context.get("task", "").lower(),
            "code_comments": True,
            "readme": True,
            "deployment_guide": context.get("deployment_required", False),
        }

    # Helper methods for proposal generation
    def _generate_approach(self, analysis: Dict[str, Any]) -> str:
        """Generate development approach."""
        complexity = analysis.get("complexity", "medium")
        language = analysis.get("language", "python")

        if complexity == "low":
            return f"Simple {language} implementation with basic structure"
        elif complexity == "medium":
            return f"Modular {language} design with clear separation of concerns"
        else:
            return f"Comprehensive {language} architecture with detailed planning"

    def _design_code_structure(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Design code structure."""
        language = analysis.get("language", "python")

        if language == "python":
            return {
                "main_file": "main.py",
                "modules": ["models.py", "services.py", "utils.py"],
                "tests": ["test_main.py", "test_services.py"],
                "docs": ["README.md", "API.md"],
            }
        else:
            return {
                "main_file": "index.js",
                "modules": ["models.js", "services.js", "utils.js"],
                "tests": ["test_main.js", "test_services.js"],
                "docs": ["README.md", "API.md"],
            }

    def _create_implementation_plan(self, analysis: Dict[str, Any]) -> List[str]:
        """Create implementation plan."""
        return [
            "Set up project structure",
            "Implement core functionality",
            "Add error handling",
            "Write tests",
            "Add documentation",
            "Code review and optimization",
        ]

    def _generate_code_template(self, analysis: Dict[str, Any]) -> str:
        """Generate code template."""
        language = analysis.get("language", "python")
        task = analysis.get("task", "")

        if language == "python" and "api" in task.lower():
            return self._generate_python_api_template()
        else:
            return self._generate_python_basic_template()

    def _generate_python_api_template(self) -> str:
        """Generate Python API template."""
        return '''"""
FastAPI Application Template

Generated by CodeConductor Agent Service
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

app = FastAPI(title="API Service", description="Generated API")

# Pydantic models
class Item(BaseModel):
    id: Optional[int] = None
    name: str
    description: Optional[str] = None

# In-memory storage
items_db = []
item_id_counter = 1

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "API Service", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "API Service"}

@app.get("/items", response_model=List[Item])
async def get_items():
    """Get all items."""
    return items_db

@app.post("/items", response_model=Item)
async def create_item(item: Item):
    """Create a new item."""
    global item_id_counter
    item_dict = item.dict()
    item_dict["id"] = item_id_counter
    item_id_counter += 1
    items_db.append(item_dict)
    return item_dict

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''

    def _generate_python_basic_template(self) -> str:
        """Generate basic Python template."""
        return '''"""
Python Application Template

Generated by CodeConductor Agent Service
"""

import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Application:
    """Main application class."""
    
    def __init__(self):
        """Initialize the application."""
        self.name = "Generated Application"
        logger.info(f"Initialized {self.name}")
    
    def run(self):
        """Run the application."""
        logger.info(f"Running {self.name}")
        return {"status": "success", "message": "Application running"}

def main():
    """Main function."""
    app = Application()
    result = app.run()
    print(f"Result: {result}")

if __name__ == "__main__":
    main()
'''

    def _define_quality_guidelines(self, analysis: Dict[str, Any]) -> List[str]:
        """Define quality guidelines."""
        language = analysis.get("language", "python")
        guidelines = []

        if language == "python":
            guidelines.extend(
                [
                    "Follow PEP 8 style guide",
                    "Use type hints",
                    "Write docstrings for all functions",
                    "Handle exceptions properly",
                    "Write unit tests",
                ]
            )
        else:
            guidelines.extend(
                [
                    "Follow language conventions",
                    "Use meaningful variable names",
                    "Handle errors properly",
                    "Write documentation",
                    "Write tests",
                ]
            )

        return guidelines

    def _create_documentation_plan(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create documentation plan."""
        return {
            "readme": "Project overview and setup instructions",
            "api_docs": "API documentation if applicable",
            "code_comments": "Inline code documentation",
            "deployment": "Deployment instructions if needed",
        }

    def _estimate_code_size(self, analysis: Dict[str, Any]) -> Dict[str, int]:
        """Estimate code size."""
        complexity = analysis.get("complexity", "medium")
        req_count = len(analysis.get("requirements", []))

        if complexity == "low":
            return {"lines": 100 + req_count * 20, "files": 3}
        elif complexity == "medium":
            return {"lines": 300 + req_count * 30, "files": 5}
        else:
            return {"lines": 600 + req_count * 40, "files": 8}

    def _calculate_confidence(self, analysis: Dict[str, Any]) -> float:
        """Calculate confidence score."""
        complexity = analysis.get("complexity", "medium")
        req_count = len(analysis.get("requirements", []))

        # Base confidence
        confidence = 0.8

        # Adjust for complexity
        if complexity == "low":
            confidence += 0.1
        elif complexity == "high":
            confidence -= 0.1

        # Adjust for requirements
        if req_count <= 5:
            confidence += 0.05
        elif req_count > 10:
            confidence -= 0.05

        return min(max(confidence, 0.0), 1.0)

    def _generate_reasoning(self, analysis: Dict[str, Any]) -> str:
        """Generate reasoning for the proposal."""
        complexity = analysis.get("complexity", "medium")
        language = analysis.get("language", "python")
        patterns = analysis.get("patterns", [])

        reasoning = f"Based on the analysis, this is a {complexity} complexity task "
        reasoning += f"requiring {language} implementation. "
        reasoning += f"Key patterns identified: {', '.join(patterns[:3])}. "
        reasoning += (
            "The proposed approach balances functionality with maintainability."
        )

        return reasoning

    def _generate_suggestions(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate improvement suggestions."""
        suggestions = []

        if analysis.get("complexity") == "high":
            suggestions.append("Consider breaking down into smaller modules")
        if len(analysis.get("requirements", [])) > 5:
            suggestions.append("Prioritize requirements by importance")
        if "api" in analysis.get("task", "").lower():
            suggestions.append("Include comprehensive API documentation")

        return suggestions

    # Helper methods for review
    def _assess_code_quality(self, code: str) -> float:
        """Assess overall code quality."""
        if not code:
            return 0.0

        score = 0.5  # Base score

        # Check for good practices
        if "def " in code:
            score += 0.1
        if "class " in code:
            score += 0.1
        if "import " in code:
            score += 0.1
        if "try:" in code:
            score += 0.1
        if "except" in code:
            score += 0.1
        if "return" in code:
            score += 0.1

        return min(score, 1.0)

    def _identify_issues(self, code: str) -> List[str]:
        """Identify potential issues in code."""
        issues = []

        if not code:
            return ["Empty code template"]

        if "TODO" in code:
            issues.append("Contains TODO comments")
        if "FIXME" in code:
            issues.append("Contains FIXME comments")
        if "print(" in code and "logging" not in code:
            issues.append("Uses print instead of logging")
        if "pass" in code:
            issues.append("Contains pass statements (may need implementation)")

        return issues

    def _generate_improvement_suggestions(self, code: str) -> List[str]:
        """Generate improvement suggestions."""
        suggestions = []

        if not code:
            return ["Add actual implementation"]

        if "logging" not in code:
            suggestions.append("Add proper logging")
        if "test" not in code.lower():
            suggestions.append("Add unit tests")
        if "docstring" not in code.lower():
            suggestions.append("Add docstrings")
        if "type" not in code.lower():
            suggestions.append("Add type hints")

        return suggestions

    def _check_best_practices(self, code: str) -> Dict[str, bool]:
        """Check adherence to best practices."""
        return {
            "has_functions": "def " in code,
            "has_classes": "class " in code,
            "has_imports": "import " in code,
            "has_error_handling": "try:" in code or "except" in code,
            "has_documentation": '"""' in code or "'''" in code,
        }

    def _analyze_performance(self, code: str) -> List[str]:
        """Analyze performance aspects."""
        notes = []

        if "for " in code and "range(" in code:
            notes.append("Uses efficient iteration")
        if "list(" in code and "map(" in code:
            notes.append("Uses functional programming patterns")
        if "async" in code:
            notes.append("Uses asynchronous programming")

        return notes

    def _analyze_security(self, code: str) -> List[str]:
        """Analyze security aspects."""
        notes = []

        if "input(" in code:
            notes.append("Uses user input - needs validation")
        if "eval(" in code:
            notes.append("Uses eval - security risk")
        if "exec(" in code:
            notes.append("Uses exec - security risk")

        return notes

    def _assess_readability(self, code: str) -> float:
        """Assess code readability."""
        if not code:
            return 0.0

        score = 0.5  # Base score

        # Check for readability factors
        lines = code.split("\n")
        avg_line_length = sum(len(line) for line in lines) / len(lines)

        if avg_line_length < 80:
            score += 0.2
        if "def " in code:
            score += 0.1
        if "class " in code:
            score += 0.1
        if '"""' in code:
            score += 0.1

        return min(score, 1.0)

    def _assess_maintainability(self, code: str) -> float:
        """Assess code maintainability."""
        if not code:
            return 0.0

        score = 0.5  # Base score

        # Check for maintainability factors
        if "def " in code:
            score += 0.2  # Functions improve maintainability
        if "class " in code:
            score += 0.1  # Classes improve structure
        if "import " in code:
            score += 0.1  # Modularity
        if '"""' in code:
            score += 0.1  # Documentation

        return min(score, 1.0)

    def _recommend_test_coverage(self, code: str) -> List[str]:
        """Recommend test coverage."""
        recommendations = []

        if "def " in code:
            recommendations.append("Unit tests for each function")
        if "class " in code:
            recommendations.append("Unit tests for each class method")
        if "api" in code.lower():
            recommendations.append("Integration tests for API endpoints")
        if "database" in code.lower():
            recommendations.append("Database integration tests")

        return recommendations

    def _identify_documentation_gaps(self, code: str) -> List[str]:
        """Identify documentation gaps."""
        gaps = []

        if "def " in code and '"""' not in code:
            gaps.append("Missing function docstrings")
        if "class " in code and '"""' not in code:
            gaps.append("Missing class docstrings")
        if "import " in code:
            gaps.append("Missing module documentation")

        return gaps

    def _provide_overall_assessment(self, code: str) -> str:
        """Provide overall assessment."""
        if not code:
            return "Empty code template - needs implementation"

        quality = self._assess_code_quality(code)
        readability = self._assess_readability(code)
        maintainability = self._assess_maintainability(code)

        avg_score = (quality + readability + maintainability) / 3

        if avg_score >= 0.8:
            return "Excellent code quality with good structure and documentation"
        elif avg_score >= 0.6:
            return "Good code quality with room for improvement"
        elif avg_score >= 0.4:
            return "Basic code structure that needs enhancement"
        else:
            return "Code needs significant improvement in structure and quality"

    def _assess_proposal_quality(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the quality of the proposal."""
        completeness = self._assess_proposal_completeness(proposal)
        feasibility = self._assess_proposal_feasibility(proposal)
        innovation = self._assess_proposal_innovation(proposal)

        return {
            "completeness": completeness,
            "feasibility": feasibility,
            "innovation": innovation,
            "overall": (completeness + feasibility + innovation) / 3,
        }

    def _assess_proposal_completeness(self, proposal: Dict[str, Any]) -> float:
        """Assess proposal completeness."""
        required_fields = [
            "approach",
            "structure",
            "implementation_plan",
            "code_template",
        ]
        present_fields = sum(1 for field in required_fields if field in proposal)
        return present_fields / len(required_fields)

    def _assess_proposal_feasibility(self, proposal: Dict[str, Any]) -> float:
        """Assess proposal feasibility."""
        # Simple heuristic based on complexity and confidence
        confidence = proposal.get("confidence", 0.5)
        complexity = proposal.get("analysis_summary", {}).get("complexity", "medium")

        feasibility = confidence

        if complexity == "high":
            feasibility *= 0.8
        elif complexity == "low":
            feasibility *= 1.1

        return min(feasibility, 1.0)

    def _assess_proposal_innovation(self, proposal: Dict[str, Any]) -> float:
        """Assess proposal innovation."""
        # Simple heuristic based on patterns and suggestions
        patterns = proposal.get("analysis_summary", {}).get("patterns", [])
        suggestions = proposal.get("suggestions", [])

        innovation = 0.5  # Base score

        if len(patterns) > 3:
            innovation += 0.2
        if len(suggestions) > 2:
            innovation += 0.2
        if "architecture" in proposal.get("approach", "").lower():
            innovation += 0.1

        return min(innovation, 1.0)


# Agent factory for creating different agent types
class AgentFactory:
    """Factory for creating different types of agents."""

    @staticmethod
    def create_agent(
        agent_type: str, name: str = None, config: Dict[str, Any] = None
    ) -> BaseAgent:
        """
        Create an agent of the specified type.

        Args:
            agent_type: Type of agent to create
            name: Agent name
            config: Agent configuration

        Returns:
            Agent instance
        """
        if agent_type == "codegen":
            return CodeGenAgent(name or "codegen_agent", config)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
