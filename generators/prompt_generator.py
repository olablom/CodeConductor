"""
Prompt Generator for CodeConductor

Converts ensemble consensus to structured prompts for Cursor.
"""

import json
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TaskType(Enum):
    FUNCTION = "function"
    CLASS = "class"
    API = "api"
    TEST = "test"
    UTILITY = "utility"
    UNKNOWN = "unknown"


@dataclass
class PromptTemplate:
    """Template for generating structured prompts."""

    name: str
    task_type: TaskType
    template: str
    required_fields: List[str]
    optional_fields: List[str]


class PromptGenerator:
    """Generates structured prompts from ensemble consensus."""

    def __init__(self):
        self.templates = self._load_templates()
        self.context_manager = ContextManager()

    def _load_templates(self) -> Dict[str, PromptTemplate]:
        """Load prompt templates for different task types."""
        return {
            "function": PromptTemplate(
                name="Function Implementation",
                task_type=TaskType.FUNCTION,
                template="""## Task: {task_description}

### Approach
{approach}

### Requirements
- **Function Name**: {function_name}
- **Parameters**: {parameters}
- **Return Type**: {return_type}
- **Dependencies**: {dependencies}

### Constraints
- Use existing patterns in codebase
- Include proper error handling
- Add type hints
- Write docstring

### Expected Files
- `{main_file}`: Main implementation
- `{test_file}`: Test suite

### Context
{context}

### Implementation Notes
{reasoning}""",
                required_fields=["task_description", "approach", "function_name"],
                optional_fields=[
                    "parameters",
                    "return_type",
                    "dependencies",
                    "context",
                    "reasoning",
                ],
            ),
            "class": PromptTemplate(
                name="Class Implementation",
                task_type=TaskType.CLASS,
                template="""## Task: {task_description}

### Approach
{approach}

### Requirements
- **Class Name**: {class_name}
- **Methods**: {methods}
- **Attributes**: {attributes}
- **Dependencies**: {dependencies}

### Constraints
- Follow existing class patterns
- Include proper error handling
- Add type hints and docstrings
- Write comprehensive tests

### Expected Files
- `{main_file}`: Class implementation
- `{test_file}`: Test suite

### Context
{context}

### Implementation Notes
{reasoning}""",
                required_fields=["task_description", "approach", "class_name"],
                optional_fields=[
                    "methods",
                    "attributes",
                    "dependencies",
                    "context",
                    "reasoning",
                ],
            ),
            "api": PromptTemplate(
                name="API Implementation",
                task_type=TaskType.API,
                template="""## Task: {task_description}

### Approach
{approach}

### Requirements
- **Endpoint**: {endpoint}
- **Method**: {method}
- **Parameters**: {parameters}
- **Response Format**: {response_format}
- **Dependencies**: {dependencies}

### Constraints
- Follow RESTful conventions
- Include proper error handling
- Add input validation
- Write integration tests

### Expected Files
- `{main_file}`: API implementation
- `{test_file}`: Test suite
- `{config_file}`: Configuration (if needed)

### Context
{context}

### Implementation Notes
{reasoning}""",
                required_fields=["task_description", "approach", "endpoint"],
                optional_fields=[
                    "method",
                    "parameters",
                    "response_format",
                    "dependencies",
                    "context",
                    "reasoning",
                ],
            ),
            "test": PromptTemplate(
                name="Test Implementation",
                task_type=TaskType.TEST,
                template="""## Task: {task_description}

### Approach
{approach}

### Requirements
- **Test Target**: {test_target}
- **Test Cases**: {test_cases}
- **Framework**: {framework}
- **Dependencies**: {dependencies}

### Constraints
- Achieve 90%+ code coverage
- Include edge cases
- Use descriptive test names
- Mock external dependencies

### Expected Files
- `{test_file}`: Test suite
- `{fixture_file}`: Test fixtures (if needed)

### Context
{context}

### Implementation Notes
{reasoning}""",
                required_fields=["task_description", "approach", "test_target"],
                optional_fields=[
                    "test_cases",
                    "framework",
                    "dependencies",
                    "context",
                    "reasoning",
                ],
            ),
            "utility": PromptTemplate(
                name="Utility Implementation",
                task_type=TaskType.UTILITY,
                template="""## Task: {task_description}

### Approach
{approach}

### Requirements
- **Utility Name**: {utility_name}
- **Functionality**: {functionality}
- **Dependencies**: {dependencies}

### Constraints
- Keep it simple and focused
- Include proper error handling
- Add type hints
- Write tests

### Expected Files
- `{main_file}`: Utility implementation
- `{test_file}`: Test suite

### Context
{context}

### Implementation Notes
{reasoning}""",
                required_fields=["task_description", "approach", "utility_name"],
                optional_fields=[
                    "functionality",
                    "dependencies",
                    "context",
                    "reasoning",
                ],
            ),
        }

    def generate_prompt(
        self,
        consensus: Dict[str, Any],
        task_description: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate structured prompt from ensemble consensus."""
        try:
            # Determine task type
            task_type = self._classify_task(task_description, consensus)

            # Get appropriate template
            template = self.templates.get(task_type.value, self.templates["function"])

            # Extract and validate fields
            template_data = self._extract_template_data(
                consensus, task_description, template
            )

            # Inject context if provided
            if context:
                template_data["context"] = self.context_manager.format_context(context)

            # Generate prompt
            prompt = template.template.format(**template_data)

            # Validate prompt length
            if len(prompt) > 4000:  # Cursor context limit
                prompt = self._truncate_prompt(prompt, 4000)

            logger.info(f"Generated prompt for {task_type.value} task")
            return prompt

        except Exception as e:
            logger.error(f"Failed to generate prompt: {e}")
            return self._generate_fallback_prompt(task_description, consensus)

    def _classify_task(
        self, task_description: str, consensus: Dict[str, Any]
    ) -> TaskType:
        """Classify task type based on description and consensus."""
        description_lower = task_description.lower()

        # Check for specific keywords
        if any(
            word in description_lower
            for word in ["test", "spec", "assert", "pytest", "unittest"]
        ):
            return TaskType.TEST
        elif any(word in description_lower for word in ["class", "object", "model"]):
            return TaskType.CLASS
        elif any(
            word in description_lower
            for word in ["api", "endpoint", "route", "http", "rest"]
        ):
            return TaskType.API
        elif any(word in description_lower for word in ["utility", "helper", "tool"]):
            return TaskType.UTILITY
        elif any(
            word in description_lower
            for word in ["function", "def ", "calculate", "compute"]
        ):
            return TaskType.FUNCTION

        # Check consensus approach
        approach = consensus.get("approach", "").lower()
        if "test" in approach or "pytest" in approach:
            return TaskType.TEST
        elif "class" in approach:
            return TaskType.CLASS
        elif "api" in approach or "endpoint" in approach or "rest" in approach:
            return TaskType.API
        elif "utility" in approach:
            return TaskType.UTILITY

        return TaskType.FUNCTION  # Default

    def _extract_template_data(
        self, consensus: Dict[str, Any], task_description: str, template: PromptTemplate
    ) -> Dict[str, str]:
        """Extract and format data for template."""
        data = {
            "task_description": task_description,
            "approach": consensus.get("approach", "Standard implementation approach"),
            "reasoning": consensus.get(
                "reasoning", "Follow best practices and existing patterns"
            ),
            "dependencies": self._format_dependencies(
                consensus.get("dependencies", [])
            ),
            "context": "Use existing project structure and coding standards",
        }

        # Extract specific fields based on task type
        if template.task_type == TaskType.FUNCTION:
            data.update(self._extract_function_data(consensus, task_description))
        elif template.task_type == TaskType.CLASS:
            data.update(self._extract_class_data(consensus, task_description))
        elif template.task_type == TaskType.API:
            data.update(self._extract_api_data(consensus, task_description))
        elif template.task_type == TaskType.TEST:
            data.update(self._extract_test_data(consensus, task_description))
        elif template.task_type == TaskType.UTILITY:
            data.update(self._extract_utility_data(consensus, task_description))

        return data

    def _extract_function_data(
        self, consensus: Dict[str, Any], task_description: str
    ) -> Dict[str, str]:
        """Extract function-specific data."""
        files_needed = consensus.get("files_needed", ["main.py"])
        main_file = files_needed[0] if files_needed else "main.py"

        return {
            "function_name": self._extract_function_name(consensus, task_description),
            "parameters": self._extract_parameters(consensus, task_description),
            "return_type": self._extract_return_type(consensus),
            "main_file": main_file,
            "test_file": f"test_{main_file.replace('.py', '')}.py",
        }

    def _extract_class_data(
        self, consensus: Dict[str, Any], task_description: str
    ) -> Dict[str, str]:
        """Extract class-specific data."""
        files_needed = consensus.get("files_needed", ["main.py"])
        main_file = files_needed[0] if files_needed else "main.py"

        return {
            "class_name": self._extract_class_name(consensus, task_description),
            "methods": self._extract_methods(consensus),
            "attributes": self._extract_attributes(consensus),
            "main_file": main_file,
            "test_file": f"test_{main_file.replace('.py', '')}.py",
        }

    def _extract_api_data(
        self, consensus: Dict[str, Any], task_description: str
    ) -> Dict[str, str]:
        """Extract API-specific data."""
        files_needed = consensus.get("files_needed", ["api.py"])
        main_file = files_needed[0] if files_needed else "api.py"

        return {
            "endpoint": self._extract_endpoint(consensus, task_description),
            "method": "GET",  # Default
            "parameters": self._extract_parameters(consensus, task_description),
            "response_format": "JSON",
            "main_file": main_file,
            "test_file": f"test_{main_file.replace('.py', '')}.py",
            "config_file": "config.py",
        }

    def _extract_test_data(
        self, consensus: Dict[str, Any], task_description: str
    ) -> Dict[str, str]:
        """Extract test-specific data."""
        files_needed = consensus.get("files_needed", ["test_main.py"])
        test_file = files_needed[0] if files_needed else "test_main.py"

        return {
            "test_target": self._extract_test_target(consensus, task_description),
            "test_cases": self._extract_test_cases(consensus),
            "framework": "pytest",
            "test_file": test_file,
            "fixture_file": "conftest.py",
        }

    def _extract_utility_data(
        self, consensus: Dict[str, Any], task_description: str
    ) -> Dict[str, str]:
        """Extract utility-specific data."""
        files_needed = consensus.get("files_needed", ["utils.py"])
        main_file = files_needed[0] if files_needed else "utils.py"

        return {
            "utility_name": self._extract_utility_name(consensus, task_description),
            "functionality": consensus.get("approach", "Utility function"),
            "main_file": main_file,
            "test_file": f"test_{main_file.replace('.py', '')}.py",
        }

    def _extract_function_name(
        self, consensus: Dict[str, Any], task_description: str
    ) -> str:
        """Extract function name from consensus."""
        approach = consensus.get("approach", "")

        # Look for specific function names in task description FIRST
        if "factorial" in task_description.lower():
            return "factorial"
        elif "calculate" in task_description.lower():
            # Extract the thing being calculated
            match = re.search(
                r"calculate\s+(?:the\s+)?(\w+)", task_description, re.IGNORECASE
            )
            if match:
                return f"calculate_{match.group(1)}"

        # Look for common patterns in approach
        if "function" in approach.lower():
            # Extract name after "function" or "def"
            match = re.search(r"(?:function|def)\s+(\w+)", approach, re.IGNORECASE)
            if match:
                return match.group(1)

        # Default based on approach
        if "recursive" in approach.lower():
            return "recursive_function"
        elif "iterative" in approach.lower():
            return "iterative_function"

        return "main_function"

    def _extract_parameters(
        self, consensus: Dict[str, Any], task_description: str
    ) -> str:
        """Extract parameters from consensus."""
        approach = consensus.get("approach", "")

        # Look for specific parameter patterns FIRST
        if "factorial" in task_description.lower():
            return "n: int"
        elif "calculate" in task_description.lower():
            # Extract what's being calculated
            if "number" in task_description.lower():
                return "number: int"
            elif "string" in task_description.lower():
                return "text: str"
            else:
                return "value: Any"

        # Look for parameter patterns in approach
        params = re.findall(r"(\w+):\s*\w+", approach)
        if params:
            return ", ".join(params)

        # Default based on task type
        return "param: Any"

    def _extract_class_name(
        self, consensus: Dict[str, Any], task_description: str
    ) -> str:
        """Extract class name from consensus."""
        approach = consensus.get("approach", "")

        # Look for class name in task description
        if "calculator" in task_description.lower():
            return "Calculator"
        elif "class" in task_description.lower():
            # Extract class name after "class"
            match = re.search(r"class\s+(\w+)", task_description, re.IGNORECASE)
            if match:
                return match.group(1)

        # Look in approach
        match = re.search(r"class\s+(\w+)", approach, re.IGNORECASE)
        if match:
            return match.group(1)

        return "MainClass"

    def _extract_endpoint(
        self, consensus: Dict[str, Any], task_description: str
    ) -> str:
        """Extract API endpoint from consensus."""
        approach = consensus.get("approach", "")

        # Look for endpoint in task description
        if "authentication" in task_description.lower():
            return "/auth/login"
        elif "user" in task_description.lower():
            return "/api/users"

        # Look in approach
        match = re.search(r"endpoint[:\s]+([/\w]+)", approach, re.IGNORECASE)
        if match:
            return match.group(1)

        return "/api/endpoint"

    def _extract_return_type(self, consensus: Dict[str, Any]) -> str:
        """Extract return type from consensus."""
        approach = consensus.get("approach", "")
        if "int" in approach.lower():
            return "int"
        elif "str" in approach.lower():
            return "str"
        elif "list" in approach.lower():
            return "List"
        elif "dict" in approach.lower():
            return "Dict"
        return "Any"

    def _extract_methods(self, consensus: Dict[str, Any]) -> str:
        """Extract methods from consensus."""
        return "__init__, main_method"

    def _extract_attributes(self, consensus: Dict[str, Any]) -> str:
        """Extract attributes from consensus."""
        return "attribute1, attribute2"

    def _extract_test_target(
        self, consensus: Dict[str, Any], task_description: str
    ) -> str:
        """Extract test target from consensus."""
        # Look for what's being tested in task description
        if "factorial" in task_description.lower():
            return "factorial"
        elif "function" in task_description.lower():
            # Extract function name
            match = re.search(
                r"test.*?(\w+)\s+function", task_description, re.IGNORECASE
            )
            if match:
                return match.group(1)

        files_needed = consensus.get("files_needed", [])
        if files_needed:
            return files_needed[0].replace(".py", "").replace("test_", "")
        return "main_module"

    def _extract_test_cases(self, consensus: Dict[str, Any]) -> str:
        """Extract test cases from consensus."""
        return "Basic functionality, Edge cases, Error handling"

    def _extract_utility_name(
        self, consensus: Dict[str, Any], task_description: str
    ) -> str:
        """Extract utility name from consensus."""
        approach = consensus.get("approach", "")

        # Look for utility name in task description
        if "utility" in task_description.lower():
            match = re.search(r"(\w+)\s+utility", task_description, re.IGNORECASE)
            if match:
                return match.group(1)

        # Look in approach
        match = re.search(r"utility\s+(\w+)", approach, re.IGNORECASE)
        if match:
            return match.group(1)

        return "utility_function"

    def _format_dependencies(self, dependencies: List[str]) -> str:
        """Format dependencies list."""
        if not dependencies:
            return "None"
        return ", ".join(dependencies)

    def _truncate_prompt(self, prompt: str, max_length: int) -> str:
        """Truncate prompt to fit within limits."""
        if len(prompt) <= max_length:
            return prompt

        # Try to truncate from the middle, keeping start and end
        half_length = max_length // 2
        start = prompt[:half_length]
        end = prompt[-half_length:]

        return f"{start}\n\n... (truncated) ...\n\n{end}"

    def _generate_fallback_prompt(
        self, task_description: str, consensus: Dict[str, Any]
    ) -> str:
        """Generate a simple fallback prompt if template generation fails."""
        return f"""## Task: {task_description}

### Approach
{consensus.get("approach", "Implement according to requirements")}

### Requirements
- Implement the requested functionality
- Follow existing code patterns
- Include proper error handling
- Write tests

### Files Needed
{", ".join(consensus.get("files_needed", ["main.py"]))}

### Dependencies
{", ".join(consensus.get("dependencies", []))}

### Notes
{consensus.get("reasoning", "Follow best practices")}"""


class ContextManager:
    """Manages context injection for prompts."""

    def format_context(self, context: Dict[str, Any]) -> str:
        """Format context for inclusion in prompt."""
        formatted = []

        if "project_structure" in context:
            formatted.append(f"**Project Structure**: {context['project_structure']}")

        if "coding_standards" in context:
            formatted.append(f"**Coding Standards**: {context['coding_standards']}")

        if "existing_patterns" in context:
            formatted.append(f"**Existing Patterns**: {context['existing_patterns']}")

        if "dependencies" in context:
            formatted.append(
                f"**Project Dependencies**: {', '.join(context['dependencies'])}"
            )

        if not formatted:
            return "Use existing project structure and coding standards"

        return "\n".join(formatted)
