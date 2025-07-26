#!/usr/bin/env python3
"""
Prompt Generator for CodeConductor MVP
Converts ensemble consensus to structured prompts for Cursor.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, Template
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class PromptContext:
    """Context information for prompt generation."""

    project_structure: str = ""
    coding_standards: List[str] = None
    existing_patterns: List[str] = None
    dependencies: List[str] = None
    max_tokens: int = 4000

    def __post_init__(self):
        if self.coding_standards is None:
            self.coding_standards = [
                "Use type hints",
                "Include docstrings",
                "Follow PEP 8",
                "Handle errors gracefully",
            ]
        if self.existing_patterns is None:
            self.existing_patterns = []
        if self.dependencies is None:
            self.dependencies = []


class PromptGenerator:
    """Generates structured prompts from ensemble consensus."""

    def __init__(self, template_dir: str = "generators/templates"):
        self.template_dir = Path(template_dir)
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self._ensure_template_dir()

    def _ensure_template_dir(self):
        """Ensure template directory exists with default templates."""
        self.template_dir.mkdir(parents=True, exist_ok=True)

        # Create default template if it doesn't exist
        default_template = self.template_dir / "prompt.md.j2"
        if not default_template.exists():
            self._create_default_template(default_template)

    def _create_default_template(self, template_path: Path):
        """Create the default Jinja2 template."""
        template_content = """## Task: {{ consensus.task | default("Implement requested functionality") }}

### Approach
{{ consensus.approach | default("Follow standard Python patterns and best practices") }}

### Requirements
{% if consensus.requirements %}
{% for req in consensus.requirements %}
- {{ req }}
{% endfor %}
{% else %}
- Implement the requested functionality
- Include proper error handling
- Write comprehensive tests
{% endif %}

{% if consensus.files_needed %}
### Expected Files
{% for file in consensus.files_needed %}
- `{{ file }}`: {{ file.split('.')[-1] | upper }} implementation
{% endfor %}
{% endif %}

{% if consensus.dependencies %}
### Dependencies
{% for dep in consensus.dependencies %}
- {{ dep }}
{% endfor %}
{% endif %}

### Constraints
{% for standard in context.coding_standards %}
- {{ standard }}
{% endfor %}

{% if context.existing_patterns %}
### Existing Patterns
{% for pattern in context.existing_patterns %}
- {{ pattern }}
{% endfor %}
{% endif %}

{% if context.project_structure %}
### Project Context
{{ context.project_structure }}
{% endif %}

### Output Format
Please provide the code in the following format:

```{{ consensus.language | default("python") }}
# Your implementation here
```

```test_{{ consensus.test_file | default("test_implementation") }}.py
# Your test cases here
```

Make sure all tests pass and the code follows the specified standards.
"""

        template_path.write_text(template_content)
        logger.info(f"✅ Created default template: {template_path}")

    def generate(
        self, consensus: Dict[str, Any], context: Optional[PromptContext] = None
    ) -> str:
        """
        Generate a structured prompt from ensemble consensus.

        Args:
            consensus: Consensus result from ensemble engine
            context: Optional context information

        Returns:
            Generated prompt string
        """
        logger.info("🎯 Generating prompt from consensus...")

        # Validate consensus
        self._validate_consensus(consensus)

        # Use default context if none provided
        if context is None:
            context = PromptContext()

        # Prepare template data
        template_data = {"consensus": consensus, "context": context}

        try:
            # Load and render template
            template = self.env.get_template("prompt.md.j2")
            prompt = template.render(**template_data)

            # Validate prompt length
            self._validate_prompt_length(prompt, context.max_tokens)

            logger.info(f"✅ Generated prompt ({len(prompt)} chars)")
            return prompt

        except Exception as e:
            logger.error(f"❌ Failed to generate prompt: {e}")
            return self._generate_fallback_prompt(consensus, context)

    def _validate_consensus(self, consensus: Dict[str, Any]):
        """Validate that consensus contains required fields."""
        required_fields = ["task", "approach"]

        for field in required_fields:
            if field not in consensus:
                logger.warning(f"⚠️ Missing required field: {field}")
                consensus[field] = f"Default {field}"

        # Ensure requirements is a list
        if "requirements" not in consensus:
            consensus["requirements"] = ["Implement requested functionality"]
        elif not isinstance(consensus["requirements"], list):
            consensus["requirements"] = [str(consensus["requirements"])]

    def _validate_prompt_length(self, prompt: str, max_tokens: int):
        """Validate that prompt doesn't exceed token limits."""
        # Rough estimation: 1 token ≈ 4 characters
        estimated_tokens = len(prompt) // 4

        if estimated_tokens > max_tokens:
            logger.warning(
                f"⚠️ Prompt may exceed token limit: {estimated_tokens} > {max_tokens}"
            )

    def _generate_fallback_prompt(
        self, consensus: Dict[str, Any], context: PromptContext
    ) -> str:
        """Generate a simple fallback prompt if template fails."""
        logger.info("🔄 Using fallback prompt generation...")

        prompt_parts = [
            f"## Task: {consensus.get('task', 'Implement functionality')}",
            f"\n### Approach\n{consensus.get('approach', 'Follow Python best practices')}",
            "\n### Requirements",
        ]

        for req in consensus.get(
            "requirements", ["Implement the requested functionality"]
        ):
            prompt_parts.append(f"- {req}")

        prompt_parts.append("\n### Constraints")
        for standard in context.coding_standards:
            prompt_parts.append(f"- {standard}")

        prompt_parts.append("\nPlease provide the implementation and tests.")

        return "\n".join(prompt_parts)

    def generate_code_generation_prompt(
        self, task: str, files_needed: List[str] = None, dependencies: List[str] = None
    ) -> str:
        """
        Generate a prompt specifically for code generation tasks.

        Args:
            task: Description of what to implement
            files_needed: List of files to create/modify
            dependencies: Required dependencies

        Returns:
            Structured prompt for code generation
        """
        consensus = {
            "task": task,
            "approach": "Implement the requested functionality following Python best practices",
            "requirements": [
                "Create clean, well-documented code",
                "Include proper error handling",
                "Write comprehensive tests",
                "Follow PEP 8 standards",
            ],
            "files_needed": files_needed or [],
            "dependencies": dependencies or [],
            "language": "python",
        }

        return self.generate(consensus)

    def generate_test_prompt(self, implementation_file: str) -> str:
        """
        Generate a prompt for creating tests for existing code.

        Args:
            implementation_file: Path to the implementation file

        Returns:
            Structured prompt for test generation
        """
        consensus = {
            "task": f"Create comprehensive tests for {implementation_file}",
            "approach": "Analyze the implementation and create thorough test cases",
            "requirements": [
                "Test all public functions and methods",
                "Include edge cases and error conditions",
                "Achieve high test coverage",
                "Use pytest framework",
            ],
            "files_needed": [f"test_{implementation_file}"],
            "language": "python",
        }

        return self.generate(consensus)


# Convenience functions
def generate_prompt(
    consensus: Dict[str, Any], context: Optional[PromptContext] = None
) -> str:
    """Generate a prompt from consensus."""
    generator = PromptGenerator()
    return generator.generate(consensus, context)


def generate_code_prompt(task: str, files_needed: List[str] = None) -> str:
    """Generate a code generation prompt."""
    generator = PromptGenerator()
    return generator.generate_code_generation_prompt(task, files_needed)


async def main():
    """Demo function to test PromptGenerator."""
    print("🎯 CodeConductor Prompt Generator Demo")
    print("=" * 50)

    generator = PromptGenerator()

    # Test 1: Simple consensus
    print("\n📝 Test 1: Simple consensus...")
    consensus = {
        "task": "Create a simple calculator class",
        "approach": "Implement basic arithmetic operations with error handling",
        "requirements": [
            "Add, subtract, multiply, divide methods",
            "Handle division by zero",
            "Include type hints",
        ],
        "files_needed": ["calculator.py", "test_calculator.py"],
        "dependencies": ["pytest"],
    }

    prompt = generator.generate(consensus)
    print("✅ Generated prompt:")
    print("-" * 30)
    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)

    # Test 2: With context
    print("\n📝 Test 2: With context...")
    context = PromptContext(
        project_structure="Simple Python project with src/ and tests/ directories",
        coding_standards=[
            "Use dataclasses",
            "Include type hints",
            "Follow black formatting",
        ],
        existing_patterns=["Use dependency injection", "Implement repository pattern"],
    )

    prompt_with_context = generator.generate(consensus, context)
    print("✅ Generated prompt with context:")
    print("-" * 30)
    print(
        prompt_with_context[:500] + "..."
        if len(prompt_with_context) > 500
        else prompt_with_context
    )

    print(
        f"\n🎉 Demo completed! Generated {len(prompt)} and {len(prompt_with_context)} character prompts"
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
