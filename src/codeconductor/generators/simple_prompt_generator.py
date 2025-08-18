#!/usr/bin/env python3
"""
Simple Prompt Generator for CodeConductor
Generates prompts for basic coding tasks without assuming specific project context
"""

import logging

logger = logging.getLogger(__name__)


class SimplePromptGenerator:
    """
    Simple prompt generator for basic coding tasks
    Doesn't assume FastAPI or any specific project structure
    """

    def __init__(self):
        self.templates = {
            "class": self._class_template,
            "function": self._function_template,
            "script": self._script_template,
            "api": self._api_template,
        }

    def generate_prompt(self, task: str, task_type: str = "class") -> str:
        """
        Generate a simple, focused prompt for the given task

        Args:
            task: The task description
            task_type: Type of task (class, function, script, api)

        Returns:
            Generated prompt string
        """
        logger.info(f"🎯 Generating simple prompt for task: {task[:50]}...")

        # Determine task type if not specified
        if task_type == "auto":
            task_type = self._detect_task_type(task)

        # Get appropriate template
        template_func = self.templates.get(task_type, self._class_template)

        # Generate prompt
        prompt = template_func(task)

        logger.info(f"✅ Generated prompt ({len(prompt)} chars)")
        return prompt

    def _detect_task_type(self, task: str) -> str:
        """Detect the type of task based on keywords"""
        task_lower = task.lower()

        if any(word in task_lower for word in ["class", "todo", "list", "manager"]):
            return "class"
        elif any(word in task_lower for word in ["function", "def", "method"]):
            return "function"
        elif any(word in task_lower for word in ["api", "endpoint", "route", "fastapi", "flask"]):
            return "api"
        elif any(word in task_lower for word in ["script", "main", "run"]):
            return "script"
        else:
            return "class"  # Default

    def _class_template(self, task: str) -> str:
        """Template for class creation tasks"""
        return f"""Create a Python class that implements the following requirements:

{task}

Requirements:
- Include proper type hints
- Add comprehensive docstrings
- Include error handling where appropriate
- Follow PEP 8 style guidelines
- Make the code production-ready

Please provide only the class implementation without any additional imports or setup code."""

    def _function_template(self, task: str) -> str:
        """Template for function creation tasks"""
        return f"""Create a Python function that implements the following requirements:

{task}

Requirements:
- Include proper type hints
- Add comprehensive docstrings
- Include error handling where appropriate
- Follow PEP 8 style guidelines
- Make the code production-ready

Please provide only the function implementation without any additional imports or setup code."""

    def _script_template(self, task: str) -> str:
        """Template for script creation tasks"""
        return f"""Create a Python script that implements the following requirements:

{task}

Requirements:
- Include proper error handling
- Add comments explaining key sections
- Follow PEP 8 style guidelines
- Make the script executable and production-ready
- Include a main function if appropriate

Please provide the complete script implementation."""

    def _api_template(self, task: str) -> str:
        """Template for API creation tasks"""
        return f"""Create a Python API implementation that satisfies the following requirements:

{task}

Requirements:
- Use FastAPI or Flask (specify which)
- Include proper error handling
- Add input validation
- Include comprehensive docstrings
- Follow REST API best practices
- Make the code production-ready

Please provide the complete API implementation including necessary imports."""

    def generate_multiple_prompts(self, task: str, count: int = 3) -> list[str]:
        """
        Generate multiple variations of prompts for the same task

        Args:
            task: The task description
            count: Number of prompt variations to generate

        Returns:
            List of generated prompts
        """
        prompts = []

        # Generate different approaches
        approaches = [
            ("class", "Create a simple, clean implementation"),
            ("class", "Create a robust implementation with error handling"),
            (
                "class",
                "Create an optimized implementation with performance considerations",
            ),
        ]

        for _i, (_task_type, approach) in enumerate(approaches[:count]):
            base_prompt = self._class_template(task)
            enhanced_prompt = f"{base_prompt}\n\nApproach: {approach}"
            prompts.append(enhanced_prompt)

        return prompts


# Convenience function
def generate_simple_prompt(task: str, task_type: str = "auto") -> str:
    """Convenience function to generate a simple prompt"""
    generator = SimplePromptGenerator()
    return generator.generate_prompt(task, task_type)
