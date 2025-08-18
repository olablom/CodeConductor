"""
Prompt Generation Components

Handles conversion of ensemble consensus to structured prompts for Cursor.
"""

from .prompt_generator import PromptContext, PromptGenerator
from .simple_prompt_generator import SimplePromptGenerator

__all__ = ["PromptGenerator", "PromptContext", "SimplePromptGenerator"]
