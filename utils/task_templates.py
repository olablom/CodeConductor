"""
Task Templates for CodeConductor

This module provides predefined task templates for common use cases,
making it easy for users to get started with proven patterns.
"""

from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class TaskTemplate:
    """Represents a task template with metadata and examples."""

    name: str
    description: str
    category: str
    template: str
    difficulty: str
    estimated_time: str
    tags: List[str]
    example_output: str


class TaskTemplateLibrary:
    """Library of predefined task templates."""

    def __init__(self):
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, TaskTemplate]:
        """Load all available task templates."""
        return {
            "shopping_cart": TaskTemplate(
                name="Shopping Cart Class",
                description="Create a Python class for managing a shopping cart with add, remove, and calculate total methods",
                category="Object-Oriented Programming",
                template="Create a Python class for managing a shopping cart with add, remove, and calculate total methods",
                difficulty="Beginner",
                estimated_time="5-10 minutes",
                tags=["OOP", "Classes", "Data Structures"],
                example_output="ShoppingCart class with dictionary storage and validation",
            ),
            "rest_api": TaskTemplate(
                name="REST API with Authentication",
                description="Build a REST API for user authentication with JWT tokens and user management",
                category="Web Development",
                template="Build a REST API for user authentication with JWT tokens and user management",
                difficulty="Intermediate",
                estimated_time="15-20 minutes",
                tags=["FastAPI", "Authentication", "JWT", "REST"],
                example_output="FastAPI app with JWT authentication and user CRUD operations",
            ),
            "data_pipeline": TaskTemplate(
                name="Data Processing Pipeline",
                description="Create a data pipeline for processing CSV files with validation and error handling",
                category="Data Science",
                template="Create a data pipeline for processing CSV files with validation and error handling",
                difficulty="Intermediate",
                estimated_time="10-15 minutes",
                tags=["Pandas", "Data Processing", "CSV", "Validation"],
                example_output="Data pipeline with pandas, validation, and error handling",
            ),
            "cli_tool": TaskTemplate(
                name="CLI Tool with Arguments",
                description="Build a command-line interface tool with argument parsing and help documentation",
                category="CLI Development",
                template="Build a command-line interface tool with argument parsing and help documentation",
                difficulty="Beginner",
                estimated_time="8-12 minutes",
                tags=["CLI", "argparse", "Command Line"],
                example_output="CLI tool with argparse, help, and subcommands",
            ),
            "ml_model": TaskTemplate(
                name="Machine Learning Model API",
                description="Create a machine learning model API with preprocessing, prediction, and model persistence",
                category="Machine Learning",
                template="Create a machine learning model API with preprocessing, prediction, and model persistence",
                difficulty="Advanced",
                estimated_time="20-25 minutes",
                tags=["ML", "API", "Scikit-learn", "Model Persistence"],
                example_output="ML API with preprocessing pipeline and model serving",
            ),
            "database_orm": TaskTemplate(
                name="Database ORM with SQLAlchemy",
                description="Implement a database ORM using SQLAlchemy with models, relationships, and migrations",
                category="Database",
                template="Implement a database ORM using SQLAlchemy with models, relationships, and migrations",
                difficulty="Intermediate",
                estimated_time="15-20 minutes",
                tags=["SQLAlchemy", "ORM", "Database", "Models"],
                example_output="SQLAlchemy models with relationships and migration setup",
            ),
            "async_web_scraper": TaskTemplate(
                name="Async Web Scraper",
                description="Build an asynchronous web scraper with rate limiting and data extraction",
                category="Web Scraping",
                template="Build an asynchronous web scraper with rate limiting and data extraction",
                difficulty="Intermediate",
                estimated_time="12-18 minutes",
                tags=["Async", "Web Scraping", "aiohttp", "Rate Limiting"],
                example_output="Async web scraper with rate limiting and data parsing",
            ),
            "caching_system": TaskTemplate(
                name="Caching System with Redis",
                description="Implement a caching system using Redis with TTL, cache invalidation, and fallback",
                category="Performance",
                template="Implement a caching system using Redis with TTL, cache invalidation, and fallback",
                difficulty="Intermediate",
                estimated_time="10-15 minutes",
                tags=["Redis", "Caching", "Performance", "TTL"],
                example_output="Redis caching system with TTL and cache invalidation",
            ),
            "file_upload_api": TaskTemplate(
                name="File Upload API",
                description="Create a file upload API with validation, storage, and download functionality",
                category="File Handling",
                template="Create a file upload API with validation, storage, and download functionality",
                difficulty="Intermediate",
                estimated_time="12-18 minutes",
                tags=["File Upload", "API", "Validation", "Storage"],
                example_output="File upload API with validation and secure storage",
            ),
            "notification_system": TaskTemplate(
                name="Real-time Notification System",
                description="Build a real-time notification system with WebSocket support and message queuing",
                category="Real-time",
                template="Build a real-time notification system with WebSocket support and message queuing",
                difficulty="Advanced",
                estimated_time="20-25 minutes",
                tags=["WebSocket", "Real-time", "Notifications", "Message Queue"],
                example_output="Real-time notification system with WebSocket and queuing",
            ),
        }

    def get_all_templates(self) -> List[TaskTemplate]:
        """Get all available templates."""
        return list(self.templates.values())

    def get_templates_by_category(self, category: str) -> List[TaskTemplate]:
        """Get templates filtered by category."""
        return [t for t in self.templates.values() if t.category == category]

    def get_templates_by_difficulty(self, difficulty: str) -> List[TaskTemplate]:
        """Get templates filtered by difficulty."""
        return [t for t in self.templates.values() if t.difficulty == difficulty]

    def get_template_by_name(self, name: str) -> TaskTemplate:
        """Get a specific template by name."""
        return self.templates.get(name)

    def search_templates(self, query: str) -> List[TaskTemplate]:
        """Search templates by name, description, or tags."""
        query = query.lower()
        results = []

        for template in self.templates.values():
            if (
                query in template.name.lower()
                or query in template.description.lower()
                or any(query in tag.lower() for tag in template.tags)
            ):
                results.append(template)

        return results

    def get_categories(self) -> List[str]:
        """Get all available categories."""
        return list(set(t.category for t in self.templates.values()))

    def get_difficulties(self) -> List[str]:
        """Get all available difficulties."""
        return list(set(t.difficulty for t in self.templates.values()))


# Global template library instance
template_library = TaskTemplateLibrary()
