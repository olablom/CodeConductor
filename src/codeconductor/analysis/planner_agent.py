"""
Planner Agent for CodeConductor
Uses Tree-sitter analysis + Local LLM for intelligent development planning
"""

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Any

# Import our Tree-sitter analyzer
from .tree_sitter_analyzer import FastAPITreeSitterAnalyzer


@dataclass
class DevelopmentPlan:
    """Represents a development plan created by the Planner Agent"""

    task_description: str
    steps: list[dict[str, Any]]
    context_used: dict[str, Any]
    cursor_prompts: list[str]
    estimated_complexity: str  # 'simple', 'medium', 'complex'
    dependencies: list[str]
    validation_criteria: list[str]


class PlannerAgent:
    """
    Intelligent planning agent that uses local LLM for reasoning
    and Tree-sitter for code understanding
    """

    def __init__(self, project_path: str, local_model: str = "mistral"):
        self.project_path = project_path
        self.local_model = local_model
        self.code_analyzer = FastAPITreeSitterAnalyzer()
        self.project_context = self._load_project_context()

    def _load_project_context(self) -> dict[str, Any]:
        """Load complete project context using Tree-sitter"""
        print(f"🔍 Loading project context from {self.project_path}...")

        # Get Tree-sitter analysis
        analysis = self.code_analyzer.analyze_project(self.project_path)

        # Get FastAPI routes
        routes = self.code_analyzer.extract_fastapi_routes(self.project_path)

        # Build comprehensive context
        context = {
            "statistics": analysis["statistics"],
            "routes": routes,
            "functions": [e for e in analysis["elements"] if e.type == "function"],
            "classes": [e for e in analysis["elements"] if e.type == "class"],
            "methods": [e for e in analysis["elements"] if e.type == "method"],
            "file_structure": self._get_file_structure(),
            "dependencies": self._extract_dependencies(),
            "patterns": self._identify_code_patterns(analysis),
        }

        return context

    def _get_file_structure(self) -> dict[str, list[str]]:
        """Get organized file structure"""
        structure = {
            "routers": [],
            "models": [],
            "services": [],
            "utils": [],
            "tests": [],
        }

        for root, dirs, files in os.walk(self.project_path):
            for file in files:
                if file.endswith(".py"):
                    rel_path = os.path.relpath(os.path.join(root, file), self.project_path)

                    if "routers" in rel_path:
                        structure["routers"].append(rel_path)
                    elif "models" in rel_path:
                        structure["models"].append(rel_path)
                    elif "services" in rel_path:
                        structure["services"].append(rel_path)
                    elif "test" in rel_path:
                        structure["tests"].append(rel_path)
                    else:
                        structure["utils"].append(rel_path)

        return structure

    def _extract_dependencies(self) -> dict[str, list[str]]:
        """Extract project dependencies from requirements.txt or pyproject.toml"""
        dependencies = {"fastapi": [], "database": [], "testing": [], "other": []}

        req_file = os.path.join(self.project_path, "requirements.txt")
        if os.path.exists(req_file):
            with open(req_file) as f:
                for line in f:
                    dep = line.strip().lower()
                    if "fastapi" in dep or "pydantic" in dep:
                        dependencies["fastapi"].append(line.strip())
                    elif "sqlalchemy" in dep or "postgres" in dep:
                        dependencies["database"].append(line.strip())
                    elif "pytest" in dep or "test" in dep:
                        dependencies["testing"].append(line.strip())
                    else:
                        dependencies["other"].append(line.strip())

        return dependencies

    def _identify_code_patterns(self, analysis: dict[str, Any]) -> dict[str, Any]:
        """Identify common patterns in the codebase"""
        patterns = {
            "authentication": False,
            "crud_operations": False,
            "pagination": False,
            "error_handling": False,
            "async_patterns": False,
        }

        # Check for patterns based on function/class names and decorators
        for element in analysis["elements"]:
            if element.type in ["function", "method"]:
                name_lower = element.name.lower()

                # Auth patterns
                if any(auth in name_lower for auth in ["login", "auth", "token", "jwt"]):
                    patterns["authentication"] = True

                # CRUD patterns
                if any(
                    crud in name_lower
                    for crud in [
                        "create",
                        "read",
                        "update",
                        "delete",
                        "get",
                        "post",
                        "put",
                    ]
                ):
                    patterns["crud_operations"] = True

                # Async patterns
                if "async" in str(element.metadata.get("decorators", [])):
                    patterns["async_patterns"] = True

        return patterns

    def create_development_plan(self, task: str) -> DevelopmentPlan:
        """
        Create an intelligent development plan using local LLM reasoning
        """
        print(f"\n🧠 Creating development plan for: {task}")

        # Step 1: Analyze task complexity
        complexity = self._analyze_task_complexity(task)

        # Step 2: Use local LLM for reasoning
        reasoning_prompt = self._build_reasoning_prompt(task)
        reasoning_result = self._local_llm_reasoning(reasoning_prompt)

        # Step 3: Generate step-by-step plan
        steps = self._generate_development_steps(task, reasoning_result)

        # Step 4: Create optimized Cursor prompts
        cursor_prompts = self._generate_cursor_prompts(steps)

        # Step 5: Define validation criteria
        validation_criteria = self._define_validation_criteria(task, steps)

        # Step 6: Identify dependencies
        dependencies = self._identify_task_dependencies(task)

        return DevelopmentPlan(
            task_description=task,
            steps=steps,
            context_used=self.project_context,
            cursor_prompts=cursor_prompts,
            estimated_complexity=complexity,
            dependencies=dependencies,
            validation_criteria=validation_criteria,
        )

    def _analyze_task_complexity(self, task: str) -> str:
        """Analyze task complexity based on keywords and scope"""
        task_lower = task.lower()

        # Complex indicators
        if any(
            word in task_lower
            for word in ["refactor", "migrate", "redesign", "integrate", "optimization"]
        ):
            return "complex"

        # Medium indicators
        if any(word in task_lower for word in ["add", "implement", "create", "update", "modify"]):
            return "medium"

        # Simple indicators
        if any(word in task_lower for word in ["fix", "change", "rename", "move", "typo"]):
            return "simple"

        return "medium"  # Default

    def _build_reasoning_prompt(self, task: str) -> str:
        """Build a comprehensive prompt for local LLM reasoning"""
        prompt = f"""Task: {task}

Project Context:
- Total Files: {self.project_context["statistics"]["total_files"]}
- Languages: {self.project_context["statistics"]["languages"]}
- Total Functions: {self.project_context["statistics"]["total_functions"]}
- Total Classes: {self.project_context["statistics"]["total_classes"]}
- FastAPI Routes: {len(self.project_context["routes"])}

Current Routes:
{self._format_routes_for_prompt()}

Code Patterns Detected:
{json.dumps(self.project_context["patterns"], indent=2)}

File Structure:
{self._format_file_structure_for_prompt()}

Based on this context, create a step-by-step plan to accomplish the task.
Consider:
1. What files need to be modified?
2. What new files need to be created?
3. What existing patterns should be followed?
4. What tests need to be updated?
5. What are potential risks or dependencies?

Provide a detailed plan with specific implementation steps.
"""
        return prompt

    def _format_routes_for_prompt(self) -> str:
        """Format routes for LLM context"""
        routes_str = ""
        for route in self.project_context["routes"][:10]:  # Limit to 10 for context size
            routes_str += f"- {route['method']} {route['path']} -> {route['function']}()\n"

        if len(self.project_context["routes"]) > 10:
            routes_str += f"... and {len(self.project_context['routes']) - 10} more routes\n"

        return routes_str

    def _format_file_structure_for_prompt(self) -> str:
        """Format file structure for LLM context"""
        structure_str = ""
        for category, files in self.project_context["file_structure"].items():
            if files:
                structure_str += f"\n{category.capitalize()}:\n"
                for file in files[:5]:  # Limit files per category
                    structure_str += f"  - {file}\n"
                if len(files) > 5:
                    structure_str += f"  ... and {len(files) - 5} more files\n"

        return structure_str

    def _local_llm_reasoning(self, prompt: str) -> str:
        """
        Use local LLM (via Ollama) for reasoning
        Falls back to mock response if Ollama not available
        """
        try:
            # Try to use Ollama with proper encoding
            result = subprocess.run(
                ["ollama", "run", self.local_model, prompt],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=30,
            )

            if result.returncode == 0:
                return result.stdout
            else:
                print(f"⚠️ Ollama error: {result.stderr}")
                return self._mock_llm_reasoning(prompt)

        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("⚠️ Ollama not available, using mock reasoning")
            return self._mock_llm_reasoning(prompt)

    def _mock_llm_reasoning(self, prompt: str) -> str:
        """Mock LLM reasoning for when Ollama is not available"""
        # Extract task from prompt
        task_line = prompt.split("\n")[0].replace("Task: ", "")

        # Generate mock reasoning based on task keywords
        if "cache" in task_line.lower():
            return """Development Plan:

1. Install Redis dependencies:
   - Add redis and fastapi-cache2 to requirements.txt

2. Create cache configuration:
   - Create utils/cache.py with Redis connection setup
   - Add cache decorator functions

3. Apply caching to GET endpoints:
   - Import cache decorator in each router file
   - Add @cache decorator to GET methods
   - Set appropriate TTL values

4. Update main.py:
   - Initialize Redis connection on startup
   - Add cache middleware

5. Testing:
   - Add cache tests
   - Verify cache invalidation logic"""

        elif "admin" in task_line.lower():
            return """Development Plan:

1. Create admin router:
   - Create routers/admin.py
   - Add admin-only authentication dependency

2. Implement admin endpoints:
   - GET /api/admin/users - List all users
   - GET /api/admin/stats - System statistics
   - POST /api/admin/users/{id}/ban - Ban user

3. Add admin authentication:
   - Create is_admin dependency
   - Check user role in JWT token

4. Update main.py:
   - Import and include admin router

5. Add tests for admin functionality"""

        else:
            return f"Generic plan for: {task_line}"

    def _generate_development_steps(self, task: str, reasoning: str) -> list[dict[str, Any]]:
        """Generate structured development steps from reasoning"""
        steps = []

        # Parse reasoning into steps (simple parsing for now)
        lines = reasoning.split("\n")
        current_step = None
        step_number = 1

        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):  # Step indicator
                if current_step and current_step["description"]:
                    steps.append(current_step)

                current_step = {
                    "number": step_number,
                    "description": line.lstrip("0123456789.- "),
                    "files_affected": [],
                    "estimated_time": "15 minutes",
                }
                step_number += 1

        if current_step and current_step["description"]:
            steps.append(current_step)

        # If no steps were parsed, create generic steps
        if not steps:
            steps = [
                {
                    "number": 1,
                    "description": f"Implement {task}",
                    "files_affected": [],
                    "estimated_time": "30 minutes",
                }
            ]

        return steps

    def _generate_cursor_prompts(self, steps: list[dict[str, Any]]) -> list[str]:
        """Generate optimized prompts for Cursor based on development steps"""
        prompts = []

        for step in steps:
            # Build context-aware prompt for each step
            prompt = f"""Implement the following in the existing FastAPI project:

Task: {step["description"]}

Project Structure:
- Authentication: JWT-based auth in routers/auth.py
- User Management: CRUD operations in routers/users.py
- Posts Management: Blog posts with ownership in routers/posts.py
- Database: Mock in-memory database (will be PostgreSQL later)

Current Implementation Details:
- OAuth2PasswordBearer for JWT tokens
- Pydantic models for validation
- Proper error handling with HTTPException
- Pagination support on list endpoints

Specific Requirements:
1. Follow the existing code patterns and style
2. Use appropriate type hints
3. Include proper error handling
4. Add docstrings to new functions
5. Maintain consistency with existing endpoints

{self._add_file_specific_context(step)}

Generate clean, production-ready code that integrates seamlessly with the existing codebase."""

            prompts.append(prompt)

        return prompts

    def _add_file_specific_context(self, step: dict[str, Any]) -> str:
        """Add file-specific context to prompt based on step"""
        context = ""

        # Add relevant code examples based on task
        if "cache" in step["description"].lower():
            context += "\nNote: The project currently has no caching. Implement Redis-based caching using fastapi-cache2."

        elif "admin" in step["description"].lower():
            context += "\nNote: Create a new admin router following the pattern of existing routers (auth.py, users.py, posts.py)."

        return context

    def _define_validation_criteria(self, task: str, steps: list[dict[str, Any]]) -> list[str]:
        """Define criteria to validate successful implementation"""
        criteria = [
            "Code follows existing project patterns",
            "All new endpoints are properly documented",
            "Type hints are used consistently",
            "Error handling is implemented",
            "No breaking changes to existing functionality",
        ]

        # Add task-specific criteria
        if "cache" in task.lower():
            criteria.extend(
                [
                    "Redis connection is properly configured",
                    "Cache invalidation logic is implemented",
                    "TTL values are appropriate for data type",
                ]
            )

        elif "admin" in task.lower():
            criteria.extend(
                [
                    "Admin authentication is properly secured",
                    "Admin endpoints check user permissions",
                    "Admin actions are logged/auditable",
                ]
            )

        return criteria

    def _identify_task_dependencies(self, task: str) -> list[str]:
        """Identify dependencies required for the task"""
        dependencies = []

        task_lower = task.lower()

        if "cache" in task_lower or "redis" in task_lower:
            dependencies.extend(["redis", "fastapi-cache2", "aioredis"])

        if "database" in task_lower or "postgres" in task_lower:
            dependencies.extend(["psycopg2-binary", "alembic"])

        if "test" in task_lower:
            dependencies.extend(["pytest", "pytest-asyncio", "httpx"])

        if "monitoring" in task_lower:
            dependencies.extend(["prometheus-client", "opentelemetry-api"])

        return dependencies

    def display_plan(self, plan: DevelopmentPlan):
        """Display the development plan in a readable format"""
        print(f"\n{'=' * 60}")
        print(f"📋 Development Plan: {plan.task_description}")
        print(f"{'=' * 60}")

        print(f"\n🎯 Complexity: {plan.estimated_complexity.upper()}")
        print(f"📁 Files in project: {plan.context_used['statistics']['total_files']}")
        print(f"🚀 Current routes: {len(plan.context_used['routes'])}")

        if plan.dependencies:
            print("\n📦 Dependencies needed:")
            for dep in plan.dependencies:
                print(f"   - {dep}")

        print("\n📝 Implementation Steps:")
        for step in plan.steps:
            print(f"\n{step['number']}. {step['description']}")
            print(f"   ⏱️  Estimated time: {step['estimated_time']}")

        print("\n✅ Validation Criteria:")
        for criteria in plan.validation_criteria:
            print(f"   - {criteria}")

        print(f"\n🤖 Cursor Prompts Generated: {len(plan.cursor_prompts)}")
        print("\nFirst prompt preview:")
        print("-" * 40)
        if plan.cursor_prompts:
            print(plan.cursor_prompts[0][:500] + "...")
        print("-" * 40)


# Example usage
if __name__ == "__main__":
    # Initialize planner for test project
    planner = PlannerAgent("test_fastapi_project")

    # Example tasks
    tasks = [
        "Add Redis caching to all GET endpoints",
        "Create admin router with user management endpoints",
        "Implement rate limiting on API endpoints",
    ]

    for task in tasks[:1]:  # Test with first task
        print(f"\n{'🚀' * 30}")
        plan = planner.create_development_plan(task)
        planner.display_plan(plan)
