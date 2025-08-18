"""
Tree-sitter based Universal Code Analyzer for CodeConductor
Provides multi-language parsing with incremental updates and error recovery
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Tree-sitter imports
try:
    import tree_sitter
    import tree_sitter_javascript
    import tree_sitter_python

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    print(
        "Warning: tree-sitter not installed. Install with: pip install tree-sitter tree-sitter-python tree-sitter-javascript"
    )


@dataclass
class CodeElement:
    """Represents a parsed code element (function, class, etc.)"""

    type: str  # 'function', 'class', 'method', 'variable', etc.
    name: str
    line_start: int
    line_end: int
    file_path: str
    language: str
    metadata: dict[str, Any]  # Additional info like params, return type, etc.


class TreeSitterAnalyzer:
    """Universal code analyzer using Tree-sitter for multi-language support"""

    def __init__(self):
        if not TREE_SITTER_AVAILABLE:
            raise ImportError(
                "Tree-sitter is required. Install with: pip install tree-sitter tree-sitter-python"
            )

        # Initialize parsers for different languages
        self.parsers = {}
        self.languages = {}

        # Python parser
        PY_LANGUAGE = tree_sitter.Language(tree_sitter_python.language())
        py_parser = tree_sitter.Parser(PY_LANGUAGE)
        self.parsers["python"] = py_parser
        self.languages["python"] = PY_LANGUAGE

        # JavaScript parser
        JS_LANGUAGE = tree_sitter.Language(tree_sitter_javascript.language())
        js_parser = tree_sitter.Parser(JS_LANGUAGE)
        self.parsers["javascript"] = js_parser
        self.parsers["js"] = js_parser
        self.languages["javascript"] = JS_LANGUAGE

        # TypeScript parser (if available)
        try:
            import tree_sitter_typescript

            TS_LANGUAGE = tree_sitter.Language(tree_sitter_typescript.language_typescript())
            ts_parser = tree_sitter.Parser(TS_LANGUAGE)
            self.parsers["typescript"] = ts_parser
            self.parsers["ts"] = ts_parser
            self.languages["typescript"] = TS_LANGUAGE
        except:
            print("TypeScript parser not available")

    def analyze_file(self, file_path: str) -> list[CodeElement]:
        """Analyze a single file and extract code elements"""
        path = Path(file_path)
        if not path.exists():
            return []

        # Determine language from file extension
        language = self._detect_language(path)
        if not language or language not in self.parsers:
            return []

        # Read file content
        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return []

        # Parse with Tree-sitter
        parser = self.parsers[language]
        tree = parser.parse(bytes(content, "utf-8"))

        # Extract elements based on language
        elements = []
        if language == "python":
            elements.extend(self._extract_python_elements(tree, content, file_path))
        elif language in ["javascript", "js", "typescript", "ts"]:
            elements.extend(self._extract_javascript_elements(tree, content, file_path))

        return elements

    def analyze_project(self, project_path: str) -> dict[str, Any]:
        """Analyze entire project with Tree-sitter"""
        project_path = Path(project_path)

        all_elements = []
        file_count = 0
        language_stats = {}

        # Find all code files
        for file_path in self._find_code_files(project_path):
            file_count += 1
            language = self._detect_language(file_path)

            # Update language statistics
            if language:
                language_stats[language] = language_stats.get(language, 0) + 1

            # Analyze file
            elements = self.analyze_file(str(file_path))
            all_elements.extend(elements)

        # Build cross-references
        cross_refs = self._build_cross_references(all_elements)

        return {
            "elements": all_elements,
            "statistics": {
                "total_files": file_count,
                "languages": language_stats,
                "total_functions": len([e for e in all_elements if e.type == "function"]),
                "total_classes": len([e for e in all_elements if e.type == "class"]),
                "total_methods": len([e for e in all_elements if e.type == "method"]),
            },
            "cross_references": cross_refs,
        }

    def _detect_language(self, file_path: Path) -> str | None:
        """Detect programming language from file extension"""
        extension_map = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".mjs": "javascript",
            ".cjs": "javascript",
        }
        return extension_map.get(file_path.suffix.lower())

    def _find_code_files(self, project_path: Path) -> list[Path]:
        """Find all code files in project, excluding common non-code directories"""
        exclude_dirs = {
            "node_modules",
            "__pycache__",
            ".git",
            ".venv",
            "venv",
            "env",
            "dist",
            "build",
            ".next",
            ".pytest_cache",
            "coverage",
        }

        code_files = []
        for root, dirs, files in os.walk(project_path):
            # Remove excluded directories from search
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                file_path = Path(root) / file
                if self._detect_language(file_path):
                    code_files.append(file_path)

        return code_files

    def _extract_python_elements(self, tree, content: str, file_path: str) -> list[CodeElement]:
        """Extract Python code elements using Tree-sitter queries"""
        elements = []

        # Walk through the AST to find functions and classes
        def walk_node(node):
            if node.type == "function_definition":
                # Check if this is a method (inside a class)
                parent = node.parent
                is_method = False
                while parent:
                    if parent.type == "class_definition":
                        is_method = True
                        break
                    parent = parent.parent

                # Get function name
                for child in node.children:
                    if child.type == "identifier":
                        func_name = content[child.start_byte : child.end_byte]

                        if not is_method:
                            elements.append(
                                CodeElement(
                                    type="function",
                                    name=func_name,
                                    line_start=node.start_point[0] + 1,
                                    line_end=node.end_point[0] + 1,
                                    file_path=file_path,
                                    language="python",
                                    metadata={
                                        "decorators": self._extract_decorators(node, content)
                                    },
                                )
                            )
                        else:
                            elements.append(
                                CodeElement(
                                    type="method",
                                    name=func_name,
                                    line_start=node.start_point[0] + 1,
                                    line_end=node.end_point[0] + 1,
                                    file_path=file_path,
                                    language="python",
                                    metadata={
                                        "decorators": self._extract_decorators(node, content)
                                    },
                                )
                            )
                        break

            elif node.type == "class_definition":
                # Get class name
                for child in node.children:
                    if child.type == "identifier":
                        class_name = content[child.start_byte : child.end_byte]
                        elements.append(
                            CodeElement(
                                type="class",
                                name=class_name,
                                line_start=node.start_point[0] + 1,
                                line_end=node.end_point[0] + 1,
                                file_path=file_path,
                                language="python",
                                metadata={
                                    "decorators": self._extract_decorators(node, content),
                                    "bases": self._extract_class_bases(node, content),
                                },
                            )
                        )
                        break

            # Recursively walk children
            for child in node.children:
                walk_node(child)

        walk_node(tree.root_node)
        return elements

    def _extract_javascript_elements(self, tree, content: str, file_path: str) -> list[CodeElement]:
        """Extract JavaScript/TypeScript code elements"""
        elements = []

        # Walk through the AST to find functions and classes
        def walk_node(node):
            if node.type == "function_declaration":
                # Get function name
                for child in node.children:
                    if child.type == "identifier":
                        func_name = content[child.start_byte : child.end_byte]
                        elements.append(
                            CodeElement(
                                type="function",
                                name=func_name,
                                line_start=node.start_point[0] + 1,
                                line_end=node.end_point[0] + 1,
                                file_path=file_path,
                                language="javascript",
                                metadata={},
                            )
                        )
                        break

            elif node.type == "class_declaration":
                # Get class name
                for child in node.children:
                    if child.type == "identifier":
                        class_name = content[child.start_byte : child.end_byte]
                        elements.append(
                            CodeElement(
                                type="class",
                                name=class_name,
                                line_start=node.start_point[0] + 1,
                                line_end=node.end_point[0] + 1,
                                file_path=file_path,
                                language="javascript",
                                metadata={},
                            )
                        )
                        break

            # Recursively walk children
            for child in node.children:
                walk_node(child)

        walk_node(tree.root_node)
        return elements

    def _extract_decorators(self, node, content: str) -> list[str]:
        """Extract decorators from a Python function or class"""
        decorators = []

        # Look for decorator nodes as siblings before the function/class
        sibling = node.prev_sibling
        while sibling and sibling.type == "decorator":
            decorator_text = content[sibling.start_byte : sibling.end_byte]
            decorators.append(decorator_text)
            sibling = sibling.prev_sibling

        return list(reversed(decorators))  # Reverse to get correct order

    def _extract_class_bases(self, class_node, content: str) -> list[str]:
        """Extract base classes from a Python class definition"""
        bases = []

        # Find the argument list (base classes) in class definition
        for child in class_node.children:
            if child.type == "argument_list":
                # Extract each argument (base class)
                for arg in child.children:
                    if arg.type == "identifier" or arg.type == "attribute":
                        bases.append(content[arg.start_byte : arg.end_byte])

        return bases

    def _build_cross_references(self, elements: list[CodeElement]) -> dict[str, Any]:
        """Build cross-references between code elements"""
        # Group elements by type
        functions = [e for e in elements if e.type == "function"]
        classes = [e for e in elements if e.type == "class"]
        methods = [e for e in elements if e.type == "method"]

        # Build import graph (simplified for now)
        cross_refs = {
            "class_hierarchy": self._build_class_hierarchy(classes),
            "function_calls": {},  # TODO: Implement function call analysis
            "imports": {},  # TODO: Implement import analysis
        }

        return cross_refs

    def _build_class_hierarchy(self, classes: list[CodeElement]) -> dict[str, list[str]]:
        """Build class inheritance hierarchy"""
        hierarchy = {}

        for cls in classes:
            bases = cls.metadata.get("bases", [])
            if bases:
                hierarchy[cls.name] = bases

        return hierarchy


# FastAPI-specific analysis extensions
class FastAPITreeSitterAnalyzer(TreeSitterAnalyzer):
    """Extended analyzer for FastAPI-specific patterns"""

    def extract_fastapi_routes(self, project_path: str) -> list[dict[str, Any]]:
        """Extract FastAPI routes using Tree-sitter"""
        project_analysis = self.analyze_project(project_path)
        routes = []

        # Find all Python files
        for element in project_analysis["elements"]:
            if element.language == "python" and element.type in ["function", "method"]:
                # Check decorators for FastAPI patterns
                decorators = element.metadata.get("decorators", [])
                for decorator in decorators:
                    if any(
                        method in decorator
                        for method in [
                            "@app.get",
                            "@app.post",
                            "@app.put",
                            "@app.delete",
                            "@router.get",
                            "@router.post",
                            "@router.put",
                            "@router.delete",
                        ]
                    ):
                        # Extract route info from decorator
                        route_info = self._parse_route_decorator(decorator, element)
                        if route_info:
                            routes.append(route_info)

        return routes

    def _parse_route_decorator(self, decorator: str, element: CodeElement) -> dict[str, Any] | None:
        """Parse FastAPI route decorator to extract route information"""
        # Match patterns like @app.get("/path") or @router.post("/path", ...)
        pattern = r'@(?:app|router)\.(get|post|put|delete|patch)\s*\(\s*["\']([^"\']+)["\']'
        match = re.search(pattern, decorator)

        if match:
            method = match.group(1).upper()
            path = match.group(2)

            return {
                "method": method,
                "path": path,
                "function": element.name,
                "file": element.file_path,
                "line": element.line_start,
                "decorators": element.metadata.get("decorators", []),
            }

        return None


# Example usage
if __name__ == "__main__":
    # Test the analyzer
    analyzer = FastAPITreeSitterAnalyzer()

    # Analyze a project
    project_path = "test_fastapi_project"
    if os.path.exists(project_path):
        print("Analyzing project with Tree-sitter...")

        # General analysis
        analysis = analyzer.analyze_project(project_path)
        print("\nProject Statistics:")
        print(f"  Total files: {analysis['statistics']['total_files']}")
        print(f"  Languages: {analysis['statistics']['languages']}")
        print(f"  Functions: {analysis['statistics']['total_functions']}")
        print(f"  Classes: {analysis['statistics']['total_classes']}")
        print(f"  Methods: {analysis['statistics']['total_methods']}")

        # FastAPI-specific analysis
        routes = analyzer.extract_fastapi_routes(project_path)
        print(f"\nFastAPI Routes Found: {len(routes)}")
        for route in sorted(routes, key=lambda x: x["path"]):
            print(
                f"  {route['method']} {route['path']} -> {route['function']}() [{route['file']}:{route['line']}]"
            )
    else:
        print(f"Project path '{project_path}' not found")
