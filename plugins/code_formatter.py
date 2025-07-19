"""
Code Formatter Plugin for CodeConductor v2.0

This plugin provides code formatting and style enforcement capabilities.
It can format code according to various style guides and provide
style recommendations.
"""

from typing import Dict, Any, List
from plugins.base import BaseToolPlugin, PluginMetadata, PluginType


class CodeFormatterPlugin(BaseToolPlugin):
    """
    Code formatting tool that can format code according to various style guides
    and provide style recommendations.
    """

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="code_formatter",
            version="1.0.0",
            description="Code formatting and style enforcement tool",
            author="CodeConductor Team",
            plugin_type=PluginType.TOOL,
            entry_point="plugins.code_formatter:CodeFormatterPlugin",
            dependencies=["black", "isort", "flake8"],
            config_schema={
                "formatter": {
                    "type": "string",
                    "default": "black",
                    "required": False,
                    "description": "Code formatter to use (black, autopep8, yapf)",
                },
                "line_length": {
                    "type": "integer",
                    "default": 88,
                    "required": False,
                    "description": "Maximum line length",
                },
                "enable_isort": {
                    "type": "boolean",
                    "default": True,
                    "required": False,
                    "description": "Enable import sorting",
                },
                "style_guide": {
                    "type": "string",
                    "default": "pep8",
                    "required": False,
                    "description": "Style guide to follow (pep8, google, facebook)",
                },
            },
            tags=["formatting", "style", "code-quality"],
            homepage="https://github.com/codeconductor/plugins",
            license="MIT",
        )

    def initialize(self) -> bool:
        """Initialize the code formatter"""
        try:
            self.formatter = self.get_config("formatter", "black")
            self.line_length = self.get_config("line_length", 88)
            self.enable_isort = self.get_config("enable_isort", True)
            self.style_guide = self.get_config("style_guide", "pep8")

            # Initialize formatting tools
            self.formatting_issues = []
            self.style_violations = []

            self.log_info("Code Formatter initialized successfully")
            return True

        except Exception as e:
            self.log_error(f"Failed to initialize Code Formatter: {e}")
            return False

    def cleanup(self) -> None:
        """Clean up code formatter resources"""
        self.formatting_issues.clear()
        self.style_violations.clear()
        self.log_info("Code Formatter cleaned up")

    def execute(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """
        Execute code formatting on input data.

        Args:
            input_data: Code string to format
            **kwargs: Additional formatting options

        Returns:
            Dictionary containing formatted code and formatting information
        """
        if not isinstance(input_data, str):
            return {
                "success": False,
                "error": "Input data must be a string",
                "formatted_code": input_data,
            }

        try:
            # Apply formatting
            formatted_code = self._format_code(input_data, **kwargs)

            # Check for style violations
            style_violations = self._check_style_violations(formatted_code)

            # Generate formatting report
            report = self._generate_formatting_report(
                input_data, formatted_code, style_violations
            )

            return {
                "success": True,
                "formatted_code": formatted_code,
                "style_violations": style_violations,
                "report": report,
                "changes_made": self._count_changes(input_data, formatted_code),
            }

        except Exception as e:
            self.log_error(f"Code formatting failed: {e}")
            return {"success": False, "error": str(e), "formatted_code": input_data}

    def _format_code(self, code: str, **kwargs) -> str:
        """Format code according to configured formatter"""
        formatted_code = code

        # Apply import sorting if enabled
        if self.enable_isort:
            formatted_code = self._sort_imports(formatted_code)

        # Apply main formatter
        if self.formatter == "black":
            formatted_code = self._format_with_black(formatted_code, **kwargs)
        elif self.formatter == "autopep8":
            formatted_code = self._format_with_autopep8(formatted_code, **kwargs)
        elif self.formatter == "yapf":
            formatted_code = self._format_with_yapf(formatted_code, **kwargs)
        else:
            # Default to basic formatting
            formatted_code = self._basic_formatting(formatted_code)

        return formatted_code

    def _sort_imports(self, code: str) -> str:
        """Sort imports using isort logic"""
        lines = code.split("\n")
        import_lines = []
        other_lines = []
        in_import_block = False

        for line in lines:
            stripped = line.strip()

            if stripped.startswith(("import ", "from ")):
                if not in_import_block:
                    in_import_block = True
                import_lines.append(line)
            else:
                if in_import_block and stripped:
                    in_import_block = False
                other_lines.append(line)

        # Sort import lines
        import_lines.sort(
            key=lambda x: (
                x.startswith("from "),  # 'from' imports come after 'import'
                x.lower(),
            )
        )

        # Reconstruct code
        result_lines = []
        if import_lines:
            result_lines.extend(import_lines)
            result_lines.append("")  # Add blank line after imports

        result_lines.extend(other_lines)

        return "\n".join(result_lines)

    def _format_with_black(self, code: str, **kwargs) -> str:
        """Format code using Black formatter logic"""
        # Simplified Black-like formatting
        lines = code.split("\n")
        formatted_lines = []

        for line in lines:
            # Remove trailing whitespace
            line = line.rstrip()

            # Ensure proper indentation
            if line.strip():
                # Count leading spaces for indentation
                indent = len(line) - len(line.lstrip())
                if indent % 4 != 0:
                    # Fix indentation to multiples of 4
                    indent = (indent // 4) * 4
                    line = " " * indent + line.strip()

            formatted_lines.append(line)

        return "\n".join(formatted_lines)

    def _format_with_autopep8(self, code: str, **kwargs) -> str:
        """Format code using autopep8 logic"""
        # Simplified autopep8-like formatting
        return self._format_with_black(code, **kwargs)  # For now, same as Black

    def _format_with_yapf(self, code: str, **kwargs) -> str:
        """Format code using YAPF logic"""
        # Simplified YAPF-like formatting
        return self._format_with_black(code, **kwargs)  # For now, same as Black

    def _basic_formatting(self, code: str) -> str:
        """Apply basic formatting rules"""
        lines = code.split("\n")
        formatted_lines = []

        for line in lines:
            # Remove trailing whitespace
            line = line.rstrip()

            # Ensure consistent indentation
            if line.strip():
                indent = len(line) - len(line.lstrip())
                if indent % 4 != 0:
                    indent = (indent // 4) * 4
                    line = " " * indent + line.strip()

            formatted_lines.append(line)

        return "\n".join(formatted_lines)

    def _check_style_violations(self, code: str) -> List[Dict[str, Any]]:
        """Check for style guide violations"""
        violations = []
        lines = code.split("\n")

        for line_num, line in enumerate(lines, 1):
            # Check line length
            if len(line) > self.line_length:
                violations.append(
                    {
                        "type": "line_too_long",
                        "severity": "medium",
                        "line": line_num,
                        "description": f"Line {line_num} is {len(line)} characters long (max: {self.line_length})",
                        "suggestion": "Break long line into multiple lines",
                    }
                )

            # Check for trailing whitespace
            if line.rstrip() != line:
                violations.append(
                    {
                        "type": "trailing_whitespace",
                        "severity": "low",
                        "line": line_num,
                        "description": f"Trailing whitespace on line {line_num}",
                        "suggestion": "Remove trailing whitespace",
                    }
                )

            # Check for mixed tabs and spaces
            if "\t" in line and " " in line[: len(line) - len(line.lstrip())]:
                violations.append(
                    {
                        "type": "mixed_indentation",
                        "severity": "medium",
                        "line": line_num,
                        "description": f"Mixed tabs and spaces on line {line_num}",
                        "suggestion": "Use consistent indentation (spaces recommended)",
                    }
                )

        return violations

    def _generate_formatting_report(
        self, original_code: str, formatted_code: str, violations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate a comprehensive formatting report"""
        original_lines = len(original_code.split("\n"))
        formatted_lines = len(formatted_code.split("\n"))
        original_chars = len(original_code)
        formatted_chars = len(formatted_code)

        return {
            "summary": {
                "original_lines": original_lines,
                "formatted_lines": formatted_lines,
                "original_chars": original_chars,
                "formatted_chars": formatted_chars,
                "violations_found": len(violations),
                "formatter_used": self.formatter,
                "line_length_limit": self.line_length,
            },
            "violations_by_type": self._group_violations_by_type(violations),
            "recommendations": self._generate_style_recommendations(violations),
        }

    def _group_violations_by_type(
        self, violations: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Group violations by type"""
        grouped = {}
        for violation in violations:
            violation_type = violation.get("type", "unknown")
            grouped[violation_type] = grouped.get(violation_type, 0) + 1
        return grouped

    def _generate_style_recommendations(
        self, violations: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate style recommendations based on violations"""
        recommendations = []

        if not violations:
            recommendations.append("✅ Code follows style guidelines")
            return recommendations

        # Count violation types
        violation_types = self._group_violations_by_type(violations)

        for violation_type, count in violation_types.items():
            if violation_type == "line_too_long":
                recommendations.append(f"📏 Break {count} long line(s) for readability")
            elif violation_type == "trailing_whitespace":
                recommendations.append(
                    f"🧹 Remove trailing whitespace from {count} line(s)"
                )
            elif violation_type == "mixed_indentation":
                recommendations.append(f"🔧 Fix mixed indentation on {count} line(s)")
            else:
                recommendations.append(f"⚠️ Address {count} {violation_type} issue(s)")

        # General recommendations
        if len(violations) > 10:
            recommendations.append("🔍 Consider using automated formatting tools")

        recommendations.append(f"📚 Follow {self.style_guide.upper()} style guide")

        return recommendations

    def _count_changes(self, original_code: str, formatted_code: str) -> Dict[str, int]:
        """Count the number of changes made during formatting"""
        original_lines = original_code.split("\n")
        formatted_lines = formatted_code.split("\n")

        changes = 0
        for orig, fmt in zip(original_lines, formatted_lines):
            if orig != fmt:
                changes += 1

        # Account for different line counts
        changes += abs(len(original_lines) - len(formatted_lines))

        return {"lines_changed": changes, "total_lines": len(formatted_lines)}
