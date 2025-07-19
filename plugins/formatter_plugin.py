"""
Code Formatter Plugin for CodeConductor v2.0

A simple code formatting plugin that improves code style and readability.
"""

from typing import Dict, Any, List
from plugins.base_simple import BaseToolPlugin


class FormatterPlugin(BaseToolPlugin):
    """Code formatting plugin that improves code style"""

    def name(self) -> str:
        return "code_formatter"

    def version(self) -> str:
        return "1.0.0"

    def description(self) -> str:
        return "Formats code for better readability and style consistency"

    def activate(self) -> None:
        """Initialize formatting tools"""
        self.style_violations = []
        self.formatted_code = ""
        print(f"📝 {self.name()} activated")

    def deactivate(self) -> None:
        """Clean up formatting resources"""
        self.style_violations.clear()
        print(f"📝 {self.name()} deactivated")

    def execute(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """Format code and return results"""
        if not isinstance(input_data, str):
            return {
                "success": False,
                "error": "Input must be a string",
                "formatted_code": input_data,
            }

        try:
            # Apply formatting
            self.formatted_code = self._format_code(input_data)

            # Check for style violations
            self.style_violations = self._check_style_violations(self.formatted_code)

            # Generate report
            report = self._generate_formatting_report(input_data, self.formatted_code)

            return {
                "success": True,
                "formatted_code": self.formatted_code,
                "style_violations": self.style_violations,
                "report": report,
                "changes_made": self._count_changes(input_data, self.formatted_code),
            }

        except Exception as e:
            return {"success": False, "error": str(e), "formatted_code": input_data}

    def _format_code(self, code: str) -> str:
        """Apply basic code formatting"""
        lines = code.split("\n")
        formatted_lines = []

        for line in lines:
            # Remove trailing whitespace
            line = line.rstrip()

            # Fix indentation (ensure multiples of 4 spaces)
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
            # Check line length (max 88 characters)
            if len(line) > 88:
                violations.append(
                    {
                        "type": "line_too_long",
                        "severity": "medium",
                        "line": line_num,
                        "description": f"Line {line_num} is {len(line)} characters long (max: 88)",
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
        self, original_code: str, formatted_code: str
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
                "violations_found": len(self.style_violations),
                "formatter_used": "basic_python",
            },
            "violations_by_type": self._group_violations_by_type(),
            "recommendations": self._generate_style_recommendations(),
        }

    def _group_violations_by_type(self) -> Dict[str, int]:
        """Group violations by type"""
        grouped = {}
        for violation in self.style_violations:
            violation_type = violation.get("type", "unknown")
            grouped[violation_type] = grouped.get(violation_type, 0) + 1
        return grouped

    def _generate_style_recommendations(self) -> List[str]:
        """Generate style recommendations based on violations"""
        recommendations = []

        if not self.style_violations:
            recommendations.append("✅ Code follows style guidelines")
            return recommendations

        # Count violation types
        violation_types = self._group_violations_by_type()

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
        if len(self.style_violations) > 10:
            recommendations.append(
                "🔍 Consider using automated formatting tools like Black"
            )

        recommendations.append("📚 Follow PEP 8 style guide")

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
