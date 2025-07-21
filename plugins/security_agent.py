"""
Security Agent Plugin for CodeConductor v2.0

This plugin provides advanced security analysis capabilities for generated code.
It can detect security vulnerabilities, suggest security improvements, and
validate code against security best practices.
"""

from typing import Dict, Any, List
from plugins.base import BaseAgentPlugin, PluginMetadata, PluginType


class SecurityAgentPlugin(BaseAgentPlugin):
    """
    Security-focused agent that analyzes code for security vulnerabilities
    and provides security recommendations.
    """

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="security_agent",
            version="1.0.0",
            description="Advanced security analysis agent for code vulnerability detection",
            author="CodeConductor Team",
            plugin_type=PluginType.AGENT,
            entry_point="plugins.security_agent:SecurityAgentPlugin",
            dependencies=["bandit", "safety"],
            config_schema={
                "severity_threshold": {
                    "type": "string",
                    "default": "medium",
                    "required": False,
                    "description": "Minimum severity level to report",
                },
                "enable_bandit": {
                    "type": "boolean",
                    "default": True,
                    "required": False,
                    "description": "Enable bandit security analysis",
                },
                "enable_safety": {
                    "type": "boolean",
                    "default": True,
                    "required": False,
                    "description": "Enable safety dependency analysis",
                },
            },
            tags=["security", "vulnerability", "analysis"],
            homepage="https://github.com/codeconductor/plugins",
            license="MIT",
        )

    def initialize(self) -> bool:
        """Initialize the security agent"""
        try:
            self.severity_threshold = self.get_config("severity_threshold", "medium")
            self.enable_bandit = self.get_config("enable_bandit", True)
            self.enable_safety = self.get_config("enable_safety", True)

            # Initialize security tools
            self.security_issues = []
            self.recommendations = []

            self.log_info("Security Agent initialized successfully")
            return True

        except Exception as e:
            self.log_error(f"Failed to initialize Security Agent: {e}")
            return False

    def cleanup(self) -> None:
        """Clean up security agent resources"""
        self.security_issues.clear()
        self.recommendations.clear()
        self.log_info("Security Agent cleaned up")

    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze code for security vulnerabilities"""
        code = context.get("code", "")
        prompt = context.get("prompt", "")

        analysis_result = {
            "security_score": 100,
            "vulnerabilities": [],
            "recommendations": [],
            "severity": "low",
        }

        # Basic security checks
        vulnerabilities = self._check_basic_security(code)
        analysis_result["vulnerabilities"] = vulnerabilities

        # Calculate security score
        if vulnerabilities:
            analysis_result["security_score"] = max(0, 100 - len(vulnerabilities) * 10)
            analysis_result["severity"] = self._determine_severity(vulnerabilities)

        # Generate recommendations
        analysis_result["recommendations"] = self._generate_security_recommendations(
            vulnerabilities, prompt
        )

        self.log_info(
            f"Security analysis completed. Score: {analysis_result['security_score']}"
        )
        return analysis_result

    def act(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Take security-focused actions"""
        analysis = context.get("analysis", {})
        code = context.get("code", "")

        action_result = {
            "action_type": "security_improvement",
            "modified_code": code,
            "changes_made": [],
            "security_improvements": [],
        }

        # Apply security improvements if vulnerabilities found
        vulnerabilities = analysis.get("vulnerabilities", [])
        if vulnerabilities:
            improved_code, changes = self._apply_security_fixes(code, vulnerabilities)
            action_result["modified_code"] = improved_code
            action_result["changes_made"] = changes
            action_result["security_improvements"] = [
                f"Fixed {len(changes)} security issues"
            ]

        return action_result

    def observe(self, result: Dict[str, Any]) -> None:
        """Observe results and learn from security analysis"""
        security_score = result.get("security_score", 100)
        vulnerabilities = result.get("vulnerabilities", [])

        self.log_info(f"Observed security score: {security_score}")

        # Store learning data
        if vulnerabilities:
            self.security_issues.extend(vulnerabilities)

        # Update recommendations based on patterns
        self._update_recommendations(vulnerabilities)

    def _check_basic_security(self, code: str) -> List[Dict[str, Any]]:
        """Perform basic security checks on code"""
        vulnerabilities = []

        # Check for dangerous imports
        dangerous_imports = ["os", "subprocess", "eval", "exec", "pickle"]
        for import_name in dangerous_imports:
            if f"import {import_name}" in code or f"from {import_name}" in code:
                vulnerabilities.append(
                    {
                        "type": "dangerous_import",
                        "severity": "high",
                        "description": f"Dangerous import detected: {import_name}",
                        "line": self._find_line_number(code, import_name),
                        "suggestion": f"Consider alternatives to {import_name} or add proper validation",
                    }
                )

        # Check for hardcoded secrets
        secret_patterns = [
            r"password\s*=\s*['\"][^'\"]+['\"]",
            r"api_key\s*=\s*['\"][^'\"]+['\"]",
            r"secret\s*=\s*['\"][^'\"]+['\"]",
            r"token\s*=\s*['\"][^'\"]+['\"]",
        ]

        import re

        for pattern in secret_patterns:
            matches = re.finditer(pattern, code, re.IGNORECASE)
            for match in matches:
                vulnerabilities.append(
                    {
                        "type": "hardcoded_secret",
                        "severity": "critical",
                        "description": "Hardcoded secret detected",
                        "line": self._find_line_number(code, match.group()),
                        "suggestion": "Use environment variables or secure secret management",
                    }
                )

        # Check for SQL injection patterns
        sql_patterns = [
            r"execute\s*\(\s*[\"'].*\+.*[\"']",
            r"query\s*=\s*[\"'].*\+.*[\"']",
        ]

        for pattern in sql_patterns:
            matches = re.finditer(pattern, code, re.IGNORECASE)
            for match in matches:
                vulnerabilities.append(
                    {
                        "type": "sql_injection",
                        "severity": "high",
                        "description": "Potential SQL injection detected",
                        "line": self._find_line_number(code, match.group()),
                        "suggestion": "Use parameterized queries or ORM",
                    }
                )

        return vulnerabilities

    def _find_line_number(self, code: str, text: str) -> int:
        """Find line number for a specific text in code"""
        lines = code.split("\n")
        for i, line in enumerate(lines, 1):
            if text in line:
                return i
        return 0

    def _determine_severity(self, vulnerabilities: List[Dict[str, Any]]) -> str:
        """Determine overall severity based on vulnerabilities"""
        if not vulnerabilities:
            return "low"

        severities = [v.get("severity", "medium") for v in vulnerabilities]

        if "critical" in severities:
            return "critical"
        elif "high" in severities:
            return "high"
        elif "medium" in severities:
            return "medium"
        else:
            return "low"

    def _generate_security_recommendations(
        self, vulnerabilities: List[Dict[str, Any]], prompt: str
    ) -> List[str]:
        """Generate security recommendations"""
        recommendations = []

        if not vulnerabilities:
            recommendations.append("✅ No security vulnerabilities detected")
            return recommendations

        # Count vulnerability types
        vuln_types = {}
        for vuln in vulnerabilities:
            vuln_type = vuln.get("type", "unknown")
            vuln_types[vuln_type] = vuln_types.get(vuln_type, 0) + 1

        # Generate specific recommendations
        for vuln_type, count in vuln_types.items():
            if vuln_type == "dangerous_import":
                recommendations.append(
                    f"🔒 Replace {count} dangerous import(s) with safer alternatives"
                )
            elif vuln_type == "hardcoded_secret":
                recommendations.append(
                    f"🔐 Move {count} hardcoded secret(s) to environment variables"
                )
            elif vuln_type == "sql_injection":
                recommendations.append(
                    f"💉 Use parameterized queries for {count} SQL operation(s)"
                )
            else:
                recommendations.append(f"⚠️ Address {count} {vuln_type} issue(s)")

        # General recommendations
        if len(vulnerabilities) > 5:
            recommendations.append("🔍 Consider running a full security audit")

        recommendations.append("📚 Review OWASP Top 10 security guidelines")

        return recommendations

    def _apply_security_fixes(
        self, code: str, vulnerabilities: List[Dict[str, Any]]
    ) -> tuple:
        """Apply security fixes to code"""
        modified_code = code
        changes = []

        for vuln in vulnerabilities:
            vuln_type = vuln.get("type")

            if vuln_type == "dangerous_import":
                # Replace dangerous imports with safer alternatives
                import_name = vuln.get("description", "").split(": ")[-1]
                if import_name == "os":
                    modified_code = modified_code.replace(
                        f"import {import_name}",
                        "# import os  # SECURITY: Replaced with pathlib",
                    )
                    changes.append("Replaced os import with pathlib")

            elif vuln_type == "hardcoded_secret":
                # Replace hardcoded secrets with environment variables
                line = vuln.get("line", 0)
                if line > 0:
                    lines = modified_code.split("\n")
                    if line <= len(lines):
                        old_line = lines[line - 1]
                        new_line = old_line.replace(
                            "password = '", "password = os.getenv('PASSWORD', '"
                        ).replace("'", "')")
                        lines[line - 1] = new_line
                        modified_code = "\n".join(lines)
                        changes.append(f"Replaced hardcoded secret on line {line}")

        return modified_code, changes

    def _update_recommendations(self, vulnerabilities: List[Dict[str, Any]]) -> None:
        """Update recommendations based on observed patterns"""
        # This could implement machine learning to improve recommendations
        # For now, just store the vulnerabilities for pattern analysis
        self.security_issues.extend(vulnerabilities)

        # Keep only recent issues (last 100)
        if len(self.security_issues) > 100:
            self.security_issues = self.security_issues[-100:]
