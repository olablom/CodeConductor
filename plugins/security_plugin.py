"""
Security Plugin for CodeConductor v2.0

A simple security analysis plugin that detects common security vulnerabilities.
"""

import re
from typing import Dict, Any, List
from plugins.base_simple import BaseAgentPlugin


class SecurityPlugin(BaseAgentPlugin):
    """Security analysis plugin that detects vulnerabilities in code"""

    def name(self) -> str:
        return "security_analyzer"

    def version(self) -> str:
        return "1.0.0"

    def description(self) -> str:
        return "Detects security vulnerabilities in generated code"

    def activate(self) -> None:
        """Initialize security analysis tools"""
        self.vulnerabilities = []
        self.security_score = 100
        print(f"🔒 {self.name()} activated")

    def deactivate(self) -> None:
        """Clean up security analysis resources"""
        self.vulnerabilities.clear()
        print(f"🔒 {self.name()} deactivated")

    def analyze(self, code: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze code for security vulnerabilities"""
        self.vulnerabilities = []

        # Check for dangerous imports
        dangerous_imports = ["os", "subprocess", "eval", "exec", "pickle"]
        for import_name in dangerous_imports:
            if f"import {import_name}" in code or f"from {import_name}" in code:
                self.vulnerabilities.append(
                    {
                        "type": "dangerous_import",
                        "severity": "high",
                        "description": f"Dangerous import: {import_name}",
                        "suggestion": f"Consider alternatives to {import_name}",
                    }
                )

        # Check for hardcoded secrets
        secret_patterns = [
            r"password\s*=\s*['\"][^'\"]+['\"]",
            r"api_key\s*=\s*['\"][^'\"]+['\"]",
            r"secret\s*=\s*['\"][^'\"]+['\"]",
            r"token\s*=\s*['\"][^'\"]+['\"]",
        ]

        for pattern in secret_patterns:
            matches = re.finditer(pattern, code, re.IGNORECASE)
            for match in matches:
                self.vulnerabilities.append(
                    {
                        "type": "hardcoded_secret",
                        "severity": "critical",
                        "description": "Hardcoded secret detected",
                        "suggestion": "Use environment variables",
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
                self.vulnerabilities.append(
                    {
                        "type": "sql_injection",
                        "severity": "high",
                        "description": "Potential SQL injection",
                        "suggestion": "Use parameterized queries",
                    }
                )

        # Calculate security score
        self.security_score = max(0, 100 - len(self.vulnerabilities) * 10)

        return {
            "security_score": self.security_score,
            "vulnerabilities": self.vulnerabilities,
            "vulnerability_count": len(self.vulnerabilities),
            "severity": self._get_overall_severity(),
            "recommendations": self._generate_recommendations(),
        }

    def _get_overall_severity(self) -> str:
        """Determine overall severity based on vulnerabilities"""
        if not self.vulnerabilities:
            return "low"

        severities = [v.get("severity", "medium") for v in self.vulnerabilities]

        if "critical" in severities:
            return "critical"
        elif "high" in severities:
            return "high"
        elif "medium" in severities:
            return "medium"
        else:
            return "low"

    def _generate_recommendations(self) -> List[str]:
        """Generate security recommendations"""
        recommendations = []

        if not self.vulnerabilities:
            recommendations.append("✅ No security vulnerabilities detected")
            return recommendations

        # Count vulnerability types
        vuln_types = {}
        for vuln in self.vulnerabilities:
            vuln_type = vuln.get("type", "unknown")
            vuln_types[vuln_type] = vuln_types.get(vuln_type, 0) + 1

        # Generate specific recommendations
        for vuln_type, count in vuln_types.items():
            if vuln_type == "dangerous_import":
                recommendations.append(f"🔒 Replace {count} dangerous import(s)")
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

        recommendations.append("📚 Review OWASP Top 10 security guidelines")

        return recommendations
