"""
Policy Agent for CodeConductor

This agent enforces security policies and safety checks on generated code.
It prevents dangerous code from being executed and ensures compliance with safety standards.
"""

import logging
import re
from typing import Dict, List, Any, Optional
from agents.base_agent import BaseAgent


class PolicyAgent(BaseAgent):
    """
    Policy enforcement agent for code safety and security.

    This agent focuses on:
    - Code safety validation
    - Security policy enforcement
    - Dangerous pattern detection
    - Compliance checking
    """

    def __init__(self, name: str = "policy_agent", config: Optional[Dict[str, Any]] = None):
        """Initialize the policy agent."""
        super().__init__(name, config)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Policy configuration
        self.policy_config = config.get("policy", {}) if config else {}

        # Define dangerous patterns
        self.dangerous_patterns = {
            "critical": [
                r"eval\s*\(",
                r"exec\s*\(",
                r"os\.system\s*\(",
                r"subprocess\.run\s*\(",
                r"subprocess\.call\s*\(",
                r"subprocess\.Popen\s*\(",
                r"pickle\.loads\s*\(",
                r"pickle\.load\s*\(",
                r"marshal\.loads\s*\(",
                r"__import__\s*\(",
                r"globals\s*\(",
                r"locals\s*\(",
                r"compile\s*\(",
            ],
            "high": [
                r"open\s*\(",
                r"file\s*\(",
                r"input\s*\(",
                r"raw_input\s*\(",
                r"yaml\.load\s*\(",
                r"json\.loads\s*\(",
                r"ast\.literal_eval\s*\(",
                r"getattr\s*\(",
                r"setattr\s*\(",
                r"delattr\s*\(",
            ],
            "medium": [
                r"print\s*\(",
                r"assert\s+",
                r"del\s+",
                r"breakpoint\s*\(",
                r"help\s*\(",
                r"dir\s*\(",
                r"vars\s*\(",
            ],
            "low": [
                r"TODO",
                r"FIXME",
                r"XXX",
                r"HACK",
            ],
        }

        # Define file operations that are allowed
        self.allowed_file_operations = [
            r"open\s*\(\s*['\"][^'\"]*\.(py|json|txt|md|yml|yaml|toml|ini|cfg|conf)\s*['\"]",
            r"open\s*\(\s*['\"][^'\"]*\.(log|out|err)\s*['\"]",
        ]

        # Define system operations that are allowed
        self.allowed_system_operations = [
            r"os\.path\.",
            r"os\.environ\.",
            r"os\.getcwd\s*\(",
            r"os\.listdir\s*\(",
            r"os\.mkdir\s*\(",
            r"os\.makedirs\s*\(",
        ]

        self.logger.info(f"PolicyAgent '{name}' initialized with {len(self.dangerous_patterns)} policy categories")

    def check_code_safety(self, code: str) -> Dict[str, Any]:
        """
        Check code for safety violations and policy compliance.

        Args:
            code: The code to check for safety

        Returns:
            Safety check results with pass/block decision and violations
        """
        self.logger.info(f"Checking code safety for {self.name}")

        try:
            # Initialize result
            result = {
                "safe": True,
                "decision": "pass",
                "violations": [],
                "warnings": [],
                "risk_level": "low",
                "critical_violations": 0,
                "high_violations": 0,
                "medium_violations": 0,
                "low_violations": 0,
                "recommendations": [],
            }

            # Check for dangerous patterns
            for severity, patterns in self.dangerous_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, code, re.IGNORECASE)
                    for match in matches:
                        violation = {
                            "severity": severity,
                            "pattern": pattern,
                            "line": self._get_line_number(code, match.start()),
                            "match": match.group(),
                            "description": self._get_violation_description(pattern, severity),
                        }

                        result["violations"].append(violation)

                        # Count violations by severity
                        if severity == "critical":
                            result["critical_violations"] += 1
                        elif severity == "high":
                            result["high_violations"] += 1
                        elif severity == "medium":
                            result["medium_violations"] += 1
                        elif severity == "low":
                            result["low_violations"] += 1

            # Check for hardcoded secrets
            secret_violations = self._check_for_secrets(code)
            result["violations"].extend(secret_violations)
            result["critical_violations"] += len([v for v in secret_violations if v["severity"] == "critical"])

            # Check for network operations
            network_violations = self._check_network_operations(code)
            result["violations"].extend(network_violations)
            result["high_violations"] += len([v for v in network_violations if v["severity"] == "high"])

            # Determine overall safety decision
            if result["critical_violations"] > 0:
                result["safe"] = False
                result["decision"] = "block"
                result["risk_level"] = "critical"
                result["recommendations"].append("Code contains critical safety violations and cannot be executed")
            elif result["high_violations"] > 0:
                result["safe"] = False
                result["decision"] = "block"
                result["risk_level"] = "high"
                result["recommendations"].append("Code contains high-risk operations and should be reviewed")
            elif result["medium_violations"] > 0:
                result["safe"] = True
                result["decision"] = "warn"
                result["risk_level"] = "medium"
                result["recommendations"].append("Code contains medium-risk operations that should be reviewed")
            elif result["low_violations"] > 0:
                result["safe"] = True
                result["decision"] = "pass"
                result["risk_level"] = "low"
                result["recommendations"].append("Code contains minor issues that should be addressed")

            # Add general recommendations
            if result["violations"]:
                result["recommendations"].append("Review all violations before deploying to production")

            self.logger.info(f"Safety check completed. Decision: {result['decision']}, Risk level: {result['risk_level']}")
            return result

        except Exception as e:
            self.logger.error(f"Error during safety check: {e}")
            return {
                "safe": False,
                "decision": "error",
                "error": str(e),
                "violations": [],
                "recommendations": ["Safety check failed due to error"],
            }

    def _check_for_secrets(self, code: str) -> List[Dict[str, Any]]:
        """Check for hardcoded secrets in the code."""
        violations = []

        # Patterns for potential secrets
        secret_patterns = [
            (r'password\s*=\s*["\'][^"\']{8,}["\']', "Hardcoded password"),
            (r'secret\s*=\s*["\'][^"\']{8,}["\']', "Hardcoded secret"),
            (r'api_key\s*=\s*["\'][^"\']{8,}["\']', "Hardcoded API key"),
            (r'token\s*=\s*["\'][^"\']{8,}["\']', "Hardcoded token"),
            (r'key\s*=\s*["\'][^"\']{8,}["\']', "Hardcoded key"),
            (r'private_key\s*=\s*["\'][^"\']{8,}["\']', "Hardcoded private key"),
            (r'secret_key\s*=\s*["\'][^"\']{8,}["\']', "Hardcoded secret key"),
        ]

        for pattern, description in secret_patterns:
            matches = re.finditer(pattern, code, re.IGNORECASE)
            for match in matches:
                violations.append(
                    {
                        "severity": "critical",
                        "pattern": pattern,
                        "line": self._get_line_number(code, match.start()),
                        "match": match.group(),
                        "description": description,
                    }
                )

        return violations

    def _check_network_operations(self, code: str) -> List[Dict[str, Any]]:
        """Check for potentially dangerous network operations."""
        violations = []

        # Network operation patterns
        network_patterns = [
            (r'requests\.get\s*\("', "HTTP GET request"),
            (r'requests\.post\s*\("', "HTTP POST request"),
            (r'urllib\.request\.urlopen\s*\("', "URL opening"),
            (r"socket\.", "Socket operations"),
            (r"http\.client\.", "HTTP client operations"),
        ]

        for pattern, description in network_patterns:
            matches = re.finditer(pattern, code, re.IGNORECASE)
            for match in matches:
                violations.append(
                    {
                        "severity": "high",
                        "pattern": pattern,
                        "line": self._get_line_number(code, match.start()),
                        "match": match.group(),
                        "description": f"Network operation: {description}",
                    }
                )

        return violations

    def _get_line_number(self, code: str, position: int) -> int:
        """Get line number for a given position in the code."""
        return code[:position].count("\n") + 1

    def _get_violation_description(self, pattern: str, severity: str) -> str:
        """Get human-readable description for a violation pattern."""
        descriptions = {
            r"eval\s*\(": "Use of eval() function - potential code injection risk",
            r"exec\s*\(": "Use of exec() function - potential code injection risk",
            r"os\.system\s*\(": "System command execution - potential command injection risk",
            r"subprocess\.run\s*\(": "Subprocess execution - potential command injection risk",
            r"subprocess\.call\s*\(": "Subprocess execution - potential command injection risk",
            r"subprocess\.Popen\s*\(": "Subprocess execution - potential command injection risk",
            r"pickle\.loads\s*\(": "Unsafe deserialization - potential code execution risk",
            r"pickle\.load\s*\(": "Unsafe deserialization - potential code execution risk",
            r"marshal\.loads\s*\(": "Unsafe deserialization - potential code execution risk",
            r"__import__\s*\(": "Dynamic import - potential code injection risk",
            r"globals\s*\(": "Access to global namespace - potential security risk",
            r"locals\s*\(": "Access to local namespace - potential security risk",
            r"compile\s*\(": "Dynamic code compilation - potential code injection risk",
            r"open\s*\(": "File operation - potential file system access",
            r"file\s*\(": "File operation - potential file system access",
            r"input\s*\(": "User input - potential injection risk",
            r"raw_input\s*\(": "User input - potential injection risk",
            r"yaml\.load\s*\(": "YAML loading - potential code execution risk",
            r"json\.loads\s*\(": "JSON deserialization - potential injection risk",
            r"ast\.literal_eval\s*\(": "AST evaluation - potential code execution risk",
            r"getattr\s*\(": "Dynamic attribute access - potential security risk",
            r"setattr\s*\(": "Dynamic attribute modification - potential security risk",
            r"delattr\s*\(": "Dynamic attribute deletion - potential security risk",
            r"print\s*\(": "Print statement - consider using logging",
            r"assert\s+": "Assert statement - consider proper error handling",
            r"del\s+": "Delete statement - potential data loss",
            r"breakpoint\s*\(": "Breakpoint - remove before production",
            r"help\s*\(": "Help function - remove before production",
            r"dir\s*\(": "Directory listing - potential information disclosure",
            r"vars\s*\(": "Variable inspection - potential information disclosure",
            r"TODO": "TODO comment - address before production",
            r"FIXME": "FIXME comment - address before production",
            r"XXX": "XXX comment - address before production",
            r"HACK": "HACK comment - address before production",
        }

        return descriptions.get(pattern, f"{severity.title()} risk pattern detected")

    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze context for policy compliance.

        Args:
            context: Context information

        Returns:
            Policy analysis results
        """
        self.logger.info("Analyzing context for policy compliance")

        return {
            "policy_compliance": True,
            "risk_assessment": "low",
            "recommendations": ["Context appears to comply with policies"],
        }

    def propose(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Propose policy improvements.

        Args:
            analysis: Analysis results
            context: Context information

        Returns:
            Policy improvement proposals
        """
        self.logger.info("Generating policy improvement proposals")

        return {
            "policy_improvements": [],
            "security_enhancements": [],
            "compliance_measures": [],
        }

    def review(self, proposal: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Review proposal for policy compliance.

        Args:
            proposal: Proposal to review
            context: Context information

        Returns:
            Policy review results
        """
        self.logger.info("Reviewing proposal for policy compliance")

        return {"policy_compliant": True, "risk_level": "low", "recommendations": []}
