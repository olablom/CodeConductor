"""
Auth Service - Policy Agent

This module contains the PolicyAgent migrated from the main CodeConductor
codebase to the microservices architecture.
"""

import logging
import re
from typing import Dict, List, Any, Optional
from datetime import datetime


class PolicyAgent:
    """
    Policy enforcement agent for code safety and security.

    This agent focuses on:
    - Code safety validation
    - Security policy enforcement
    - Dangerous pattern detection
    - Compliance checking
    """

    def __init__(
        self, name: str = "policy_agent", config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the policy agent."""
        self.name = name
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

        self.logger.info(
            f"PolicyAgent '{name}' initialized with {len(self.dangerous_patterns)} policy categories"
        )

    def check_code_safety(self, code: str) -> Dict[str, Any]:
        """
        Check code for safety violations and policy compliance.

        Args:
            code: The code to check

        Returns:
            Dictionary with safety analysis results
        """
        if not code:
            return {
                "safe": True,
                "violations": [],
                "risk_level": "none",
                "recommendations": [],
            }

        violations = []
        risk_level = "none"

        # Check for dangerous patterns
        for severity, patterns in self.dangerous_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, code, re.IGNORECASE)
                for match in matches:
                    line_num = self._get_line_number(code, match.start())
                    violation = {
                        "type": "dangerous_pattern",
                        "severity": severity,
                        "pattern": pattern,
                        "line": line_num,
                        "position": match.start(),
                        "description": self._get_violation_description(
                            pattern, severity
                        ),
                    }
                    violations.append(violation)

                    # Update risk level
                    if severity == "critical":
                        risk_level = "critical"
                    elif severity == "high" and risk_level not in ["critical"]:
                        risk_level = "high"
                    elif severity == "medium" and risk_level not in [
                        "critical",
                        "high",
                    ]:
                        risk_level = "medium"
                    elif severity == "low" and risk_level not in [
                        "critical",
                        "high",
                        "medium",
                    ]:
                        risk_level = "low"

        # Check for secrets
        secret_violations = self._check_for_secrets(code)
        violations.extend(secret_violations)

        # Check for network operations
        network_violations = self._check_network_operations(code)
        violations.extend(network_violations)

        # Determine overall safety
        safe = (
            len([v for v in violations if v["severity"] in ["critical", "high"]]) == 0
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(violations, risk_level)

        return {
            "safe": safe,
            "violations": violations,
            "risk_level": risk_level,
            "recommendations": recommendations,
            "total_violations": len(violations),
            "critical_violations": len(
                [v for v in violations if v["severity"] == "critical"]
            ),
            "high_violations": len([v for v in violations if v["severity"] == "high"]),
            "medium_violations": len(
                [v for v in violations if v["severity"] == "medium"]
            ),
            "low_violations": len([v for v in violations if v["severity"] == "low"]),
        }

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """
        Evaluate a context for approval.

        Args:
            context: Context containing code and metadata

        Returns:
            True if approved, False if rejected
        """
        try:
            # Extract code from context
            code = context.get("code", "")
            task_type = context.get("task_type", "unknown")
            risk_level = context.get("risk_level", "medium")

            # Check code safety
            safety_result = self.check_code_safety(code)

            # Auto-approve based on risk level and safety
            if risk_level == "low" and safety_result["safe"]:
                self.logger.info(f"Auto-approved {task_type} (low risk, safe code)")
                return True

            # Auto-reject critical violations
            if safety_result["critical_violations"] > 0:
                self.logger.warning(f"Auto-rejected {task_type} (critical violations)")
                return False

            # Auto-reject high risk with high violations
            if risk_level == "high" and safety_result["high_violations"] > 0:
                self.logger.warning(
                    f"Auto-rejected {task_type} (high risk with violations)"
                )
                return False

            # For medium risk or unsafe code, require human approval
            self.logger.info(
                f"Requiring human approval for {task_type} (risk: {risk_level}, safe: {safety_result['safe']})"
            )
            return False

        except Exception as e:
            self.logger.error(f"Error in policy evaluation: {e}")
            return False

    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze context for policy compliance.

        Args:
            context: Context to analyze

        Returns:
            Analysis results
        """
        code = context.get("code", "")
        task_type = context.get("task_type", "unknown")

        safety_result = self.check_code_safety(code)

        return {
            "agent_name": self.name,
            "task_type": task_type,
            "safety_analysis": safety_result,
            "policy_compliant": safety_result["safe"],
            "risk_assessment": safety_result["risk_level"],
            "recommendations": safety_result["recommendations"],
            "timestamp": datetime.now().isoformat(),
        }

    def propose(
        self, analysis: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Propose policy actions based on analysis.

        Args:
            analysis: Previous analysis results
            context: Original context

        Returns:
            Policy proposal
        """
        safety_result = analysis.get("safety_analysis", {})
        risk_level = safety_result.get("risk_level", "medium")

        # Determine approval strategy
        if risk_level == "critical":
            strategy = "reject"
        elif risk_level == "high":
            strategy = "human_approval"
        elif risk_level == "medium":
            strategy = "conditional_approval"
        else:
            strategy = "auto_approve"

        return {
            "agent_name": self.name,
            "strategy": strategy,
            "approved": strategy == "auto_approve",
            "requires_human": strategy in ["human_approval", "conditional_approval"],
            "risk_level": risk_level,
            "violations": safety_result.get("violations", []),
            "recommendations": safety_result.get("recommendations", []),
            "confidence": 0.9 if strategy == "auto_approve" else 0.7,
            "reasoning": f"Policy evaluation based on {risk_level} risk level",
        }

    def review(
        self, proposal: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Review a proposal for final approval.

        Args:
            proposal: Proposal to review
            context: Original context

        Returns:
            Review results
        """
        # Final policy check
        code = context.get("code", "")
        safety_result = self.check_code_safety(code)

        # Override proposal if critical violations found
        if safety_result["critical_violations"] > 0:
            proposal["approved"] = False
            proposal["strategy"] = "reject"
            proposal["reasoning"] = "Critical safety violations detected"

        return {
            "agent_name": self.name,
            "final_approval": proposal["approved"],
            "strategy": proposal["strategy"],
            "risk_level": safety_result["risk_level"],
            "violations": safety_result["violations"],
            "recommendations": safety_result["recommendations"],
            "confidence": proposal.get("confidence", 0.8),
            "reasoning": proposal.get("reasoning", "Policy review completed"),
            "timestamp": datetime.now().isoformat(),
        }

    def _check_for_secrets(self, code: str) -> List[Dict[str, Any]]:
        """Check for potential secrets in code."""
        violations = []

        # Common secret patterns
        secret_patterns = [
            r"password\s*=\s*['\"][^'\"]+['\"]",
            r"api_key\s*=\s*['\"][^'\"]+['\"]",
            r"secret\s*=\s*['\"][^'\"]+['\"]",
            r"token\s*=\s*['\"][^'\"]+['\"]",
            r"key\s*=\s*['\"][^'\"]{20,}['\"]",  # Long keys
        ]

        for pattern in secret_patterns:
            matches = re.finditer(pattern, code, re.IGNORECASE)
            for match in matches:
                line_num = self._get_line_number(code, match.start())
                violations.append(
                    {
                        "type": "potential_secret",
                        "severity": "high",
                        "pattern": pattern,
                        "line": line_num,
                        "position": match.start(),
                        "description": "Potential secret or API key found",
                    }
                )

        return violations

    def _check_network_operations(self, code: str) -> List[Dict[str, Any]]:
        """Check for potentially dangerous network operations."""
        violations = []

        # Network operation patterns
        network_patterns = [
            r"requests\.get\s*\(",
            r"requests\.post\s*\(",
            r"urllib\.request\.urlopen\s*\(",
            r"socket\.connect\s*\(",
            r"http\.client\.HTTPConnection\s*\(",
        ]

        for pattern in network_patterns:
            matches = re.finditer(pattern, code, re.IGNORECASE)
            for match in matches:
                line_num = self._get_line_number(code, match.start())
                violations.append(
                    {
                        "type": "network_operation",
                        "severity": "medium",
                        "pattern": pattern,
                        "line": line_num,
                        "position": match.start(),
                        "description": "Network operation detected",
                    }
                )

        return violations

    def _get_line_number(self, code: str, position: int) -> int:
        """Get line number for a position in code."""
        return code[:position].count("\n") + 1

    def _get_violation_description(self, pattern: str, severity: str) -> str:
        """Get human-readable description of violation."""
        descriptions = {
            r"eval\s*\(": "Use of eval() - potential code injection",
            r"exec\s*\(": "Use of exec() - potential code injection",
            r"os\.system\s*\(": "Use of os.system() - potential command injection",
            r"subprocess\.": "Use of subprocess - potential command injection",
            r"pickle\.": "Use of pickle - potential deserialization attack",
            r"__import__\s*\(": "Use of __import__() - potential code injection",
            r"open\s*\(": "File operation detected",
            r"input\s*\(": "User input detected",
            r"print\s*\(": "Print statement detected",
            r"TODO": "TODO comment found",
            r"FIXME": "FIXME comment found",
        }

        for p, desc in descriptions.items():
            if re.search(p, pattern, re.IGNORECASE):
                return f"{desc} ({severity} risk)"

        return f"Pattern '{pattern}' detected ({severity} risk)"

    def _generate_recommendations(
        self, violations: List[Dict[str, Any]], risk_level: str
    ) -> List[str]:
        """Generate recommendations based on violations."""
        recommendations = []

        if risk_level == "critical":
            recommendations.append(
                "CRITICAL: Code contains dangerous patterns that could lead to security vulnerabilities"
            )
            recommendations.append(
                "Review and remove all eval(), exec(), os.system(), and subprocess calls"
            )

        if risk_level == "high":
            recommendations.append(
                "HIGH RISK: Code contains potentially dangerous operations"
            )
            recommendations.append(
                "Consider using safer alternatives for file and network operations"
            )

        if any(v["type"] == "potential_secret" for v in violations):
            recommendations.append(
                "Remove hardcoded secrets and use environment variables instead"
            )

        if any(v["type"] == "network_operation" for v in violations):
            recommendations.append(
                "Review network operations for security implications"
            )

        if risk_level in ["medium", "low"]:
            recommendations.append(
                "Code appears relatively safe but review for best practices"
            )

        return recommendations

    def _get_timestamp(self) -> str:
        """Get current timestamp as string."""
        return datetime.now().isoformat()
