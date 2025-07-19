"""
PolicyAgent - Security and safety checks for generated code.

This agent validates generated code against security policies before execution.
Blocks dangerous operations, license violations, and other safety concerns.
"""

import re
import ast
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class BlockReason(Enum):
    """Reasons why code was blocked by PolicyAgent."""

    DANGEROUS_SYSTEM_CALL = "dangerous_system_call"
    FILE_OPERATION = "file_operation"
    NETWORK_ACCESS = "network_access"
    LICENSE_VIOLATION = "license_violation"
    TOO_LARGE = "too_large"
    FORBIDDEN_IMPORT = "forbidden_import"
    SUSPICIOUS_PATTERN = "suspicious_pattern"


@dataclass
class PolicyViolation:
    """Represents a policy violation found in code."""

    reason: BlockReason
    line_number: int
    code_snippet: str
    severity: str  # "high", "medium", "low"
    description: str


class PolicyAgent:
    """
    Security agent that validates generated code against safety policies.

    Checks for:
    - Dangerous system calls (os.system, subprocess, etc.)
    - File operations that could be harmful
    - Network access patterns
    - License violations (GPL, AGPL headers)
    - Code size limits
    - Forbidden imports
    - Suspicious patterns
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize PolicyAgent with configuration."""
        self.config = config or {}

        # Security patterns
        self.dangerous_patterns = {
            BlockReason.DANGEROUS_SYSTEM_CALL: [
                r"os\.system\s*\(",
                r"subprocess\.(run|call|Popen)\s*\(",
                r"exec\s*\(",
                r"eval\s*\(",
                r"__import__\s*\(",
                r"compile\s*\(",
            ],
            BlockReason.FILE_OPERATION: [
                r'open\s*\(\s*[\'"][^\'"]*\.(key|pem|p12|pfx|env|secret|password|token)[\'"]',
                r'open\s*\(\s*[\'"][^\'"]*\.(py|txt|json|yaml|yml|ini|cfg|conf)[\'"]\s*,\s*[\'"]w[\'"]',
                r'with\s+open\s*\(\s*[\'"][^\'"]*\.(py|txt|json|yaml|yml|ini|cfg|conf)[\'"]\s*,\s*[\'"]w[\'"]',
            ],
            BlockReason.NETWORK_ACCESS: [
                r"requests\.(get|post|put|delete|patch)\s*\(",
                r"urllib\.(request|urlopen)\s*\(",
                r"socket\.(socket|connect)\s*\(",
                r"http\.(client|server)\s*\(",
            ],
            BlockReason.FORBIDDEN_IMPORT: [
                r"import\s+(torch|tensorflow|sklearn|pandas|numpy|matplotlib|seaborn)",
                r"from\s+(torch|tensorflow|sklearn|pandas|numpy|matplotlib|seaborn)\s+import",
            ],
            BlockReason.SUSPICIOUS_PATTERN: [
                r"rm\s+-rf",
                r'format\s*\(\s*[\'"][^\'"]*\{[^\'"]*\}',
                r'f[\'"][^\'"]*\{[^\'"]*\}',
                r"__[a-zA-Z_]+__",  # Magic methods
            ],
        }

        # License violation patterns
        self.license_patterns = {
            BlockReason.LICENSE_VIOLATION: [
                r"GNU\s+(General|Lesser)\s+Public\s+License",
                r"GPL\s+v[23]",
                r"LGPL\s+v[23]",
                r"AGPL\s+v[23]",
                r"Affero\s+General\s+Public\s+License",
                r"Copyright\s+\(c\)\s+\d{4}\s+[A-Za-z\s]+",
            ]
        }

        # Configuration limits
        self.max_lines = self.config.get("max_lines", 100)
        self.max_characters = self.config.get("max_characters", 5000)

        # Compile regex patterns for performance
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile all regex patterns for better performance."""
        self.compiled_patterns = {}

        for reason, patterns in self.dangerous_patterns.items():
            self.compiled_patterns[reason] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]

        for reason, patterns in self.license_patterns.items():
            self.compiled_patterns[reason] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]

    def validate_code(
        self, code: str, prompt: str = ""
    ) -> Tuple[bool, List[PolicyViolation]]:
        """
        Validate code against all security policies.

        Args:
            code: The code to validate
            prompt: The original prompt (for context)

        Returns:
            Tuple of (is_safe, violations)
        """
        violations = []

        # Check code size limits
        size_violations = self._check_size_limits(code)
        violations.extend(size_violations)

        # Check for dangerous patterns
        pattern_violations = self._check_dangerous_patterns(code)
        violations.extend(pattern_violations)

        # Check for license violations
        license_violations = self._check_license_violations(code)
        violations.extend(license_violations)

        # Check syntax (basic validation)
        syntax_violations = self._check_syntax(code)
        violations.extend(syntax_violations)

        # Code is safe if no violations (all violations should block)
        is_safe = len(violations) == 0

        return is_safe, violations

    def _check_size_limits(self, code: str) -> List[PolicyViolation]:
        """Check if code exceeds size limits."""
        violations = []
        lines = code.split("\n")

        if len(lines) > self.max_lines:
            violations.append(
                PolicyViolation(
                    reason=BlockReason.TOO_LARGE,
                    line_number=len(lines),
                    code_snippet=f"Code has {len(lines)} lines (max: {self.max_lines})",
                    severity="medium",
                    description=f"Code exceeds maximum line count of {self.max_lines}",
                )
            )

        if len(code) > self.max_characters:
            violations.append(
                PolicyViolation(
                    reason=BlockReason.TOO_LARGE,
                    line_number=0,
                    code_snippet=f"Code has {len(code)} characters (max: {self.max_characters})",
                    severity="medium",
                    description=f"Code exceeds maximum character count of {self.max_characters}",
                )
            )

        return violations

    def _check_dangerous_patterns(self, code: str) -> List[PolicyViolation]:
        """Check for dangerous patterns in code."""
        violations = []
        lines = code.split("\n")

        for line_num, line in enumerate(lines, 1):
            for reason, patterns in self.compiled_patterns.items():
                for pattern in patterns:
                    if pattern.search(line):
                        severity = (
                            "high"
                            if reason
                            in [
                                BlockReason.DANGEROUS_SYSTEM_CALL,
                                BlockReason.FILE_OPERATION,
                            ]
                            else "medium"
                        )

                        violations.append(
                            PolicyViolation(
                                reason=reason,
                                line_number=line_num,
                                code_snippet=line.strip(),
                                severity=severity,
                                description=f"Found {reason.value.replace('_', ' ')} on line {line_num}",
                            )
                        )

        return violations

    def _check_license_violations(self, code: str) -> List[PolicyViolation]:
        """Check for license violations in code."""
        violations = []
        lines = code.split("\n")

        for line_num, line in enumerate(lines, 1):
            for reason, patterns in self.compiled_patterns.items():
                if reason == BlockReason.LICENSE_VIOLATION:
                    for pattern in patterns:
                        if pattern.search(line):
                            violations.append(
                                PolicyViolation(
                                    reason=reason,
                                    line_number=line_num,
                                    code_snippet=line.strip(),
                                    severity="medium",
                                    description=f"Found license violation on line {line_num}",
                                )
                            )

        return violations

    def _check_syntax(self, code: str) -> List[PolicyViolation]:
        """Check basic Python syntax."""
        violations = []

        try:
            ast.parse(code)
        except SyntaxError as e:
            violations.append(
                PolicyViolation(
                    reason=BlockReason.SUSPICIOUS_PATTERN,
                    line_number=e.lineno or 0,
                    code_snippet=str(e),
                    severity="medium",
                    description=f"Syntax error: {e.msg}",
                )
            )

        return violations

    def get_block_summary(self, violations: List[PolicyViolation]) -> Dict:
        """Get a summary of blocked code for logging."""
        if not violations:
            return {"blocked": False, "reasons": []}

        reasons = {}
        for violation in violations:
            reason = violation.reason.value
            if reason not in reasons:
                reasons[reason] = 0
            reasons[reason] += 1

        return {
            "blocked": True,
            "reasons": reasons,
            "total_violations": len(violations),
            "high_severity": len([v for v in violations if v.severity == "high"]),
            "medium_severity": len([v for v in violations if v.severity == "medium"]),
            "low_severity": len([v for v in violations if v.severity == "low"]),
        }


# Global instance for easy access
policy_agent = PolicyAgent()


def validate_code_safety(
    code: str, prompt: str = ""
) -> Tuple[bool, List[PolicyViolation]]:
    """
    Convenience function to validate code safety.

    Args:
        code: The code to validate
        prompt: The original prompt (for context)

    Returns:
        Tuple of (is_safe, violations)
    """
    return policy_agent.validate_code(code, prompt)


if __name__ == "__main__":
    # Test the PolicyAgent
    test_codes = [
        # Safe code
        """def hello():
    print("Hello, World!")
    return "Hello, World!"
""",
        # Dangerous code
        """import os
def dangerous():
    os.system("rm -rf /")
    return "Deleted everything!"
""",
        # File operation
        """def write_secret():
    with open("secret.key", "w") as f:
        f.write("password123")
    return "Secret written"
""",
        # License violation
        """# GNU General Public License v3.0
def gpl_function():
    return "This is GPL code"
""",
    ]

    for i, code in enumerate(test_codes, 1):
        print(f"\n=== Test {i} ===")
        print(f"Code:\n{code}")

        is_safe, violations = validate_code_safety(code)
        print(f"Safe: {is_safe}")

        if violations:
            print("Violations:")
            for v in violations:
                print(f"  - {v.reason.value}: {v.description} (line {v.line_number})")
        else:
            print("No violations found")
