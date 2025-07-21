"""
PolicyLoader - Dynamic YAML-based policy management for CodeConductor v2.0

This module provides functionality to load, validate, and manage policy configurations
from YAML files, allowing users to customize security, quality, and style policies
without modifying code.
"""

import yaml
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class PolicySeverity(Enum):
    """Policy violation severity levels"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class PolicyViolation:
    """Represents a policy violation"""

    rule_name: str
    description: str
    severity: PolicySeverity
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    suggestion: Optional[str] = None


class PolicyLoader:
    """
    Loads and manages policy configurations from YAML files.

    Supports dynamic policy loading, validation, and enforcement modes.
    """

    def __init__(self, path: str = "config/policies.yaml"):
        self.path = Path(path)
        self.policies: Dict[str, Any] = {}
        self._load_policies()

    def _load_policies(self) -> None:
        """Load policies from YAML file"""
        if not self.path.exists():
            raise FileNotFoundError(f"Policy file not found: {self.path}")

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self.policies = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in policy file: {e}")

        self._validate_policies()

    def _validate_policies(self) -> None:
        """Validate policy structure and values"""
        required_sections = ["security", "quality", "style"]

        for section in required_sections:
            if section not in self.policies:
                raise ValueError(f"Missing required policy section: {section}")

        # Validate severity levels
        if "severity_levels" in self.policies:
            for level, rules in self.policies["severity_levels"].items():
                if not isinstance(rules, list):
                    raise ValueError(
                        f"Severity level {level} must contain a list of rules"
                    )

    def reload(self) -> Dict[str, Any]:
        """Reload policies from file"""
        self._load_policies()
        return self.policies

    def get_policy(self, section: str, key: str, default: Any = None) -> Any:
        """Get a specific policy value"""
        return self.policies.get(section, {}).get(key, default)

    def get_severity_level(self, rule_name: str) -> PolicySeverity:
        """Get severity level for a specific rule"""
        severity_levels = self.policies.get("severity_levels", {})

        for level, rules in severity_levels.items():
            if rule_name in rules:
                return PolicySeverity(level)

        return PolicySeverity.MEDIUM  # Default severity

    def get_enforcement_mode(self) -> str:
        """Get current enforcement mode"""
        return self.policies.get("enforcement", {}).get("mode", "strict")

    def get_max_violations(self) -> int:
        """Get maximum allowed violations before blocking"""
        return self.policies.get("enforcement", {}).get("max_violations", 5)

    def is_auto_fix_enabled(self) -> bool:
        """Check if auto-fix is enabled"""
        return self.policies.get("enforcement", {}).get("auto_fix", False)

    def get_blocked_imports(self) -> List[str]:
        """Get list of blocked imports"""
        return self.get_policy("security", "blocked_imports", [])

    def get_blocked_patterns(self) -> List[str]:
        """Get list of blocked patterns"""
        return self.get_policy("security", "blocked_patterns", [])

    def get_forbidden_functions(self) -> List[str]:
        """Get list of forbidden functions"""
        return self.get_policy("quality", "forbidden_functions", [])

    def get_forbidden_variable_names(self) -> List[str]:
        """Get list of forbidden variable names"""
        return self.get_policy("style", "forbidden_variable_names", [])

    def get_max_line_length(self) -> int:
        """Get maximum line length"""
        return self.get_policy("style", "max_line_length", 88)

    def get_max_complexity(self) -> int:
        """Get maximum complexity"""
        return self.get_policy("quality", "max_complexity", 10)

    def get_min_test_coverage(self) -> int:
        """Get minimum test coverage"""
        return self.get_policy("quality", "min_test_coverage", 80)

    def check_import_violations(self, code: str) -> List[PolicyViolation]:
        """Check for blocked import violations"""
        violations = []
        blocked_imports = self.get_blocked_imports()

        lines = code.split("\n")
        for line_num, line in enumerate(lines, 1):
            line = line.strip()

            # Check for import statements
            if line.startswith("import ") or line.startswith("from "):
                for blocked_import in blocked_imports:
                    if (
                        f"import {blocked_import}" in line
                        or f"from {blocked_import}" in line
                    ):
                        violations.append(
                            PolicyViolation(
                                rule_name="blocked_imports",
                                description=f"Blocked import detected: {blocked_import}",
                                severity=self.get_severity_level("blocked_imports"),
                                line_number=line_num,
                                code_snippet=line,
                                suggestion=f"Remove or replace import of {blocked_import}",
                            )
                        )

        return violations

    def check_pattern_violations(self, code: str) -> List[PolicyViolation]:
        """Check for blocked pattern violations"""
        violations = []
        blocked_patterns = self.get_blocked_patterns()

        lines = code.split("\n")
        for line_num, line in enumerate(lines, 1):
            for pattern in blocked_patterns:
                if pattern in line:
                    violations.append(
                        PolicyViolation(
                            rule_name="blocked_patterns",
                            description=f"Blocked pattern detected: {pattern}",
                            severity=self.get_severity_level("blocked_patterns"),
                            line_number=line_num,
                            code_snippet=line,
                            suggestion=f"Remove or replace pattern: {pattern}",
                        )
                    )

        return violations

    def check_function_violations(self, code: str) -> List[PolicyViolation]:
        """Check for forbidden function violations"""
        violations = []
        forbidden_functions = self.get_forbidden_functions()

        lines = code.split("\n")
        for line_num, line in enumerate(lines, 1):
            for func in forbidden_functions:
                # Check for function calls
                if f"{func}(" in line:
                    violations.append(
                        PolicyViolation(
                            rule_name="forbidden_functions",
                            description=f"Forbidden function detected: {func}",
                            severity=self.get_severity_level("forbidden_functions"),
                            line_number=line_num,
                            code_snippet=line,
                            suggestion=f"Replace {func} with appropriate alternative",
                        )
                    )

        return violations

    def check_variable_name_violations(self, code: str) -> List[PolicyViolation]:
        """Check for forbidden variable name violations"""
        violations = []
        forbidden_names = self.get_forbidden_variable_names()

        lines = code.split("\n")
        for line_num, line in enumerate(lines, 1):
            for name in forbidden_names:
                # Check for variable assignments
                pattern = rf"\b{re.escape(name)}\s*="
                if re.search(pattern, line):
                    violations.append(
                        PolicyViolation(
                            rule_name="forbidden_variable_names",
                            description=f"Forbidden variable name detected: {name}",
                            severity=self.get_severity_level(
                                "forbidden_variable_names"
                            ),
                            line_number=line_num,
                            code_snippet=line,
                            suggestion=f"Use a more descriptive variable name instead of {name}",
                        )
                    )

        return violations

    def check_line_length_violations(self, code: str) -> List[PolicyViolation]:
        """Check for line length violations"""
        violations = []
        max_length = self.get_max_line_length()

        lines = code.split("\n")
        for line_num, line in enumerate(lines, 1):
            if len(line) > max_length:
                violations.append(
                    PolicyViolation(
                        rule_name="max_line_length",
                        description=f"Line too long: {len(line)} characters (max: {max_length})",
                        severity=self.get_severity_level("max_line_length"),
                        line_number=line_num,
                        code_snippet=line[:max_length] + "..."
                        if len(line) > max_length
                        else line,
                        suggestion="Break long line into multiple lines",
                    )
                )

        return violations

    def analyze_code(self, code: str) -> Dict[str, Any]:
        """
        Comprehensive code analysis against all policies

        Returns:
            Dict containing violations, summary, and recommendations
        """
        all_violations = []

        # Run all policy checks
        all_violations.extend(self.check_import_violations(code))
        all_violations.extend(self.check_pattern_violations(code))
        all_violations.extend(self.check_function_violations(code))
        all_violations.extend(self.check_variable_name_violations(code))
        all_violations.extend(self.check_line_length_violations(code))

        # Group violations by severity
        violations_by_severity = {
            PolicySeverity.CRITICAL: [],
            PolicySeverity.HIGH: [],
            PolicySeverity.MEDIUM: [],
            PolicySeverity.LOW: [],
        }

        for violation in all_violations:
            violations_by_severity[violation.severity].append(violation)

        # Determine if code should be blocked
        enforcement_mode = self.get_enforcement_mode()
        max_violations = self.get_max_violations()

        critical_violations = len(violations_by_severity[PolicySeverity.CRITICAL])
        high_violations = len(violations_by_severity[PolicySeverity.HIGH])

        should_block = False
        if enforcement_mode == "strict":
            should_block = critical_violations > 0 or high_violations > 0
        elif enforcement_mode == "warning":
            should_block = critical_violations > 0
        elif enforcement_mode == "lenient":
            should_block = critical_violations > 2

        # Check total violation count
        if len(all_violations) > max_violations:
            should_block = True

        return {
            "violations": all_violations,
            "violations_by_severity": violations_by_severity,
            "summary": {
                "total_violations": len(all_violations),
                "critical": critical_violations,
                "high": high_violations,
                "medium": len(violations_by_severity[PolicySeverity.MEDIUM]),
                "low": len(violations_by_severity[PolicySeverity.LOW]),
            },
            "should_block": should_block,
            "enforcement_mode": enforcement_mode,
            "recommendations": self._generate_recommendations(all_violations),
        }

    def _generate_recommendations(self, violations: List[PolicyViolation]) -> List[str]:
        """Generate recommendations based on violations"""
        recommendations = []

        if not violations:
            recommendations.append("✅ No policy violations detected!")
            return recommendations

        # Group violations by type
        violation_types = {}
        for violation in violations:
            if violation.rule_name not in violation_types:
                violation_types[violation.rule_name] = []
            violation_types[violation.rule_name].append(violation)

        # Generate specific recommendations
        for rule_name, rule_violations in violation_types.items():
            count = len(rule_violations)
            if rule_name == "blocked_imports":
                recommendations.append(
                    f"🔒 Remove {count} blocked import(s) for security"
                )
            elif rule_name == "blocked_patterns":
                recommendations.append(f"🚫 Remove {count} dangerous pattern(s)")
            elif rule_name == "forbidden_functions":
                recommendations.append(
                    f"⚠️ Replace {count} forbidden function(s) with better alternatives"
                )
            elif rule_name == "forbidden_variable_names":
                recommendations.append(
                    f"📝 Use more descriptive variable names ({count} violation(s))"
                )
            elif rule_name == "max_line_length":
                recommendations.append(f"📏 Break {count} long line(s) for readability")

        return recommendations

    def save_policies(self, yaml_content: str) -> None:
        """Save policies back to YAML file from string content"""
        try:
            # Validate YAML first
            parsed_policies = yaml.safe_load(yaml_content)
            if not isinstance(parsed_policies, dict):
                raise ValueError("YAML content must be a dictionary")

            # Save to file
            with open(self.path, "w", encoding="utf-8") as f:
                f.write(yaml_content)

            # Update internal policies
            self.policies = parsed_policies
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML: {e}")
        except Exception as e:
            raise ValueError(f"Failed to save policies: {e}")

    def get_policy_summary(self) -> Dict[str, Any]:
        """Get a summary of current policies"""
        return {
            "file_path": str(self.path),
            "sections": list(self.policies.keys()),
            "enforcement_mode": self.get_enforcement_mode(),
            "auto_fix_enabled": self.is_auto_fix_enabled(),
            "max_violations": self.get_max_violations(),
            "blocked_imports_count": len(self.get_blocked_imports()),
            "blocked_patterns_count": len(self.get_blocked_patterns()),
            "forbidden_functions_count": len(self.get_forbidden_functions()),
            "forbidden_variable_names_count": len(self.get_forbidden_variable_names()),
        }
