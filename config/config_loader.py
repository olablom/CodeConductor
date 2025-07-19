"""
Configuration Loader for CodeConductor

Loads and manages configuration from YAML files.
Provides easy access to all system settings.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import os


class ConfigLoader:
    """Loads and manages configuration settings"""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self.config = {}
        self.load_config()

    def load_config(self):
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    def reload_config(self):
        """Reload configuration from file"""
        self.load_config()

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """Get configuration for a specific agent"""
        return self.get(f"agents.{agent_name}", {})

    def get_rl_config(self) -> Dict[str, Any]:
        """Get reinforcement learning configuration"""
        return self.get("reinforcement_learning", {})

    def get_cursor_config(self) -> Dict[str, Any]:
        """Get Cursor integration configuration"""
        return self.get("cursor_integration", {})

    def get_use_case_config(self, use_case: str) -> Dict[str, Any]:
        """Get configuration for a specific use case"""
        return self.get(f"use_cases.{use_case}", {})

    def get_testing_config(self) -> Dict[str, Any]:
        """Get testing configuration"""
        return self.get("testing", {})

    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        return self.get("database", {})

    def get_human_gate_config(self) -> Dict[str, Any]:
        """Get human-in-the-loop configuration"""
        return self.get("human_gate", {})

    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration"""
        return self.get("performance", {})

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self.get("logging", {})

    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration"""
        return self.get("security", {})

    def get_demo_config(self) -> Dict[str, Any]:
        """Get demo configuration"""
        return self.get("demo", {})

    def is_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled"""
        return self.get(f"{feature}.enabled", False)

    def get_all_use_cases(self) -> Dict[str, Any]:
        """Get all available use cases"""
        return self.get("use_cases", {})

    def get_optimization_strategies(self) -> list:
        """Get available RL optimization strategies"""
        return self.get("reinforcement_learning.optimization_strategies", [])

    def get_test_frameworks(self) -> list:
        """Get available test frameworks"""
        return self.get("testing.frameworks", [])

    def get_allowed_extensions(self) -> list:
        """Get allowed file extensions for code generation"""
        return self.get("security.allowed_extensions", [])

    def get_sample_projects(self) -> list:
        """Get sample projects for demo"""
        return self.get("demo.sample_projects", [])

    def get_development_tools(self) -> list:
        """Get development tools configuration"""
        return self.get("development.tools", [])

    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration and return any issues"""
        issues = []
        warnings = []

        # Check required sections
        required_sections = [
            "system",
            "agents",
            "reinforcement_learning",
            "human_gate",
            "cursor_integration",
            "database",
        ]

        for section in required_sections:
            if not self.get(section):
                issues.append(f"Missing required configuration section: {section}")

        # Check agent configurations
        required_agents = ["codegen_agent", "architect_agent", "reviewer_agent"]
        for agent in required_agents:
            agent_config = self.get_agent_config(agent)
            if not agent_config:
                issues.append(f"Missing configuration for required agent: {agent}")
            elif "confidence_threshold" not in agent_config:
                warnings.append(f"Agent {agent} missing confidence_threshold")

        # Check RL configuration
        rl_config = self.get_rl_config()
        if rl_config.get("enabled", False):
            if "learning_rate" not in rl_config:
                issues.append("RL enabled but learning_rate not specified")
            if "algorithm" not in rl_config:
                issues.append("RL enabled but algorithm not specified")

        # Check database configuration
        db_config = self.get_database_config()
        if not db_config.get("path"):
            warnings.append("Database path not specified")

        # Check Cursor configuration
        cursor_config = self.get_cursor_config()
        if cursor_config.get("enabled", False):
            if not cursor_config.get("cursor_path"):
                warnings.append("Cursor enabled but cursor_path not specified")

        return {"valid": len(issues) == 0, "issues": issues, "warnings": warnings}

    def get_environment_config(self) -> Dict[str, Any]:
        """Get environment-specific configuration"""
        environment = self.get("system.environment", "development")

        # Override with environment variables
        env_config = {}

        # Database
        db_url = os.getenv("DATABASE_URL")
        if db_url:
            env_config["database"] = {"url": db_url}

        # Logging
        log_level = os.getenv("LOG_LEVEL")
        if log_level:
            env_config["logging"] = {"level": log_level}

        # Cursor
        cursor_path = os.getenv("CURSOR_PATH")
        if cursor_path:
            env_config["cursor_integration"] = {"cursor_path": cursor_path}

        return env_config

    def export_config(self, file_path: str):
        """Export current configuration to file"""
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)

    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values"""

        def update_nested_dict(base_dict, update_dict):
            for key, value in update_dict.items():
                if (
                    isinstance(value, dict)
                    and key in base_dict
                    and isinstance(base_dict[key], dict)
                ):
                    update_nested_dict(base_dict[key], value)
                else:
                    base_dict[key] = value

        update_nested_dict(self.config, updates)

    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of the current configuration"""
        return {
            "system": {
                "name": self.get("system.name"),
                "version": self.get("system.version"),
                "environment": self.get("system.environment"),
            },
            "features": {
                "rl_enabled": self.is_enabled("reinforcement_learning"),
                "human_gate_enabled": self.is_enabled("human_gate"),
                "cursor_enabled": self.is_enabled("cursor_integration"),
                "automated_tests": self.get("testing.automated_tests", False),
            },
            "agents": list(self.get("agents", {}).keys()),
            "use_cases": list(self.get("use_cases", {}).keys()),
            "test_frameworks": [f["name"] for f in self.get_test_frameworks()],
            "optimization_strategies": [
                s["name"] for s in self.get_optimization_strategies()
            ],
        }


# Global configuration instance
config = ConfigLoader()


def get_config() -> ConfigLoader:
    """Get global configuration instance"""
    return config


def reload_config():
    """Reload global configuration"""
    config.reload_config()
