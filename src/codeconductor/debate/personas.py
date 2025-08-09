"""
Utilities for loading agent personas from YAML and constructing LocalAIAgent instances.
"""

from __future__ import annotations

import yaml
from typing import Dict, Any, List, Optional

from .local_ai_agent import LocalAIAgent


DEFAULT_PERSONAS_YAML = {
    "architect": {
        "style": "skeptical-architect",
        "rules": [
            "challenge assumptions",
            "prefer small, incremental changes",
        ],
    },
    "coder": {
        "style": "pragmatic-coder",
        "rules": [
            "write tests first if missing",
            "optimize for maintainability",
        ],
    },
    "reviewer": {
        "style": "security-stickler",
        "rules": [
            "check for OWASP Top 10 vulnerabilities",
            "prefer explicit over implicit",
        ],
    },
}


def load_personas_yaml(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """Load personas mapping from YAML file; return defaults if path is None.

    Structure:
      role:
        style: str
        rules: [str, ...]
    """
    if not path:
        return DEFAULT_PERSONAS_YAML
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError("Personas YAML must be a mapping of roles")
        return data  # type: ignore[return-value]


def build_agents_from_personas(
    personas: Dict[str, Dict[str, Any]], roles: Optional[List[str]] = None
) -> List[LocalAIAgent]:
    """Create LocalAIAgent list from personas mapping.

    If roles is provided, only include those keys in that order. Otherwise include all keys.
    """
    selected_roles = roles or list(personas.keys())
    agents: List[LocalAIAgent] = []
    for role in selected_roles:
        spec = personas.get(role) or {}
        style = spec.get("style", role)
        rules = spec.get("rules", [])
        if not isinstance(rules, list):
            rules = [str(rules)]

        persona_text = (
            f"You are {role.title()} – style: {style}.\n"
            + ("Guidelines:\n" + "\n".join(f"- {r}" for r in rules) if rules else "")
        ).strip()

        agent = LocalAIAgent(role.title(), persona_text)
        agents.append(agent)
    return agents
