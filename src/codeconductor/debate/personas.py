"""
Utilities for loading agent personas from YAML and constructing LocalAIAgent instances.
"""

from __future__ import annotations

import yaml
from typing import Dict, Any, List, Optional
from pathlib import Path
from functools import lru_cache

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


# Back-compat API expected by tests
@lru_cache(maxsize=None)
def _load_all_personas() -> Dict[str, Dict[str, Any]]:
    """Load all persona YAMLs into a dict keyed by persona name."""
    data: Dict[str, Dict[str, Any]] = {}

    # 1) Samlad YAML (om du har en personas.yaml)
    base_dir = Path(__file__).resolve().parent
    yml = base_dir / "personas.yaml"
    if yml.exists():
        with yml.open("r", encoding="utf-8") as f:
            obj = yaml.safe_load(f) or {}
        if isinstance(obj, dict):
            for name, cfg in obj.items():
                if isinstance(cfg, dict):
                    data[name] = cfg

    # 2) Lägg till defaults om inga YAML-filer hittades
    if not data:
        data.update(DEFAULT_PERSONAS_YAML)

    return data


def list_personas() -> Dict[str, Dict[str, Any]]:
    """Return a copy of loaded personas (for testing/introspection)."""
    return dict(_load_all_personas())


def get_persona_prompt(name: str) -> str:
    """
    Back-compat API expected by tests.
    Returns the textual prompt for the given persona.
    """
    personas = _load_all_personas()
    cfg = personas.get(name)
    if not cfg:
        raise KeyError(f"Persona '{name}' not found")
    
    # Vanliga fältnamn: "prompt", "system", "template"
    for key in ("prompt", "system", "template"):
        if isinstance(cfg.get(key), str) and cfg[key].strip():
            return cfg[key].strip()
    
    # Tillåt 'messages' [{role, content}] – slå ihop till en systemprompt
    msgs = cfg.get("messages")
    if isinstance(msgs, list) and msgs:
        parts = []
        for m in msgs:
            if isinstance(m, dict) and isinstance(m.get("content"), str):
                parts.append(m["content"])
        if parts:
            return "\n\n".join(parts).strip()
    
    # Fallback: bygg prompt från style och rules
    style = cfg.get("style", name)
    rules = cfg.get("rules", [])
    if isinstance(rules, list) and rules:
        prompt = f"You are {name.title()} – style: {style}.\n"
        prompt += "Guidelines:\n" + "\n".join(f"- {r}" for r in rules)
        return prompt.strip()
    
    # Sista utväg: returnera namnet
    return f"You are {name.title()}."


# Om din nuvarande kod använder ett nytt API-namn, exponera alias:
def load_persona(name: str) -> str: 
    return get_persona_prompt(name)


__all__ = [
    "load_personas_yaml", "build_agents_from_personas",
    "get_persona_prompt", "list_personas", "load_persona"
]
