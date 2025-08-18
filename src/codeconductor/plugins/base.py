from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PluginContext:
    task_description: str
    code: str | None = None
    metadata: dict[str, Any] | None = None


class AgentPlugin:
    """
    Minimal plugin interface. Implementors should override the hooks they need.
    """

    name: str = "UnnamedPlugin"
    version: str = "0.1.0"

    def on_pre_generate(self, context: PluginContext) -> None:
        pass

    def on_post_generate(self, context: PluginContext) -> None:
        pass

    def on_validate(self, context: PluginContext) -> None:
        pass
