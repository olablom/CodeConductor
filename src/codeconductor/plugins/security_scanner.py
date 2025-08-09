from __future__ import annotations

from .base import AgentPlugin, PluginContext


class SecurityScannerPlugin(AgentPlugin):
    name = "SecurityScanner"
    version = "0.1.0"

    def on_validate(self, context: PluginContext) -> None:
        code = context.code or ""
        issues = []
        if "exec(" in code or "eval(" in code:
            issues.append("Avoid exec/eval usage")
        if "subprocess.Popen(" in code and "shell=True" in code:
            issues.append("Avoid shell=True in subprocess")
        if "password=" in code or "SECRET_KEY" in code:
            issues.append("Hardcoded secrets detected")

        if issues:
            meta = context.metadata or {}
            meta.setdefault("security_issues", []).extend(issues)
            context.metadata = meta
