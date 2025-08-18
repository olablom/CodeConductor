#!/usr/bin/env python3
# filename: src/codeconductor/debate/base_agent.py
from __future__ import annotations
import os
from typing import Any, Dict
from ..utils.async_tools import ensure_async


class BaseAIAgent:
    def __init__(self, name: str = "Agent"):
        self.name = name
        # Wrappa metoderna EFTER att subklassen har definierat dem
        # Detta görs i en separat metod som subklassen anropar

    def _wrap_methods(self):
        """Wrappa alla metoder med ensure_async - anropa detta i subklassens
        __init__ sist"""
        # Spara original-metoderna
        self._propose_original = self.propose
        self._rebuttal_original = self.rebuttal
        self._finalize_original = self.finalize

        # Wrappa med ensure_async
        self.propose = ensure_async(self._propose_original)
        self.rebuttal = ensure_async(self._rebuttal_original)
        self.finalize = ensure_async(self._finalize_original)

    def _check_gpu_disabled(self) -> bool:
        """Kontrollera om GPU är inaktiverad för tester"""
        return os.getenv("CC_GPU_DISABLED", "0") == "1"

    # Subklasser definierar dessa (sync eller async går bra)
    def propose(self, prompt: str, **kw) -> Dict[str, Any]:
        if self._check_gpu_disabled():
            return {
                "content": "[MOCKED] " + str(prompt),
                "agent": self.name,
                "type": "proposal",
            }
        raise NotImplementedError("Subklasser måste implementera propose")

    def rebuttal(self, state: Dict[str, Any], **kw) -> Dict[str, Any]:
        if self._check_gpu_disabled():
            return {
                "content": "[MOCKED] rebuttal for " + str(state),
                "agent": self.name,
                "type": "rebuttal",
            }
        raise NotImplementedError("Subklasser måste implementera rebuttal")

    def finalize(self, state: Dict[str, Any], **kw) -> Dict[str, Any]:
        if self._check_gpu_disabled():
            return {
                "content": "[MOCKED] final for " + str(state),
                "agent": self.name,
                "type": "final",
            }
        raise NotImplementedError("Subklasser måste implementera finalize")
