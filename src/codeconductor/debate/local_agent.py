"""
Local AI Agent for CodeConductor Debate System

Adapts the AI Project Advisor agent system to use local models instead of OpenAI.
"""

import logging
import os
from typing import Any

from ..ensemble.ensemble_engine import EnsembleEngine
from ..ensemble.model_manager import ModelManager

logger = logging.getLogger(__name__)


class LocalAIAgent:
    """Local AI Agent that uses CodeConductor's ensemble system"""

    def __init__(
        self, name: str, persona: str, model_manager: ModelManager | None = None
    ):
        self.name = name
        self.persona = persona
        self.conversation_history = [{"role": "system", "content": persona}]
        self.model_manager = model_manager or ModelManager()
        self.ensemble_engine = EnsembleEngine()

    def add_message(self, role: str, content: str):
        """Add a message to conversation history"""
        self.conversation_history.append({"role": role, "content": content})

    def _check_gpu_disabled(self) -> bool:
        """Kontrollera om GPU är inaktiverad för tester"""
        return os.getenv("CC_GPU_DISABLED", "0") == "1"

    async def generate_response(self, user_input: str) -> str:
        """Generate response using local ensemble"""
        # Kontrollera GPU_DISABLED först
        if self._check_gpu_disabled():
            return f"[MOCKED] {self.name} response to: {user_input}"

        self.add_message("user", user_input)

        try:
            # Use the public process_request method
            response = await self.ensemble_engine.process_request(
                task_description=user_input,
                timeout=30.0,
                prefer_fast_models=False,
                enable_fallback=True,
            )

            if response and "generated_code" in response:
                reply = response["generated_code"]
                self.add_message("assistant", reply)
                return reply
            else:
                logger.error("Failed to get response from ensemble for " + self.name)
            return "Error: Could not generate response for " + self.name

        except Exception as e:
            logger.error("Error generating response for " + self.name + ": " + str(e))
            return "Error: " + str(e)

    # Nya debate-metoder för att vara kompatibel med CodeConductorDebateManager
    def propose(self, prompt: str, **kw) -> dict[str, Any]:
        """Generate initial proposal - kompatibel med nya debate-systemet"""
        if self._check_gpu_disabled():
            return {
                "content": "[MOCKED] " + prompt,
                "agent": self.name,
                "type": "proposal",
            }

        # Använd generate_response som fallback
        try:
            import asyncio

            response = asyncio.run(self.generate_response(prompt))
            return {"content": response, "agent": self.name, "type": "proposal"}
        except Exception as e:
            return {
                "content": f"Error: {str(e)}",
                "agent": self.name,
                "type": "proposal",
            }

    def rebuttal(self, state: dict[str, Any], **kw) -> dict[str, Any]:
        """Generate rebuttal - kompatibel med nya debate-systemet"""
        if self._check_gpu_disabled():
            return {
                "content": "[MOCKED] rebuttal for debate state",
                "agent": self.name,
                "type": "rebuttal",
            }

        # Skapa prompt från state
        prompt = "Provide your rebuttal to the other proposals."
        if isinstance(state, dict) and "prompt" in state:
            prompt = state["prompt"]

        try:
            import asyncio

            response = asyncio.run(self.generate_response(prompt))
            return {"content": response, "agent": self.name, "type": "rebuttal"}
        except Exception as e:
            return {
                "content": f"Error: {str(e)}",
                "agent": self.name,
                "type": "rebuttal",
            }

    def finalize(self, state: dict[str, Any], **kw) -> dict[str, Any]:
        """Generate final recommendation - kompatibel med nya debate-systemet"""
        if self._check_gpu_disabled():
            return {
                "content": "[MOCKED] final recommendation for debate",
                "agent": self.name,
                "type": "final",
            }

        # Skapa prompt från state
        prompt = (
            "Based on the debate so far, what is your final recommendation "
            "for the code implementation?"
        )
        if isinstance(state, dict) and "prompt" in state:
            prompt = state["prompt"]

        try:
            import asyncio

            response = asyncio.run(self.generate_response(prompt))
            return {"content": response, "agent": self.name, "type": "final"}
        except Exception as e:
            return {"content": "Error: " + str(e), "agent": self.name, "type": "final"}

    def get_conversation_history(self) -> list:
        """Get full conversation history"""
        return self.conversation_history.copy()

    def reset_conversation(self):
        """Reset conversation history but keep persona"""
        self.conversation_history = [{"role": "system", "content": self.persona}]
