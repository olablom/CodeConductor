"""
Local AI Agent for CodeConductor Debate System

Adapts the AI Project Advisor agent system to use local models instead of OpenAI.
"""

import logging

from ..ensemble.ensemble_engine import EnsembleEngine
from ..ensemble.model_manager import ModelManager

logger = logging.getLogger(__name__)


class LocalAIAgent:
    """Local AI Agent that uses CodeConductor's ensemble system"""

    def __init__(self, name: str, persona: str, model_manager: ModelManager | None = None):
        self.name = name
        self.persona = persona
        self.conversation_history = [{"role": "system", "content": persona}]
        self.model_manager = model_manager or ModelManager()
        self.ensemble_engine = EnsembleEngine()

    def add_message(self, role: str, content: str):
        """Add a message to conversation history"""
        self.conversation_history.append({"role": role, "content": content})

    async def generate_response(self, user_input: str) -> str:
        """Generate response using local ensemble"""
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
                logger.error(f"Failed to get response from ensemble for {self.name}")
                return f"Error: Could not generate response for {self.name}"

        except Exception as e:
            logger.error(f"Error generating response for {self.name}: {e}")
            return f"Error: {str(e)}"

    def get_conversation_history(self) -> list:
        """Get full conversation history"""
        return self.conversation_history.copy()

    def reset_conversation(self):
        """Reset conversation history but keep persona"""
        self.conversation_history = [{"role": "system", "content": self.persona}]
