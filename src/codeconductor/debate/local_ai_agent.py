"""
Local AI Agent for CodeConductor

A simple agent that uses local models, following the same structure as the OpenAI version.
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from .base_agent import BaseAIAgent

if TYPE_CHECKING:  # type hints only, avoid heavy imports at runtime
    from ..ensemble.single_model_engine import SingleModelEngine

logger = logging.getLogger(__name__)


class LocalAIAgent(BaseAIAgent):
    """
    Simple AI agent that uses local models, similar to the OpenAI version.

    This follows the same structure as ai-project-advisor/src/core/ai_agent.py
    but uses our local model instead of OpenAI.
    """

    def __init__(self, name: str, persona: str):
        super().__init__(name)  # Anropa BaseAIAgent.__init__
        self.persona = persona
        self.conversation_history = [{"role": "system", "content": persona}]
        self.shared_engine = None  # Will be set by debate manager

        # VIKTIGT: Wrappa metoderna EFTER att alla är definierade
        self._wrap_methods()

    def set_shared_engine(self, engine: "SingleModelEngine"):
        """Set the shared engine for this agent."""
        self.shared_engine = engine

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history."""
        self.conversation_history.append({"role": role, "content": content})

    def propose(self, prompt: str, **kw) -> dict[str, Any]:
        """Generate initial proposal"""
        # Kontrollera GPU_DISABLED först
        if self._check_gpu_disabled():
            return {
                "content": "[MOCKED] " + prompt,
                "agent": self.name,
                "type": "proposal",
            }

        # Riktig implementation endast om GPU är tillgänglig
        if self.shared_engine is None:
            return {
                "content": "Error: No shared engine set for " + self.name,
                "agent": self.name,
                "type": "proposal",
            }

        try:
            response = asyncio.run(self.generate_response(prompt))
            return {"content": response, "agent": self.name, "type": "proposal"}
        except Exception as e:
            return {
                "content": "Error: " + str(e),
                "agent": self.name,
                "type": "proposal",
            }

    def rebuttal(self, state: dict[str, Any], **kw) -> dict[str, Any]:
        """Generate rebuttal based on other agents' proposals"""
        # Kontrollera GPU_DISABLED först
        if self._check_gpu_disabled():
            return {
                "content": "[MOCKED] rebuttal for debate state",
                "agent": self.name,
                "type": "rebuttal",
            }

        prompt = (
            "Based on the debate state: " + str(state) +
            ", provide your rebuttal."
        )

        if self.shared_engine is None:
            return {
                "content": "Error: No shared engine set for " + self.name,
                "agent": self.name,
                "type": "rebuttal",
            }

        try:
            response = asyncio.run(self.generate_response(prompt))
            return {"content": response, "agent": self.name, "type": "rebuttal"}
        except Exception as e:
            return {
                "content": "Error: " + str(e),
                "agent": self.name,
                "type": "rebuttal",
            }

    def finalize(self, state: dict[str, Any], **kw) -> dict[str, Any]:
        """Generate final recommendation"""
        # Kontrollera GPU_DISABLED först
        if self._check_gpu_disabled():
            return {
                "content": "[MOCKED] final recommendation for debate",
                "agent": self.name,
                "type": "final",
            }

        prompt = (
            "Based on the debate state: "
            + str(state)
            + ", provide your final recommendation."
        )

        if self.shared_engine is None:
            return {
                "content": "Error: No shared engine set for " + self.name,
                "agent": self.name,
                "type": "final",
            }

        try:
            response = asyncio.run(self.generate_response(prompt))
            return {"content": response, "agent": self.name, "type": "final"}
        except Exception as e:
            return {"content": "Error: " + str(e), "agent": self.name, "type": "final"}

    async def generate_response(self, user_prompt: str, timeout: float = 120.0) -> str:
        """
        Generate a response using the local model.

        Args:
            user_prompt: The user's prompt
            timeout: Timeout in seconds (increased from 30 to 120)

        Returns:
            The generated response
        """
        # Kontrollera GPU_DISABLED först
        if self._check_gpu_disabled():
            return "[MOCKED] " + self.name + " response to: " + user_prompt

        try:
            # Create the full prompt with persona
            full_prompt = (
                "System: "
                + self.persona
                + "\n\nUser: "
                + user_prompt
                + "\n\n"
                + self.name
                + ":"
            )

            logger.info(
                "[PERSONA] "
                + self.name
                + ": "
                + (self.persona.splitlines()[0] if self.persona else "")
            )
            logger.info(f"{self.name} generating response...")

            # Use the shared engine to generate response
            try:
                from ..ensemble.single_model_engine import (
                    SingleModelRequest,
                )  # lazy import
            except Exception:
                # Minimal fallback to avoid heavy imports during CI quick mode
                class SingleModelRequest:  # type: ignore
                    def __init__(self, task_description: str, timeout: float):
                        self.task_description = task_description
                        self.timeout = timeout

            request = SingleModelRequest(task_description=full_prompt, timeout=timeout)

            response = await asyncio.wait_for(
                self.shared_engine.process_request(request), timeout=timeout
            )

            if response and response.content:
                logger.info(f"{self.name} generated response successfully")
                return response.content
            else:
                logger.warning(f"{self.name} got empty response")
                return f"{self.name} generated an empty response."

        except TimeoutError:
            logger.error(f"{self.name} timed out after {timeout} seconds")
            return f"{self.name} timed out during response generation."
        except Exception as e:
            logger.error(f"{self.name} error: {str(e)}")
            return f"{self.name} encountered an error: {str(e)}"


class LocalDebateManager:
    """
    Simple debate manager that uses local agents, similar to the OpenAI version.

    This follows the same structure as ai-project-advisor/src/core/debate_manager.py
    but uses our local agents instead of OpenAI.
    """

    def __init__(self, agents: list[LocalAIAgent]):
        self.agents = agents
        self.full_transcript = []
        self.shared_engine = None

    def set_shared_engine(self, engine: "SingleModelEngine"):
        """Set the shared engine for all agents."""
        self.shared_engine = engine
        for agent in self.agents:
            agent.set_shared_engine(engine)

    async def conduct_debate(
        self,
        user_prompt: str,
        timeout_per_turn: float = 60.0,
        rounds: int = 1,
    ) -> list[dict[str, str]]:
        """
        Conduct a debate between all agents.

        Args:
            user_prompt: The user's prompt to debate
            timeout_per_turn: Timeout for each agent's turn (default 60 seconds)
            rounds: How many phases to run (1 = proposals only, 2 = +rebuttals, 3 = +final)

        Returns:
            List of debate responses
        """
        debate_responses = []

        # Phase 1: Initial proposals
        logger.info("Starting debate - Phase 1: Initial proposals")
        for agent in self.agents:
            try:
                logger.info(f"{agent.name} making proposal...")
                response = await asyncio.wait_for(
                    agent.generate_response(user_prompt, timeout=timeout_per_turn),
                    timeout=timeout_per_turn,
                )
                debate_responses.append(
                    {"agent": agent.name, "turn": "proposal", "content": response}
                )
                logger.info(f"{agent.name} proposal completed")
            except TimeoutError:
                logger.error(f"{agent.name} timed out during proposal")
                debate_responses.append(
                    {
                        "agent": agent.name,
                        "turn": "proposal",
                        "content": (
                            f"{agent.name} timed out during proposal generation."
                        ),
                    }
                )
            except Exception as e:
                logger.error(f"{agent.name} error during proposal: {e}")
                debate_responses.append(
                    {
                        "agent": agent.name,
                        "turn": "proposal",
                        "content": f"{agent.name} encountered an error: {str(e)}",
                    }
                )

        # Phase 2: Rebuttals (only if rounds >= 2)
        if rounds >= 2:
            logger.info("Debate - Phase 2: Rebuttals")
            for agent in self.agents:
                try:
                    # Create rebuttal prompt with other agents' proposals
                    other_proposals = [
                        r
                        for r in debate_responses
                        if r["turn"] == "proposal" and r["agent"] != agent.name
                    ]
                    rebuttal_prompt = (
                        f"User: {user_prompt}\n\nOther agents' proposals:\n"
                    )
                    for prop in other_proposals:
                        rebuttal_prompt += (
                            f"- {prop['agent']}: " f"{prop['content'][:200]}...\n\n"
                        )
                    rebuttal_prompt += (
                        "Please provide your rebuttal to these proposals:"
                    )

                    logger.info(f"{agent.name} making rebuttal...")
                    response = await asyncio.wait_for(
                        agent.generate_response(
                            rebuttal_prompt, timeout=timeout_per_turn
                        ),
                        timeout=timeout_per_turn,
                    )
                    debate_responses.append(
                        {
                            "agent": agent.name,
                            "turn": "rebuttal",
                            "content": response,
                        }
                    )
                    logger.info(f"{agent.name} rebuttal completed")
                except TimeoutError:
                    logger.error(f"{agent.name} timed out during rebuttal")
                    debate_responses.append(
                        {
                            "agent": agent.name,
                            "turn": "rebuttal",
                            "content": (
                                f"{agent.name} timed out during rebuttal generation."
                            ),
                        }
                    )
                except Exception as e:
                    logger.error(f"{agent.name} error during rebuttal: {e}")
                    debate_responses.append(
                        {
                            "agent": agent.name,
                            "turn": "rebuttal",
                            "content": f"{agent.name} encountered an error: {str(e)}",
                        }
                    )

        # Phase 3: Final recommendations (only if rounds >= 3)
        if rounds >= 3:
            logger.info("Debate - Phase 3: Final recommendations")
            for agent in self.agents:
                try:
                    # Create final recommendation prompt
                    final_prompt = (
                        f"User: {user_prompt}\n\nBased on all the proposals and "
                        f"rebuttals, provide your final recommendation:"
                    )

                    logger.info(f"{agent.name} making final recommendation...")
                    response = await asyncio.wait_for(
                        agent.generate_response(final_prompt, timeout=timeout_per_turn),
                        timeout=timeout_per_turn,
                    )
                    debate_responses.append(
                        {
                            "agent": agent.name,
                            "turn": "final_recommendation",
                            "content": response,
                        }
                    )
                    logger.info(f"{agent.name} final recommendation completed")
                except TimeoutError:
                    logger.error(f"{agent.name} timed out during final recommendation")
                    debate_responses.append(
                        {
                            "agent": agent.name,
                            "turn": "final_recommendation",
                            "content": (
                            f"{agent.name} timed out during final recommendation "
                            f"generation."
                        ),
                        }
                    )
                except Exception as e:
                    logger.error(f"{agent.name} error during final recommendation: {e}")
                    debate_responses.append(
                        {
                            "agent": agent.name,
                            "turn": "final_recommendation",
                            "content": f"{agent.name} encountered an error: {str(e)}",
                        }
                    )

        logger.info("Debate completed successfully!")
        return debate_responses

    def get_transcript(self) -> list[dict[str, Any]]:
        """Get the full debate transcript."""
        return self.full_transcript
