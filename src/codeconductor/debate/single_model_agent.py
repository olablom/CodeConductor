"""
Single Model Agent for CodeConductor

A simplified agent that uses only one model, similar to OpenAI API approach.
"""

import asyncio
import logging
from pathlib import Path

from ..ensemble.single_model_engine import SingleModelEngine, SingleModelRequest

logger = logging.getLogger(__name__)


class SingleModelAIAgent:
    """
    Simplified AI agent that uses a single model.

    This is similar to how OpenAI API works - one model, multiple agents.
    """

    def __init__(
        self,
        name: str,
        persona: str,
        preferred_model: str = "meta-llama-3.1-8b-instruct",
    ):
        self.name = name
        self.persona = persona
        self.preferred_model = preferred_model
        self.single_model_engine = SingleModelEngine(preferred_model)

    async def initialize(self) -> bool:
        """Initialize the agent."""
        try:
            logger.info(f"🚀 Initializing {self.name} agent...")
            success = await self.single_model_engine.initialize()
            if success:
                logger.info(f"✅ {self.name} agent initialized successfully")
            else:
                logger.error(f"❌ {self.name} agent initialization failed")
            return success
        except Exception as e:
            logger.error(f"❌ {self.name} agent initialization error: {e}")
            return False

    async def generate_response(self, user_prompt: str) -> str:
        """Generate a response using the single model."""
        try:
            # Combine persona and user prompt
            full_prompt = f"{self.persona}\n\nUser: {user_prompt}\n\n{self.name}:"

            # Create request
            request = SingleModelRequest(task_description=full_prompt, timeout=30.0)

            # Process request
            response = await self.single_model_engine.process_request(request)

            logger.info(f"✅ {self.name} generated response in {response.execution_time:.2f}s")
            return response.content

        except Exception as e:
            logger.error(f"❌ {self.name} response generation failed: {e}")
            return f"Error: {str(e)}"

    async def cleanup(self):
        """Clean up the agent."""
        try:
            logger.info(f"🧹 Cleaning up {self.name} agent...")
            await self.single_model_engine.cleanup()
            logger.info(f"✅ {self.name} agent cleanup completed")
        except Exception as e:
            logger.error(f"❌ {self.name} agent cleanup failed: {e}")


class SingleModelDebateManager:
    """
    Simplified debate manager that uses single model agents.
    """

    def __init__(self, agents: list):
        self.agents = agents
        self.full_transcript = []

    async def conduct_debate(self, user_prompt: str) -> dict:
        """Conduct a multi-agent debate using single model agents."""

        # Initialize all agents
        for agent in self.agents:
            await agent.initialize()

        try:
            # 1. Initial proposals from each agent
            print("\n--- Initial Proposals ---\n")
            for agent in self.agents:
                try:
                    response = await asyncio.wait_for(
                        agent.generate_response(user_prompt), timeout=30.0
                    )
                    self.full_transcript.append(
                        {"agent": agent.name, "turn": "proposal", "content": response}
                    )
                    print(f"{agent.name}:\n{response}\n")
                except TimeoutError:
                    print(f"⏰ {agent.name} timed out during proposal")
                    response = f"{agent.name} timed out during proposal generation."
                    self.full_transcript.append(
                        {"agent": agent.name, "turn": "proposal", "content": response}
                    )
                except Exception as e:
                    print(f"❌ {agent.name} error during proposal: {e}")
                    response = f"{agent.name} encountered an error: {str(e)}"
                    self.full_transcript.append(
                        {"agent": agent.name, "turn": "proposal", "content": response}
                    )

            # 2. Rebuttal round
            print("\n--- Rebuttal Round ---\n")
            for agent in self.agents:
                try:
                    others = [a for a in self.agents if a != agent]
                    rebuttal_prompt = "Here are the other agents' proposals:\n"

                    for other in others:
                        last_proposal = next(
                            (
                                entry["content"]
                                for entry in reversed(self.full_transcript)
                                if entry["agent"] == other.name and entry["turn"] == "proposal"
                            ),
                            "",
                        )
                        rebuttal_prompt += f"{other.name}: {last_proposal}\n"

                    rebuttal_prompt += "Please provide your rebuttal or counter-argument."
                    response = await asyncio.wait_for(
                        agent.generate_response(rebuttal_prompt), timeout=30.0
                    )
                    self.full_transcript.append(
                        {"agent": agent.name, "turn": "rebuttal", "content": response}
                    )
                    print(f"{agent.name} (rebuttal):\n{response}\n")
                except TimeoutError:
                    print(f"⏰ {agent.name} timed out during rebuttal")
                    response = f"{agent.name} timed out during rebuttal generation."
                    self.full_transcript.append(
                        {"agent": agent.name, "turn": "rebuttal", "content": response}
                    )
                except Exception as e:
                    print(f"❌ {agent.name} error during rebuttal: {e}")
                    response = f"{agent.name} encountered an error during rebuttal: {str(e)}"
                    self.full_transcript.append(
                        {"agent": agent.name, "turn": "rebuttal", "content": response}
                    )

            # 3. Final recommendations
            print("\n--- Final Recommendations ---\n")
            for agent in self.agents:
                try:
                    final_prompt = "Based on the debate so far, what is your final recommendation for the code implementation?"
                    response = await asyncio.wait_for(
                        agent.generate_response(final_prompt), timeout=30.0
                    )
                    self.full_transcript.append(
                        {
                            "agent": agent.name,
                            "turn": "final_recommendation",
                            "content": response,
                        }
                    )
                    print(f"{agent.name} (final recommendation):\n{response}\n")
                except TimeoutError:
                    print(f"⏰ {agent.name} timed out during final recommendation")
                    response = f"{agent.name} timed out during final recommendation generation."
                    self.full_transcript.append(
                        {
                            "agent": agent.name,
                            "turn": "final_recommendation",
                            "content": response,
                        }
                    )
                except Exception as e:
                    print(f"❌ {agent.name} error during final recommendation: {e}")
                    response = (
                        f"{agent.name} encountered an error during final recommendation: {str(e)}"
                    )
                    self.full_transcript.append(
                        {
                            "agent": agent.name,
                            "turn": "final_recommendation",
                            "content": response,
                        }
                    )

            return {
                "transcript": self.full_transcript,
                "agents": [agent.name for agent in self.agents],
                "total_turns": len(self.full_transcript),
            }

        except Exception as e:
            print(f"❌ Debate failed: {e}")
            return {
                "transcript": self.full_transcript,
                "agents": [agent.name for agent in self.agents],
                "total_turns": len(self.full_transcript),
                "error": str(e),
            }
        finally:
            # Cleanup all agents
            for agent in self.agents:
                await agent.cleanup()

    def extract_consensus(self) -> str:
        """Extract consensus from final recommendations."""
        final_recommendations = [
            entry["content"]
            for entry in self.full_transcript
            if entry["turn"] == "final_recommendation"
        ]

        if not final_recommendations:
            return "No final recommendations found"

        consensus = "\n\n".join(
            [
                f"## {entry['agent']}\n{entry['content']}"
                for entry in self.full_transcript
                if entry["turn"] == "final_recommendation"
            ]
        )

        return consensus

    def save_transcript(self, filename: str = "debate_transcript.json") -> Path:
        """Save debate transcript to file"""
        import json
        from datetime import datetime
        from pathlib import Path

        # Skapa artifacts/runs om den inte finns
        output_dir = Path("artifacts/runs")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / filename

        transcript_data = {
            "agents": ([agent.name for agent in self.agents] if hasattr(self, "agents") else []),
            "turns": self.full_transcript,
            "timestamp": datetime.now().isoformat(),
            "model": getattr(self, "model_name", "unknown"),
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(transcript_data, f, indent=2, ensure_ascii=False)

        return output_file
