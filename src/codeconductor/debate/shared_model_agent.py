"""
Shared Model Agent for CodeConductor

All agents share the same model instance, similar to OpenAI API approach.
"""

import asyncio
import logging
from typing import Optional
from pathlib import Path

from ..ensemble.single_model_engine import SingleModelEngine, SingleModelRequest

logger = logging.getLogger(__name__)


class SharedModelAIAgent:
    """
    AI agent that shares a model with other agents.
    
    This prevents the VRAM explosion from multiple model instances.
    """
    
    def __init__(self, name: str, persona: str, shared_engine: SingleModelEngine):
        self.name = name
        self.persona = persona
        self.shared_engine = shared_engine  # All agents share the same engine
        
    async def generate_response(self, user_prompt: str) -> str:
        """Generate a response using the shared model."""
        try:
            # Combine persona and user prompt
            full_prompt = f"{self.persona}\n\nUser: {user_prompt}\n\n{self.name}:"
            
            # Create request
            request = SingleModelRequest(
                task_description=full_prompt,
                timeout=30.0
            )
            
            # Process request using shared engine
            response = await self.shared_engine.process_request(request)
            
            logger.info(f"✅ {self.name} generated response in {response.execution_time:.2f}s")
            return response.content
            
        except Exception as e:
            logger.error(f"❌ {self.name} response generation failed: {e}")
            return f"Error: {str(e)}"


class SharedModelDebateManager:
    """
    Debate manager that uses agents sharing a single model.
    """
    
    def __init__(self, agents: list, shared_engine: SingleModelEngine):
        self.agents = agents
        self.shared_engine = shared_engine
        self.full_transcript = []
        
    async def conduct_debate(self, user_prompt: str) -> dict:
        """Conduct a multi-agent debate using shared model agents."""
        
        try:
            # 1. Initial proposals from each agent
            print("\n--- Initial Proposals ---\n")
            for agent in self.agents:
                try:
                    response = await asyncio.wait_for(
                        agent.generate_response(user_prompt),
                        timeout=30.0
                    )
                    self.full_transcript.append(
                        {"agent": agent.name, "turn": "proposal", "content": response}
                    )
                    print(f"{agent.name}:\n{response}\n")
                except asyncio.TimeoutError:
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
                        agent.generate_response(rebuttal_prompt),
                        timeout=30.0
                    )
                    self.full_transcript.append(
                        {"agent": agent.name, "turn": "rebuttal", "content": response}
                    )
                    print(f"{agent.name} (rebuttal):\n{response}\n")
                except asyncio.TimeoutError:
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
                        agent.generate_response(final_prompt),
                        timeout=30.0
                    )
                    self.full_transcript.append(
                        {
                            "agent": agent.name,
                            "turn": "final_recommendation",
                            "content": response,
                        }
                    )
                    print(f"{agent.name} (final recommendation):\n{response}\n")
                except asyncio.TimeoutError:
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
                    response = f"{agent.name} encountered an error during final recommendation: {str(e)}"
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
                "error": str(e)
            }
        finally:
            # Cleanup shared engine once
            await self.shared_engine.cleanup()
    
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
        from pathlib import Path
        from datetime import datetime
        
        # Skapa artifacts/runs om den inte finns
        output_dir = Path("artifacts/runs")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / filename
        
        transcript_data = {
            "agents": [agent.name for agent in self.agents] if hasattr(self, 'agents') else [],
            "turns": self.full_transcript,
            "timestamp": datetime.now().isoformat(),
            "model": getattr(self, 'model_name', 'unknown')
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(transcript_data, f, indent=2, ensure_ascii=False)
        
        return output_file 