#!/usr/bin/env python3
"""
CodeConductor Debate Manager - Multi-Agent Debate System
"""

import asyncio
import yaml
import json
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass

from .local_agent import LocalAIAgent
from .personas import get_persona_prompt

logger = logging.getLogger(__name__)


class SingleModelDebateManager:
    """Base debate manager for single model debates"""
    
    def __init__(self, *args, **kwargs):
        self.transcript = []
        # optional attrs used by save_transcript()
        self.agents = getattr(self, "agents", [])
        self.model_name = getattr(self, "model_name", "unknown")
        
    def save_transcript(self, filename: str = "debate_transcript.json"):
        """
        Save debate transcript to artifacts/runs/<filename> as JSON.
        Safe to call even if agents/model_name are unset.
        """
        output_dir = Path("artifacts/runs")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / filename

        payload = {
            "agents": [getattr(a, "name", str(a)) for a in self.agents] if self.agents else [],
            "turns": self.transcript,
            "timestamp": datetime.now().isoformat(),
            "model": self.model_name or "unknown",
        }
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return output_file


class CodeConductorDebateManager:
    """Debate manager for code generation tasks"""

    def __init__(self, agents: Optional[List[LocalAIAgent]] = None):
        self.agents = agents if agents is not None else self._default_agents()
        self.full_transcript = []

    def _default_agents(self) -> List[LocalAIAgent]:
        """Minimal set för tester; kan vara mockade eller enkla lokala agenter"""
        from .local_ai_agent import LocalAIAgent
        return [
            LocalAIAgent(name="Architect", persona="You are an Architect."),
            LocalAIAgent(name="Coder", persona="You are a Coder."),
        ]

    async def conduct_debate(self, user_prompt: str) -> Dict[str, Any]:
        """Conduct a multi-agent debate for code generation"""

        try:
            # 1. Initial proposals from each agent
            print("\n--- Initial Proposals ---\n")
            for agent in self.agents:
                try:
                    response = await asyncio.wait_for(
                        agent.generate_response(user_prompt),
                        timeout=30.0  # 30 second timeout per agent
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

            # 2. Rebuttal round: each agent sees others' proposals
            print("\n--- Rebuttal Round ---\n")
            for agent in self.agents:
                try:
                    others = [a for a in self.agents if a != agent]
                    rebuttal_prompt = "Here are the other agents' proposals:\n"

                    for other in others:
                        # Find the last proposal from this agent
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
                        timeout=30.0  # 30 second timeout per agent
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
                        timeout=30.0  # 30 second timeout per agent
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

    def get_transcript(self) -> List[Dict[str, Any]]:
        """Get full debate transcript"""
        return self.full_transcript.copy()

    def save_transcript(self, filename: str = "code_debate_transcript.yaml"):
        """Save transcript to YAML and JSON files"""
        # Save as YAML
        with open(filename, "w", encoding="utf-8") as f:
            yaml.dump(self.full_transcript, f, allow_unicode=True, sort_keys=False)

        # Save as JSON
        json_filename = filename.replace(".yaml", ".json")
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(self.full_transcript, f, indent=2, ensure_ascii=False)

        logger.info(f"Transcript saved to {filename} and {json_filename}")
        
        # Also save to artifacts/runs/ for consistency
        output_dir = Path("artifacts/runs")
        output_dir.mkdir(parents=True, exist_ok=True)
        artifacts_file = output_dir / f"debate_transcript_{int(time.time())}.json"
        
        payload = {
            "agents": [agent.name for agent in self.agents] if hasattr(self, 'agents') else [],
            "turns": self.full_transcript,
            "timestamp": datetime.now().isoformat(),
            "model": getattr(self, 'model_name', 'unknown'),
        }
        with artifacts_file.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        
        return artifacts_file

    def extract_consensus(self) -> str:
        """Extract consensus from final recommendations"""
        final_recommendations = [
            entry["content"]
            for entry in self.full_transcript
            if entry["turn"] == "final_recommendation"
        ]

        if not final_recommendations:
            return "No final recommendations found"

        # Simple consensus: combine all final recommendations
        consensus = "\n\n".join(
            [
                f"## {entry['agent']}\n{entry['content']}"
                for entry in self.full_transcript
                if entry["turn"] == "final_recommendation"
            ]
        )

        return consensus
