#!/usr/bin/env python3
"""
CodeConductor Debate Manager - Multi-Agent Debate System (FIXED VERSION)
"""

import asyncio
import inspect
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .local_agent import LocalAIAgent

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
            "agents": (
                [getattr(a, "name", str(a)) for a in self.agents] if self.agents else []
            ),
            "turns": self.transcript,
            "timestamp": datetime.now().isoformat(),
            "model": self.model_name or "unknown",
        }
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return output_file


class CodeConductorDebateManager:
    """Debate manager for code generation tasks (FIXED VERSION)"""

    def __init__(self, agents: list[LocalAIAgent] | None = None):
        self.agents = agents if agents is not None else self._default_agents()
        self.full_transcript = []

    def _awaitable(self, x: Any):
        """Normalisera till awaitable"""
        if inspect.isawaitable(x):
            return x

        async def _ret(v=x):
            return v

        return _ret()

    async def _run_phase(self, calls: list[Any]) -> list[Any]:
        """Kör en fas av debatten med normaliserade awaitables"""
        return await asyncio.gather(
            *[self._awaitable(c) for c in calls], return_exceptions=False
        )

    def _default_agents(self) -> list[LocalAIAgent]:
        """Minimal set för tester; kan vara mockade eller enkla lokala agenter"""
        from .local_ai_agent import LocalAIAgent

        return [
            LocalAIAgent(name="Architect", persona="You are an Architect."),
            LocalAIAgent(name="Coder", persona="You are a Coder."),
        ]

    def _check_agent_capabilities(self, agent) -> dict[str, bool]:
        """Kontrollera vilka metoder agenten har"""
        return {
            "has_propose": hasattr(agent, "propose") and callable(agent.propose),
            "has_rebuttal": hasattr(agent, "rebuttal") and callable(agent.rebuttal),
            "has_finalize": hasattr(agent, "finalize") and callable(agent.finalize),
            "has_generate_response": hasattr(agent, "generate_response")
            and callable(agent.generate_response),
        }

    def _get_agent_response(self, agent, method: str, *args, **kwargs):
        """Hämta svar från agent baserat på tillgängliga metoder"""
        capabilities = self._check_agent_capabilities(agent)

        if method == "propose" and capabilities["has_propose"]:
            return agent.propose(*args, **kwargs)
        elif method == "rebuttal" and capabilities["has_rebuttal"]:
            return agent.rebuttal(*args, **kwargs)
        elif method == "finalize" and capabilities["has_finalize"]:
            return agent.finalize(*args, **kwargs)
        elif capabilities["has_generate_response"]:
            # Fallback till generate_response om de specifika metoderna saknas
            if method == "propose":
                prompt = args[0] if args else "Provide your initial proposal."
            elif method == "rebuttal":
                prompt = "Provide your rebuttal to the other proposals."
            elif method == "finalize":
                prompt = "Provide your final recommendation."
            else:
                prompt = "Provide your response."

            # Använd generate_response som fallback
            return agent.generate_response(prompt)
        else:
            raise AttributeError(
                f"Agent {agent.name} has no suitable method for {method}"
            )

    async def conduct_debate(self, user_prompt: str) -> dict[str, Any]:
        """Conduct a multi-agent debate for code generation (FIXED VERSION)"""

        try:
            # 1. Initial proposals from each agent
            print("\n--- Initial Proposals ---\n")

            # Använd adapter för att hantera olika agent-typer
            proposal_calls = [
                self._get_agent_response(agent, "propose", user_prompt)
                for agent in self.agents
            ]
            proposals = await self._run_phase(proposal_calls)

            # Hantera resultaten
            for i, response in enumerate(proposals):
                if isinstance(response, Exception):
                    print(f"❌ {self.agents[i].name} error during proposal: {response}")
                    response = (
                        f"{self.agents[i].name} encountered an error: {str(response)}"
                    )
                else:
                    # Hantera både dict-format och str-format
                    if isinstance(response, dict) and "content" in response:
                        content = response["content"]
                    else:
                        content = str(response)
                    print(f"{self.agents[i].name}:\n{content}\n")

                self.full_transcript.append(
                    {
                        "agent": self.agents[i].name,
                        "turn": "proposal",
                        "content": content if "content" in locals() else str(response),
                    }
                )

            # 2. Rebuttal round: each agent sees others' proposals
            print("\n--- Rebuttal Round ---\n")

            # Samla alla rebuttal-anrop med adapter
            rebuttal_calls = []
            for agent in self.agents:
                others = [a for a in self.agents if a != agent]
                rebuttal_prompt = "Here are the other agents' proposals:\n"

                for other in others:
                    # Find the last proposal from this agent
                    last_proposal = next(
                        (
                            entry["content"]
                            for entry in reversed(self.full_transcript)
                            if entry["agent"] == other.name
                            and entry["turn"] == "proposal"
                        ),
                        "",
                    )
                    rebuttal_prompt += f"{other.name}: {last_proposal}\n"

                rebuttal_prompt += "Please provide your rebuttal or counter-argument."

                # Skapa state för rebuttal
                state = {
                    "prompt": rebuttal_prompt,
                    "proposals": [
                        entry["content"]
                        for entry in self.full_transcript
                        if entry["turn"] == "proposal"
                    ],
                }
                rebuttal_calls.append(
                    self._get_agent_response(agent, "rebuttal", state)
                )

            # Kör alla rebuttals parallellt
            rebuttals = await self._run_phase(rebuttal_calls)

            # Hantera resultaten
            for i, response in enumerate(rebuttals):
                if isinstance(response, Exception):
                    print(f"❌ {self.agents[i].name} error during rebuttal: {response}")
                    response = (
                        f"{self.agents[i].name} encountered an error during rebuttal: "
                        f"{str(response)}"
                    )
                else:
                    # Hantera både dict-format och str-format
                    if isinstance(response, dict) and "content" in response:
                        content = response["content"]
                    else:
                        content = str(response)
                    print(f"{self.agents[i].name} (rebuttal):\n{content}\n")

                self.full_transcript.append(
                    {
                        "agent": self.agents[i].name,
                        "turn": "rebuttal",
                        "content": content if "content" in locals() else str(response),
                    }
                )

            # 3. Final recommendations
            print("\n--- Final Recommendations ---\n")

            # Samla alla final-recommendation-anrop med adapter
            final_calls = []
            for agent in self.agents:
                # Skapa state för final recommendation
                state = {
                    "prompt": (
                        "Based on the debate so far, what is your final recommendation "
                        "for the code implementation?"
                    ),
                    "proposals": [
                        entry["content"]
                        for entry in self.full_transcript
                        if entry["turn"] == "proposal"
                    ],
                    "rebuttals": [
                        entry["content"]
                        for entry in self.full_transcript
                        if entry["turn"] == "rebuttal"
                    ],
                }
                final_calls.append(self._get_agent_response(agent, "finalize", state))

            # Kör alla final recommendations parallellt
            finals = await self._run_phase(final_calls)

            # Hantera resultaten
            for i, response in enumerate(finals):
                if isinstance(response, Exception):
                    print(
                        f"❌ {self.agents[i].name} error during final recommendation: "
                        f"{response}"
                    )
                    response = (
                        f"{self.agents[i].name} encountered an error during final "
                        f"recommendation generation."
                    )
                else:
                    # Hantera både dict-format och str-format
                    if isinstance(response, dict) and "content" in response:
                        content = response["content"]
                    else:
                        content = str(response)
                    print(f"{self.agents[i].name} (final recommendation):\n{content}\n")

                self.full_transcript.append(
                    {
                        "agent": self.agents[i].name,
                        "turn": "final_recommendation",
                        "content": content if "content" in locals() else str(response),
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

    def get_transcript(self) -> list[dict[str, Any]]:
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
            "agents": (
                [agent.name for agent in self.agents] if hasattr(self, "agents") else []
            ),
            "turns": self.full_transcript,
            "timestamp": datetime.now().isoformat(),
            "model": getattr(self, "model_name", "unknown"),
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
