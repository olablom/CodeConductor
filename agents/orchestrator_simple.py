"""
Simplified AgentOrchestrator with Plugin Support

Uses the user's elegant plugin architecture for CodeConductor v2.0.
"""

from typing import Dict, Any, List
import json
from pathlib import Path
from datetime import datetime

from agents.code_gen import CodeGenAgent
from agents.architect import ArchitectAgent
from agents.reviewer import ReviewerAgent
from agents.prompt_optimizer import prompt_optimizer
from plugins.base_simple import PluginManager, BaseAgentPlugin


class SimpleAgentOrchestrator:
    """Koordinerar multi-agent diskussion med plugin-stöd."""

    def __init__(self, enable_plugins: bool = True):
        self.codegen_agent = CodeGenAgent()
        self.architect_agent = ArchitectAgent()
        self.reviewer_agent = ReviewerAgent()

        # Agent registry
        self.agents = {
            "codegen": self.codegen_agent,
            "architect": self.architect_agent,
            "reviewer": self.reviewer_agent,
        }

        # Plugin system
        self.enable_plugins = enable_plugins
        self.plugin_manager = None
        self.plugin_agents = []

        if self.enable_plugins:
            self._initialize_plugins()

    def _initialize_plugins(self) -> None:
        """Initialize plugin system and load agent plugins"""
        try:
            self.plugin_manager = PluginManager()

            # Discover and activate plugins
            self.plugin_manager.discover()
            self.plugin_manager.activate_all()

            # Get agent plugins
            self.plugin_agents = self.plugin_manager.get_agent_plugins()

            print(f"🎯 Loaded {len(self.plugin_agents)} plugin agents")

        except Exception as e:
            print(f"⚠️ Plugin system initialization failed: {e}")
            self.enable_plugins = False

    def facilitate_discussion(
        self, prompt: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Facilitarar multi-agent diskussion och syntetiserar konsensus.

        Args:
            prompt: Kodkravet att diskutera
            context: Ytterligare kontext (valfritt)

        Returns:
            Dictionary med konsensus och optimerad approach
        """
        if context is None:
            context = {}

        print("🤖 Starting multi-agent discussion...")
        print(f"📝 Prompt: {prompt}")
        print("=" * 60)

        # 1. Samla analyser från alla agents (core + plugins)
        analyses = {}

        # Core agents
        for agent_name, agent in self.agents.items():
            print(f"\n🔍 {agent.role} analyzing...")
            try:
                analysis = agent.analyze(prompt, context)
                analyses[agent_name] = analysis
                print(
                    f"✅ {agent.name}: {analysis.get('recommendation', 'analysis complete')}"
                )
            except Exception as e:
                print(f"❌ {agent.name} failed: {e}")
                analyses[agent_name] = agent._fallback_analysis(prompt)

        # Plugin agents
        if self.enable_plugins and self.plugin_agents:
            for plugin_agent in self.plugin_agents:
                plugin_name = plugin_agent.name()
                print(f"\n🔌 {plugin_name} analyzing...")
                try:
                    plugin_context = {
                        "prompt": prompt,
                        "code": context.get("code", ""),
                        "discussion_history": context.get("discussion_history", []),
                        "core_analyses": analyses,
                    }

                    plugin_analysis = plugin_agent.analyze(
                        context.get("code", ""), plugin_context
                    )
                    analyses[f"plugin_{plugin_name}"] = plugin_analysis
                    print(
                        f"✅ {plugin_name}: {plugin_analysis.get('description', 'analysis complete')}"
                    )

                except Exception as e:
                    print(f"❌ Plugin {plugin_name} failed: {e}")
                    analyses[f"plugin_{plugin_name}"] = {
                        "error": str(e),
                        "recommendation": "Plugin analysis failed",
                    }

        # 2. Syntetisera konsensus
        consensus = self._synthesize_consensus(analyses, prompt)

        # 3. Låt PromptOptimizer förfina
        optimized = self._optimize_with_rl(consensus, prompt)

        # 4. Skapa final proposal
        final_proposal = self._create_final_proposal(analyses, consensus, optimized)

        print("\n" + "=" * 60)
        print("🎯 MULTI-AGENT CONSENSUS REACHED")
        print("=" * 60)

        return final_proposal

    def _synthesize_consensus(
        self, analyses: Dict[str, Any], prompt: str
    ) -> Dict[str, Any]:
        """Syntetiserar konsensus från alla agent-analyser."""

        # Extrahera viktiga komponenter
        approaches = []
        patterns = []
        risks = []
        recommendations = []

        for agent_name, analysis in analyses.items():
            if "approach" in analysis:
                approaches.append(analysis["approach"])
            if "patterns" in analysis:
                patterns.extend(analysis["patterns"])
            if "risks" in analysis:
                risks.extend(analysis["risks"])
            if "recommendation" in analysis:
                recommendations.append(analysis["recommendation"])

            # Handle plugin-specific analysis results
            if agent_name.startswith("plugin_"):
                if "security_score" in analysis:
                    recommendations.append(
                        f"Security score: {analysis['security_score']}"
                    )
                if "vulnerabilities" in analysis:
                    risks.extend(
                        [
                            f"Security: {v.get('description', 'vulnerability')}"
                            for v in analysis.get("vulnerabilities", [])
                        ]
                    )
                if "recommendations" in analysis:
                    recommendations.extend(analysis["recommendations"])

        # Skapa konsensus
        consensus = {
            "synthesized_approach": self._combine_approaches(approaches),
            "recommended_patterns": list(set(patterns))[:3],  # Top 3 unika patterns
            "identified_risks": list(set(risks))[:5],  # Top 5 unika risks
            "consensus_recommendation": self._select_best_recommendation(
                recommendations
            ),
            "confidence": self._calculate_consensus_confidence(analyses),
        }

        return consensus

    def _combine_approaches(self, approaches: List[str]) -> str:
        """Kombinerar olika approaches till en syntetiserad approach."""
        if not approaches:
            return "Standard implementation with error handling"

        # Enkel kombination - ta första som inte är fallback
        for approach in approaches:
            if approach and "Standard" not in approach:
                return approach

        return approaches[0] if approaches else "Standard implementation"

    def _select_best_recommendation(self, recommendations: List[str]) -> str:
        """Väljer bästa rekommendationen baserat på frekvens."""
        if not recommendations:
            return "defensive_programming"

        # Räkna frekvens
        from collections import Counter

        counter = Counter(recommendations)
        return counter.most_common(1)[0][0]

    def _calculate_consensus_confidence(self, analyses: Dict[str, Any]) -> float:
        """Beräknar konsensus-konfidens baserat på agent-konfidens."""
        confidences = []
        for analysis in analyses.values():
            if "confidence" in analysis:
                confidences.append(analysis["confidence"])

        if not confidences:
            return 0.7  # Default confidence

        return sum(confidences) / len(confidences)

    def _optimize_with_rl(
        self, consensus: Dict[str, Any], prompt: str
    ) -> Dict[str, Any]:
        """Använder PromptOptimizer för att förfina konsensus."""

        # Skapa en optimerad prompt baserat på konsensus
        optimized_prompt = f"""
Based on multi-agent consensus:

{prompt}

Consensus Approach: {consensus["synthesized_approach"]}
Recommended Patterns: {", ".join(consensus["recommended_patterns"])}
Identified Risks: {", ".join(consensus["identified_risks"])}

Generate optimized implementation following the consensus.
"""

        # Skapa temporär prompt-fil
        temp_prompt_path = Path("data/temp_consensus.md")
        temp_prompt_path.parent.mkdir(parents=True, exist_ok=True)
        temp_prompt_path.write_text(optimized_prompt)

        try:
            # Använd PromptOptimizer för att förfina
            optimized_code = prompt_optimizer.mutate_prompt(
                optimized_prompt, "add_examples"
            )

            return {
                "optimized_prompt": optimized_code,
                "optimization_applied": "add_examples",
                "rl_score": consensus["confidence"],
            }
        except Exception as e:
            print(f"❌ RL optimization failed: {e}")
            return {
                "optimized_prompt": optimized_prompt,
                "optimization_applied": "none",
                "rl_score": consensus["confidence"],
            }
        finally:
            # Cleanup
            if temp_prompt_path.exists():
                temp_prompt_path.unlink()

    def _create_final_proposal(
        self,
        analyses: Dict[str, Any],
        consensus: Dict[str, Any],
        optimized: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Skapar final proposal för human approval."""

        return {
            "proposal_id": f"proposal_{len(analyses)}_{int(consensus['confidence'] * 100)}_{int(datetime.now().timestamp())}",
            "prompt": analyses.get("codegen", {}).get("approach", "Unknown"),
            "approach": consensus["synthesized_approach"],
            "confidence": consensus["confidence"],
            "patterns": consensus["recommended_patterns"],
            "risks": consensus["identified_risks"],
            "rl_score": optimized["rl_score"],
            "optimization": optimized["optimization_applied"],
            "agent_analyses": analyses,
            "consensus": consensus,
            "optimized": optimized,
            "timestamp": "2025-07-19T18:00:00Z",
        }

    def get_agent_summary(self) -> Dict[str, Any]:
        """Returnerar sammanfattning av alla agents (core + plugins)."""
        summary = {
            "total_agents": len(self.agents) + len(self.plugin_agents),
            "core_agents": len(self.agents),
            "plugin_agents": len(self.plugin_agents),
            "agents": {},
        }

        # Core agents
        for name, agent in self.agents.items():
            summary["agents"][name] = {
                "name": agent.name,
                "role": agent.role,
                "type": "core",
            }

        # Plugin agents
        for plugin_agent in self.plugin_agents:
            name = plugin_agent.name()
            summary["agents"][name] = {
                "name": plugin_agent.name(),
                "role": plugin_agent.description(),
                "type": "plugin",
                "version": plugin_agent.version(),
            }

        return summary

    def get_plugin_info(self) -> Dict[str, Any]:
        """Get detailed plugin information."""
        if not self.plugin_manager:
            return {"plugins_enabled": False}

        return self.plugin_manager.get_plugin_info()

    def cleanup(self) -> None:
        """Clean up orchestrator and plugins"""
        if self.plugin_manager:
            self.plugin_manager.deactivate_all()
