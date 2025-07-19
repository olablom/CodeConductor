"""
Distributed AgentOrchestrator using Celery for parallel execution
"""

import time
import logging
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from omegaconf import OmegaConf

from integrations.celery_app import celery_app, is_celery_available, get_celery_stats
from agents.celery_agents import (
    codegen_analyze_task,
    architect_analyze_task,
    reviewer_analyze_task,
    policy_check_task,
    orchestrator_discussion_task,
    parallel_analysis_task,
)

logger = logging.getLogger(__name__)


class DistributedAgentOrchestrator:
    """Distributed orchestrator using Celery for parallel agent execution"""

    def __init__(self, enable_plugins: bool = True):
        self.cfg = OmegaConf.load("config/base.yaml")
        self.enable_plugins = enable_plugins
        self.distributed_enabled = self.cfg.distributed.enabled

        # Check if Celery is available
        if self.distributed_enabled and not is_celery_available():
            logger.warning(
                "Distributed mode enabled but Celery broker not available. Falling back to local mode."
            )
            self.distributed_enabled = False

        # Initialize plugin system if enabled
        self.plugin_manager = None
        if self.enable_plugins:
            self._initialize_plugins()

    def _initialize_plugins(self) -> None:
        """Initialize plugin system"""
        try:
            from plugins.base_simple import PluginManager

            self.plugin_manager = PluginManager()
            self.plugin_manager.discover()
            self.plugin_manager.activate_all()
            logger.info(f"Loaded {len(self.plugin_manager.get_plugins())} plugins")
        except Exception as e:
            logger.error(f"Plugin initialization failed: {e}")
            self.enable_plugins = False

    def facilitate_discussion(
        self, prompt: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Facilitate multi-agent discussion using distributed execution"""
        if context is None:
            context = {}

        logger.info("🤖 Starting distributed multi-agent discussion...")
        logger.info(f"📝 Prompt: {prompt}")
        logger.info(f"🔧 Distributed mode: {self.distributed_enabled}")

        if self.distributed_enabled:
            return self._distributed_discussion(prompt, context)
        else:
            return self._local_discussion(prompt, context)

    def _distributed_discussion(
        self, prompt: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run discussion using distributed Celery tasks"""
        start_time = time.time()

        # Start all agent analysis tasks in parallel
        logger.info("🚀 Starting parallel agent analysis tasks...")

        tasks = {
            "codegen": codegen_analyze_task.delay(prompt, context),
            "architect": architect_analyze_task.delay(prompt, context),
            "reviewer": reviewer_analyze_task.delay(prompt, context),
        }

        # Wait for all tasks to complete
        results = {}
        for agent_name, task in tasks.items():
            try:
                logger.info(f"⏳ Waiting for {agent_name} analysis...")
                result = task.get(timeout=self.cfg.distributed.task_timeout)
                results[agent_name] = result
                logger.info(f"✅ {agent_name} analysis completed")
            except Exception as e:
                logger.error(f"❌ {agent_name} analysis failed: {e}")
                results[agent_name] = {
                    "error": str(e),
                    "recommendation": "Analysis failed",
                }

        # Run plugin analysis if enabled
        if self.enable_plugins and self.plugin_manager:
            plugin_results = self._run_plugin_analysis(prompt, context, results)
            results.update(plugin_results)

        # Synthesize consensus
        consensus = self._synthesize_consensus(results, prompt)

        # Optimize with RL
        optimized = self._optimize_with_rl(consensus, prompt)

        # Create final proposal
        final_proposal = self._create_final_proposal(results, consensus, optimized)

        execution_time = time.time() - start_time
        logger.info(
            f"⏱️ Distributed discussion completed in {execution_time:.2f} seconds"
        )

        return final_proposal

    def _local_discussion(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run discussion using local execution (fallback)"""
        from agents.orchestrator_simple import SimpleAgentOrchestrator

        logger.info("🔄 Falling back to local execution...")
        orchestrator = SimpleAgentOrchestrator(enable_plugins=self.enable_plugins)
        return orchestrator.facilitate_discussion(prompt, context)

    def _run_plugin_analysis(
        self, prompt: str, context: Dict[str, Any], core_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run plugin analysis in parallel"""
        if not self.plugin_manager:
            return {}

        plugin_results = {}
        plugin_agents = self.plugin_manager.get_agent_plugins()

        for plugin_agent in plugin_agents:
            plugin_name = plugin_agent.name()
            try:
                logger.info(f"🔌 Running {plugin_name} analysis...")
                plugin_context = {
                    "prompt": prompt,
                    "code": context.get("code", ""),
                    "core_analyses": core_results,
                }

                result = plugin_agent.analyze(context.get("code", ""), plugin_context)
                plugin_results[f"plugin_{plugin_name}"] = result
                logger.info(f"✅ {plugin_name} analysis completed")

            except Exception as e:
                logger.error(f"❌ Plugin {plugin_name} analysis failed: {e}")
                plugin_results[f"plugin_{plugin_name}"] = {
                    "error": str(e),
                    "recommendation": "Plugin analysis failed",
                }

        return plugin_results

    def _synthesize_consensus(
        self, analyses: Dict[str, Any], prompt: str
    ) -> Dict[str, Any]:
        """Synthesize consensus from all agent analyses"""
        approaches = []
        patterns = []
        risks = []
        recommendations = []

        for agent_name, analysis in analyses.items():
            if isinstance(analysis, dict):
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

        consensus = {
            "synthesized_approach": self._combine_approaches(approaches),
            "recommended_patterns": list(set(patterns))[:3],
            "identified_risks": list(set(risks))[:5],
            "consensus_recommendation": self._select_best_recommendation(
                recommendations
            ),
            "confidence": self._calculate_consensus_confidence(analyses),
        }

        return consensus

    def _combine_approaches(self, approaches: List[str]) -> str:
        """Combine different approaches into a synthesized approach"""
        if not approaches:
            return "Standard implementation with error handling"

        for approach in approaches:
            if approach and "Standard" not in approach:
                return approach

        return approaches[0] if approaches else "Standard implementation"

    def _select_best_recommendation(self, recommendations: List[str]) -> str:
        """Select the best recommendation from all agents"""
        if not recommendations:
            return "Implement with standard best practices"

        # Simple heuristic: prefer non-fallback recommendations
        for rec in recommendations:
            if rec and "standard" not in rec.lower() and "fallback" not in rec.lower():
                return rec

        return recommendations[0]

    def _calculate_consensus_confidence(self, analyses: Dict[str, Any]) -> float:
        """Calculate confidence score based on agreement"""
        successful_analyses = sum(
            1 for a in analyses.values() if isinstance(a, dict) and "error" not in a
        )
        total_analyses = len(analyses)

        if total_analyses == 0:
            return 0.0

        base_confidence = successful_analyses / total_analyses

        # Boost confidence if we have plugin analysis
        plugin_count = sum(1 for name in analyses.keys() if name.startswith("plugin_"))
        if plugin_count > 0:
            base_confidence += 0.1

        return min(base_confidence, 1.0)

    def _optimize_with_rl(
        self, consensus: Dict[str, Any], prompt: str
    ) -> Dict[str, Any]:
        """Optimize consensus using reinforcement learning"""
        try:
            from agents.prompt_optimizer import prompt_optimizer

            optimized_prompt = prompt_optimizer.optimize(prompt, consensus)

            return {
                "optimized_prompt": optimized_prompt,
                "optimization_score": consensus.get("confidence", 0.5),
                "original_consensus": consensus,
            }
        except Exception as e:
            logger.error(f"RL optimization failed: {e}")
            return consensus

    def _create_final_proposal(
        self,
        analyses: Dict[str, Any],
        consensus: Dict[str, Any],
        optimized: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create final proposal from all results"""
        return {
            "consensus": consensus,
            "optimized": optimized,
            "agent_analyses": analyses,
            "execution_mode": "distributed" if self.distributed_enabled else "local",
            "timestamp": time.time(),
            "plugin_count": len(
                [k for k in analyses.keys() if k.startswith("plugin_")]
            ),
        }

    def get_distributed_stats(self) -> Dict[str, Any]:
        """Get distributed execution statistics"""
        if not self.distributed_enabled:
            return {"distributed_enabled": False}

        try:
            celery_stats = get_celery_stats()
            return {
                "distributed_enabled": True,
                "celery_stats": celery_stats,
                "broker_available": is_celery_available(),
                "worker_concurrency": self.cfg.distributed.worker_concurrency,
                "task_timeout": self.cfg.distributed.task_timeout,
            }
        except Exception as e:
            logger.error(f"Failed to get distributed stats: {e}")
            return {"distributed_enabled": True, "error": str(e)}

    def cleanup(self) -> None:
        """Cleanup resources"""
        if self.plugin_manager:
            self.plugin_manager.deactivate_all()
