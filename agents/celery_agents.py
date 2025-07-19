"""
Celery tasks for agent operations in CodeConductor
"""

import json
import logging
from typing import Dict, Any
from celery import current_task

from integrations.celery_app import celery_app
from agents.code_gen import CodeGenAgent
from agents.architect import ArchitectAgent
from agents.reviewer import ReviewerAgent
from agents.policy_agent import PolicyAgent

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="agents.codegen_analyze")
def codegen_analyze_task(
    self, prompt: str, context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Celery task for CodeGenAgent analysis"""
    try:
        logger.info(f"Starting CodeGenAgent analysis for task {self.request.id}")

        agent = CodeGenAgent()
        result = agent.analyze(prompt, context or {})

        # Update task state
        self.update_state(
            state="SUCCESS",
            meta={"status": "completed", "agent": "CodeGenAgent", "result": result},
        )

        logger.info(f"CodeGenAgent analysis completed for task {self.request.id}")
        return result

    except Exception as e:
        logger.error(f"CodeGenAgent analysis failed for task {self.request.id}: {e}")
        self.update_state(
            state="FAILURE",
            meta={"status": "failed", "agent": "CodeGenAgent", "error": str(e)},
        )
        raise


@celery_app.task(bind=True, name="agents.codegen_act")
def codegen_act_task(
    self, prompt: str, strategy: str, context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Celery task for CodeGenAgent action"""
    try:
        logger.info(f"Starting CodeGenAgent action for task {self.request.id}")

        agent = CodeGenAgent()
        result = agent.act({"prompt": prompt, "strategy": strategy, **(context or {})})

        self.update_state(
            state="SUCCESS",
            meta={"status": "completed", "agent": "CodeGenAgent", "result": result},
        )

        logger.info(f"CodeGenAgent action completed for task {self.request.id}")
        return result

    except Exception as e:
        logger.error(f"CodeGenAgent action failed for task {self.request.id}: {e}")
        self.update_state(
            state="FAILURE",
            meta={"status": "failed", "agent": "CodeGenAgent", "error": str(e)},
        )
        raise


@celery_app.task(bind=True, name="agents.architect_analyze")
def architect_analyze_task(
    self, prompt: str, context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Celery task for ArchitectAgent analysis"""
    try:
        logger.info(f"Starting ArchitectAgent analysis for task {self.request.id}")

        agent = ArchitectAgent()
        result = agent.analyze(prompt, context or {})

        self.update_state(
            state="SUCCESS",
            meta={"status": "completed", "agent": "ArchitectAgent", "result": result},
        )

        logger.info(f"ArchitectAgent analysis completed for task {self.request.id}")
        return result

    except Exception as e:
        logger.error(f"ArchitectAgent analysis failed for task {self.request.id}: {e}")
        self.update_state(
            state="FAILURE",
            meta={"status": "failed", "agent": "ArchitectAgent", "error": str(e)},
        )
        raise


@celery_app.task(bind=True, name="agents.reviewer_analyze")
def reviewer_analyze_task(
    self, prompt: str, context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Celery task for ReviewerAgent analysis"""
    try:
        logger.info(f"Starting ReviewerAgent analysis for task {self.request.id}")

        agent = ReviewerAgent()
        result = agent.analyze(prompt, context or {})

        self.update_state(
            state="SUCCESS",
            meta={"status": "completed", "agent": "ReviewerAgent", "result": result},
        )

        logger.info(f"ReviewerAgent analysis completed for task {self.request.id}")
        return result

    except Exception as e:
        logger.error(f"ReviewerAgent analysis failed for task {self.request.id}: {e}")
        self.update_state(
            state="FAILURE",
            meta={"status": "failed", "agent": "ReviewerAgent", "error": str(e)},
        )
        raise


@celery_app.task(bind=True, name="agents.policy_check")
def policy_check_task(
    self, code: str, context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Celery task for PolicyAgent safety check"""
    try:
        logger.info(f"Starting PolicyAgent check for task {self.request.id}")

        agent = PolicyAgent()
        result = agent.check_safety(code, context or {})

        self.update_state(
            state="SUCCESS",
            meta={"status": "completed", "agent": "PolicyAgent", "result": result},
        )

        logger.info(f"PolicyAgent check completed for task {self.request.id}")
        return result

    except Exception as e:
        logger.error(f"PolicyAgent check failed for task {self.request.id}: {e}")
        self.update_state(
            state="FAILURE",
            meta={"status": "failed", "agent": "PolicyAgent", "error": str(e)},
        )
        raise


@celery_app.task(bind=True, name="agents.orchestrator_discussion")
def orchestrator_discussion_task(
    self, prompt: str, context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Celery task for orchestrating multi-agent discussion"""
    try:
        logger.info(f"Starting multi-agent discussion for task {self.request.id}")

        from agents.orchestrator_simple import SimpleAgentOrchestrator

        orchestrator = SimpleAgentOrchestrator()
        result = orchestrator.facilitate_discussion(prompt, context or {})

        self.update_state(
            state="SUCCESS",
            meta={
                "status": "completed",
                "orchestrator": "SimpleAgentOrchestrator",
                "result": result,
            },
        )

        logger.info(f"Multi-agent discussion completed for task {self.request.id}")
        return result

    except Exception as e:
        logger.error(f"Multi-agent discussion failed for task {self.request.id}: {e}")
        self.update_state(
            state="FAILURE",
            meta={
                "status": "failed",
                "orchestrator": "SimpleAgentOrchestrator",
                "error": str(e),
            },
        )
        raise


@celery_app.task(bind=True, name="agents.parallel_analysis")
def parallel_analysis_task(
    self, prompt: str, context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Celery task for running all agent analyses in parallel"""
    try:
        logger.info(f"Starting parallel agent analysis for task {self.request.id}")

        # Start all analysis tasks in parallel
        codegen_task = codegen_analyze_task.delay(prompt, context)
        architect_task = architect_analyze_task.delay(prompt, context)
        reviewer_task = reviewer_analyze_task.delay(prompt, context)

        # Wait for all results
        results = {
            "codegen": codegen_task.get(),
            "architect": architect_task.get(),
            "reviewer": reviewer_task.get(),
        }

        self.update_state(
            state="SUCCESS",
            meta={"status": "completed", "parallel_analysis": True, "results": results},
        )

        logger.info(f"Parallel agent analysis completed for task {self.request.id}")
        return results

    except Exception as e:
        logger.error(f"Parallel agent analysis failed for task {self.request.id}: {e}")
        self.update_state(
            state="FAILURE",
            meta={"status": "failed", "parallel_analysis": True, "error": str(e)},
        )
        raise
