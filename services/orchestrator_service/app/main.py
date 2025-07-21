"""
Orchestrator Service - Main FastAPI Application

This service coordinates multi-agent discussions and manages consensus
between different AI agents for code generation and analysis tasks.
"""

import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from app.agents.orchestrator import AgentOrchestrator, OrchestratorFactory
from app.schemas import (
    DiscussionRequest,
    DiscussionResponse,
    DiscussionSummary,
    OrchestratorStatistics,
    HealthResponse,
    OrchestrateRequest,
    OrchestrateResponse,
    ConsensusStrategy,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global orchestrator instance
orchestrator: AgentOrchestrator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Orchestrator Service...")

    # Initialize orchestrator
    global orchestrator
    try:
        orchestrator = OrchestratorFactory.create_orchestrator("standard")
        logger.info("Initialized AgentOrchestrator")
    except Exception as e:
        logger.error(f"Failed to initialize AgentOrchestrator: {e}")

    logger.info("Orchestrator Service started successfully")

    yield

    # Shutdown
    logger.info("Shutting down Orchestrator Service...")
    orchestrator = None


# Create FastAPI app
app = FastAPI(
    title="CodeConductor Orchestrator Service",
    description="Microservice for coordinating multi-agent discussions",
    version="2.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        service="Orchestrator Service",
        version="2.0.0",
        available_agents=orchestrator.config["available_agents"]
        if orchestrator
        else [],
        consensus_strategies=[strategy.value for strategy in ConsensusStrategy],
    )


@app.post("/discussions/start", response_model=DiscussionResponse)
async def start_discussion(request: DiscussionRequest):
    """
    Start a multi-agent discussion for the given task.

    This endpoint coordinates a discussion between multiple agents,
    collects their analyses and proposals, and attempts to reach consensus.
    """
    try:
        if not orchestrator:
            raise HTTPException(status_code=503, detail="Orchestrator not initialized")

        # Convert request to orchestrator format
        task_context = request.task_context.dict()
        agents = request.agents
        config = request.config.dict() if request.config else None

        # Run discussion
        result = await orchestrator.run_discussion(
            task_context=task_context,
            agents=agents,
            max_rounds=config.get("max_rounds") if config else None,
        )

        # Convert result to response format
        discussion_summary = DiscussionSummary(
            total_rounds=result["discussion_summary"]["total_rounds"],
            agents_used=result["discussion_summary"]["agents_used"],
            consensus_reached=result["discussion_summary"]["consensus_reached"],
            final_round=None,  # Could be added if needed
            discussion_quality=result["discussion_summary"]["discussion_quality"],
        )

        response = DiscussionResponse(
            consensus_reached=result["consensus_reached"],
            final_consensus=result["final_consensus"],
            discussion_rounds=result["discussion_rounds"],
            total_rounds=result["total_rounds"],
            agents_used=result["agents_used"],
            consensus_strategy=result["consensus_strategy"],
            discussion_summary=discussion_summary,
            metadata=result["metadata"],
        )

        logger.info(
            f"Discussion completed. Consensus reached: {result['consensus_reached']}"
        )
        return response

    except Exception as e:
        logger.error(f"Discussion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Discussion failed: {str(e)}")


@app.get("/discussions/history")
async def get_discussion_history():
    """
    Get the complete discussion history.

    Returns all discussion rounds and their results.
    """
    try:
        if not orchestrator:
            raise HTTPException(status_code=503, detail="Orchestrator not initialized")

        history = orchestrator.get_discussion_history()

        # Convert to serializable format
        history_data = []
        for round_data in history:
            history_data.append(
                {
                    "round_id": round_data.round_id,
                    "task_context": round_data.task_context,
                    "analyses": round_data.analyses,
                    "proposals": round_data.proposals,
                    "consensus": round_data.consensus,
                    "metadata": round_data.metadata,
                }
            )

        return {"discussion_history": history_data, "total_rounds": len(history_data)}

    except Exception as e:
        logger.error(f"Failed to get discussion history: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get discussion history: {str(e)}"
        )


@app.get("/statistics", response_model=OrchestratorStatistics)
async def get_statistics():
    """
    Get orchestrator statistics.

    Returns performance statistics for all agents and the orchestrator.
    """
    try:
        if not orchestrator:
            raise HTTPException(status_code=503, detail="Orchestrator not initialized")

        stats = orchestrator.get_agent_statistics()
        return OrchestratorStatistics(**stats)

    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get statistics: {str(e)}"
        )


@app.post("/discussions/reset")
async def reset_discussion():
    """
    Reset the discussion history.

    Clears all discussion data and starts fresh.
    """
    try:
        if not orchestrator:
            raise HTTPException(status_code=503, detail="Orchestrator not initialized")

        orchestrator.reset_discussion()
        logger.info("Discussion history reset")

        return {"message": "Discussion history reset successfully"}

    except Exception as e:
        logger.error(f"Failed to reset discussion: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to reset discussion: {str(e)}"
        )


@app.get("/discussions/{discussion_id}")
async def get_discussion_status(discussion_id: str):
    """
    Get status of a specific discussion.

    This is a placeholder endpoint for future implementation
    where discussions might have persistent IDs.
    """
    try:
        if not orchestrator:
            raise HTTPException(status_code=503, detail="Orchestrator not initialized")

        # For now, return current discussion status
        history = orchestrator.get_discussion_history()

        return {
            "discussion_id": discussion_id,
            "status": "completed" if history else "not_started",
            "total_rounds": len(history),
            "last_activity": history[-1].metadata["timestamp"] if history else None,
        }

    except Exception as e:
        logger.error(f"Failed to get discussion status: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get discussion status: {str(e)}"
        )


# Legacy endpoints for backward compatibility
@app.post("/orchestrate", response_model=OrchestrateResponse)
async def orchestrate_legacy(request: OrchestrateRequest):
    """
    Legacy orchestration endpoint.

    This endpoint maintains backward compatibility with the old API.
    """
    try:
        if not orchestrator:
            raise HTTPException(status_code=503, detail="Orchestrator not initialized")

        # Convert legacy request to new format
        if not request.tasks:
            return OrchestrateResponse(result=[])

        # Use the first task as the main task
        main_task = request.tasks[0]
        task_context = {
            "task": main_task.name,
            "requirements": main_task.params.get("requirements", []),
            "language": main_task.params.get("language", "python"),
        }

        # Run discussion
        result = await orchestrator.run_discussion(task_context=task_context)

        return OrchestrateResponse(result=result)

    except Exception as e:
        logger.error(f"Legacy orchestration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Orchestration failed: {str(e)}")


@app.post("/orchestrate/", response_model=OrchestrateResponse)
async def orchestrate_legacy_alt(request: OrchestrateRequest):
    """
    Alternative legacy orchestration endpoint.

    This is the same as /orchestrate but with trailing slash.
    """
    return await orchestrate_legacy(request)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True, log_level="info")
