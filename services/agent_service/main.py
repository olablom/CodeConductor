"""
Agent Service - Main FastAPI Application

This service provides agent functionality for code generation, analysis, and review.
It integrates the core agent classes from the CodeConductor codebase.
"""

import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
from typing import Dict, Any, List

from app.agents import AgentFactory, BaseAgent
from app.schemas import (
    AgentAnalysisRequest,
    AgentAnalysisResponse,
    AgentProposalRequest,
    AgentProposalResponse,
    AgentReviewRequest,
    AgentReviewResponse,
    AgentStatusResponse,
    HealthResponse,
    ErrorResponse,
    AgentType,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global agent instances
agents: Dict[str, BaseAgent] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Agent Service...")

    # Initialize default agents
    try:
        agents["codegen"] = AgentFactory.create_agent("codegen", "codegen_agent")
        logger.info("Initialized CodeGenAgent")
    except Exception as e:
        logger.error(f"Failed to initialize CodeGenAgent: {e}")

    logger.info("Agent Service started successfully")

    yield

    # Shutdown
    logger.info("Shutting down Agent Service...")
    agents.clear()


# Create FastAPI app
app = FastAPI(
    title="CodeConductor Agent Service",
    description="Microservice for AI agent functionality",
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
        service="Agent Service",
        version="2.0.0",
        agents_available=list(agents.keys()),
    )


@app.post("/agents/analyze", response_model=AgentAnalysisResponse)
async def analyze_task(request: AgentAnalysisRequest):
    """
    Analyze a task using the specified agent.

    This endpoint uses the agent to analyze the task context and extract
    key insights, patterns, and recommendations.
    """
    try:
        agent_type = request.agent_type.value

        # Get or create agent
        if agent_type not in agents:
            agents[agent_type] = AgentFactory.create_agent(
                agent_type, f"{agent_type}_agent", request.config
            )

        agent = agents[agent_type]

        # Convert task context to dict
        context = request.task_context.dict()

        # Perform analysis
        analysis = agent.analyze(context)

        # Convert analysis to response format
        response = AgentAnalysisResponse(
            agent_name=agent.name,
            task=analysis.get("task", ""),
            language=analysis.get("language", "python"),
            requirements=analysis.get("requirements", []),
            patterns=analysis.get("patterns", []),
            challenges=analysis.get("challenges", []),
            complexity=analysis.get("complexity", "medium"),
            recommended_approach=analysis.get("recommended_approach", ""),
            quality_focus=analysis.get("quality_focus", []),
            performance_needs=analysis.get("performance_needs", []),
            security_needs=analysis.get("security_needs", []),
            testing_strategy=analysis.get("testing_strategy", ""),
            documentation_needs=analysis.get("documentation_needs", {}),
        )

        logger.info(f"Analysis completed by {agent.name}")
        return response

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/agents/propose", response_model=AgentProposalResponse)
async def generate_proposal(request: AgentProposalRequest):
    """
    Generate a proposal using the specified agent.

    This endpoint uses the agent to generate a detailed proposal based on
    the analysis results and original task context.
    """
    try:
        agent_type = request.agent_type.value

        # Get or create agent
        if agent_type not in agents:
            agents[agent_type] = AgentFactory.create_agent(
                agent_type, f"{agent_type}_agent", request.config
            )

        agent = agents[agent_type]

        # Convert task context to dict
        context = request.task_context.dict()

        # Generate proposal
        proposal = agent.propose(request.analysis, context)

        # Convert proposal to response format
        response = AgentProposalResponse(
            agent_name=agent.name,
            approach=proposal.get("approach", ""),
            structure=proposal.get("structure", {}),
            implementation_plan=proposal.get("implementation_plan", []),
            code_template=proposal.get("code_template", ""),
            quality_guidelines=proposal.get("quality_guidelines", []),
            documentation_plan=proposal.get("documentation_plan", {}),
            size_estimate=proposal.get("size_estimate", {}),
            confidence=proposal.get("confidence", 0.0),
            reasoning=proposal.get("reasoning", ""),
            suggestions=proposal.get("suggestions", []),
            analysis_summary=proposal.get("analysis_summary", {}),
        )

        logger.info(
            f"Proposal generated by {agent.name} with confidence {proposal.get('confidence', 0.0):.2f}"
        )
        return response

    except Exception as e:
        logger.error(f"Proposal generation failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Proposal generation failed: {str(e)}"
        )


@app.post("/agents/review", response_model=AgentReviewResponse)
async def review_proposal(request: AgentReviewRequest):
    """
    Review a proposal using the specified agent.

    This endpoint uses the agent to review a proposal and provide
    feedback on quality, security, and best practices.
    """
    try:
        agent_type = request.agent_type.value

        # Get or create agent
        if agent_type not in agents:
            agents[agent_type] = AgentFactory.create_agent(
                agent_type, f"{agent_type}_agent", request.config
            )

        agent = agents[agent_type]

        # Convert task context to dict
        context = request.task_context.dict()

        # Perform review
        review = agent.review(request.proposal, context)

        # Convert review to response format
        response = AgentReviewResponse(
            agent_name=agent.name,
            quality_score=review.get("quality_score", 0.0),
            issues=review.get("issues", []),
            improvements=review.get("improvements", []),
            best_practices=review.get("best_practices", {}),
            performance_notes=review.get("performance_notes", []),
            security_notes=review.get("security_notes", []),
            readability_score=review.get("readability_score", 0.0),
            maintainability_score=review.get("maintainability_score", 0.0),
            test_recommendations=review.get("test_recommendations", []),
            documentation_gaps=review.get("documentation_gaps", []),
            overall_assessment=review.get("overall_assessment", ""),
            proposal_quality=review.get("proposal_quality", {}),
            review_summary=review.get("review_summary", {}),
        )

        logger.info(
            f"Review completed by {agent.name} with overall score {review.get('review_summary', {}).get('overall_score', 0.0):.2f}"
        )
        return response

    except Exception as e:
        logger.error(f"Review failed: {e}")
        raise HTTPException(status_code=500, detail=f"Review failed: {str(e)}")


@app.get("/agents/{agent_type}/status", response_model=AgentStatusResponse)
async def get_agent_status(agent_type: str):
    """
    Get the status of a specific agent.

    Returns the current status and configuration of the specified agent.
    """
    try:
        if agent_type not in agents:
            raise HTTPException(
                status_code=404, detail=f"Agent type '{agent_type}' not found"
            )

        agent = agents[agent_type]
        status = agent.get_status()

        response = AgentStatusResponse(
            agent_name=status.get("name", ""),
            config=status.get("config", {}),
            has_message_bus=status.get("has_message_bus", False),
            has_llm_client=status.get("has_llm_client", False),
            status="active",
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent status: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get agent status: {str(e)}"
        )


@app.get("/agents", response_model=List[str])
async def list_agents():
    """
    List all available agent types.

    Returns a list of agent types that are currently available.
    """
    return list(agents.keys())


@app.post("/agents/{agent_type}/analyze")
async def analyze_with_agent_type(
    agent_type: str, request: AgentAnalysisRequest, background_tasks: BackgroundTasks
):
    """
    Analyze a task using a specific agent type.

    This is a convenience endpoint that allows specifying the agent type
    in the URL path instead of the request body.
    """
    # Override the agent type in the request
    request.agent_type = AgentType(agent_type)
    return await analyze_task(request)


@app.post("/agents/{agent_type}/propose")
async def propose_with_agent_type(
    agent_type: str, request: AgentProposalRequest, background_tasks: BackgroundTasks
):
    """
    Generate a proposal using a specific agent type.

    This is a convenience endpoint that allows specifying the agent type
    in the URL path instead of the request body.
    """
    # Override the agent type in the request
    request.agent_type = AgentType(agent_type)
    return await generate_proposal(request)


@app.post("/agents/{agent_type}/review")
async def review_with_agent_type(
    agent_type: str, request: AgentReviewRequest, background_tasks: BackgroundTasks
):
    """
    Review a proposal using a specific agent type.

    This is a convenience endpoint that allows specifying the agent type
    in the URL path instead of the request body.
    """
    # Override the agent type in the request
    request.agent_type = AgentType(agent_type)
    return await review_proposal(request)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True, log_level="info")
