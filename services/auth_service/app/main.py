"""
Auth Service - Main FastAPI Application

This service handles authentication, authorization, and approval workflows
for the CodeConductor system, including policy enforcement and human approval gates.
"""

import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from app.agents.policy_agent import PolicyAgent
from app.agents.human_gate import HumanGate
from app.schemas import (
    ApprovalRequest,
    ApprovalResponse,
    ApprovalStrategy,
    RiskLevel,
    PolicyAnalysis,
    PolicyProposal,
    PolicyReview,
    SafetyAnalysis,
    CodeSafetyRequest,
    CodeSafetyResponse,
    HealthResponse,
    HumanApprovalStats,
    HumanApprovalSummary,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global agent instances
policy_agent: PolicyAgent = None
human_gate: HumanGate = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Auth Service...")

    # Initialize agents
    global policy_agent, human_gate
    try:
        policy_agent = PolicyAgent(name="auth_policy_agent")
        human_gate = HumanGate()
        logger.info("Initialized PolicyAgent and HumanGate")
    except Exception as e:
        logger.error(f"Failed to initialize agents: {e}")

    logger.info("Auth Service started successfully")

    yield

    # Shutdown
    logger.info("Shutting down Auth Service...")
    policy_agent = None
    human_gate = None


# Create FastAPI app
app = FastAPI(
    title="CodeConductor Auth Service",
    description="Authentication, authorization, and approval service",
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
    approval_stats = None
    if human_gate:
        approval_stats = HumanApprovalStats(**human_gate.get_approval_stats())

    return HealthResponse(
        status="healthy",
        service="Auth Service",
        version="2.0.0",
        policy_agent_ready=policy_agent is not None,
        human_gate_ready=human_gate is not None,
        approval_stats=approval_stats,
    )


@app.post("/auth/approve", response_model=ApprovalResponse)
async def approve_request(request: ApprovalRequest):
    """
    Request approval for a context.

    This endpoint evaluates the context using the policy agent and
    may require human approval for high-risk operations.
    """
    try:
        if not policy_agent:
            raise HTTPException(status_code=503, detail="Policy agent not initialized")

        # Prepare context for policy evaluation
        context = request.context.copy()
        if request.code:
            context["code"] = request.code
        if request.task_type:
            context["task_type"] = request.task_type
        if request.risk_level:
            context["risk_level"] = request.risk_level.value
        if request.confidence:
            context["confidence"] = request.confidence
        if request.agent_analyses:
            context["agent_analyses"] = request.agent_analyses

        # First, try policy agent auto-approval
        approved = policy_agent.evaluate(context)

        if approved:
            # Auto-approved by policy agent
            strategy = ApprovalStrategy.AUTO_APPROVE
            reasoning = "Auto-approved by policy agent"
            confidence = 0.9
        else:
            # Policy agent requires human approval
            if human_gate:
                approved = await human_gate.request_approval(context)
                strategy = (
                    ApprovalStrategy.HUMAN_APPROVAL
                    if approved
                    else ApprovalStrategy.REJECT
                )
                reasoning = "Human approval decision"
                confidence = 0.8 if approved else 0.9
            else:
                # No human gate available, reject
                approved = False
                strategy = ApprovalStrategy.REJECT
                reasoning = "Human approval required but not available"
                confidence = 0.9

        # Get safety analysis for response
        code = context.get("code", "")
        safety_result = policy_agent.check_code_safety(code)

        # Determine risk level
        risk_level = RiskLevel(safety_result.get("risk_level", "medium"))

        return ApprovalResponse(
            approved=approved,
            strategy=strategy,
            risk_level=risk_level,
            violations=safety_result.get("violations", []),
            recommendations=safety_result.get("recommendations", []),
            confidence=confidence,
            reasoning=reasoning,
            timestamp=policy_agent._get_timestamp(),
        )

    except Exception as e:
        logger.error(f"Approval request failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Approval request failed: {str(e)}"
        )


@app.post("/auth/analyze", response_model=PolicyAnalysis)
async def analyze_policy(request: ApprovalRequest):
    """
    Analyze context for policy compliance.

    This endpoint performs a detailed policy analysis without making
    approval decisions.
    """
    try:
        if not policy_agent:
            raise HTTPException(status_code=503, detail="Policy agent not initialized")

        # Prepare context
        context = request.context.copy()
        if request.code:
            context["code"] = request.code
        if request.task_type:
            context["task_type"] = request.task_type

        # Perform analysis
        analysis = policy_agent.analyze(context)

        # Convert to response format
        safety_analysis = SafetyAnalysis(**analysis["safety_analysis"])

        return PolicyAnalysis(
            agent_name=analysis["agent_name"],
            task_type=analysis["task_type"],
            safety_analysis=safety_analysis,
            policy_compliant=analysis["policy_compliant"],
            risk_assessment=RiskLevel(analysis["risk_assessment"]),
            recommendations=analysis["recommendations"],
            timestamp=analysis["timestamp"],
        )

    except Exception as e:
        logger.error(f"Policy analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Policy analysis failed: {str(e)}")


@app.post("/auth/propose", response_model=PolicyProposal)
async def propose_policy(request: ApprovalRequest):
    """
    Propose policy actions based on analysis.

    This endpoint generates a policy proposal without making final decisions.
    """
    try:
        if not policy_agent:
            raise HTTPException(status_code=503, detail="Policy agent not initialized")

        # Prepare context
        context = request.context.copy()
        if request.code:
            context["code"] = request.code
        if request.task_type:
            context["task_type"] = request.task_type

        # First analyze
        analysis = policy_agent.analyze(context)

        # Then propose
        proposal = policy_agent.propose(analysis, context)

        return PolicyProposal(
            agent_name=proposal["agent_name"],
            strategy=ApprovalStrategy(proposal["strategy"]),
            approved=proposal["approved"],
            requires_human=proposal["requires_human"],
            risk_level=RiskLevel(proposal["risk_level"]),
            violations=proposal["violations"],
            recommendations=proposal["recommendations"],
            confidence=proposal["confidence"],
            reasoning=proposal["reasoning"],
        )

    except Exception as e:
        logger.error(f"Policy proposal failed: {e}")
        raise HTTPException(status_code=500, detail=f"Policy proposal failed: {str(e)}")


@app.post("/auth/review", response_model=PolicyReview)
async def review_policy(request: ApprovalRequest):
    """
    Perform final policy review.

    This endpoint performs the final review and makes the approval decision.
    """
    try:
        if not policy_agent:
            raise HTTPException(status_code=503, detail="Policy agent not initialized")

        # Prepare context
        context = request.context.copy()
        if request.code:
            context["code"] = request.code
        if request.task_type:
            context["task_type"] = request.task_type

        # First analyze and propose
        analysis = policy_agent.analyze(context)
        proposal = policy_agent.propose(analysis, context)

        # Then review
        review = policy_agent.review(proposal, context)

        return PolicyReview(
            agent_name=review["agent_name"],
            final_approval=review["final_approval"],
            strategy=ApprovalStrategy(review["strategy"]),
            risk_level=RiskLevel(review["risk_level"]),
            violations=review["violations"],
            recommendations=review["recommendations"],
            confidence=review["confidence"],
            reasoning=review["reasoning"],
            timestamp=review["timestamp"],
        )

    except Exception as e:
        logger.error(f"Policy review failed: {e}")
        raise HTTPException(status_code=500, detail=f"Policy review failed: {str(e)}")


@app.post("/auth/safety", response_model=CodeSafetyResponse)
async def check_code_safety(request: CodeSafetyRequest):
    """
    Check code for safety violations.

    This endpoint performs a detailed safety analysis of code.
    """
    try:
        if not policy_agent:
            raise HTTPException(status_code=503, detail="Policy agent not initialized")

        # Perform safety analysis
        safety_result = policy_agent.check_code_safety(request.code)

        return CodeSafetyResponse(
            safety_analysis=SafetyAnalysis(**safety_result),
            timestamp=policy_agent._get_timestamp(),
        )

    except Exception as e:
        logger.error(f"Code safety check failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Code safety check failed: {str(e)}"
        )


@app.get("/auth/stats", response_model=HumanApprovalStats)
async def get_approval_stats():
    """
    Get human approval statistics.

    Returns statistics about human approval decisions.
    """
    try:
        if not human_gate:
            raise HTTPException(status_code=503, detail="Human gate not initialized")

        stats = human_gate.get_approval_stats()
        return HumanApprovalStats(**stats)

    except Exception as e:
        logger.error(f"Failed to get approval stats: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get approval stats: {str(e)}"
        )


@app.get("/auth/summary", response_model=HumanApprovalSummary)
async def get_approval_summary():
    """
    Get human approval summary.

    Returns a comprehensive summary of approval decisions and trends.
    """
    try:
        if not human_gate:
            raise HTTPException(status_code=503, detail="Human gate not initialized")

        summary = human_gate.get_approval_summary()
        return HumanApprovalSummary(**summary)

    except Exception as e:
        logger.error(f"Failed to get approval summary: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get approval summary: {str(e)}"
        )


@app.post("/auth/reset")
async def reset_approval_history():
    """
    Reset approval history.

    Clears all approval history and starts fresh.
    """
    try:
        if not human_gate:
            raise HTTPException(status_code=503, detail="Human gate not initialized")

        human_gate.reset_approval_history()
        logger.info("Approval history reset")

        return {"message": "Approval history reset successfully"}

    except Exception as e:
        logger.error(f"Failed to reset approval history: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to reset approval history: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8005, reload=True, log_level="info")
