#!/usr/bin/env python3
"""
Orchestrator Service - Koordinerar anrop mellan agent-tjänsterna
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import logging
import uvicorn
import httpx
from datetime import datetime
import uuid

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CodeConductor Orchestrator Service",
    description="Service för koordinering av AI-agenter och arbetsflöden",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
AGENT_SERVICE_URL = "http://localhost:8001"
DATA_SERVICE_URL = "http://localhost:8003"
QUEUE_SERVICE_URL = "http://localhost:8004"


# Pydantic models
class DiscussionRequest(BaseModel):
    task_context: Dict[str, Any]
    agents: List[str]  # ["codegen", "architect", "review"]
    max_rounds: int = 3
    consensus_strategy: str = "weighted_majority"


class DiscussionResponse(BaseModel):
    discussion_id: str
    consensus: Optional[Dict[str, Any]]
    discussion_rounds: int
    consensus_reached: bool
    final_proposals: List[Dict[str, Any]]
    execution_time: float
    timestamp: datetime


class WorkflowRequest(BaseModel):
    workflow_type: str  # "code_generation", "code_review", "architecture_design"
    input_data: Dict[str, Any]
    steps: List[Dict[str, Any]]


class WorkflowResponse(BaseModel):
    workflow_id: str
    status: str  # "running", "completed", "failed"
    current_step: int
    total_steps: int
    results: Dict[str, Any]
    execution_time: float
    timestamp: datetime


# In-memory storage (replace with database in production)
discussions = {}
workflows = {}


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Orchestrator Service",
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now(),
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "active_discussions": len(discussions),
        "active_workflows": len(workflows),
        "agent_service": await check_service_health(AGENT_SERVICE_URL),
        "data_service": await check_service_health(DATA_SERVICE_URL),
        "queue_service": await check_service_health(QUEUE_SERVICE_URL),
    }


@app.post("/discussions/start", response_model=DiscussionResponse)
async def start_discussion(
    request: DiscussionRequest, background_tasks: BackgroundTasks
):
    """Starta en multi-agent diskussion"""
    try:
        discussion_id = str(uuid.uuid4())
        logger.info(
            f"Starting discussion {discussion_id} with agents: {request.agents}"
        )

        # Start discussion in background
        background_tasks.add_task(
            run_discussion,
            discussion_id,
            request.task_context,
            request.agents,
            request.max_rounds,
            request.consensus_strategy,
        )

        # Store initial state
        discussions[discussion_id] = {
            "status": "running",
            "agents": request.agents,
            "max_rounds": request.max_rounds,
            "start_time": datetime.now(),
            "rounds": [],
        }

        return DiscussionResponse(
            discussion_id=discussion_id,
            consensus=None,
            discussion_rounds=0,
            consensus_reached=False,
            final_proposals=[],
            execution_time=0.0,
            timestamp=datetime.now(),
        )

    except Exception as e:
        logger.error(f"Error starting discussion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/discussions/{discussion_id}", response_model=DiscussionResponse)
async def get_discussion(discussion_id: str):
    """Hämta status för en diskussion"""
    if discussion_id not in discussions:
        raise HTTPException(status_code=404, detail="Discussion not found")

    discussion = discussions[discussion_id]

    return DiscussionResponse(
        discussion_id=discussion_id,
        consensus=discussion.get("consensus"),
        discussion_rounds=len(discussion.get("rounds", [])),
        consensus_reached=discussion.get("consensus_reached", False),
        final_proposals=discussion.get("final_proposals", []),
        execution_time=(datetime.now() - discussion["start_time"]).total_seconds(),
        timestamp=datetime.now(),
    )


@app.post("/workflows/start", response_model=WorkflowResponse)
async def start_workflow(request: WorkflowRequest, background_tasks: BackgroundTasks):
    """Starta ett arbetsflöde"""
    try:
        workflow_id = str(uuid.uuid4())
        logger.info(f"Starting workflow {workflow_id}: {request.workflow_type}")

        # Start workflow in background
        background_tasks.add_task(
            run_workflow,
            workflow_id,
            request.workflow_type,
            request.input_data,
            request.steps,
        )

        # Store initial state
        workflows[workflow_id] = {
            "status": "running",
            "workflow_type": request.workflow_type,
            "current_step": 0,
            "total_steps": len(request.steps),
            "start_time": datetime.now(),
            "results": {},
        }

        return WorkflowResponse(
            workflow_id=workflow_id,
            status="running",
            current_step=0,
            total_steps=len(request.steps),
            results={},
            execution_time=0.0,
            timestamp=datetime.now(),
        )

    except Exception as e:
        logger.error(f"Error starting workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/workflows/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(workflow_id: str):
    """Hämta status för ett arbetsflöde"""
    if workflow_id not in workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")

    workflow = workflows[workflow_id]

    return WorkflowResponse(
        workflow_id=workflow_id,
        status=workflow["status"],
        current_step=workflow["current_step"],
        total_steps=workflow["total_steps"],
        results=workflow["results"],
        execution_time=(datetime.now() - workflow["start_time"]).total_seconds(),
        timestamp=datetime.now(),
    )


@app.get("/discussions")
async def list_discussions():
    """Lista alla aktiva diskussioner"""
    return {
        "discussions": [
            {
                "id": discussion_id,
                "status": discussion["status"],
                "agents": discussion["agents"],
                "rounds": len(discussion.get("rounds", [])),
                "start_time": discussion["start_time"],
            }
            for discussion_id, discussion in discussions.items()
        ]
    }


@app.get("/workflows")
async def list_workflows():
    """Lista alla aktiva arbetsflöden"""
    return {
        "workflows": [
            {
                "id": workflow_id,
                "status": workflow["status"],
                "type": workflow["workflow_type"],
                "current_step": workflow["current_step"],
                "total_steps": workflow["total_steps"],
                "start_time": workflow["start_time"],
            }
            for workflow_id, workflow in workflows.items()
        ]
    }


# Background tasks
async def run_discussion(
    discussion_id: str,
    task_context: Dict[str, Any],
    agents: List[str],
    max_rounds: int,
    consensus_strategy: str,
):
    """Run discussion in background"""
    try:
        logger.info(f"Running discussion {discussion_id}")

        rounds = []
        consensus = None
        consensus_reached = False

        for round_num in range(max_rounds):
            logger.info(f"Discussion {discussion_id}, round {round_num + 1}")

            # Get analysis from each agent
            analyses = []
            for agent in agents:
                analysis = await call_agent_service(agent, "analyze", task_context)
                analyses.append(analysis)

            # Get proposals from each agent
            proposals = []
            for agent in agents:
                proposal = await call_agent_service(agent, "propose", task_context)
                proposals.append(proposal)

            # Try to reach consensus
            consensus = await reach_consensus(proposals, consensus_strategy)

            round_data = {
                "round": round_num + 1,
                "analyses": analyses,
                "proposals": proposals,
                "consensus": consensus,
            }
            rounds.append(round_data)

            # Update discussion state
            discussions[discussion_id]["rounds"] = rounds
            discussions[discussion_id]["consensus"] = consensus

            if consensus:
                consensus_reached = True
                break

            # Update task context for next round
            task_context["previous_round"] = round_data

        # Finalize discussion
        discussions[discussion_id]["status"] = "completed"
        discussions[discussion_id]["consensus_reached"] = consensus_reached
        discussions[discussion_id]["final_proposals"] = (
            proposals if not consensus_reached else []
        )

        logger.info(
            f"Discussion {discussion_id} completed. Consensus reached: {consensus_reached}"
        )

    except Exception as e:
        logger.error(f"Error in discussion {discussion_id}: {e}")
        discussions[discussion_id]["status"] = "failed"
        discussions[discussion_id]["error"] = str(e)


async def run_workflow(
    workflow_id: str,
    workflow_type: str,
    input_data: Dict[str, Any],
    steps: List[Dict[str, Any]],
):
    """Run workflow in background"""
    try:
        logger.info(f"Running workflow {workflow_id}")

        results = {}

        for step_num, step in enumerate(steps):
            logger.info(f"Workflow {workflow_id}, step {step_num + 1}")

            # Update workflow state
            workflows[workflow_id]["current_step"] = step_num + 1

            # Execute step
            step_result = await execute_workflow_step(step, input_data, results)
            results[f"step_{step_num + 1}"] = step_result

            # Update workflow results
            workflows[workflow_id]["results"] = results

        # Finalize workflow
        workflows[workflow_id]["status"] = "completed"

        logger.info(f"Workflow {workflow_id} completed")

    except Exception as e:
        logger.error(f"Error in workflow {workflow_id}: {e}")
        workflows[workflow_id]["status"] = "failed"
        workflows[workflow_id]["error"] = str(e)


# Helper functions
async def call_agent_service(
    agent_type: str, operation: str, task_context: Dict[str, Any]
) -> Dict[str, Any]:
    """Call agent service"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{AGENT_SERVICE_URL}/agents/{operation}",
            json={"agent_type": agent_type, "task_context": task_context},
        )
        return response.json()


async def reach_consensus(
    proposals: List[Dict[str, Any]], strategy: str
) -> Optional[Dict[str, Any]]:
    """Reach consensus among proposals"""
    if not proposals:
        return None

    # Simple consensus logic (replace with actual logic)
    if strategy == "weighted_majority":
        # Use proposal with highest confidence
        best_proposal = max(proposals, key=lambda p: p.get("confidence", 0))
        if best_proposal.get("confidence", 0) > 0.7:
            return best_proposal

    return None


async def execute_workflow_step(
    step: Dict[str, Any], input_data: Dict[str, Any], previous_results: Dict[str, Any]
) -> Dict[str, Any]:
    """Execute a workflow step"""
    step_type = step.get("type", "agent_call")

    if step_type == "agent_call":
        agent = step.get("agent")
        operation = step.get("operation", "analyze")
        return await call_agent_service(agent, operation, input_data)
    elif step_type == "data_operation":
        # Call data service
        return {"status": "data_operation_completed"}
    else:
        return {"status": "unknown_step_type"}


async def check_service_health(service_url: str) -> str:
    """Check if a service is healthy"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{service_url}/health", timeout=1.0)
            return "healthy" if response.status_code == 200 else "unhealthy"
    except:
        return "unreachable"


if __name__ == "__main__":
    logger.info("Starting Orchestrator Service...")
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True, log_level="info")
