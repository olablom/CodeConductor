#!/usr/bin/env python3
"""
Agent Service - Hanterar alla agent-relaterade operationer
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import logging
import uvicorn
import asyncio
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CodeConductor Agent Service",
    description="Service för hantering av AI-agenter (CodeGen, Architect, Review, Policy, Q-Learning)",
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


# Pydantic models
class AgentRequest(BaseModel):
    agent_type: str  # "codegen", "architect", "review", "policy", "qlearning"
    task_context: Dict[str, Any]
    config: Optional[Dict[str, Any]] = None


class AgentResponse(BaseModel):
    agent_name: str
    result: Dict[str, Any]
    confidence: float
    execution_time: float
    timestamp: datetime


class AgentStatus(BaseModel):
    agent_name: str
    status: str  # "idle", "busy", "error"
    last_activity: datetime
    queue_size: int


# In-memory storage (replace with database in production)
agent_states = {}
agent_queue = []


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Agent Service",
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now(),
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "agents_available": list(agent_states.keys()),
        "queue_size": len(agent_queue),
        "uptime": "TODO: implement uptime tracking",
    }


@app.post("/agents/analyze", response_model=AgentResponse)
async def analyze_task(request: AgentRequest, background_tasks: BackgroundTasks):
    """Analysera en uppgift med specifik agent"""
    try:
        logger.info(f"Analyzing task with agent: {request.agent_type}")

        # Simulate agent analysis (replace with actual agent logic)
        result = await simulate_agent_analysis(request.agent_type, request.task_context)

        response = AgentResponse(
            agent_name=request.agent_type,
            result=result,
            confidence=result.get("confidence", 0.5),
            execution_time=result.get("execution_time", 0.1),
            timestamp=datetime.now(),
        )

        # Store result in background
        background_tasks.add_task(store_agent_result, request.agent_type, response)

        return response

    except Exception as e:
        logger.error(f"Error in agent analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agents/propose", response_model=AgentResponse)
async def propose_solution(request: AgentRequest, background_tasks: BackgroundTasks):
    """Föreslå en lösning med specifik agent"""
    try:
        logger.info(f"Proposing solution with agent: {request.agent_type}")

        # Simulate agent proposal (replace with actual agent logic)
        result = await simulate_agent_proposal(request.agent_type, request.task_context)

        response = AgentResponse(
            agent_name=request.agent_type,
            result=result,
            confidence=result.get("confidence", 0.5),
            execution_time=result.get("execution_time", 0.1),
            timestamp=datetime.now(),
        )

        # Store result in background
        background_tasks.add_task(store_agent_result, request.agent_type, response)

        return response

    except Exception as e:
        logger.error(f"Error in agent proposal: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agents/status", response_model=List[AgentStatus])
async def get_agent_status():
    """Hämta status för alla agenter"""
    status_list = []
    for agent_name, state in agent_states.items():
        status_list.append(
            AgentStatus(
                agent_name=agent_name,
                status=state.get("status", "idle"),
                last_activity=state.get("last_activity", datetime.now()),
                queue_size=len(
                    [q for q in agent_queue if q.get("agent") == agent_name]
                ),
            )
        )
    return status_list


@app.post("/agents/{agent_type}/reset")
async def reset_agent(agent_type: str):
    """Återställ en specifik agent"""
    try:
        if agent_type in agent_states:
            agent_states[agent_type] = {
                "status": "idle",
                "last_activity": datetime.now(),
                "reset_count": agent_states[agent_type].get("reset_count", 0) + 1,
            }
            logger.info(f"Reset agent: {agent_type}")
            return {"message": f"Agent {agent_type} reset successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Agent {agent_type} not found")
    except Exception as e:
        logger.error(f"Error resetting agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Background tasks
async def store_agent_result(agent_type: str, response: AgentResponse):
    """Store agent result in background"""
    agent_states[agent_type] = {
        "status": "idle",
        "last_activity": datetime.now(),
        "last_result": response.dict(),
    }
    logger.info(f"Stored result for agent: {agent_type}")


# Simulation functions (replace with actual agent logic)
async def simulate_agent_analysis(
    agent_type: str, task_context: Dict[str, Any]
) -> Dict[str, Any]:
    """Simulate agent analysis"""
    await asyncio.sleep(0.1)  # Simulate processing time

    return {
        "analysis": f"Analysis by {agent_type} agent",
        "confidence": 0.8,
        "execution_time": 0.1,
        "insights": ["insight1", "insight2"],
        "recommendations": ["rec1", "rec2"],
    }


async def simulate_agent_proposal(
    agent_type: str, task_context: Dict[str, Any]
) -> Dict[str, Any]:
    """Simulate agent proposal"""
    await asyncio.sleep(0.1)  # Simulate processing time

    return {
        "proposal": f"Proposal by {agent_type} agent",
        "confidence": 0.7,
        "execution_time": 0.1,
        "approach": "recommended_approach",
        "estimated_effort": "medium",
    }


if __name__ == "__main__":
    logger.info("Starting Agent Service...")
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True, log_level="info")
