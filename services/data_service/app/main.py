"""
Data Service - Main FastAPI Application

This service handles data persistence, Q-learning, bandit algorithms, and prompt optimization
for the CodeConductor system.
"""

import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
from typing import Dict

try:
    import numpy as np
except ImportError:
    # Simple fallback
    np = None
from datetime import datetime

from app.agents.linucb_bandit import LinUCBBandit
from app.agents.qlearning_agent import QLearningAgent
from app.agents.prompt_optimizer import PromptOptimizerAgent
from app.schemas import (
    BanditChooseRequest,
    BanditChooseResponse,
    BanditUpdateRequest,
    BanditUpdateResponse,
    BanditStatsResponse,
    QLearningRunRequest,
    QLearningRunResponse,
    QLearningUpdateRequest,
    QLearningUpdateResponse,
    QLearningStatsResponse,
    QLearningAction,
    PromptOptimizeRequest,
    PromptOptimizeResponse,
    PromptOptimizerStatsResponse,
    HealthResponse,
    ModelStateRequest,
    ModelStateResponse,
    ModelResetRequest,
    ModelResetResponse,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global agent instances
bandits: Dict[str, LinUCBBandit] = {}
qlearning_agents: Dict[str, QLearningAgent] = {}
prompt_optimizers: Dict[str, PromptOptimizerAgent] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Data Service...")

    # Initialize default instances
    try:
        # Default bandit
        bandits["default"] = LinUCBBandit(d=10, alpha=1.0, name="default_bandit")

        # Default Q-learning agent
        qlearning_agents["qlearning_agent"] = QLearningAgent(name="qlearning_agent")

        # Default prompt optimizer
        prompt_optimizers["prompt_optimizer"] = PromptOptimizerAgent(
            name="prompt_optimizer"
        )

        logger.info("Initialized default agents and bandits")
    except Exception as e:
        logger.error(f"Failed to initialize agents: {e}")

    logger.info("Data Service started successfully")

    yield

    # Shutdown
    logger.info("Shutting down Data Service...")
    bandits.clear()
    qlearning_agents.clear()
    prompt_optimizers.clear()


# Create FastAPI app
app = FastAPI(
    title="CodeConductor Data Service",
    description="Data persistence, Q-learning, bandits, and prompt optimization service",
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
        service="Data Service",
        version="2.0.0",
        bandits_ready=len(bandits) > 0,
        qlearning_ready=len(qlearning_agents) > 0,
        prompt_optimizer_ready=len(prompt_optimizers) > 0,
        active_bandits=list(bandits.keys()),
        active_agents=list(qlearning_agents.keys()) + list(prompt_optimizers.keys()),
    )


# Bandit endpoints
@app.post("/bandits/choose", response_model=BanditChooseResponse)
async def choose_bandit_arm(request: BanditChooseRequest):
    """
    Choose an arm from a bandit algorithm.

    This endpoint uses the LinUCB bandit to select the best arm
    based on the provided feature vector.
    """
    try:
        bandit_name = request.bandit_name

        # Get or create bandit
        if bandit_name not in bandits:
            if request.config:
                bandits[bandit_name] = LinUCBBandit(
                    d=request.config.d, alpha=request.config.alpha, name=bandit_name
                )
            else:
                bandits[bandit_name] = LinUCBBandit(
                    d=len(request.features), name=bandit_name
                )

        bandit = bandits[bandit_name]

        # Convert features to array
        features = request.features if np is None else np.array(request.features)

        # Select arm
        selected_arm = bandit.select_arm(request.arms, features)

        # Get UCB values for all arms
        ucb_values = {arm: bandit.get_ucb(arm, features) for arm in request.arms}

        # Get confidence intervals
        confidence_intervals = bandit.get_confidence_intervals(request.arms, features)

        return BanditChooseResponse(
            selected_arm=selected_arm,
            ucb_values=ucb_values,
            confidence_intervals=confidence_intervals,
            bandit_name=bandit_name,
            exploration=False,  # LinUCB doesn't have explicit exploration flag
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(f"Bandit arm selection failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Bandit arm selection failed: {str(e)}"
        )


@app.post("/bandits/update", response_model=BanditUpdateResponse)
async def update_bandit(request: BanditUpdateRequest):
    """
    Update bandit with observed reward.

    This endpoint updates the bandit model with the reward
    observed from pulling the selected arm.
    """
    try:
        bandit_name = request.bandit_name

        if bandit_name not in bandits:
            raise HTTPException(
                status_code=404, detail=f"Bandit '{bandit_name}' not found"
            )

        bandit = bandits[bandit_name]

        # Convert features to array
        features = request.features if np is None else np.array(request.features)

        # Update bandit
        bandit.update(request.arm, features, request.reward)

        return BanditUpdateResponse(
            updated=True,
            arm=request.arm,
            reward=request.reward,
            bandit_name=bandit_name,
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(f"Bandit update failed: {e}")
        raise HTTPException(status_code=500, detail=f"Bandit update failed: {str(e)}")


@app.get("/bandits/{bandit_name}/stats", response_model=BanditStatsResponse)
async def get_bandit_stats(bandit_name: str):
    """
    Get statistics for a specific bandit.

    Returns detailed statistics about the bandit's performance.
    """
    try:
        if bandit_name not in bandits:
            raise HTTPException(
                status_code=404, detail=f"Bandit '{bandit_name}' not found"
            )

        bandit = bandits[bandit_name]
        stats = bandit.get_arm_statistics()

        return BanditStatsResponse(
            bandit_name=bandit_name,
            total_pulls=stats["total_pulls"],
            arm_count=stats["arm_count"],
            arms=stats["arms"],
            config={"d": bandit.d, "alpha": bandit.alpha, "name": bandit.name},
        )

    except Exception as e:
        logger.error(f"Failed to get bandit stats: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get bandit stats: {str(e)}"
        )


# Q-learning endpoints
@app.post("/qlearning/run", response_model=QLearningRunResponse)
async def run_qlearning(request: QLearningRunRequest):
    """
    Run Q-learning episode.

    This endpoint uses the Q-learning agent to select an action
    based on the provided context.
    """
    try:
        agent_name = request.agent_name

        # Get or create Q-learning agent
        if agent_name not in qlearning_agents:
            config = request.config.dict() if request.config else {}
            qlearning_agents[agent_name] = QLearningAgent(
                name=agent_name, config=config
            )

        agent = qlearning_agents[agent_name]

        # Get state and select action
        state = agent.get_state(request.context)
        selected_action = agent.select_action(state)

        # Get Q-value
        q_value = agent.get_q_value(state, selected_action)

        # Convert QAction to QLearningAction
        qlearning_action = QLearningAction(
            agent_combination=selected_action.agent_combination,
            prompt_strategy=selected_action.prompt_strategy,
            iteration_count=selected_action.iteration_count,
            confidence_threshold=selected_action.confidence_threshold,
        )

        return QLearningRunResponse(
            agent_name=agent_name,
            selected_action=qlearning_action,
            q_value=q_value,
            epsilon=agent.epsilon,
            exploration=agent.epsilon > 0.1,  # Simple exploration detection
            confidence=1.0 - agent.epsilon,
            reasoning=f"Q-learning selection with epsilon={agent.epsilon:.4f}",
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(f"Q-learning run failed: {e}")
        raise HTTPException(status_code=500, detail=f"Q-learning run failed: {str(e)}")


@app.post("/qlearning/update", response_model=QLearningUpdateResponse)
async def update_qlearning(request: QLearningUpdateRequest):
    """
    Update Q-learning with observed reward.

    This endpoint updates the Q-learning agent with the reward
    observed from taking the selected action.
    """
    try:
        agent_name = request.agent_name

        if agent_name not in qlearning_agents:
            raise HTTPException(
                status_code=404, detail=f"Q-learning agent '{agent_name}' not found"
            )

        agent = qlearning_agents[agent_name]

        # Get states
        state = agent.get_state(request.context)
        next_state = (
            agent.get_state(request.next_context) if request.next_context else state
        )

        # Get old Q-value
        old_q_value = agent.get_q_value(state, request.action)

        # Update Q-value
        agent.update_q_value(state, request.action, request.reward, next_state)

        # Get new Q-value
        new_q_value = agent.get_q_value(state, request.action)

        return QLearningUpdateResponse(
            updated=True,
            agent_name=agent_name,
            old_q_value=old_q_value,
            new_q_value=new_q_value,
            reward=request.reward,
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(f"Q-learning update failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Q-learning update failed: {str(e)}"
        )


@app.get("/qlearning/{agent_name}/stats", response_model=QLearningStatsResponse)
async def get_qlearning_stats(agent_name: str):
    """
    Get statistics for a specific Q-learning agent.

    Returns detailed statistics about the agent's learning progress.
    """
    try:
        if agent_name not in qlearning_agents:
            raise HTTPException(
                status_code=404, detail=f"Q-learning agent '{agent_name}' not found"
            )

        agent = qlearning_agents[agent_name]
        stats = agent.get_learning_statistics()

        return QLearningStatsResponse(
            agent_name=agent_name,
            total_episodes=stats["total_episodes"],
            successful_episodes=stats["successful_episodes"],
            success_rate=stats["success_rate"],
            epsilon=stats["epsilon"],
            q_table_size=stats["q_table_size"],
            average_q_value=stats["average_q_value"],
            min_q_value=stats["min_q_value"],
            max_q_value=stats["max_q_value"],
            total_visits=stats["total_visits"],
            visit_distribution=stats["visit_distribution"],
            learning_rate=stats["learning_rate"],
            discount_factor=stats["discount_factor"],
        )

    except Exception as e:
        logger.error(f"Failed to get Q-learning stats: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get Q-learning stats: {str(e)}"
        )


# Prompt optimization endpoints
@app.post("/prompt/optimize", response_model=PromptOptimizeResponse)
async def optimize_prompt(request: PromptOptimizeRequest):
    """
    Optimize prompt using Q-learning.

    This endpoint uses the prompt optimizer to mutate a prompt
    based on previous outcomes and current context.
    """
    try:
        agent_name = request.agent_name

        # Get or create prompt optimizer
        if agent_name not in prompt_optimizers:
            prompt_optimizers[agent_name] = PromptOptimizerAgent(name=agent_name)

        optimizer = prompt_optimizers[agent_name]

        # Create context for optimization
        context = {
            "task_id": request.task_id,
            "arm_prev": request.arm_prev,
            "passed": request.passed,
            "blocked": request.blocked,
            "complexity": request.complexity,
            "model_source": request.model_source,
            "prompt": request.original_prompt,
        }

        # Get optimization proposal
        proposal = optimizer.propose({}, context)

        return PromptOptimizeResponse(
            agent_name=agent_name,
            selected_action=proposal["selected_action"],
            original_prompt=proposal["original_prompt"],
            mutated_prompt=proposal["mutated_prompt"],
            mutation=proposal["mutation"],
            q_value=proposal["q_value"],
            epsilon=proposal["epsilon"],
            exploration=proposal["exploration"],
            confidence=proposal["confidence"],
            reasoning=proposal["reasoning"],
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(f"Prompt optimization failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Prompt optimization failed: {str(e)}"
        )


@app.get("/prompt/{agent_name}/stats", response_model=PromptOptimizerStatsResponse)
async def get_prompt_optimizer_stats(agent_name: str):
    """
    Get statistics for a specific prompt optimizer.

    Returns detailed statistics about the optimizer's performance.
    """
    try:
        if agent_name not in prompt_optimizers:
            raise HTTPException(
                status_code=404, detail=f"Prompt optimizer '{agent_name}' not found"
            )

        optimizer = prompt_optimizers[agent_name]
        summary = optimizer.get_q_table_summary()
        action_stats = optimizer.get_action_stats()

        return PromptOptimizerStatsResponse(
            agent_name=agent_name,
            total_states=summary["total_states"],
            total_entries=summary["total_entries"],
            average_q_value=summary["average_q_value"],
            min_q_value=summary["min_q_value"],
            max_q_value=summary["max_q_value"],
            epsilon=summary["epsilon"],
            total_episodes=summary["total_episodes"],
            successful_episodes=summary["successful_episodes"],
            success_rate=summary["success_rate"],
            action_stats=action_stats,
        )

    except Exception as e:
        logger.error(f"Failed to get prompt optimizer stats: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get prompt optimizer stats: {str(e)}"
        )


# Model state management endpoints
@app.post("/models/state", response_model=ModelStateResponse)
async def get_model_state(request: ModelStateRequest):
    """
    Get model state for persistence.

    Returns the current state of a model for backup/restore.
    """
    try:
        model_name = request.model_name
        model_type = request.model_type

        if model_type == "bandit":
            if model_name not in bandits:
                raise HTTPException(
                    status_code=404, detail=f"Bandit '{model_name}' not found"
                )
            state = bandits[model_name].get_model_state()
        elif model_type == "qlearning":
            if model_name not in qlearning_agents:
                raise HTTPException(
                    status_code=404, detail=f"Q-learning agent '{model_name}' not found"
                )
            state = qlearning_agents[model_name].get_model_state()
        elif model_type == "prompt_optimizer":
            if model_name not in prompt_optimizers:
                raise HTTPException(
                    status_code=404, detail=f"Prompt optimizer '{model_name}' not found"
                )
            state = prompt_optimizers[model_name].get_model_state()
        else:
            raise HTTPException(
                status_code=400, detail=f"Unknown model type: {model_type}"
            )

        return ModelStateResponse(
            model_name=model_name,
            model_type=model_type,
            state=state,
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(f"Failed to get model state: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get model state: {str(e)}"
        )


@app.post("/models/reset", response_model=ModelResetResponse)
async def reset_model(request: ModelResetRequest):
    """
    Reset model to initial state.

    Clears all learning progress and resets the model.
    """
    try:
        model_name = request.model_name
        model_type = request.model_type

        if model_type == "bandit":
            if model_name not in bandits:
                raise HTTPException(
                    status_code=404, detail=f"Bandit '{model_name}' not found"
                )
            bandits[model_name].reset()
        elif model_type == "qlearning":
            if model_name not in qlearning_agents:
                raise HTTPException(
                    status_code=404, detail=f"Q-learning agent '{model_name}' not found"
                )
            qlearning_agents[model_name].reset_q_table()
        elif model_type == "prompt_optimizer":
            if model_name not in prompt_optimizers:
                raise HTTPException(
                    status_code=404, detail=f"Prompt optimizer '{model_name}' not found"
                )
            prompt_optimizers[model_name].reset()
        else:
            raise HTTPException(
                status_code=400, detail=f"Unknown model type: {model_type}"
            )

        return ModelResetResponse(
            reset=True,
            model_name=model_name,
            model_type=model_type,
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(f"Failed to reset model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset model: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8003, reload=True, log_level="info")
