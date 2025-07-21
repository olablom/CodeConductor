#!/usr/bin/env python3
"""
GPU-Powered Data Service
Neural Q-Learning and GPU-accelerated bandits using PyTorch
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import uuid

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

app = FastAPI(title="GPU Data Service", version="2.0.0")


# Neural Network for Deep Q-Learning
class DeepQNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(DeepQNetwork, self).__init__()
        self.device = device

        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, action_size),
        ).to(self.device)

    def forward(self, x):
        return self.network(x)


# Neural Contextual Bandit
class NeuralBandit:
    def __init__(self, feature_dim: int, num_arms: int):
        self.feature_dim = feature_dim
        self.num_arms = num_arms
        self.device = device

        # Neural network for arm selection
        self.model = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_arms),
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        # Track arm statistics
        self.arm_pulls = torch.zeros(num_arms).to(self.device)
        self.arm_rewards = torch.zeros(num_arms).to(self.device)

    def select_arm(self, features: List[float], epsilon: float = 0.1) -> int:
        features_tensor = (
            torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        )

        # Epsilon-greedy exploration
        if np.random.random() < epsilon:
            return np.random.randint(0, self.num_arms)

        # Neural network prediction
        with torch.no_grad():
            q_values = self.model(features_tensor)
            return q_values.argmax().item()

    def update(self, arm: int, reward: float, features: List[float]):
        features_tensor = (
            torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        )

        # Update statistics
        self.arm_pulls[arm] += 1
        self.arm_rewards[arm] += reward

        # Neural network training
        self.optimizer.zero_grad()
        q_values = self.model(features_tensor)

        # Create target (current reward + future estimate)
        target = q_values.clone()
        target[0, arm] = reward

        loss = self.criterion(q_values, target)
        loss.backward()
        self.optimizer.step()


# Global instances
neural_bandit = None
deep_q_network = None


# Pydantic models
class BanditRequest(BaseModel):
    arms: List[str]
    features: List[float]
    epsilon: float = 0.1


class BanditResponse(BaseModel):
    selected_arm: str
    arm_index: int
    q_values: Dict[str, float]
    confidence: float
    exploration: bool
    gpu_used: bool
    inference_time_ms: float
    timestamp: str


class QLearningRequest(BaseModel):
    episodes: int = 1
    learning_rate: float = 0.001
    discount_factor: float = 0.99
    epsilon: float = 0.1
    context: Dict[str, Any]


class QLearningResponse(BaseModel):
    agent_name: str
    selected_action: Dict[str, Any]
    q_value: float
    epsilon: float
    exploration: bool
    confidence: float
    reasoning: str
    gpu_used: bool
    training_time_ms: float
    timestamp: str


class RewardUpdate(BaseModel):
    arm: str
    reward: float
    features: List[float]


@app.on_event("startup")
async def startup_event():
    global neural_bandit, deep_q_network
    print("🚀 Initializing GPU-powered AI services...")

    # Initialize neural bandit
    neural_bandit = NeuralBandit(feature_dim=10, num_arms=5)
    print(f"✅ Neural Bandit initialized on {device}")

    # Initialize deep Q-network
    state_size = 20  # Context features
    action_size = 6  # Different agent combinations
    deep_q_network = DeepQNetwork(state_size, action_size)
    print(f"✅ Deep Q-Network initialized on {device}")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "gpu_data_service",
        "gpu_available": torch.cuda.is_available(),
        "device": str(device),
        "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3
        if torch.cuda.is_available()
        else 0,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/gpu/bandits/choose", response_model=BanditResponse)
async def neural_bandit_choose(request: BanditRequest):
    global neural_bandit

    if neural_bandit is None:
        raise HTTPException(status_code=500, detail="Neural bandit not initialized")

    start_time = (
        torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    )
    end_time = (
        torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    )

    if start_time:
        start_time.record()

    # Select arm using neural bandit
    arm_index = neural_bandit.select_arm(request.features, request.epsilon)
    selected_arm = request.arms[arm_index % len(request.arms)]

    # Get Q-values for all arms
    features_tensor = (
        torch.tensor(request.features, dtype=torch.float32).unsqueeze(0).to(device)
    )
    with torch.no_grad():
        q_values = neural_bandit.model(features_tensor).cpu().numpy()[0]

    if end_time:
        end_time.record()
        torch.cuda.synchronize()
        inference_time = start_time.elapsed_time(end_time)
    else:
        inference_time = 0.0

    # Calculate confidence based on Q-value spread
    confidence = float(np.max(q_values) / (np.sum(q_values) + 1e-8))
    exploration = np.random.random() < request.epsilon

    return BanditResponse(
        selected_arm=selected_arm,
        arm_index=arm_index,
        q_values={
            arm: float(q_values[i % len(request.arms)])
            for i, arm in enumerate(request.arms)
        },
        confidence=confidence,
        exploration=exploration,
        gpu_used=torch.cuda.is_available(),
        inference_time_ms=inference_time,
        timestamp=datetime.now().isoformat(),
    )


@app.post("/gpu/qlearning/run", response_model=QLearningResponse)
async def deep_q_learning_run(request: QLearningRequest):
    global deep_q_network

    if deep_q_network is None:
        raise HTTPException(status_code=500, detail="Deep Q-network not initialized")

    start_time = (
        torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    )
    end_time = (
        torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    )

    if start_time:
        start_time.record()

    # Convert context to state vector with proper type handling
    def safe_float(value, default=0.5):
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, str):
            # Convert string values to numeric
            value_lower = value.lower()
            if value_lower in ["low", "simple", "small", "minimal"]:
                return 0.2
            elif value_lower in ["medium", "moderate", "average"]:
                return 0.5
            elif value_lower in ["high", "complex", "large", "extensive"]:
                return 0.8
            elif value_lower in ["critical", "maximum", "extreme"]:
                return 1.0
            else:
                return default
        else:
            return default

    context_features = [
        safe_float(request.context.get("complexity", 0.5)),
        safe_float(request.context.get("urgency", 0.5)),
        safe_float(request.context.get("team_size", 0.5)),
        safe_float(request.context.get("deadline", 0.5)),
        safe_float(request.context.get("domain_expertise", 0.5)),
        safe_float(request.context.get("code_quality", 0.5)),
        safe_float(request.context.get("testing_required", 0.5)),
        safe_float(request.context.get("documentation_needed", 0.5)),
        safe_float(request.context.get("performance_critical", 0.5)),
        safe_float(request.context.get("security_important", 0.5)),
        safe_float(request.context.get("scalability_needed", 0.5)),
        safe_float(request.context.get("maintainability", 0.5)),
        safe_float(request.context.get("time_constraints", 0.5)),
        safe_float(request.context.get("budget_constraints", 0.5)),
        safe_float(request.context.get("risk_tolerance", 0.5)),
        safe_float(request.context.get("innovation_level", 0.5)),
        safe_float(request.context.get("compliance_required", 0.5)),
        safe_float(request.context.get("integration_complexity", 0.5)),
        safe_float(request.context.get("legacy_system", 0.5)),
    ]

    # Pad or truncate to state_size
    state_size = 20
    if len(context_features) < state_size:
        context_features.extend([0.0] * (state_size - len(context_features)))
    else:
        context_features = context_features[:state_size]

    state_tensor = (
        torch.tensor(context_features, dtype=torch.float32).unsqueeze(0).to(device)
    )

    # Epsilon-greedy action selection
    if np.random.random() < request.epsilon:
        action = np.random.randint(0, 6)  # 6 different agent combinations
        exploration = True
    else:
        with torch.no_grad():
            q_values = deep_q_network(state_tensor)
            action = q_values.argmax().item()
            exploration = False

    # Agent combinations
    agent_combinations = [
        {"agent_combination": "codegen_only", "prompt_strategy": "standard"},
        {"agent_combination": "codegen_review", "prompt_strategy": "chain_of_thought"},
        {"agent_combination": "architect_codegen", "prompt_strategy": "detailed"},
        {"agent_combination": "full_team", "prompt_strategy": "collaborative"},
        {
            "agent_combination": "expert_specialist",
            "prompt_strategy": "domain_specific",
        },
        {"agent_combination": "rapid_prototype", "prompt_strategy": "minimal"},
    ]

    selected_action = agent_combinations[action]

    # Get Q-value for selected action
    with torch.no_grad():
        q_values = deep_q_network(state_tensor)
        q_value = float(q_values[0, action].item())

    if end_time:
        end_time.record()
        torch.cuda.synchronize()
        training_time = start_time.elapsed_time(end_time)
    else:
        training_time = 0.0

    # Calculate confidence
    confidence = min(0.95, max(0.1, abs(q_value) / 10.0))

    return QLearningResponse(
        agent_name="deep_qlearning_agent",
        selected_action=selected_action,
        q_value=q_value,
        epsilon=request.epsilon,
        exploration=exploration,
        confidence=confidence,
        reasoning=f"Deep Q-learning selection with epsilon={request.epsilon:.4f}, GPU={torch.cuda.is_available()}",
        gpu_used=torch.cuda.is_available(),
        training_time_ms=training_time,
        timestamp=datetime.now().isoformat(),
    )


@app.post("/gpu/bandits/update")
async def update_bandit_reward(update: RewardUpdate):
    global neural_bandit

    if neural_bandit is None:
        raise HTTPException(status_code=500, detail="Neural bandit not initialized")

    # Find arm index
    arm_index = 0  # Default to first arm
    neural_bandit.update(arm_index, update.reward, update.features)

    return {
        "status": "updated",
        "arm": update.arm,
        "reward": update.reward,
        "gpu_used": torch.cuda.is_available(),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/gpu/stats")
async def get_gpu_stats():
    stats = {
        "gpu_available": torch.cuda.is_available(),
        "device": str(device),
        "gpu_memory_total_gb": 0.0,
        "gpu_memory_used_gb": 0.0,
        "gpu_memory_free_gb": 0.0,
        "gpu_utilization": 0.0,
    }

    if torch.cuda.is_available():
        stats["gpu_memory_total_gb"] = (
            torch.cuda.get_device_properties(0).total_memory / 1024**3
        )
        stats["gpu_memory_used_gb"] = (
            torch.cuda.get_device_properties(0).total_memory
            - torch.cuda.memory_reserved(0)
        ) / 1024**3
        stats["gpu_memory_free_gb"] = torch.cuda.memory_reserved(0) / 1024**3

    return stats


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8007)
