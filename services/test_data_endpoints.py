#!/usr/bin/env python3
"""
Test Data Service Endpoints

Simple script to test the Data Service endpoints.
"""

import requests
import json


def test_health():
    """Test health endpoint."""
    print("=== Testing Health Endpoint ===")
    response = requests.get("http://localhost:9006/health")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(json.dumps(data, indent=2))
    else:
        print(f"Error: {response.text}")
    print()


def test_bandit_choose():
    """Test bandit choose endpoint."""
    print("=== Testing Bandit Choose Endpoint ===")
    data = {
        "arms": ["arm1", "arm2", "arm3"],
        "features": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "bandit_name": "test_bandit",
    }
    response = requests.post("http://localhost:9006/bandits/choose", json=data)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(json.dumps(data, indent=2))
    else:
        print(f"Error: {response.text}")
    print()


def test_qlearning_run():
    """Test Q-learning run endpoint."""
    print("=== Testing Q-Learning Run Endpoint ===")
    data = {
        "context": {
            "task_type": "api_creation",
            "complexity": "medium",
            "language": "python",
            "agent_count": 2,
        },
        "agent_name": "test_qlearning",
    }
    response = requests.post("http://localhost:9006/qlearning/run", json=data)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(json.dumps(data, indent=2))
    else:
        print(f"Error: {response.text}")
    print()


def test_prompt_optimize():
    """Test prompt optimization endpoint."""
    print("=== Testing Prompt Optimization Endpoint ===")
    data = {
        "original_prompt": "Create a simple API",
        "task_id": "test_task_001",
        "arm_prev": "add_type_hints",
        "passed": True,
        "blocked": False,
        "complexity": 0.5,
        "model_source": "gpt-4",
        "agent_name": "test_optimizer",
    }
    response = requests.post("http://localhost:9006/prompt/optimize", json=data)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(json.dumps(data, indent=2))
    else:
        print(f"Error: {response.text}")
    print()


if __name__ == "__main__":
    print("🚀 Testing Data Service Endpoints")
    print("=" * 50)

    test_health()
    test_bandit_choose()
    test_qlearning_run()
    test_prompt_optimize()

    print("✅ Testing complete!")
