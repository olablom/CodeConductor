#!/usr/bin/env python3
"""
Mock Metrics Service for CodeConductor
Generates fake metrics for testing Grafana dashboard
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import FastAPI
from fastapi.responses import Response
import time
import random
import threading

app = FastAPI(title="Mock Metrics Service")

# Mock metrics
REQUEST_COUNT = Counter(
    "codeconductor_requests_total", "Total requests", ["service", "endpoint"]
)
INFERENCE_DURATION = Histogram(
    "codeconductor_inference_duration_seconds",
    "Inference duration",
    ["service", "algorithm"],
)
GPU_MEMORY = Gauge("codeconductor_gpu_memory_bytes", "GPU memory usage", ["type"])
BANDIT_SELECTIONS = Counter(
    "codeconductor_bandit_selections_total", "Bandit arm selections", ["arm"]
)

# System metrics
CPU_USAGE = Gauge("codeconductor_cpu_usage_percent", "CPU usage percentage")
MEMORY_USAGE = Gauge("codeconductor_memory_usage_bytes", "Memory usage in bytes")
ACTIVE_CONNECTIONS = Gauge("codeconductor_active_connections", "Active connections")


def generate_mock_metrics():
    """Generate realistic mock metrics"""
    while True:
        # Simulate AI inference
        gpu_duration = random.uniform(0.001, 0.01)  # 1-10ms
        cpu_duration = random.uniform(0.01, 0.1)  # 10-100ms

        INFERENCE_DURATION.labels(
            service="gpu-data", algorithm="neural_bandit"
        ).observe(gpu_duration)
        INFERENCE_DURATION.labels(service="cpu-data", algorithm="q_learning").observe(
            cpu_duration
        )

        # Simulate requests
        REQUEST_COUNT.labels(service="gpu-data", endpoint="/bandits/choose").inc()
        REQUEST_COUNT.labels(service="cpu-data", endpoint="/qlearning/run").inc()

        # Simulate bandit selections
        arms = ["exploration", "exploitation", "random", "neural"]
        selected_arm = random.choice(arms)
        BANDIT_SELECTIONS.labels(arm=selected_arm).inc()

        # Simulate GPU memory
        allocated = random.uniform(1000000000, 5000000000)  # 1-5GB
        reserved = random.uniform(2000000000, 8000000000)  # 2-8GB
        GPU_MEMORY.labels(type="allocated").set(allocated)
        GPU_MEMORY.labels(type="reserved").set(reserved)

        # Simulate system metrics
        CPU_USAGE.set(random.uniform(20, 80))
        MEMORY_USAGE.set(random.uniform(1000000000, 4000000000))
        ACTIVE_CONNECTIONS.set(random.randint(5, 50))

        time.sleep(5)  # Update every 5 seconds


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type="text/plain")


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "mock_metrics"}


if __name__ == "__main__":
    # Start metrics generation in background
    metrics_thread = threading.Thread(target=generate_mock_metrics, daemon=True)
    metrics_thread.start()

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
