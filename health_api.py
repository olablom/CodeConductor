#!/usr/bin/env python3
"""
Health API for CodeConductor - Provides health endpoints for external monitoring
"""

from flask import Flask, jsonify, request
from ensemble.model_manager import ModelManager
from prometheus_client import Counter, Histogram, generate_latest
import time
import asyncio

app = Flask(__name__)

# Initialize model manager
model_manager = ModelManager()

# Prometheus metrics
model_requests = Counter("model_requests_total", "Total requests per model", ["model"])
model_response_time = Histogram(
    "model_response_time_seconds", "Model response time", ["model"]
)
ensemble_success = Counter(
    "ensemble_success_total", "Total successful ensemble runs", ["status"]
)
codeconductor_response_time = Histogram(
    "codeconductor_response_time_seconds", "Total pipeline response time", ["task"]
)


def get_healthy_models():
    """Synchronous wrapper to get healthy models"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            models = loop.run_until_complete(model_manager.list_models())
            healthy_models = []
            for model in models:
                is_healthy = loop.run_until_complete(model_manager.check_health(model))
                healthy_models.append(
                    {
                        "name": model.name,
                        "healthy": is_healthy,
                        "last_response_time": 0.0,  # Placeholder
                    }
                )
            return healthy_models
        finally:
            loop.close()
    except Exception as e:
        print(f"Error getting healthy models: {e}")
        return []


@app.route("/health", methods=["GET"])
def health_check():
    start_time = time.time()
    healthy_models = get_healthy_models()
    response_time = time.time() - start_time
    status = "healthy" if healthy_models else "unhealthy"

    ensemble_success.labels(status=status).inc()
    return jsonify(
        {
            "status": status,
            "models": [
                {
                    "name": m["name"],
                    "healthy": m["healthy"],
                    "last_response_time": m.get("last_response_time", 0),
                }
                for m in healthy_models
            ],
            "response_time": response_time,
        }
    ), 200


@app.route("/health/models", methods=["GET"])
def health_models():
    healthy_models = get_healthy_models()
    return jsonify(
        {
            "models": [
                {"name": m["name"], "healthy": m["healthy"]} for m in healthy_models
            ]
        }
    ), 200


@app.route("/health/ensemble", methods=["GET"])
def health_ensemble():
    healthy_models = get_healthy_models()
    return jsonify(
        {"ensemble_status": "healthy" if len(healthy_models) >= 3 else "degraded"}
    ), 200


@app.route("/metrics", methods=["GET"])
def metrics():
    from flask import Response
    return Response(generate_latest(), mimetype='text/plain')


@app.route("/ready", methods=["GET"])
def ready():
    healthy_models = get_healthy_models()
    return jsonify(
        {"status": "ready" if healthy_models else "not_ready"}
    ), 200 if healthy_models else 503


@app.route("/live", methods=["GET"])
def live():
    return jsonify({"status": "live"}), 200


@app.route("/pipeline_metrics", methods=["POST"])
def pipeline_metrics():
    data = request.get_json()
    task = data.get("task", "unknown")
    total_time = data.get("total_time", 0.0)
    codeconductor_response_time.labels(task=task).observe(total_time)
    return jsonify({"message": "Pipeline metric recorded"}), 200


if __name__ == "__main__":
    app.run(port=8081)
