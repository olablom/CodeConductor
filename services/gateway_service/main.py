#!/usr/bin/env python3
"""
Gateway Service - API Gateway för CodeConductor
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import httpx
import logging
import uvicorn
from datetime import datetime
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CodeConductor API Gateway",
    description="API Gateway för routing till mikrotjänster",
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

# Service routing configuration
SERVICES = {
    "agent": "http://agent-service:8001",
    "orchestrator": "http://orchestrator-service:8002",
    "data": "http://data-service:8003",
    "auth": "http://auth-service:8005",
}


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "API Gateway",
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now(),
        "available_services": list(SERVICES.keys()),
    }


@app.get("/health")
async def health_check():
    """Detailed health check with service status"""
    service_status = {}

    async with httpx.AsyncClient() as client:
        for service_name, service_url in SERVICES.items():
            try:
                response = await client.get(f"{service_url}/health", timeout=5.0)
                service_status[service_name] = (
                    "healthy" if response.status_code == 200 else "unhealthy"
                )
            except Exception as e:
                service_status[service_name] = "unreachable"
                logger.warning(f"Service {service_name} unreachable: {e}")

    return {
        "status": "healthy",
        "gateway": "healthy",
        "services": service_status,
        "timestamp": datetime.now(),
    }


@app.get("/api/v1/agents/{path:path}")
async def route_to_agent_service(path: str, request: Request):
    """Route agent-related requests to Agent Service"""
    return await route_request("agent", path, request)


@app.get("/api/v1/orchestrator/{path:path}")
async def route_to_orchestrator_service(path: str, request: Request):
    """Route orchestrator-related requests to Orchestrator Service"""
    return await route_request("orchestrator", path, request)


@app.get("/api/v1/data/{path:path}")
async def route_to_data_service(path: str, request: Request):
    """Route data-related requests to Data Service"""
    return await route_request("data", path, request)


@app.get("/api/v1/auth/{path:path}")
async def route_to_auth_service(path: str, request: Request):
    """Route auth-related requests to Auth Service"""
    return await route_request("auth", path, request)


@app.post("/api/v1/agents/{path:path}")
async def route_to_agent_service_post(path: str, request: Request):
    """Route POST requests to Agent Service"""
    return await route_request("agent", path, request)


@app.post("/api/v1/orchestrator/{path:path}")
async def route_to_orchestrator_service_post(path: str, request: Request):
    """Route POST requests to Orchestrator Service"""
    return await route_request("orchestrator", path, request)


@app.post("/api/v1/data/{path:path}")
async def route_to_data_service_post(path: str, request: Request):
    """Route POST requests to Data Service"""
    return await route_request("data", path, request)


@app.post("/api/v1/auth/{path:path}")
async def route_to_auth_service_post(path: str, request: Request):
    """Route POST requests to Auth Service"""
    return await route_request("auth", path, request)


async def route_request(service_name: str, path: str, request: Request):
    """Route request to appropriate service"""
    try:
        if service_name not in SERVICES:
            raise HTTPException(
                status_code=404, detail=f"Service {service_name} not found"
            )

        service_url = SERVICES[service_name]
        target_url = f"{service_url}/{path}"

        # Get request body if present
        body = None
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
            except:
                pass

        # Get query parameters
        query_params = dict(request.query_params)

        # Forward request to service
        async with httpx.AsyncClient() as client:
            response = await client.request(
                method=request.method,
                url=target_url,
                params=query_params,
                content=body,
                headers=dict(request.headers),
                timeout=30.0,
            )

            # Return response from service
            return response.json()

    except httpx.RequestError as e:
        logger.error(f"Error routing to {service_name}: {e}")
        raise HTTPException(
            status_code=503, detail=f"Service {service_name} unavailable"
        )
    except Exception as e:
        logger.error(f"Unexpected error routing to {service_name}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/v1/status")
async def get_system_status():
    """Get overall system status"""
    status = {
        "gateway": "healthy",
        "services": {},
        "overall": "healthy",
        "timestamp": datetime.now(),
    }

    async with httpx.AsyncClient() as client:
        for service_name, service_url in SERVICES.items():
            try:
                response = await client.get(f"{service_url}/", timeout=5.0)
                if response.status_code == 200:
                    data = response.json()
                    status["services"][service_name] = data.get("status", "unknown")
                else:
                    status["services"][service_name] = "unhealthy"
            except Exception as e:
                status["services"][service_name] = "unreachable"
                logger.warning(f"Service {service_name} unreachable: {e}")

    # Determine overall status
    unhealthy_services = [s for s in status["services"].values() if s != "healthy"]
    if unhealthy_services:
        status["overall"] = "degraded"
        if len(unhealthy_services) == len(SERVICES):
            status["overall"] = "unhealthy"

    return status


@app.get("/api/v1/metrics")
async def get_metrics():
    """Get basic metrics"""
    return {
        "gateway_requests": "TODO: implement request counting",
        "service_health": "TODO: implement health metrics",
        "response_times": "TODO: implement timing metrics",
        "error_rates": "TODO: implement error tracking",
    }


if __name__ == "__main__":
    logger.info("Starting API Gateway...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
