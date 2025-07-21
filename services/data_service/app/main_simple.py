"""
Data Service - Simple Main FastAPI Application

This is a simplified version for testing basic functionality.
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Data Service (Simple)...")
    logger.info("Data Service started successfully")

    yield

    # Shutdown
    logger.info("Shutting down Data Service...")


# Create FastAPI app
app = FastAPI(
    title="CodeConductor Data Service (Simple)",
    description="Simple data service for testing",
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


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Data Service (Simple)",
        "version": "2.0.0",
        "bandits_ready": True,
        "qlearning_ready": True,
        "prompt_optimizer_ready": True,
        "active_bandits": ["default"],
        "active_agents": ["qlearning_agent", "prompt_optimizer"],
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Data Service is running!"}


if __name__ == "__main__":
    uvicorn.run(
        "main_simple:app", host="0.0.0.0", port=8003, reload=True, log_level="info"
    )
