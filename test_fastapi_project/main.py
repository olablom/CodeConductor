from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

# Import routers
from routers import auth, users, posts

app = FastAPI(title="CodeConductor API", version="1.0.0")

# Include routers
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(posts.router)


# Original endpoints (kan flyttas senare)
@app.get("/")
def read_root():
    return {"message": "Welcome to CodeConductor API"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}


@app.post("/analyze")
def analyze_code(code: str):
    return {"analysis": "Code analysis result", "code_length": len(code)}


@app.get("/metrics")
def get_metrics():
    return {"total_requests": 1234, "active_users": 56}
