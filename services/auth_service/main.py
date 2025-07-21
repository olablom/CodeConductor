#!/usr/bin/env python3
"""
Auth Service - Hanterar autentisering och godkännande
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging
import uvicorn
from datetime import datetime, timedelta
import uuid

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CodeConductor Auth Service",
    description="Service för autentisering och mänskligt godkännande",
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
class ApprovalRequest(BaseModel):
    request_type: str  # "code_generation", "deployment", "configuration_change"
    description: str
    requester: str
    urgency: str = "normal"  # low, normal, high, critical
    metadata: Optional[Dict[str, Any]] = None


class ApprovalResponse(BaseModel):
    approval_id: str
    status: str  # "pending", "approved", "rejected"
    request_type: str
    description: str
    requester: str
    created_at: datetime
    approved_at: Optional[datetime] = None
    approved_by: Optional[str] = None
    comments: Optional[str] = None


class AuthRequest(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    token: str
    user_id: str
    username: str
    expires_at: datetime


# In-memory storage (replace with database in production)
approvals = {}
users = {
    "admin": {
        "password": "admin123",  # In production, use hashed passwords
        "role": "admin",
        "permissions": ["read", "write", "approve"],
    },
    "developer": {
        "password": "dev123",
        "role": "developer",
        "permissions": ["read", "write"],
    },
    "reviewer": {
        "password": "review123",
        "role": "reviewer",
        "permissions": ["read", "approve"],
    },
}

# Simple token storage (in production, use JWT)
active_tokens = {}


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Auth Service",
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now(),
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "active_approvals": len(
            [a for a in approvals.values() if a["status"] == "pending"]
        ),
        "active_tokens": len(active_tokens),
        "registered_users": len(users),
        "uptime": "TODO: implement uptime tracking",
    }


@app.post("/auth/login", response_model=AuthResponse)
async def login(request: AuthRequest):
    """Logga in användare"""
    try:
        logger.info(f"Login attempt for user: {request.username}")

        if request.username not in users:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        user = users[request.username]
        if user["password"] != request.password:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # Generate token
        token = str(uuid.uuid4())
        expires_at = datetime.now() + timedelta(hours=24)

        active_tokens[token] = {
            "user_id": request.username,
            "username": request.username,
            "role": user["role"],
            "permissions": user["permissions"],
            "expires_at": expires_at,
        }

        logger.info(f"User {request.username} logged in successfully")

        return AuthResponse(
            token=token,
            user_id=request.username,
            username=request.username,
            expires_at=expires_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/auth/logout")
async def logout(token: str):
    """Logga ut användare"""
    try:
        if token in active_tokens:
            user = active_tokens.pop(token)
            logger.info(f"User {user['username']} logged out")
            return {"message": "Logged out successfully"}
        else:
            raise HTTPException(status_code=401, detail="Invalid token")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/auth/validate")
async def validate_token(token: str):
    """Validera token"""
    try:
        if token not in active_tokens:
            raise HTTPException(status_code=401, detail="Invalid token")

        token_data = active_tokens[token]
        if token_data["expires_at"] < datetime.now():
            # Token expired
            active_tokens.pop(token)
            raise HTTPException(status_code=401, detail="Token expired")

        return {
            "valid": True,
            "user_id": token_data["user_id"],
            "username": token_data["username"],
            "role": token_data["role"],
            "permissions": token_data["permissions"],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token validation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/approvals/request", response_model=ApprovalResponse)
async def request_approval(request: ApprovalRequest):
    """Begär godkännande"""
    try:
        logger.info(f"Approval request: {request.request_type} by {request.requester}")

        approval_id = str(uuid.uuid4())

        approval = ApprovalResponse(
            approval_id=approval_id,
            status="pending",
            request_type=request.request_type,
            description=request.description,
            requester=request.requester,
            created_at=datetime.now(),
            approved_at=None,
            approved_by=None,
            comments=None,
        )

        approvals[approval_id] = approval.dict()

        logger.info(f"Approval request created: {approval_id}")

        return approval

    except Exception as e:
        logger.error(f"Approval request error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/approvals/{approval_id}", response_model=ApprovalResponse)
async def get_approval(approval_id: str):
    """Hämta godkännandestatus"""
    if approval_id not in approvals:
        raise HTTPException(status_code=404, detail="Approval not found")

    return ApprovalResponse(**approvals[approval_id])


@app.post("/approvals/{approval_id}/approve")
async def approve_request(
    approval_id: str, approver: str, comments: Optional[str] = None
):
    """Godkänn en begäran"""
    try:
        if approval_id not in approvals:
            raise HTTPException(status_code=404, detail="Approval not found")

        approval = approvals[approval_id]
        if approval["status"] != "pending":
            raise HTTPException(status_code=400, detail="Approval is not pending")

        approval["status"] = "approved"
        approval["approved_at"] = datetime.now()
        approval["approved_by"] = approver
        approval["comments"] = comments

        logger.info(f"Approval {approval_id} approved by {approver}")

        return {"message": "Approval granted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Approval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/approvals/{approval_id}/reject")
async def reject_request(
    approval_id: str, rejector: str, comments: Optional[str] = None
):
    """Avvisa en begäran"""
    try:
        if approval_id not in approvals:
            raise HTTPException(status_code=404, detail="Approval not found")

        approval = approvals[approval_id]
        if approval["status"] != "pending":
            raise HTTPException(status_code=400, detail="Approval is not pending")

        approval["status"] = "rejected"
        approval["approved_at"] = datetime.now()
        approval["approved_by"] = rejector
        approval["comments"] = comments

        logger.info(f"Approval {approval_id} rejected by {rejector}")

        return {"message": "Approval rejected successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Rejection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/approvals")
async def list_approvals(status: Optional[str] = None):
    """Lista alla godkännanden"""
    all_approvals = list(approvals.values())

    if status:
        all_approvals = [a for a in all_approvals if a["status"] == status]

    return {"approvals": all_approvals, "total": len(all_approvals)}


if __name__ == "__main__":
    logger.info("Starting Auth Service...")
    uvicorn.run("main:app", host="0.0.0.0", port=8005, reload=True, log_level="info")
