#!/usr/bin/env python3
"""
Test FastAPI application for ProjectAnalyzer testing.

This app includes various route types to test the analyzer:
- Basic CRUD operations
- Authentication routes
- Different HTTP methods
- Path parameters
- Query parameters
"""

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.security import HTTPBearer
from pydantic import BaseModel
from typing import List, Optional
import uuid
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="Test API for CodeConductor",
    description="A test API to verify ProjectAnalyzer functionality",
    version="1.0.0"
)

# Security
security = HTTPBearer()

# Pydantic models
class User(BaseModel):
    name: str
    email: str
    age: Optional[int] = None

class Post(BaseModel):
    title: str
    content: str
    author_id: int

class LoginRequest(BaseModel):
    email: str
    password: str

# In-memory storage (for testing)
users_db = {}
posts_db = {}

# Authentication routes
@app.post("/api/auth/login")
async def login(login_data: LoginRequest):
    """User login endpoint"""
    if login_data.email == "admin@test.com" and login_data.password == "password":
        return {"access_token": "fake-jwt-token", "token_type": "bearer"}
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/api/auth/logout")
async def logout():
    """User logout endpoint"""
    return {"message": "Successfully logged out"}

@app.post("/api/auth/refresh")
async def refresh_token():
    """Refresh JWT token"""
    return {"access_token": "new-fake-jwt-token", "token_type": "bearer"}

# User management routes
@app.get("/api/users")
async def get_users(skip: int = Query(0, ge=0), limit: int = Query(10, ge=1, le=100)):
    """Get all users with pagination"""
    user_list = list(users_db.values())
    return {"users": user_list[skip:skip + limit], "total": len(user_list)}

@app.post("/api/users")
async def create_user(user: User):
    """Create a new user"""
    user_id = str(uuid.uuid4())
    user_data = user.dict()
    user_data["id"] = user_id
    user_data["created_at"] = datetime.now().isoformat()
    users_db[user_id] = user_data
    return {"message": "User created", "user": user_data}

@app.get("/api/users/{user_id}")
async def get_user(user_id: str):
    """Get user by ID"""
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    return users_db[user_id]

@app.put("/api/users/{user_id}")
async def update_user(user_id: str, user: User):
    """Update user by ID"""
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    user_data = user.dict()
    user_data["id"] = user_id
    user_data["updated_at"] = datetime.now().isoformat()
    users_db[user_id] = user_data
    return {"message": "User updated", "user": user_data}

@app.patch("/api/users/{user_id}")
async def partial_update_user(user_id: str, user: User):
    """Partially update user by ID"""
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    current_user = users_db[user_id]
    update_data = user.dict(exclude_unset=True)
    current_user.update(update_data)
    current_user["updated_at"] = datetime.now().isoformat()
    return {"message": "User partially updated", "user": current_user}

@app.delete("/api/users/{user_id}")
async def delete_user(user_id: str):
    """Delete user by ID"""
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    del users_db[user_id]
    return {"message": "User deleted"}

# Post management routes
@app.get("/api/posts")
async def get_posts(skip: int = Query(0, ge=0), limit: int = Query(10, ge=1, le=100)):
    """Get all posts with pagination"""
    post_list = list(posts_db.values())
    return {"posts": post_list[skip:skip + limit], "total": len(post_list)}

@app.post("/api/posts")
async def create_post(post: Post):
    """Create a new post"""
    post_id = str(uuid.uuid4())
    post_data = post.dict()
    post_data["id"] = post_id
    post_data["created_at"] = datetime.now().isoformat()
    posts_db[post_id] = post_data
    return {"message": "Post created", "post": post_data}

@app.get("/api/posts/{post_id}")
async def get_post(post_id: str):
    """Get post by ID"""
    if post_id not in posts_db:
        raise HTTPException(status_code=404, detail="Post not found")
    return posts_db[post_id]

@app.put("/api/posts/{post_id}")
async def update_post(post_id: str, post: Post):
    """Update post by ID"""
    if post_id not in posts_db:
        raise HTTPException(status_code=404, detail="Post not found")
    post_data = post.dict()
    post_data["id"] = post_id
    post_data["updated_at"] = datetime.now().isoformat()
    posts_db[post_id] = post_data
    return {"message": "Post updated", "post": post_data}

@app.delete("/api/posts/{post_id}")
async def delete_post(post_id: str):
    """Delete post by ID"""
    if post_id not in posts_db:
        raise HTTPException(status_code=404, detail="Post not found")
    del posts_db[post_id]
    return {"message": "Post deleted"}

# Health check and status routes
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/status")
async def api_status():
    """API status endpoint"""
    return {
        "status": "running",
        "version": "1.0.0",
        "users_count": len(users_db),
        "posts_count": len(posts_db)
    }

# Admin routes
@app.get("/api/admin/users")
async def admin_get_users():
    """Admin endpoint to get all users (protected)"""
    return {"users": list(users_db.values()), "total": len(users_db)}

@app.get("/api/admin/stats")
async def admin_stats():
    """Admin endpoint for statistics"""
    return {
        "total_users": len(users_db),
        "total_posts": len(posts_db),
        "created_at": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 