from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import jwt

router = APIRouter(
    prefix="/api/posts",
    tags=["posts"],
    responses={404: {"description": "Not found"}},
)

# OAuth2 scheme for JWT tokens
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

# Pydantic models
class PostBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1)
    tags: Optional[List[str]] = []
    published: bool = False

class PostCreate(PostBase):
    pass

class PostUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    content: Optional[str] = Field(None, min_length=1)
    tags: Optional[List[str]] = None
    published: Optional[bool] = None

class PostResponse(PostBase):
    id: int
    author_id: int
    author_username: str
    created_at: datetime
    updated_at: datetime
    views: int = 0
    likes: int = 0

    class Config:
        from_attributes = True

class PostListResponse(BaseModel):
    posts: List[PostResponse]
    total: int
    page: int
    limit: int
    has_next: bool
    has_prev: bool

# Mock database
posts_db = {
    1: {
        "id": 1,
        "title": "Getting Started with FastAPI",
        "content": "FastAPI is a modern web framework for building APIs...",
        "tags": ["python", "fastapi", "tutorial"],
        "published": True,
        "author_id": 1,
        "author_username": "johndoe",
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
        "views": 150,
        "likes": 25
    },
    2: {
        "id": 2,
        "title": "React Best Practices",
        "content": "Here are some React best practices to follow...",
        "tags": ["javascript", "react", "frontend"],
        "published": True,
        "author_id": 1,
        "author_username": "johndoe",
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
        "views": 200,
        "likes": 40
    }
}

# Dependency to get current user from JWT token
async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, "SECRET_KEY", algorithms=["HS256"])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        return {"username": username, "id": 1}  # Mock user data
    except jwt.PyJWTError:
        raise credentials_exception

# Optional auth dependency (for public endpoints that may have auth)
async def get_optional_user(token: Optional[str] = Depends(oauth2_scheme)) -> Optional[dict]:
    if not token:
        return None
    try:
        return await get_current_user(token)
    except:
        return None

# Public endpoints - no auth required
@router.get("/", response_model=PostListResponse)
async def list_posts(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    tag: Optional[str] = None,
    published_only: bool = True
):
    """List all posts with pagination - Public endpoint"""
    # Filter posts
    filtered_posts = []
    for post in posts_db.values():
        if published_only and not post["published"]:
            continue
        if tag and tag not in post["tags"]:
            continue
        filtered_posts.append(post)
    
    # Sort by creation date (newest first)
    filtered_posts.sort(key=lambda x: x["created_at"], reverse=True)
    
    # Paginate
    start = (page - 1) * limit
    end = start + limit
    paginated_posts = filtered_posts[start:end]
    
    return PostListResponse(
        posts=[PostResponse(**post) for post in paginated_posts],
        total=len(filtered_posts),
        page=page,
        limit=limit,
        has_next=end < len(filtered_posts),
        has_prev=page > 1
    )

@router.get("/search", response_model=PostListResponse)
async def search_posts(
    q: str = Query(..., min_length=1),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100)
):
    """Search posts by title or content - Public endpoint"""
    search_query = q.lower()
    matching_posts = []
    
    for post in posts_db.values():
        if not post["published"]:
            continue
        if (search_query in post["title"].lower() or 
            search_query in post["content"].lower() or
            any(search_query in tag for tag in post["tags"])):
            matching_posts.append(post)
    
    # Paginate results
    start = (page - 1) * limit
    end = start + limit
    paginated_posts = matching_posts[start:end]
    
    return PostListResponse(
        posts=[PostResponse(**post) for post in paginated_posts],
        total=len(matching_posts),
        page=page,
        limit=limit,
        has_next=end < len(matching_posts),
        has_prev=page > 1
    )

@router.get("/{post_id}", response_model=PostResponse)
async def get_post(post_id: int, current_user: Optional[dict] = Depends(get_optional_user)):
    """Get a specific post - Public for published, auth required for drafts"""
    if post_id not in posts_db:
        raise HTTPException(status_code=404, detail="Post not found")
    
    post = posts_db[post_id]
    
    # Check if post is published or user is the author
    if not post["published"]:
        if not current_user or current_user["id"] != post["author_id"]:
            raise HTTPException(status_code=404, detail="Post not found")
    
    # Increment view count
    post["views"] += 1
    
    return PostResponse(**post)

# Protected endpoints - require JWT auth
@router.post("/", response_model=PostResponse, status_code=status.HTTP_201_CREATED)
async def create_post(
    post: PostCreate,
    current_user: dict = Depends(get_current_user)
):
    """Create a new post - Requires authentication"""
    new_id = max(posts_db.keys()) + 1 if posts_db else 1
    
    new_post = {
        "id": new_id,
        **post.dict(),
        "author_id": current_user["id"],
        "author_username": current_user["username"],
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
        "views": 0,
        "likes": 0
    }
    
    posts_db[new_id] = new_post
    return PostResponse(**new_post)

@router.put("/{post_id}", response_model=PostResponse)
async def update_post(
    post_id: int,
    post_update: PostUpdate,
    current_user: dict = Depends(get_current_user)
):
    """Update a post - Only author can update"""
    if post_id not in posts_db:
        raise HTTPException(status_code=404, detail="Post not found")
    
    post = posts_db[post_id]
    
    # Check ownership
    if post["author_id"] != current_user["id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this post"
        )
    
    # Update fields
    update_data = post_update.dict(exclude_unset=True)
    post.update(update_data)
    post["updated_at"] = datetime.now()
    
    return PostResponse(**post)

@router.delete("/{post_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_post(
    post_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Delete a post - Only author can delete"""
    if post_id not in posts_db:
        raise HTTPException(status_code=404, detail="Post not found")
    
    post = posts_db[post_id]
    
    # Check ownership
    if post["author_id"] != current_user["id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this post"
        )
    
    # Remove from database
    del posts_db[post_id]
    return {"message": "Post deleted successfully"}

@router.post("/{post_id}/like", response_model=dict)
async def like_post(
    post_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Like a post - Requires authentication"""
    if post_id not in posts_db:
        raise HTTPException(status_code=404, detail="Post not found")
    
    post = posts_db[post_id]
    
    # Check if post is published
    if not post["published"]:
        raise HTTPException(status_code=404, detail="Post not found")
    
    # Increment likes (in real app, track who liked to prevent duplicates)
    post["likes"] += 1
    
    return {"message": "Post liked", "likes": post["likes"]}

@router.get("/user/{user_id}", response_model=PostListResponse)
async def get_user_posts(
    user_id: int,
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    current_user: Optional[dict] = Depends(get_optional_user)
):
    """Get posts by a specific user - Shows drafts only to post owner"""
    user_posts = []
    
    for post in posts_db.values():
        if post["author_id"] != user_id:
            continue
        
        # Only show unpublished posts to the author
        if not post["published"] and (not current_user or current_user["id"] != user_id):
            continue
            
        user_posts.append(post)
    
    # Sort by creation date (newest first)
    user_posts.sort(key=lambda x: x["created_at"], reverse=True)
    
    # Paginate
    start = (page - 1) * limit
    end = start + limit
    paginated_posts = user_posts[start:end]
    
    return PostListResponse(
        posts=[PostResponse(**post) for post in paginated_posts],
        total=len(user_posts),
        page=page,
        limit=limit,
        has_next=end < len(user_posts),
        has_prev=page > 1
    ) 