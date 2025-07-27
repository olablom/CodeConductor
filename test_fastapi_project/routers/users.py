from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr
from typing import List, Optional
from datetime import datetime
import jwt

router = APIRouter(
    prefix="/api/users",
    tags=["users"],
    responses={404: {"description": "Not found"}},
)

# OAuth2 scheme for JWT tokens
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


# Pydantic models
class UserBase(BaseModel):
    username: str
    email: EmailStr
    full_name: Optional[str] = None


class UserCreate(UserBase):
    password: str


class UserUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    password: Optional[str] = None


class UserResponse(UserBase):
    id: int
    created_at: datetime
    is_active: bool = True

    class Config:
        from_attributes = True


class UserListResponse(BaseModel):
    users: List[UserResponse]
    total: int
    page: int
    limit: int


# Mock database
users_db = {
    1: {
        "id": 1,
        "username": "johndoe",
        "email": "john@example.com",
        "full_name": "John Doe",
        "created_at": datetime.now(),
        "is_active": True,
        "hashed_password": "hashed_secret",
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
        # In production, verify with proper secret and algorithm
        payload = jwt.decode(token, "SECRET_KEY", algorithms=["HS256"])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        # In production, look up user from database
        return {"username": username, "id": 1}
    except jwt.PyJWTError:
        raise credentials_exception


# Public endpoint - no auth required
@router.post(
    "/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED
)
async def register_user(user: UserCreate):
    """Register a new user - Public endpoint"""
    # Check if user already exists
    for u in users_db.values():
        if u["username"] == user.username or u["email"] == user.email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username or email already registered",
            )

    # Create new user
    new_id = max(users_db.keys()) + 1 if users_db else 1
    new_user = {
        "id": new_id,
        "username": user.username,
        "email": user.email,
        "full_name": user.full_name,
        "created_at": datetime.now(),
        "is_active": True,
        "hashed_password": f"hashed_{user.password}",  # In production, use proper hashing
    }
    users_db[new_id] = new_user

    return UserResponse(**new_user)


# Protected endpoints - require JWT auth
@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(current_user: dict = Depends(get_current_user)):
    """Get current user profile"""
    user_id = current_user["id"]
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")

    return UserResponse(**users_db[user_id])


@router.get("/", response_model=UserListResponse)
async def list_users(
    page: int = 1, limit: int = 10, current_user: dict = Depends(get_current_user)
):
    """List all users with pagination"""
    # Calculate pagination
    start = (page - 1) * limit
    end = start + limit

    # Get users without passwords
    all_users = list(users_db.values())
    paginated_users = all_users[start:end]

    return UserListResponse(
        users=[UserResponse(**user) for user in paginated_users],
        total=len(all_users),
        page=page,
        limit=limit,
    )


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(user_id: int, current_user: dict = Depends(get_current_user)):
    """Get a specific user by ID"""
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")

    return UserResponse(**users_db[user_id])


@router.put("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    user_update: UserUpdate,
    current_user: dict = Depends(get_current_user),
):
    """Update user information"""
    # Check if user can only update their own profile
    if current_user["id"] != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this user",
        )

    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")

    # Update user fields
    user = users_db[user_id]
    update_data = user_update.dict(exclude_unset=True)

    if "password" in update_data:
        update_data["hashed_password"] = f"hashed_{update_data.pop('password')}"

    user.update(update_data)

    return UserResponse(**user)


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(user_id: int, current_user: dict = Depends(get_current_user)):
    """Delete a user (soft delete)"""
    # Check if user can only delete their own profile
    if current_user["id"] != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this user",
        )

    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")

    # Soft delete - just mark as inactive
    users_db[user_id]["is_active"] = False

    return {"message": "User deleted successfully"}
