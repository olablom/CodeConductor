from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional
import uuid
import jwt
import hashlib
import os
from datetime import datetime, timedelta
import pika
import json

app = FastAPI(title="User Service", version="1.0.0")

# JWT Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# RabbitMQ Configuration
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5672"))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "admin")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "admin123")

# In-memory storage for demo
users_db = {}

# Security
security = HTTPBearer()


class UserCreate(BaseModel):
    username: str
    email: str
    password: str


class UserLogin(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    id: str
    username: str
    email: str


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


def get_rabbitmq_connection():
    """Get RabbitMQ connection"""
    try:
        credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=RABBITMQ_HOST, port=RABBITMQ_PORT, credentials=credentials
            )
        )
        return connection
    except Exception as e:
        print(f"RabbitMQ connection failed: {e}")
        return None


def publish_user_event(event_type: str, user_data: dict):
    """Publish user event to RabbitMQ"""
    connection = get_rabbitmq_connection()
    if connection:
        try:
            channel = connection.channel()
            channel.queue_declare(queue="user_events", durable=True)

            message = {
                "event_type": event_type,
                "user_id": user_data.get("id"),
                "username": user_data.get("username"),
                "email": user_data.get("email"),
                "timestamp": datetime.utcnow().isoformat(),
            }

            channel.basic_publish(
                exchange="",
                routing_key="user_events",
                body=json.dumps(message),
                properties=pika.BasicProperties(
                    delivery_mode=2
                ),  # make message persistent
            )
            print(f"Published {event_type} event for user {user_data.get('username')}")
        except Exception as e:
            print(f"Failed to publish event: {e}")
        finally:
            connection.close()


def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> TokenData:
    """Verify JWT token"""
    try:
        payload = jwt.decode(
            credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM]
        )
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        token_data = TokenData(username=username)
        return token_data
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


@app.post("/register", response_model=UserResponse)
def register_user(user: UserCreate):
    """Register a new user"""
    # Check if username already exists
    for existing_user in users_db.values():
        if existing_user["username"] == user.username:
            raise HTTPException(status_code=400, detail="Username already registered")

    user_id = str(uuid.uuid4())
    hashed_password = hash_password(user.password)

    user_data = {
        "id": user_id,
        "username": user.username,
        "email": user.email,
        "password": hashed_password,
    }

    users_db[user_id] = user_data

    # Publish user created event
    publish_user_event("user_created", user_data)

    return UserResponse(id=user_id, username=user.username, email=user.email)


@app.post("/login", response_model=Token)
def login_user(user: UserLogin):
    """Login user and return JWT token"""
    # Find user by username
    user_data = None
    for user_record in users_db.values():
        if user_record["username"] == user.username:
            user_data = user_record
            break

    if not user_data:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Verify password
    if user_data["password"] != hash_password(user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )

    # Publish user login event
    publish_user_event("user_login", user_data)

    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/users/me", response_model=UserResponse)
def get_current_user(current_user: TokenData = Depends(verify_token)):
    """Get current user information"""
    # Find user by username
    for user_data in users_db.values():
        if user_data["username"] == current_user.username:
            return UserResponse(
                id=user_data["id"],
                username=user_data["username"],
                email=user_data["email"],
            )

    raise HTTPException(status_code=404, detail="User not found")


@app.get("/users/{user_id}", response_model=UserResponse)
def get_user(user_id: str):
    """Get user by ID"""
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")

    user_data = users_db[user_id]
    return UserResponse(
        id=user_data["id"], username=user_data["username"], email=user_data["email"]
    )


@app.get("/users", response_model=List[UserResponse])
def list_users():
    """List all users"""
    return [
        UserResponse(
            id=user_data["id"], username=user_data["username"], email=user_data["email"]
        )
        for user_data in users_db.values()
    ]


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "user-service"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
