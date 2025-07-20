from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional
import uuid
import jwt
import os
from datetime import datetime, timedelta
import pika
import json

app = FastAPI(title="Order Service", version="1.0.0")

# JWT Configuration (same as user service)
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"

# RabbitMQ Configuration
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5672"))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "admin")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "admin123")

# In-memory storage for demo
orders_db = {}

# Security
security = HTTPBearer()


class OrderCreate(BaseModel):
    item: str
    quantity: int
    price: Optional[float] = None


class OrderResponse(BaseModel):
    id: str
    user_id: str
    username: str
    item: str
    quantity: int
    price: Optional[float]
    status: str
    created_at: str


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


def publish_order_event(event_type: str, order_data: dict):
    """Publish order event to RabbitMQ"""
    connection = get_rabbitmq_connection()
    if connection:
        try:
            channel = connection.channel()
            channel.queue_declare(queue="order_events", durable=True)

            message = {
                "event_type": event_type,
                "order_id": order_data.get("id"),
                "user_id": order_data.get("user_id"),
                "username": order_data.get("username"),
                "item": order_data.get("item"),
                "quantity": order_data.get("quantity"),
                "status": order_data.get("status"),
                "timestamp": datetime.utcnow().isoformat(),
            }

            channel.basic_publish(
                exchange="",
                routing_key="order_events",
                body=json.dumps(message),
                properties=pika.BasicProperties(
                    delivery_mode=2
                ),  # make message persistent
            )
            print(f"Published {event_type} event for order {order_data.get('id')}")
        except Exception as e:
            print(f"Failed to publish event: {e}")
        finally:
            connection.close()


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


def get_user_id_from_username(username: str) -> str:
    """Get user ID from username (simplified - in real app would query user service)"""
    # For demo purposes, we'll use a simple mapping
    # In production, this would make an API call to the user service
    return f"user-{hash(username) % 10000}"


@app.post("/orders", response_model=OrderResponse)
def create_order(order: OrderCreate, current_user: TokenData = Depends(verify_token)):
    """Create a new order"""
    order_id = str(uuid.uuid4())
    user_id = get_user_id_from_username(current_user.username)

    order_data = {
        "id": order_id,
        "user_id": user_id,
        "username": current_user.username,
        "item": order.item,
        "quantity": order.quantity,
        "price": order.price or 10.0,  # Default price
        "status": "pending",
        "created_at": datetime.utcnow().isoformat(),
    }

    orders_db[order_id] = order_data

    # Publish order created event
    publish_order_event("order_created", order_data)

    return OrderResponse(**order_data)


@app.get("/orders/{order_id}", response_model=OrderResponse)
def get_order(order_id: str, current_user: TokenData = Depends(verify_token)):
    """Get order by ID"""
    if order_id not in orders_db:
        raise HTTPException(status_code=404, detail="Order not found")

    order_data = orders_db[order_id]

    # Check if user owns this order
    if order_data["username"] != current_user.username:
        raise HTTPException(status_code=403, detail="Not authorized to view this order")

    return OrderResponse(**order_data)


@app.get("/orders", response_model=List[OrderResponse])
def list_orders(current_user: TokenData = Depends(verify_token)):
    """List all orders for current user"""
    user_orders = [
        OrderResponse(**order_data)
        for order_data in orders_db.values()
        if order_data["username"] == current_user.username
    ]
    return user_orders


@app.put("/orders/{order_id}/status")
def update_order_status(
    order_id: str, status: str, current_user: TokenData = Depends(verify_token)
):
    """Update order status"""
    if order_id not in orders_db:
        raise HTTPException(status_code=404, detail="Order not found")

    order_data = orders_db[order_id]

    # Check if user owns this order
    if order_data["username"] != current_user.username:
        raise HTTPException(
            status_code=403, detail="Not authorized to update this order"
        )

    # Validate status
    valid_statuses = ["pending", "processing", "shipped", "delivered", "cancelled"]
    if status not in valid_statuses:
        raise HTTPException(
            status_code=400, detail=f"Invalid status. Must be one of: {valid_statuses}"
        )

    # Update status
    order_data["status"] = status
    orders_db[order_id] = order_data

    # Publish order status updated event
    publish_order_event("order_status_updated", order_data)

    return {"message": f"Order status updated to {status}", "order_id": order_id}


@app.delete("/orders/{order_id}")
def cancel_order(order_id: str, current_user: TokenData = Depends(verify_token)):
    """Cancel an order"""
    if order_id not in orders_db:
        raise HTTPException(status_code=404, detail="Order not found")

    order_data = orders_db[order_id]

    # Check if user owns this order
    if order_data["username"] != current_user.username:
        raise HTTPException(
            status_code=403, detail="Not authorized to cancel this order"
        )

    # Check if order can be cancelled
    if order_data["status"] in ["shipped", "delivered"]:
        raise HTTPException(
            status_code=400, detail="Cannot cancel shipped or delivered orders"
        )

    # Update status to cancelled
    order_data["status"] = "cancelled"
    orders_db[order_id] = order_data

    # Publish order cancelled event
    publish_order_event("order_cancelled", order_data)

    return {"message": "Order cancelled successfully", "order_id": order_id}


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "order-service"}


@app.get("/stats")
def get_stats():
    """Get order statistics"""
    total_orders = len(orders_db)
    status_counts = {}

    for order in orders_db.values():
        status = order["status"]
        status_counts[status] = status_counts.get(status, 0) + 1

    return {"total_orders": total_orders, "status_distribution": status_counts}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
