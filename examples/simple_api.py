#!/usr/bin/env python3
"""
Simple FastAPI App for Smoke Testing
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

app = FastAPI(title="Simple API", version="1.0.0")

# In-memory storage
data_store = {}

class Item(BaseModel):
    name: str
    description: str = None
    price: float = 0.0

@app.get("/")
def read_root():
    """Root endpoint."""
    return {"message": "Simple API is running"}

@app.get("/items")
def get_items():
    """Get all items."""
    return {"items": list(data_store.values())}

@app.get("/items/{item_id}")
def get_item(item_id: int):
    """Get a specific item."""
    if item_id not in data_store:
        raise HTTPException(status_code=404, detail="Item not found")
    return data_store[item_id]

@app.post("/items")
def create_item(item: Item):
    """Create a new item."""
    item_id = len(data_store) + 1
    item_data = item.dict()
    item_data["id"] = item_id
    data_store[item_id] = item_data
    return item_data

@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    """Update an existing item."""
    if item_id not in data_store:
        raise HTTPException(status_code=404, detail="Item not found")
    
    item_data = item.dict()
    item_data["id"] = item_id
    data_store[item_id] = item_data
    return item_data

@app.delete("/items/{item_id}")
def delete_item(item_id: int):
    """Delete an item."""
    if item_id not in data_store:
        raise HTTPException(status_code=404, detail="Item not found")
    
    deleted_item = data_store.pop(item_id)
    return {"message": "Item deleted", "item": deleted_item}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 