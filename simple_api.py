#!/usr/bin/env python3
"""
Simple API for testing purposes.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

# In-memory storage
items = []

class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None

@app.get("/")
async def root():
    return {"message": "Simple API is running"}

@app.get("/items")
async def get_items():
    return {"items": items}

@app.post("/items")
async def create_item(item: Item):
    item_dict = item.dict()
    item_dict["id"] = len(items) + 1
    items.append(item_dict)
    return item_dict

@app.get("/items/{item_id}")
async def get_item(item_id: int):
    if item_id <= 0 or item_id > len(items):
        raise HTTPException(status_code=404, detail="Item not found")
    return items[item_id - 1]

@app.get("/health")
async def health_check():
    from datetime import datetime
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 