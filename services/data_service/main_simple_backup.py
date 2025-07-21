#!/usr/bin/env python3
"""
Data Service - Abstraktion över databasen
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging
import uvicorn
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CodeConductor Data Service",
    description="Service för databasabstraktion och CRUD-operationer",
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
class DataRequest(BaseModel):
    collection: str
    data: Dict[str, Any]
    operation: str = "create"  # create, read, update, delete


class DataResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    message: str
    timestamp: datetime


# In-memory storage (replace with database in production)
data_store = {"discussions": {}, "workflows": {}, "agents": {}, "approvals": {}}


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Data Service",
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now(),
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "collections": list(data_store.keys()),
        "total_records": sum(len(collection) for collection in data_store.values()),
        "uptime": "TODO: implement uptime tracking",
    }


@app.post("/data", response_model=DataResponse)
async def handle_data_operation(request: DataRequest):
    """Hantera CRUD-operationer"""
    try:
        logger.info(f"Data operation: {request.operation} on {request.collection}")

        if request.collection not in data_store:
            data_store[request.collection] = {}

        collection = data_store[request.collection]

        if request.operation == "create":
            # Generate ID
            record_id = str(len(collection) + 1)
            collection[record_id] = {
                "id": record_id,
                **request.data,
                "created_at": datetime.now().isoformat(),
            }

            return DataResponse(
                success=True,
                data=collection[record_id],
                message=f"Record created with ID: {record_id}",
                timestamp=datetime.now(),
            )

        elif request.operation == "read":
            if "id" in request.data:
                record_id = str(request.data["id"])
                if record_id in collection:
                    return DataResponse(
                        success=True,
                        data=collection[record_id],
                        message="Record retrieved successfully",
                        timestamp=datetime.now(),
                    )
                else:
                    raise HTTPException(status_code=404, detail="Record not found")
            else:
                # Return all records in collection
                return DataResponse(
                    success=True,
                    data={"records": list(collection.values())},
                    message=f"Retrieved {len(collection)} records",
                    timestamp=datetime.now(),
                )

        elif request.operation == "update":
            if "id" not in request.data:
                raise HTTPException(status_code=400, detail="ID required for update")

            record_id = str(request.data["id"])
            if record_id not in collection:
                raise HTTPException(status_code=404, detail="Record not found")

            # Update record
            update_data = {k: v for k, v in request.data.items() if k != "id"}
            collection[record_id].update(update_data)
            collection[record_id]["updated_at"] = datetime.now().isoformat()

            return DataResponse(
                success=True,
                data=collection[record_id],
                message="Record updated successfully",
                timestamp=datetime.now(),
            )

        elif request.operation == "delete":
            if "id" not in request.data:
                raise HTTPException(status_code=400, detail="ID required for delete")

            record_id = str(request.data["id"])
            if record_id not in collection:
                raise HTTPException(status_code=404, detail="Record not found")

            deleted_record = collection.pop(record_id)

            return DataResponse(
                success=True,
                data=deleted_record,
                message="Record deleted successfully",
                timestamp=datetime.now(),
            )

        else:
            raise HTTPException(
                status_code=400, detail=f"Unknown operation: {request.operation}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in data operation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collections")
async def list_collections():
    """Lista alla tillgängliga collections"""
    return {
        "collections": [
            {"name": name, "record_count": len(collection)}
            for name, collection in data_store.items()
        ]
    }


@app.get("/collections/{collection_name}")
async def get_collection_info(collection_name: str):
    """Hämta information om en collection"""
    if collection_name not in data_store:
        raise HTTPException(status_code=404, detail="Collection not found")

    collection = data_store[collection_name]
    return {
        "name": collection_name,
        "record_count": len(collection),
        "records": list(collection.values()),
    }


@app.delete("/collections/{collection_name}")
async def clear_collection(collection_name: str):
    """Rensa en collection"""
    if collection_name not in data_store:
        raise HTTPException(status_code=404, detail="Collection not found")

    record_count = len(data_store[collection_name])
    data_store[collection_name].clear()

    return {
        "message": f"Collection '{collection_name}' cleared",
        "deleted_records": record_count,
    }


if __name__ == "__main__":
    logger.info("Starting Data Service...")
    uvicorn.run("main:app", host="0.0.0.0", port=8003, reload=True, log_level="info")
