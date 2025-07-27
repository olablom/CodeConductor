from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()


class User(BaseModel):
    name: str
    email: str


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/api/users")
async def get_users():
    return {"users": []}


@app.post("/api/users")
async def create_user(user: User):
    return {"id": 1, "name": user.name}


@app.get("/api/users/{user_id}")
async def get_user(user_id: int):
    return {"id": user_id}
