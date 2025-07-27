#!/usr/bin/env python3

from fastapi import FastAPI

app = FastAPI()

@app.get("/test")
def test_function():
    return {"message": "test"}

@app.post("/test2")
def test_function2():
    return {"message": "test2"} 