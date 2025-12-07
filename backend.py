from fastapi import FastAPI

api = FastAPI()

@api.get("/api/hello")
def hello():
    return {"message": "Hello from FastAPI"}
