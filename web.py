from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

@app.post("/file")
async def upload_file(file: bytes = None):
    
    return { "file" : file }
