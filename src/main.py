from fastapi import FastAPI

from src.routers import process
from src.routers import user
from src.settings import DEBUG

app = FastAPI(debug=DEBUG)

app.include_router(process.router)
app.include_router(user.router)
