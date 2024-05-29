from fastapi import FastAPI

from src.api.routers import process, user
from src.settings import DEBUG

app = FastAPI(debug=DEBUG)

app.include_router(process.router)
app.include_router(user.router)
