from fastapi import FastAPI

from api.routers import process
from api.settings import DEBUG

app = FastAPI(debug=DEBUG)

app.include_router(process.router)
