from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.routers import process
from src.routers import user
from src.settings import DEBUG

app = FastAPI(debug=DEBUG)
if DEBUG:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(process.router)
app.include_router(user.router)
