import uvicorn

from src.database import Base, engine
from src.main import app
from src.settings import DEBUG

if __name__ == "__main__":
    if DEBUG:
        Base.metadata.create_all(engine)
    uvicorn.run(app, host="0.0.0.0", port=8000)
