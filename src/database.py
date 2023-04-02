import os

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# SQLALCHEMY_DATABASE_URL = "sqlite:///./sql_app.db"
from src.settings import DEBUG
from src.utils import custom_json_serializer_sa_pydantic_models

POSTGRES_HOST = os.environ.get('POSTGRES_HOST', '127.0.0.1')
PG_PORT = os.environ.get('PG_PORT', 5432)
POSTGRES_USER = os.environ.get('POSTGRES_USER', 'joint_user')
POSTGRES_PASSWORD = os.environ.get('POSTGRES_PASSWORD', 'pass123')
POSTGRES_DB = os.environ.get('POSTGRES_DB', 'joint_dev')

SQLALCHEMY_DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}/{POSTGRES_DB}"


engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    json_serializer=custom_json_serializer_sa_pydantic_models,
    echo=True if DEBUG else False
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, expire_on_commit=False)

Base = declarative_base()


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
