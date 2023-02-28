import os

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# SQLALCHEMY_DATABASE_URL = "sqlite:///./sql_app.db"
from api.settings import DEBUG
from api.utils import custom_json_serializer_sa_pydantic_models

PG_HOST = os.environ.get('PG_HOST', '127.0.0.1')
PG_PORT = os.environ.get('PG_PORT', 5432)
PG_USER = os.environ.get('PG_USER', 'joint_user')
PG_PASS = os.environ.get('PG_PASS', 'pass123')
PG_DBNAME = os.environ.get('PG_DBNAME', 'joint_dev')

SQLALCHEMY_DATABASE_URL = f"postgresql://{PG_USER}:{PG_PASS}@{PG_HOST}/{PG_DBNAME}"


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
