import os

# to get a string like this run:
# openssl rand -hex 32
SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
ALGORITHM = "HS256"
DEBUG = 'aypa' in os.environ.get('USER')

PG_HOST = os.environ.get('PG_HOST', '127.0.0.1')
PG_PORT = os.environ.get('PG_PORT', 5432)
PG_USER = os.environ.get('PG_USER', 'joint_user')
PG_PASS = os.environ.get('PG_PASS', 'pass123')
PG_DBNAME = os.environ.get('PG_DBNAME', 'joint_dev')

SQLALCHEMY_DATABASE_URL = f"postgresql://{PG_USER}:{PG_PASS}@{PG_HOST}/{PG_DBNAME}"
