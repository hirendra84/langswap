import os

# to get a string like this run:
# openssl rand -hex 32
SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
ALGORITHM = "HS256"
DEBUG = 'aypa' in os.environ.get('USER', '')
DEBUG = DEBUG or os.environ.get('DEBUG')
DEBUG=False
LOCAL_DEBUG = False or os.environ.get('LOCAL_DEBUG')
BACKEND_URL = os.environ.get('BACKEND_URL', 'http://localhost:8000')
BASE_WORKING_DIR = os.environ.get('BASE_WORKING_DIR', '/Users/nikolaypakhtusov/data')

POSTGRES_HOST = os.environ.get('POSTGRES_HOST', '127.0.0.1')
PG_PORT = os.environ.get('PG_PORT', 5432)
POSTGRES_USER = os.environ.get('POSTGRES_USER', 'joint_user')
POSTGRES_PASSWORD = os.environ.get('POSTGRES_PASSWORD', 'pass123')
POSTGRES_DB = os.environ.get('POSTGRES_DB', 'joint_dev')

SQLALCHEMY_DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}/{POSTGRES_DB}"
