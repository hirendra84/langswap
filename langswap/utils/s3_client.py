import os

import boto3 as boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()
def get_s3_client() -> boto3.client:
    return boto3.client(
        's3',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
        config=Config(signature_version='s3v4'),
        region_name='eu-central-1'
    )
