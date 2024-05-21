import json
import os
import uuid
from io import BytesIO

import boto3 as boto3
import pydantic
import requests
from botocore.config import Config

BUCKET = 'ds-dev-video-storage'


def custom_json_serializer_sa_pydantic_models(*args, **kwargs) -> str:
    """
    Encodes json in the same way that pydantic does.
    """
    return json.dumps(*args, default=pydantic.json.pydantic_encoder, **kwargs)


def upload_file_to_s3(file: BytesIO, object_name: str) -> str:

    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.environ.get('aws_access_key_id', '***REDACTED-AWS-KEY-ID***'),
        aws_secret_access_key=os.environ.get('aws_secret_access_key',
                                             '***REDACTED-AWS-SECRET***'),
        config=Config(signature_version='s3v4'),
        region_name='eu-central-1'
    )
    s3_client.upload_fileobj(file, BUCKET, object_name)
    s3_url = s3_client.generate_presigned_url(
        ClientMethod='get_object',
        Params={
            'Bucket': BUCKET,
            'Key': object_name
        },
        ExpiresIn=60 * 60 * 24  # 24 hours
    )

    return s3_url


def download_from_s3(url: str, save_path: str, chunk_size: int = 8192) -> str:
    """
    Download file from url to save_path

    Args:
        url (str): URL to download file from
        save_path (Path): Path to save file to
        chunk_size: size of downloading chunk

    Returns:
        save_path (Path): Path to saved file
    """
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
    return save_path



def generate_public_id() -> str:
    return str(uuid.uuid1())


def is_valid_email(email: str) -> bool:
    pass