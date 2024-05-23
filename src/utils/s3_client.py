import os

import boto3 as boto3
from botocore.config import Config


def get_s3_client() -> boto3.client:
    return boto3.client(
        's3',
        aws_access_key_id=os.environ.get('aws_access_key_id', '***REDACTED-AWS-KEY-ID***'),
        aws_secret_access_key=os.environ.get('aws_secret_access_key',
                                             '***REDACTED-AWS-SECRET***'),
        config=Config(signature_version='s3v4'),
        region_name='eu-central-1'
    )
