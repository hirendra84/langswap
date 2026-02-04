import json
import uuid

import pydantic
import requests


def custom_json_serializer_sa_pydantic_models(*args, **kwargs) -> str:
    """
    Encodes json in the same way that pydantic does.
    """
    return json.dumps(*args, default=pydantic.json.pydantic_encoder, **kwargs)


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