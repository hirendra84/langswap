import io
from abc import ABC
from syslog import LOG_DEBUG
from dotenv import load_dotenv

import boto3
import requests
import os.path
import urllib.request
import requests
from langswap.pipeline_models.models import RemoteFile
from typing import List
import os
from pathlib import Path

load_dotenv()
BUCKET = os.getenv('BUCKET', 'debug-bucket-langswap-bucket')



def download_s3_directory(s3_client, bucket_name, s3_prefix, local_dir):

    
    os.makedirs(local_dir, exist_ok=True)
    
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix)
    
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                relative_path = os.path.relpath(key, s3_prefix)
                local_path = os.path.join(local_dir, relative_path)
                
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                
                if not key.endswith('/'):
                    s3_client.download_file(bucket_name, key, local_path)



class FileRepository(ABC):

    def __init__(self, public_id, base_directory: str, s3_client: boto3.client):
        ...

    def materialize_file(self, file: RemoteFile) -> RemoteFile:
        """ Saves file from file.s3_url to file.file_path """
        ...

    def get_file(self, file_name: str) -> RemoteFile:
        """ Creates empty RemoteFile object with file.file_path = directory / file_name
         and file.name == file_name and """
        ...

    def save_file(self, file: RemoteFile, force: bool = False):
        """ Uploads file to s3 using _s3_client.
        :returns: RemoteFile with file.s3_url"""
        ...

    def save_file_from_stream(self, file: RemoteFile, stream: io.BytesIO):
        """ Saves :param: stream to file.file_path """
        ...

    def subdir(self, dir_name: str) -> str:
        """ Creates folder self.directory / dir_name.
        :returns: new folder path"""
        ...
    
    def save_dir(self, dir_name: str) -> List[RemoteFile]:
        """ Creates folder self.directory / dir_name.
        :returns: new folder path"""
        ...

    @property
    def directory(self) -> str:
        ...


class RemoteFileRepository(FileRepository):
    _directory: str
    _s3_client: boto3.client
    _download_chunk_size = 8192

    _cached_files: dict[str, RemoteFile]

    def __init__(self, public_id, base_directory: str, s3_client: boto3.client):
        super().__init__(public_id, base_directory, s3_client)
        self._directory = os.path.join(base_directory, public_id)
        self._s3_client = s3_client
        self._cached_files = {}
        os.makedirs(self._directory, exist_ok=True)

    def materialize_file(self, file: RemoteFile) -> RemoteFile:
        if file.name in self._cached_files:
            return self._cached_files[file.name]

        file_path = os.path.join(self._directory, file.name)
        with requests.get(file.s3_url, stream=True) as response:
            response.raise_for_status()
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        remote_file = RemoteFile(
            name=file.name,
            file_path=file_path,
            s3_url=file.s3_url,
        )

        self._cached_files[file.name] = remote_file

        return remote_file

    def get_file(self, file_name: str):
        cached = self._cached_files.get(file_name)
        if cached is not None:
            return cached
        remote_file = RemoteFile(
            name=file_name,
            file_path=os.path.join(self._directory, file_name)
        )
        self._cached_files[file_name] = remote_file
        return remote_file

    def save_file(self, file: RemoteFile, force: bool = False):
        print(f"bucket: {BUCKET}, file: {file.file_path}")
        if file.s3_url and not force:
            return file

        with open(file.file_path, 'rb') as f:
            self._s3_client.upload_fileobj(io.BytesIO(f.read()), BUCKET, file.file_path)
            s3_url = self._s3_client.generate_presigned_url(
                ClientMethod='get_object',
                Params={
                    'Bucket': BUCKET,
                    'Key': file.file_path
                },
                ExpiresIn=60 * 60 * 48  # 48 hours
            )
        file.s3_url = s3_url
        return file

    def save_file_from_stream(self, file: RemoteFile, stream: io.BytesIO):
        print(f"BUCKET: {BUCKET}")
        self._s3_client.upload_fileobj(stream, BUCKET, file.name)
        s3_url = self._s3_client.generate_presigned_url(
            ClientMethod='get_object',
            Params={
                'Bucket': BUCKET,
                'Key': file.name
            },
            ExpiresIn=60 * 60 * 24  # 24 hours
        )
        return RemoteFile(
            name=file.name,
            file_path=os.path.join(self._directory, file.name),
            s3_url=s3_url,
        )

    @property
    def directory(self) -> str:
        return self._directory

    def subdir(self, dir_name: str) -> str:
        new_dir = os.path.join(self._directory, dir_name)
        os.makedirs(new_dir, exist_ok=True)
        return new_dir

    def save_dir(self, dir_name):
        # print(dir_name)
        #list_files = list(map(lambda x: str(x).replace(dir_name, ""), Path(dir_name).rglob("*.*")))
        list_files = list(map(str, Path(dir_name).rglob("*.*")))
        remove_file_collection = []
        for file_name in list_files:
            # print(file_name, self._directory)
            local_file = RemoteFile(
                name=file_name,
                file_path=file_name, #os.path.join(self._directory, file_name)
            )
            remote_file = self.save_file(local_file)
            remove_file_collection.append(remote_file)
        return list_files

class LocalOnlyFileRepository(FileRepository):
    """File repository that works entirely locally without S3."""
    _directory: str
    _cached_files: dict[str, RemoteFile]

    def __init__(self, public_id: str, base_directory: str):
        self._directory = os.path.join(base_directory, public_id)
        self._cached_files = {}
        os.makedirs(self._directory, exist_ok=True)

    def materialize_file(self, file: RemoteFile) -> RemoteFile:
        if file.name in self._cached_files:
            return self._cached_files[file.name]

        file_path = os.path.join(self._directory, file.name)
        # If file_path is already set and exists, use it
        if file.file_path and os.path.exists(file.file_path):
            import shutil
            if file.file_path != file_path:
                shutil.copy2(file.file_path, file_path)
        elif file.s3_url:
            # Download from URL
            with requests.get(file.s3_url, stream=True) as response:
                response.raise_for_status()
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

        remote_file = RemoteFile(
            name=file.name,
            file_path=file_path,
            s3_url=file.s3_url or f"file://{file_path}",
        )
        self._cached_files[file.name] = remote_file
        return remote_file

    def get_file(self, file_name: str):
        cached = self._cached_files.get(file_name)
        if cached is not None:
            return cached
        remote_file = RemoteFile(
            name=file_name,
            file_path=os.path.join(self._directory, file_name)
        )
        self._cached_files[file_name] = remote_file
        return remote_file

    def save_file(self, file: RemoteFile, force: bool = False):
        # Local-only: just return the file as-is
        if not file.s3_url:
            file.s3_url = f"file://{file.file_path}"
        return file

    def save_file_from_stream(self, file: RemoteFile, stream: io.BytesIO):
        file_path = os.path.join(self._directory, file.name)
        with open(file_path, 'wb') as f:
            while True:
                chunk = stream.read(8192)
                if not chunk:
                    break
                f.write(chunk)
        return RemoteFile(
            name=file.name,
            file_path=file_path,
            s3_url=f"file://{file_path}",
        )

    @property
    def directory(self) -> str:
        return self._directory

    def subdir(self, dir_name: str) -> str:
        new_dir = os.path.join(self._directory, dir_name)
        os.makedirs(new_dir, exist_ok=True)
        return new_dir

    def save_dir(self, dir_name):
        list_files = list(map(str, Path(dir_name).rglob("*.*")))
        result = []
        for file_name in list_files:
            local_file = RemoteFile(
                name=file_name,
                file_path=file_name,
                s3_url=f"file://{file_name}",
            )
            result.append(local_file)
        return list_files


class LocalFileRepository(FileRepository):
    _directory: str
    _download_chunk_size = 8192
    _s3_client: boto3.client

    _cached_files: dict[str, RemoteFile]

    def __init__(self, public_id, base_directory: str, s3_client: boto3.client):
        super().__init__(public_id, base_directory, s3_client)

        self._directory = os.path.join(base_directory, public_id)
        self._s3_client = s3_client
        self._cached_files = {}
        os.makedirs(self._directory, exist_ok=True)

    def materialize_file(self, file: RemoteFile) -> RemoteFile:
        if file.name in self._cached_files:
            return self._cached_files[file.name]

        file_path = os.path.join(self._directory, file.name)
        print(file_path)
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'} 
        if not os.path.exists(file_path):
            last_error = None
            for i in range(5):
                try:
                    with urllib.request.urlopen(file.s3_url) as response:
                        status_code = response.getcode()
                        # Print the status code
                        print('Status Code:', status_code) 
                        with open(file_path, 'wb') as f:
                            while True:
                                chunk = response.read(1024)
                                if chunk:
                                    f.write(chunk)
                                else:
                                    break
                    break
                except Exception as e:
                    print(e)
                    last_error = e
                    import time
                    time.sleep(5)

            if last_error is not None:
                raise last_error
        remote_file = RemoteFile(
            name=file.name,
            file_path=file_path,
            s3_url=file.s3_url,
        )

        self._cached_files[file.name] = remote_file

        return remote_file

    def get_file(self, file_name: str):
        cached = self._cached_files.get(file_name)
        if cached is not None:
            return cached
        remote_file = RemoteFile(
            name=file_name,
            file_path=os.path.join(self._directory, file_name)
        )
        self._cached_files[file_name] = remote_file
        return remote_file

    def save_file(self, file: RemoteFile, force: bool = False):
        if not force:
            return file

        with open(file.file_path, 'rb') as f:
            self._s3_client.upload_fileobj(io.BytesIO(f.read()), BUCKET, file.file_path)
            s3_url = self._s3_client.generate_presigned_url(
                ClientMethod='get_object',
                Params={
                    'Bucket': BUCKET,
                    'Key': file.file_path
                },
                ExpiresIn=60 * 60 * 14  # 14 days
            )
        file.s3_url = s3_url
        return file

    def save_file_from_stream(self, file: RemoteFile, stream: io.BytesIO):
        with open(file.file_path, 'wb') as f:
            while True:
                chunk = stream.read(self._download_chunk_size)
                if not chunk:
                    break
                f.write(chunk)
        return RemoteFile(
            name=file.name,
            file_path=os.path.join(self._directory, file.name),
            s3_url=file.s3_url or 'fake_url',
        )

    @property
    def directory(self) -> str:
        return self._directory

    def subdir(self, dir_name: str) -> str:

        new_dir = os.path.join(self._directory, dir_name)
        os.makedirs(new_dir, exist_ok=True)
        return new_dir

    def save_dir(self, dir_name):
        #list_files = list(map(lambda x: str(x).replace(dir_name, ""), Path(dir_name).rglob("*.*")))
        #list_files = list(map(str, Path(dir_name).rglob("*.*")))
        list_files = list(map(str, Path(dir_name).rglob("*.*")))
        remove_file_collection = []
        for file_name in list_files:
            #print(file_name, self._directory)
            local_file = RemoteFile(
                name=file_name,
                file_path=file_name, #os.path.join(self._directory, file_name)
            )
            remote_file = self.save_file(local_file)
            remove_file_collection.append(remote_file)
        return list_files