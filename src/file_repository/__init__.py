import io
from abc import ABC

import boto3
import attr
import requests
import os.path

BUCKET = 'ds-dev-video-storage'


@attr.s(auto_attribs=True)
class RemoteFile:
    name: str
    file_path: str = attr.ib(default='')
    s3_url: str = attr.ib(default='')


class FileRepository(ABC):

    def __init__(self, public_id, base_directory: str, s3_client: boto3.client):
        ...

    def materialize_file(self, file: RemoteFile) -> RemoteFile:
        ...

    def get_file(self, file_name: str) -> RemoteFile:
        ...

    def save_file(self, file: RemoteFile, force: bool = False):
        ...

    def save_file_from_stream(self, file: RemoteFile, stream: io.BytesIO):
        ...

    def subdir(self, dir_name: str) -> str:
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
                for chunk in response.iter_content(chunk_size=self._download_chunk_size):
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
        if not os.path.exists(file_path):
            with requests.get(file.s3_url, stream=True) as response:
                response.raise_for_status()
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=self._download_chunk_size):
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
                ExpiresIn=60 * 60 * 48  # 48 hours
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
            s3_url=file.s3_url,
        )

    @property
    def directory(self) -> str:
        return self._directory

    def subdir(self, dir_name: str) -> str:

        new_dir = os.path.join(self._directory, dir_name)
        os.makedirs(new_dir, exist_ok=True)
        return new_dir

