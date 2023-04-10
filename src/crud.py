from io import BytesIO
from logging import getLogger

from celery import chain
from fastapi import UploadFile
from pytube import YouTube
from sqlalchemy.orm import Session

from src import models
from src import schemas
from src.utils.common import upload_file_to_s3, generate_public_id
from src.utils.ml_processing import tasks as ml_tasks
from src.utils.youtube import _get_suitable_yt_stream, _validate_yt_link

logger = getLogger()


async def process_video(db: Session, file: UploadFile) -> models.ProcessedObject:
    # TODO: content_type (MIME type / media type) (e.g. image/jpeg)
    # 10MB = 10 * 1024 (bytes) * 1024 (kb) * 1024 (mb)
    max_size_bytes = 10 * (1024 ** 3)
    content = await file.read(max_size_bytes)
    content = BytesIO(content)

    public_id = generate_public_id()
    s3_url = upload_file_to_s3(content, public_id)

    obj = models.ProcessedObject(
        source_link=s3_url,
        original_name=file.filename,
        public_id=public_id,
    )
    db.add(obj)
    db.commit()
    db.refresh(obj)

    ml_pipeline = chain(ml_tasks.speech_to_text.s(obj.public_id, obj.source_link),
                        ml_tasks.speaker_encoder.s(obj.public_id, obj.source_link),
                        ml_tasks.text_to_speech.s(obj.public_id, obj.source_link))
    ml_pipeline()

    return obj


async def process_video_by_link(db: Session, data: schemas.CreateProcessedObjectByLink) -> models.ProcessedObject:
    link = await _validate_yt_link(data.link)
    logger.info(f"Validated link: {link}")

    yt = YouTube(link)
    stream = await _get_suitable_yt_stream(yt)

    content = BytesIO()
    stream.stream_to_buffer(content)

    public_id = generate_public_id()
    s3_url = upload_file_to_s3(content, public_id)

    obj = models.ProcessedObject(
        source_link=s3_url,
        original_name=yt.title,
        public_id=public_id,
    )
    db.add(obj)
    db.commit()
    db.refresh(obj)

    await _run_ml_pipeline(obj)

    return obj


async def get_object(db: Session, object_id: str) -> models.ProcessedObject | None:
    a = db.query(models.ProcessedObject)\
        .filter(models.ProcessedObject.public_id == object_id)\
        .one_or_none()
    return a


async def update_object(db: Session, object_id: str, data: schemas.UpdProcessedObject):
    a = db.query(models.ProcessedObject)\
        .filter(models.ProcessedObject.public_id == object_id)\
        .update({k: v for k, v in data.dict().items()
                 if k in data.__fields_set__})
    db.commit()
    return a


async def create_user(db: Session, data: schemas.CreateUser) -> models.User:
    public_id = generate_public_id()

    obj = models.User(
        email=data.email,
        public_id=public_id,
    )
    db.add(obj)
    db.commit()
    db.refresh(obj)

    return obj


async def _run_ml_pipeline(obj: models.ProcessedObject):
    ml_pipeline = chain(ml_tasks.speech_to_text.s(obj.public_id, obj.source_link),
                        ml_tasks.speaker_encoder.s(obj.public_id, obj.source_link),
                        ml_tasks.text_to_speech.s(obj.public_id, obj.source_link))
    return ml_pipeline()


