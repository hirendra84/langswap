from io import BytesIO
from logging import getLogger

from celery import chain
from fastapi import UploadFile
from sqlalchemy.orm import Session

from src import models
from src import schemas
from src.utils.common import upload_file_to_s3, generate_public_id
from src.utils.ml_processing import tasks as ml_tasks
from src.utils.youtube import get_yt_stream_and_name

logger = getLogger()


async def process_video(db: Session, file: UploadFile) -> models.ProcessedObject:
    # TODO: content_type (MIME type / media type) (e.g. image/jpeg)
    # 100MB = 100 * 1024 (bytes) * 1024 (kb) * 1024 (mb)
    max_size_bytes = 100 * (1024 ** 3)
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
    logger.error(f"Video link: {data.link}")
    video_data, video_title = get_yt_stream_and_name(data.link)

    public_id = generate_public_id()
    s3_url = upload_file_to_s3(video_data, public_id)

    obj = models.ProcessedObject(
        source_link=s3_url,
        original_name=video_title,
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


