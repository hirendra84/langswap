import uuid
from io import BytesIO

from celery import chain
from fastapi import UploadFile
from sqlalchemy.orm import Session

from . import models
from . import schemas
from .utils.common import upload_file_to_s3
from .utils.ml_processing import tasks as ml_tasks


async def process_video(db: Session, file: UploadFile):
    # TODO: content_type (MIME type / media type) (e.g. image/jpeg)
    # 10MB = 10 * 1024 (bytes) * 1024 (kb) * 1024 (mb)
    max_size_bytes = 10 * (1024 ** 3)
    content = await file.read(max_size_bytes)
    content = BytesIO(content)

    public_id = str(uuid.uuid1())
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


async def get_object(db: Session, object_id: str):
    a = db.query(models.ProcessedObject)\
        .filter(models.ProcessedObject.public_id == object_id)\
        .one_or_none()
    return a


async def update_object(db: Session, object_id: str, data: schemas.UpdProcessedObject):
    a = db.query(models.ProcessedObject)\
        .filter(models.ProcessedObject.public_id == object_id)\
        .update(data.dict())
    db.commit()
    return a
