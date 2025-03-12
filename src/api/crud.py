from io import BytesIO
from logging import getLogger

from celery import chain
from fastapi import UploadFile
from sqlalchemy.orm import Session

from src.api import models, schemas
from src.pipeline_models.enums import ProcessStatus
from src.file_repository import FileRepository, file_repo_klass
from src.pipeline_models.models import VideoTranslation
from src.settings import DEBUG, BASE_WORKING_DIR
from src.utils.common import generate_public_id
from src.ml import tasks as ml_tasks
from src.utils.s3_client import get_s3_client
from src.utils.youtube import get_yt_stream_and_name

logger = getLogger()


async def process_video(db: Session, file: UploadFile) -> models.ProcessedObject:
    # TODO: content_type (MIME type / media type) (e.g. image/jpeg)
    # 100MB = 100 * 1024 (kb) * 1024 (mb)
    max_size_bytes = 100 * (1024 ** 2)
    content = await file.read(max_size_bytes)
    content = BytesIO(content)

    public_id = generate_public_id()

    file_repo = file_repo_klass(
            public_id,
            base_directory=BASE_WORKING_DIR,
            s3_client=get_s3_client()
        )

    uploaded_video = file_repo.save_file_from_stream(
        file_repo.get_file('uploaded_video'),
        content
    )
    print(uploaded_video.s3_url)

    obj = models.ProcessedObject(
        source_link=uploaded_video.s3_url,
        original_name=file.filename,
        public_id=public_id,
        status=ProcessStatus.uploaded,
    )

    if not DEBUG:
        db.add(obj)
        db.commit()
        db.refresh(obj)

    video_translation = VideoTranslation(
        source_file=uploaded_video,
        public_id=obj.public_id,
    )

    ml_pipeline = chain(ml_tasks.speech_to_text.s(video_translation),
                        ml_tasks.translate.s(),
                        ml_tasks.text_to_speech.s())
    if DEBUG:
        ml_pipeline.apply()
    else:
        ml_pipeline()

    return obj


async def process_video_by_link(db: Session, data: schemas.CreateProcessedObjectByLink) -> models.ProcessedObject:
    logger.error(f"Video link: {data.link}")
    video_data, video_title = get_yt_stream_and_name(data.link)

    public_id = generate_public_id()

    file_repo = file_repo_klass(
        public_id,
        base_directory=BASE_WORKING_DIR,
        s3_client=get_s3_client()
    )

    uploaded_video = file_repo.save_file_from_stream(
        file_repo.get_file('uploaded_video'),
        video_data,
    )
    print(uploaded_video.s3_url)

    obj = models.ProcessedObject(
        source_link=uploaded_video.s3_url,
        original_name=video_title,
        public_id=public_id,
        status=ProcessStatus.uploaded,
    )

    if not DEBUG:
        db.add(obj)
        db.commit()
        db.refresh(obj)

    video_translation = VideoTranslation(
        source_file=uploaded_video,
        public_id=obj.public_id,
    )

    ml_pipeline = chain(ml_tasks.speech_to_text.s(video_translation),
                        ml_tasks.translate.s(),
                        ml_tasks.text_to_speech.s())
    if DEBUG:
        ml_pipeline.apply()
    else:
        ml_pipeline()

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


