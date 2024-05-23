from io import BytesIO
from logging import getLogger

from celery import chain
from fastapi import UploadFile
from sqlalchemy.orm import Session

from src import models
from src import schemas
from src.file_repository import FileRepository, RemoteFileRepository, LocalFileRepository
from src.settings import DEBUG, LOCAL_DEBUG, BASE_WORKING_DIR
from src.utils.common import generate_public_id
from src.utils.ml_processing import tasks as ml_tasks
from src.utils.s3_client import get_s3_client
from src.utils.youtube import get_yt_stream_and_name

logger = getLogger()

if LOCAL_DEBUG:
    file_repo_klass = LocalFileRepository
else:
    file_repo_klass = RemoteFileRepository


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

    obj = models.ProcessedObject(
        source_link=uploaded_video.s3_url,
        original_name=file.filename,
        public_id=public_id,
    )

    if not DEBUG:
        db.add(obj)
        db.commit()
        db.refresh(obj)

        ml_pipeline = chain(ml_tasks.speech_to_text.s(obj.public_id, uploaded_video),
                            ml_tasks.speaker_encoder.s(obj.public_id, obj.source_link),
                            ml_tasks.text_to_speech.s(obj.public_id, obj.source_link))
    if DEBUG:
        ml_tasks.speech_to_text(public_id, uploaded_video, file_repo)
        # ml_pipeline.apply()
    else:
        ml_pipeline()

    return obj


async def process_video_by_link(db: Session, data: schemas.CreateProcessedObjectByLink) -> models.ProcessedObject:
    logger.error(f"Video link: {data.link}")
    video_data, video_title = get_yt_stream_and_name(data.link)

    public_id = generate_public_id()
    file_repo = FileRepository(
        '',
        get_s3_client()
    )

    video_file = file_repo.get_file('uploaded_video')

    video_file = file_repo.save_file_from_stream(video_file, video_data)

    obj = models.ProcessedObject(
        source_link=video_file.s3_url,
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


