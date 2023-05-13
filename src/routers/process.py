import logging

from fastapi import Depends, HTTPException, APIRouter, UploadFile, File
from sqlalchemy.orm import Session
from starlette import status

from src import schemas, crud
from src.database import get_db
from src.models import ProcessedObject

router = APIRouter()


@router.post("/video", response_model=schemas.RespProcessedObject)
async def process_video(
        data: UploadFile = File(...),
        db: Session = Depends(get_db)
):
    try:
        obj: ProcessedObject = await crud.process_video(db, data)
        print(obj.prepared_link)
    except Exception as e:
        logging.exception(e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Oops there is a problem! We are already trying to fix it',
        )

    return obj


@router.get("/video/{object_id:str}/", response_model=schemas.RespProcessedObject)
async def get_object(
        object_id,
        db: Session = Depends(get_db)
):
    try:
        obj: ProcessedObject | None = await crud.get_object(db, object_id)
    except Exception as e:
        logging.exception(e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Oops there is a problem! We are already trying to fix it',
        )
    if obj is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Video not found',
        )
    return obj


@router.put("/video/{object_id:str}")
async def update_object(
        object_id,
        data: schemas.UpdProcessedObject,
        db: Session = Depends(get_db)):
    # TODO: Вообще-то тут нужен какой-то ключ, а то как-то не секурно
    try:
        await crud.update_object(db, object_id, data)
    except Exception as e:
        logging.exception(e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Ny wse po pizde poshlo, update otvalilsa',
        )


@router.post("/video/by-link")
async def upload_video_by_link(
        data: schemas.CreateProcessedObjectByLink,
        db: Session = Depends(get_db)):
    try:
        obj: ProcessedObject = await crud.process_video_by_link(db, data)
    except Exception as e:
        logging.exception(e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Oops there is a problem! We are already trying to fix it',
        )
    return obj
