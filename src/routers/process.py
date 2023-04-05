
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
        print(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Oops there is a problem! We are already trying to fix it',
        )

    return obj


@router.get("/video/{object_id:str}", response_model=schemas.RespProcessedObject)
async def get_object(
        object_id,
        db: Session = Depends(get_db)
):
    try:
        obj: ProcessedObject = await crud.get_object(db, object_id)
        print(obj.prepared_link)
        print(schemas.RespProcessedObject.from_orm(obj))
    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Oops there is a problem! We are already trying to fix it',
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
        print(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Ny wse po pizde poshlo, update otvalilsa',
        )


