from fastapi import Depends, HTTPException, APIRouter
from sqlalchemy.orm import Session
from starlette import status

from src import schemas, crud
from src.database import get_db
from src.models import User

router = APIRouter()


@router.post("/user", response_model=schemas.RespUser)
async def create_user(
        user: schemas.CreateUser,
        db: Session = Depends(get_db),
):
    try:
        obj: User = await crud.create_user(db, user)
    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Oops there is a problem! We are already trying to fix it',
        )

    return obj
