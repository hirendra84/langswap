from typing import Optional

from pydantic import BaseModel, EmailStr, HttpUrl

from src.enums import ProcessStatus


class RespProcessedObject(BaseModel):
    source_link: str
    original_name: str
    status: ProcessStatus
    prepared_link: str
    public_id: str

    class Config:
        orm_mode = True


class UpdProcessedObject(BaseModel):
    status: Optional[ProcessStatus]
    prepared_link: Optional[str]
    public_id: str


class CreateUser(BaseModel):
    email: EmailStr


class RespUser(BaseModel):
    email: EmailStr
    public_id: str

    class Config:
        orm_mode = True


class CreateProcessedObjectByLink(BaseModel):
    link: HttpUrl

    class Config:
        orm_mode = True
