from typing import Optional

from pydantic import BaseModel

from api.enums import ProcessStatus


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

