from sqlalchemy import Column, Integer, String, Enum
from .database import Base

from .enums import ProcessStatus


class ProcessedObject(Base):
    __tablename__ = "processed_object"

    id = Column(Integer, primary_key=True, index=True)
    source_link = Column(String)
    original_name = Column(String)
    status = Column(Enum(ProcessStatus), default=ProcessStatus.uploaded)
    prepared_link = Column(String, default='')
    public_id = Column(String(36), nullable=False, unique=True, index=True)
