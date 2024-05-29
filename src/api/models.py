from sqlalchemy import Column, Integer, String, Enum, DateTime, func
from src.api.database import Base

from src.pipeline_models.enums import ProcessStatus


class ProcessedObject(Base):
    __tablename__ = "processed_object"

    id = Column(Integer, primary_key=True, index=True)
    source_link = Column(String)
    original_name = Column(String)
    status = Column(Enum(ProcessStatus), default=ProcessStatus.uploaded)
    progress = Column(Integer, default=0)
    # translated = Column(ARRAY(String))
    # recognized = Column(ARRAY(String))
    prepared_link = Column(String, default='')
    public_id = Column(String(36), nullable=False, unique=True, index=True)


class User(Base):
    __tablename__ = "user"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    public_id = Column(String(36), nullable=False, unique=True, index=True)
