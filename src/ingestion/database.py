"""SQLAlchemy ORM model and helpers for image metadata storage."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import (
    Boolean,
    Column,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .metadata import ImageMetadata


class Base(DeclarativeBase):
    pass


class ImageRecord(Base):
    __tablename__ = "image_metadata"

    image_id = Column(String, primary_key=True)
    source_filename = Column(String, nullable=False)
    source_path = Column(Text, nullable=False)
    file_size_kb = Column(Float)
    width = Column(Integer)
    height = Column(Integer)
    image_mode = Column(String)
    image_format = Column(String)
    laplacian_variance = Column(Float)
    mean_brightness = Column(Float)
    brightness_std = Column(Float)
    quality_flag = Column(String)
    is_processable = Column(Boolean)
    ingested_at = Column(String)

    capture_date = Column(String, nullable=True)
    well_name = Column(String, nullable=True)
    completion_zone = Column(String, nullable=True)
    screen_type = Column(String, nullable=True)
    screen_position = Column(String, nullable=True)
    scale_reference_present = Column(Boolean, nullable=True)
    notes = Column(Text, nullable=True)


def init_db(db_path: str | Path) -> sessionmaker:
    """Create engine, run migrations, return a Session factory."""
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def upsert_image_metadata(session: Session, meta: ImageMetadata) -> ImageRecord:
    """Insert or replace an ImageRecord from an ImageMetadata dataclass."""
    record = session.get(ImageRecord, meta.image_id)
    if record is None:
        record = ImageRecord()
        session.add(record)

    for attr, val in vars(meta).items():
        setattr(record, attr, val)

    session.flush()
    return record


def get_all_images(session: Session) -> list[ImageRecord]:
    return session.query(ImageRecord).order_by(ImageRecord.source_filename).all()
