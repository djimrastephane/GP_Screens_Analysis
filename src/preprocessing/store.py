"""SQLAlchemy ORM model for preprocessed image records."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import Boolean, Column, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# Re-use the same engine/Base as ingestion so both tables live in one DB file.
# We import the Base from ingestion to share schema migrations.
from src.ingestion.database import Base


class PreprocessedRecord(Base):
    __tablename__ = "preprocessed_images"

    image_id = Column(String, primary_key=True)
    source_filename = Column(String, nullable=False)
    preprocessed_png_path = Column(Text, nullable=True)
    preprocessed_npy_path = Column(Text, nullable=True)
    original_width = Column(Integer)
    original_height = Column(Integer)
    target_width = Column(Integer)
    target_height = Column(Integer)
    scale = Column(Float)
    pad_top = Column(Integer)
    pad_bottom = Column(Integer)
    pad_left = Column(Integer)
    pad_right = Column(Integer)
    enhancements_applied = Column(String)   # comma-separated list or 'none'
    normalisation = Column(String)          # float32 or imagenet
    preprocessed_at = Column(String)


def ensure_table(db_path: str | Path) -> sessionmaker:
    """Create preprocessed_images table if absent, return session factory."""
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def upsert_preprocessed(session: Session, data: dict) -> PreprocessedRecord:
    record = session.get(PreprocessedRecord, data["image_id"])
    if record is None:
        record = PreprocessedRecord()
        session.add(record)
    for k, v in data.items():
        setattr(record, k, v)
    session.flush()
    return record


def get_all_preprocessed(session: Session) -> list[PreprocessedRecord]:
    return (
        session.query(PreprocessedRecord)
        .order_by(PreprocessedRecord.source_filename)
        .all()
    )
