"""Track annotation output paths in the database."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import Column, String, Text, create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.ingestion.database import Base


class AnnotationRecord(Base):
    __tablename__ = "annotation_runs"

    image_id = Column(String, primary_key=True)
    source_filename = Column(String)
    annotated_path = Column(Text, nullable=True)
    panel_path = Column(Text, nullable=True)
    run_timestamp = Column(String)


def ensure_table(db_path: str | Path) -> sessionmaker:
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def upsert_annotation(
    session: Session,
    image_id: str,
    source_filename: str,
    annotated_path: str | None,
    panel_path: str | None,
    run_timestamp: str,
) -> AnnotationRecord:
    record = session.get(AnnotationRecord, image_id)
    if record is None:
        record = AnnotationRecord(image_id=image_id)
        session.add(record)
    record.source_filename = source_filename
    record.annotated_path = annotated_path
    record.panel_path = panel_path
    record.run_timestamp = run_timestamp
    session.flush()
    return record


def get_all_annotations(session: Session) -> list[AnnotationRecord]:
    return (
        session.query(AnnotationRecord)
        .order_by(AnnotationRecord.source_filename)
        .all()
    )
