"""SQLAlchemy ORM for detection results."""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import Column, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.ingestion.database import Base


class DetectionRunRecord(Base):
    """One row per image per detector run."""
    __tablename__ = "detection_runs"

    run_id = Column(String, primary_key=True)   # f"{image_id}__{detector_name}"
    image_id = Column(String, nullable=False, index=True)
    source_filename = Column(String)
    detector_name = Column(String)
    model_version = Column(String, nullable=True)
    detection_count = Column(Integer)
    max_confidence = Column(Float)
    detections_json = Column(Text)              # JSON list of Detection.to_dict()
    image_width = Column(Integer)
    image_height = Column(Integer)
    run_timestamp = Column(String)
    notes = Column(Text, nullable=True)


def ensure_table(db_path: str | Path) -> sessionmaker:
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def upsert_detection_run(session: Session, result) -> DetectionRunRecord:
    """Persist a DetectionResult to the database."""
    run_id = f"{result.image_id}__{result.detector_name}"
    record = session.get(DetectionRunRecord, run_id)
    if record is None:
        record = DetectionRunRecord(run_id=run_id)
        session.add(record)

    record.image_id = result.image_id
    record.source_filename = result.source_filename
    record.detector_name = result.detector_name
    record.model_version = result.model_version
    record.detection_count = result.count
    record.max_confidence = round(result.max_confidence, 4)
    record.detections_json = json.dumps([d.to_dict() for d in result.detections])
    record.image_width = result.image_width
    record.image_height = result.image_height
    record.run_timestamp = result.run_timestamp
    record.notes = result.notes

    session.flush()
    return record


def get_detection_runs(
    session: Session, image_id: str | None = None
) -> list[DetectionRunRecord]:
    q = session.query(DetectionRunRecord)
    if image_id:
        q = q.filter(DetectionRunRecord.image_id == image_id)
    return q.order_by(DetectionRunRecord.source_filename).all()
