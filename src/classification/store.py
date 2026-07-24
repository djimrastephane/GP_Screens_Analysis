"""SQLAlchemy ORM for classification results."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import Boolean, Column, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.ingestion.database import Base


class ClassificationRunRecord(Base):
    __tablename__ = "classification_runs"

    run_id = Column(String, primary_key=True)   # f"{image_id}__{classifier_name}"
    image_id = Column(String, nullable=False, index=True)
    source_filename = Column(String)
    classifier_name = Column(String)
    n_classifications = Column(Integer)
    n_requires_review = Column(Integer)
    dominant_failure_type = Column(String)
    overall_severity = Column(String)
    composite_defect_area_pct = Column(Float)
    classifications_json = Column(Text)         # list of ClassificationResult.to_dict()
    run_timestamp = Column(String)


def ensure_table(db_path: str | Path) -> sessionmaker:
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def upsert_classification_run(
    session: Session, result: "ImageClassificationResult"
) -> ClassificationRunRecord:
    from src.classification.base import ImageClassificationResult  # type hint

    run_id = f"{result.image_id}__{result.classifier_name}"
    record = session.get(ClassificationRunRecord, run_id)
    if record is None:
        record = ClassificationRunRecord(run_id=run_id)
        session.add(record)

    record.image_id = result.image_id
    record.source_filename = result.source_filename
    record.classifier_name = result.classifier_name
    record.n_classifications = result.count
    record.n_requires_review = result.review_count
    record.dominant_failure_type = result.dominant_failure_type
    record.overall_severity = result.overall_severity
    record.composite_defect_area_pct = round(result.composite_defect_area_pct, 4)
    record.classifications_json = json.dumps([c.to_dict() for c in result.classifications])
    record.run_timestamp = result.run_timestamp

    session.flush()
    return record


def get_classification_runs(
    session: Session, image_id: str | None = None
) -> list[ClassificationRunRecord]:
    q = session.query(ClassificationRunRecord)
    if image_id:
        q = q.filter(ClassificationRunRecord.image_id == image_id)
    return q.order_by(ClassificationRunRecord.source_filename).all()


class ReviewStatus(Base):
    """Human review decision for one detection, independent of classifier re-runs."""
    __tablename__ = "review_status"

    id = Column(String, primary_key=True)   # f"{image_id}__{detection_index}"
    image_id = Column(String, nullable=False, index=True)
    detection_index = Column(Integer, nullable=False)
    reviewed = Column(Boolean, default=False, nullable=False)
    note = Column(Text, nullable=True)
    reviewed_at = Column(String, nullable=True)   # ISO-8601 UTC, set when reviewed=True


def set_reviewed(
    session: Session,
    image_id: str,
    detection_index: int,
    reviewed: bool = True,
    note: str | None = None,
) -> ReviewStatus:
    rid = f"{image_id}__{detection_index}"
    record = session.get(ReviewStatus, rid)
    if record is None:
        record = ReviewStatus(id=rid, image_id=image_id, detection_index=detection_index)
        session.add(record)
    record.reviewed = reviewed
    if note is not None:
        record.note = note
    record.reviewed_at = datetime.now(timezone.utc).isoformat() if reviewed else None
    session.flush()
    return record


def get_review_status(
    session: Session, image_id: str | None = None
) -> list[ReviewStatus]:
    q = session.query(ReviewStatus)
    if image_id:
        q = q.filter(ReviewStatus.image_id == image_id)
    return q.all()
