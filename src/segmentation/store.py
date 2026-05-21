"""SQLAlchemy ORM for segmentation results."""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import Column, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.ingestion.database import Base


class SegmentationRunRecord(Base):
    __tablename__ = "segmentation_runs"

    run_id = Column(String, primary_key=True)   # f"{image_id}__{segmenter_name}"
    image_id = Column(String, nullable=False, index=True)
    source_filename = Column(String)
    segmenter_name = Column(String)
    model_version = Column(String, nullable=True)
    mask_png_path = Column(Text, nullable=True)      # composite mask
    overlay_png_path = Column(Text, nullable=True)
    n_masks = Column(Integer)
    composite_defect_area_px = Column(Integer)
    composite_defect_area_pct = Column(Float)
    per_mask_json = Column(Text)   # list of SegmentationMask.to_dict()
    run_timestamp = Column(String)


def ensure_table(db_path: str | Path) -> sessionmaker:
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def upsert_segmentation_run(session: Session, result, mask_path=None, overlay_path=None) -> SegmentationRunRecord:
    run_id = f"{result.image_id}__{result.segmenter_name}"
    record = session.get(SegmentationRunRecord, run_id)
    if record is None:
        record = SegmentationRunRecord(run_id=run_id)
        session.add(record)

    record.image_id = result.image_id
    record.source_filename = result.source_filename
    record.segmenter_name = result.segmenter_name
    record.model_version = result.model_version
    record.mask_png_path = str(mask_path) if mask_path else None
    record.overlay_png_path = str(overlay_path) if overlay_path else None
    record.n_masks = result.count
    record.composite_defect_area_px = result.composite_defect_area_px
    record.composite_defect_area_pct = round(result.composite_defect_area_pct, 4)
    record.per_mask_json = json.dumps([m.to_dict() for m in result.masks])
    record.run_timestamp = result.run_timestamp

    session.flush()
    return record


def get_segmentation_runs(
    session: Session, image_id: str | None = None
) -> list[SegmentationRunRecord]:
    q = session.query(SegmentationRunRecord)
    if image_id:
        q = q.filter(SegmentationRunRecord.image_id == image_id)
    return q.order_by(SegmentationRunRecord.source_filename).all()
