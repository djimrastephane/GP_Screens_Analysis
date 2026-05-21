"""SQLAlchemy ORM for quantification reports."""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import Boolean, Column, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.ingestion.database import Base


class QuantificationRecord(Base):
    __tablename__ = "quantification_reports"

    image_id = Column(String, primary_key=True)
    source_filename = Column(String)
    image_width = Column(Integer)
    image_height = Column(Integer)
    screen_region_area_px = Column(Integer)
    scale_calibrated = Column(Boolean)
    pixels_per_mm = Column(Float, nullable=True)
    n_defects = Column(Integer)
    composite_defect_area_px = Column(Integer)
    erosion_pct = Column(Float)
    defect_density_per_10k_px = Column(Float)
    n_requires_review = Column(Integer)
    overall_severity = Column(String)
    dominant_failure_type = Column(String)
    failure_type_breakdown_json = Column(Text)
    severity_breakdown_json = Column(Text)
    defects_json = Column(Text)
    run_timestamp = Column(String)


def ensure_table(db_path: str | Path) -> sessionmaker:
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def upsert_report(session: Session, report: "QuantificationReport") -> QuantificationRecord:
    record = session.get(QuantificationRecord, report.image_id)
    if record is None:
        record = QuantificationRecord(image_id=report.image_id)
        session.add(record)

    record.source_filename = report.source_filename
    record.image_width = report.image_width
    record.image_height = report.image_height
    record.screen_region_area_px = report.screen_region_area_px
    record.scale_calibrated = report.scale_calibrated
    record.pixels_per_mm = report.pixels_per_mm
    record.n_defects = report.n_defects
    record.composite_defect_area_px = report.composite_defect_area_px
    record.erosion_pct = round(report.erosion_pct, 4)
    record.defect_density_per_10k_px = round(report.defect_density_per_10k_px, 4)
    record.n_requires_review = report.n_requires_review
    record.overall_severity = report.overall_severity
    record.dominant_failure_type = report.dominant_failure_type
    record.failure_type_breakdown_json = json.dumps(report.failure_type_breakdown)
    record.severity_breakdown_json = json.dumps(report.severity_breakdown)
    record.defects_json = json.dumps([d.to_dict() for d in report.defects])
    record.run_timestamp = report.run_timestamp

    session.flush()
    return record


def get_all_reports(session: Session) -> list[QuantificationRecord]:
    return (
        session.query(QuantificationRecord)
        .order_by(QuantificationRecord.source_filename)
        .all()
    )
