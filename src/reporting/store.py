"""Track generated report file paths in the database."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import Column, Integer, String, Text, create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.ingestion.database import Base


class ReportRecord(Base):
    __tablename__ = "report_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    report_type = Column(String)          # "per_image" | "summary"
    image_id = Column(String, nullable=True)
    source_filename = Column(String, nullable=True)
    pdf_path = Column(Text)
    n_pages = Column(Integer, nullable=True)
    run_timestamp = Column(String)


def ensure_table(db_path: str | Path) -> sessionmaker:
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def insert_report(
    session: Session,
    report_type: str,
    pdf_path: str,
    run_timestamp: str,
    image_id: str | None = None,
    source_filename: str | None = None,
) -> ReportRecord:
    record = ReportRecord(
        report_type=report_type,
        image_id=image_id,
        source_filename=source_filename,
        pdf_path=pdf_path,
        run_timestamp=run_timestamp,
    )
    session.add(record)
    session.flush()
    return record


def get_reports(
    session: Session, report_type: str | None = None
) -> list[ReportRecord]:
    q = session.query(ReportRecord)
    if report_type:
        q = q.filter(ReportRecord.report_type == report_type)
    return q.order_by(ReportRecord.run_timestamp.desc()).all()
