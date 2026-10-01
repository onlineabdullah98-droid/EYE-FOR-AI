"""Prediction history storage via SQLAlchemy.

Defaults to a local SQLite file. For PostgreSQL set the environment variable::

    EYEFORAI_DATABASE_URL=postgresql+psycopg2://user:password@localhost:5432/eyeforai
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import DateTime, Float, Integer, String, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from ..config import DATABASE_URL


class Base(DeclarativeBase):
    pass


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    filename: Mapped[str] = mapped_column(String(255))
    image_sha256: Mapped[str] = mapped_column(String(64), index=True)
    label: Mapped[str] = mapped_column(String(16))
    confidence: Mapped[float] = mapped_column(Float)
    prob_fake: Mapped[float] = mapped_column(Float)
    model_name: Mapped[str] = mapped_column(String(64))
    inference_ms: Mapped[float] = mapped_column(Float)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "time": self.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "filename": self.filename,
            "label": self.label,
            "confidence": round(self.confidence * 100, 2),
            "prob_fake": round(self.prob_fake * 100, 2),
            "model": self.model_name,
            "ms": round(self.inference_ms, 1),
        }


class HistoryStore:
    def __init__(self, url: str | None = None):
        url = url or DATABASE_URL
        if url.startswith("sqlite:///"):
            Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(url, pool_pre_ping=True)
        Base.metadata.create_all(self.engine)

    def add(self, *, filename: str, image_bytes: bytes, label: str, confidence: float, prob_fake: float,
            model_name: str, inference_ms: float) -> int:
        record = PredictionRecord(
            filename=filename[:255],
            image_sha256=hashlib.sha256(image_bytes).hexdigest(),
            label=label,
            confidence=confidence,
            prob_fake=prob_fake,
            model_name=model_name,
            inference_ms=inference_ms,
        )
        with Session(self.engine) as session:
            session.add(record)
            session.commit()
            return record.id

    def recent(self, limit: int = 50) -> list[dict]:
        with Session(self.engine) as session:
            rows = session.scalars(select(PredictionRecord).order_by(PredictionRecord.id.desc()).limit(limit))
            return [r.to_dict() for r in rows]

    def stats(self) -> dict[str, int]:
        with Session(self.engine) as session:
            rows = session.execute(select(PredictionRecord.label, func.count()).group_by(PredictionRecord.label))
            counts = {label: n for label, n in rows}
        return {"total": sum(counts.values()), "REAL": counts.get("REAL", 0), "FAKE": counts.get("FAKE", 0),
                "UNCERTAIN": counts.get("UNCERTAIN", 0)}

    def clear(self) -> None:
        with Session(self.engine) as session:
            session.query(PredictionRecord).delete()
            session.commit()
