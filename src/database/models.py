import datetime
from typing import List, Optional
from sqlalchemy import String, Integer, Float, DateTime, ForeignKey, Boolean, JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class Camera(Base):
    __tablename__ = "cameras"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    source: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(50), nullable=False)
    target_fps: Mapped[int] = mapped_column(Integer, default=10)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow)

class SecurityEvent(Base):
    __tablename__ = "security_events"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[str] = mapped_column(String(50), nullable=False)
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)  # intrusion, loitering, abandoned_object, theft, fire
    track_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    description: Mapped[str] = mapped_column(String(255), nullable=False)
    timestamp: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    evidence_clip_path: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

class CountingEvent(Base):
    __tablename__ = "counting_events"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[str] = mapped_column(String(50), nullable=False)
    line_name: Mapped[str] = mapped_column(String(50), nullable=False)
    track_id: Mapped[int] = mapped_column(Integer, nullable=False)
    direction: Mapped[str] = mapped_column(String(10), nullable=False)  # in, out
    timestamp: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)

class ZoneVisit(Base):
    __tablename__ = "zone_visits"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[str] = mapped_column(String(50), nullable=False)
    zone_name: Mapped[str] = mapped_column(String(50), nullable=False)
    track_id: Mapped[int] = mapped_column(Integer, nullable=False)
    entry_time: Mapped[datetime.datetime] = mapped_column(DateTime, nullable=False)
    exit_time: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime, nullable=True)
    dwell_duration: Mapped[float] = mapped_column(Float, default=0.0)
    loitering_triggered: Mapped[bool] = mapped_column(Boolean, default=False)

class PerformanceMetric(Base):
    __tablename__ = "performance_metrics"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[str] = mapped_column(String(50), nullable=False)
    engine_mode: Mapped[str] = mapped_column(String(20), nullable=False)  # baseline, adaptive
    timestamp: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow)
    cpu_utilization_pct: Mapped[float] = mapped_column(Float)
    ram_utilization_pct: Mapped[float] = mapped_column(Float)
    gpu_utilization_pct: Mapped[float] = mapped_column(Float)
    vram_allocated_bytes: Mapped[float] = mapped_column(Float)
    processed_fps: Mapped[float] = mapped_column(Float)
    inference_latency_ms: Mapped[float] = mapped_column(Float)
    e2e_latency_ms: Mapped[float] = mapped_column(Float)

class BenchmarkRun(Base):
    __tablename__ = "benchmark_runs"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[str] = mapped_column(String(50), nullable=False)
    timestamp: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow)
    duration_seconds: Mapped[float] = mapped_column(Float)
    baseline_metrics: Mapped[dict] = mapped_column(JSON)
    adaptive_metrics: Mapped[dict] = mapped_column(JSON)
    report_path: Mapped[str] = mapped_column(String(255))

class ModelActivation(Base):
    __tablename__ = "model_activations"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[str] = mapped_column(String(50), nullable=False)
    model_name: Mapped[str] = mapped_column(String(50), nullable=False)
    action: Mapped[str] = mapped_column(String(50), nullable=False)  # load, unload
    timestamp: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow)
    details: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
