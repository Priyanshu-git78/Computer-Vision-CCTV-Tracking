import datetime
from sqlalchemy.orm import Session
from sqlalchemy import select, desc
from typing import List, Optional
from src.database import models

# ==========================================
# Camera Operations
# ==========================================

def get_cameras(db: Session) -> List[models.Camera]:
    return list(db.scalars(select(models.Camera)).all())

def get_camera_by_id(db: Session, camera_id: str) -> Optional[models.Camera]:
    return db.scalars(select(models.Camera).where(models.Camera.camera_id == camera_id)).first()

def create_camera(db: Session, camera_id: str, source: str, role: str, target_fps: int = 10, enabled: bool = True) -> models.Camera:
    db_camera = get_camera_by_id(db, camera_id)
    if db_camera:
        # Update existing
        db_camera.source = source
        db_camera.role = role
        db_camera.target_fps = target_fps
        db_camera.enabled = enabled
    else:
        # Create new
        db_camera = models.Camera(
            camera_id=camera_id,
            source=source,
            role=role,
            target_fps=target_fps,
            enabled=enabled
        )
        db.add(db_camera)
    db.commit()
    db.refresh(db_camera)
    return db_camera


# ==========================================
# Security Event Operations
# ==========================================

def create_security_event(
    db: Session, 
    camera_id: str, 
    event_type: str, 
    track_id: Optional[int], 
    description: str, 
    confidence: float = 1.0,
    evidence_clip_path: Optional[str] = None
) -> models.SecurityEvent:
    event = models.SecurityEvent(
        camera_id=camera_id,
        event_type=event_type,
        track_id=track_id,
        description=description,
        confidence=confidence,
        evidence_clip_path=evidence_clip_path
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return event

def get_security_events(db: Session, limit: int = 100) -> List[models.SecurityEvent]:
    return list(db.scalars(
        select(models.SecurityEvent)
        .order_by(desc(models.SecurityEvent.timestamp))
        .limit(limit)
    ).all())


# ==========================================
# Counting Event Operations
# ==========================================

def create_counting_event(
    db: Session,
    camera_id: str,
    line_name: str,
    track_id: int,
    direction: str,
    confidence: float = 1.0
) -> models.CountingEvent:
    event = models.CountingEvent(
        camera_id=camera_id,
        line_name=line_name,
        track_id=track_id,
        direction=direction,
        confidence=confidence
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return event

def get_counting_events(db: Session, camera_id: Optional[str] = None, limit: int = 100) -> List[models.CountingEvent]:
    stmt = select(models.CountingEvent).order_by(desc(models.CountingEvent.timestamp))
    if camera_id:
        stmt = stmt.where(models.CountingEvent.camera_id == camera_id)
    return list(db.scalars(stmt.limit(limit)).all())


# ==========================================
# Zone Visit Operations
# ==========================================

def create_zone_visit(
    db: Session,
    camera_id: str,
    zone_name: str,
    track_id: int,
    entry_time: datetime.datetime,
    exit_time: Optional[datetime.datetime] = None,
    dwell_duration: float = 0.0,
    loitering_triggered: bool = False
) -> models.ZoneVisit:
    visit = models.ZoneVisit(
        camera_id=camera_id,
        zone_name=zone_name,
        track_id=track_id,
        entry_time=entry_time,
        exit_time=exit_time,
        dwell_duration=dwell_duration,
        loitering_triggered=loitering_triggered
    )
    db.add(visit)
    db.commit()
    db.refresh(visit)
    return visit

def get_zone_visits(db: Session, zone_name: Optional[str] = None, limit: int = 100) -> List[models.ZoneVisit]:
    stmt = select(models.ZoneVisit).order_by(desc(models.ZoneVisit.entry_time))
    if zone_name:
        stmt = stmt.where(models.ZoneVisit.zone_name == zone_name)
    return list(db.scalars(stmt.limit(limit)).all())


# ==========================================
# Performance Metrics Operations
# ==========================================

def create_performance_metric(
    db: Session,
    camera_id: str,
    engine_mode: str,
    cpu_utilization_pct: float,
    ram_utilization_pct: float,
    gpu_utilization_pct: float,
    vram_allocated_bytes: float,
    processed_fps: float,
    inference_latency_ms: float,
    e2e_latency_ms: float
) -> models.PerformanceMetric:
    metric = models.PerformanceMetric(
        camera_id=camera_id,
        engine_mode=engine_mode,
        cpu_utilization_pct=cpu_utilization_pct,
        ram_utilization_pct=ram_utilization_pct,
        gpu_utilization_pct=gpu_utilization_pct,
        vram_allocated_bytes=vram_allocated_bytes,
        processed_fps=processed_fps,
        inference_latency_ms=inference_latency_ms,
        e2e_latency_ms=e2e_latency_ms
    )
    db.add(metric)
    db.commit()
    db.refresh(metric)
    return metric

def get_performance_metrics(db: Session, camera_id: str, limit: int = 100) -> List[models.PerformanceMetric]:
    return list(db.scalars(
        select(models.PerformanceMetric)
        .where(models.PerformanceMetric.camera_id == camera_id)
        .order_by(desc(models.PerformanceMetric.timestamp))
        .limit(limit)
    ).all())


# ==========================================
# Benchmark Runs Operations
# ==========================================

def create_benchmark_run(
    db: Session,
    camera_id: str,
    duration_seconds: float,
    baseline_metrics: dict,
    adaptive_metrics: dict,
    report_path: str
) -> models.BenchmarkRun:
    run = models.BenchmarkRun(
        camera_id=camera_id,
        duration_seconds=duration_seconds,
        baseline_metrics=baseline_metrics,
        adaptive_metrics=adaptive_metrics,
        report_path=report_path
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run

def get_benchmark_runs(db: Session, limit: int = 20) -> List[models.BenchmarkRun]:
    return list(db.scalars(
        select(models.BenchmarkRun)
        .order_by(desc(models.BenchmarkRun.timestamp))
        .limit(limit)
    ).all())


# ==========================================
# Model Activation Operations
# ==========================================

def create_model_activation(
    db: Session,
    camera_id: str,
    model_name: str,
    action: str,
    details: Optional[dict] = None
) -> models.ModelActivation:
    act = models.ModelActivation(
        camera_id=camera_id,
        model_name=model_name,
        action=action,
        details=details
    )
    db.add(act)
    db.commit()
    db.refresh(act)
    return act
