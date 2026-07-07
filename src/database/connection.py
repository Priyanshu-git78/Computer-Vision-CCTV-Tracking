import os
import logging
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from src.database.models import Base

load_dotenv()

logger = logging.getLogger(__name__)

# Fetch database URL with SQLite fallback
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./security_ecosystem.db")

# SQLite specific argument for multithread access
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

logger.info(f"Initializing database engine with URL: {DATABASE_URL}")
engine = create_engine(DATABASE_URL, connect_args=connect_args)

# Create session maker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initializes the database tables."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize database tables: {e}")
        raise e

def get_db():
    """Database session dependency generator for FastAPI endpoints."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
