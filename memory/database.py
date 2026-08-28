import logging
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from memory.config import settings
from memory.models import Base

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _get_engine():
    """
    Creates SQLAlchemy engine with automatic fallback to SQLite
    if PostgreSQL is unreachable or connection fails.
    """
    url = settings.database_url
    # Adapt 'postgresql://' to 'postgresql+psycopg://' if psycopg3 is used
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)

    try:
        engine = create_engine(url, pool_pre_ping=True)
        # Test connection
        with engine.connect() as conn:
            pass
        logger.info(f"Connected successfully to database: {url.split('@')[-1] if '@' in url else url}")
        return engine
    except Exception as e:
        logger.warning(
            f"Could not connect to configured database ({settings.database_url}): {e}. "
            "Falling back to local SQLite database (sqlite:///agent_memory.db)."
        )
        fallback_url = "sqlite:///agent_memory.db"
        return create_engine(fallback_url, connect_args={"check_same_thread": False})


engine = _get_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, expire_on_commit=False, bind=engine)


def init_db() -> None:
    """Initialize database tables defined in models."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized successfully.")


@contextmanager
def get_db():
    """Context manager for obtaining a transactional database session."""
    session: Session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
