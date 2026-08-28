import logging
from contextlib import contextmanager
from sqlalchemy import create_engine, text, JSON
from sqlalchemy.orm import sessionmaker, Session
from memory.config import settings
from memory.models import Base, Episode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HAS_NATIVE_VECTOR_EXTENSION = False


def _get_engine():
    """
    Creates SQLAlchemy engine with automatic fallback to SQLite
    if PostgreSQL is unreachable or connection fails.
    """
    url = settings.database_url
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)

    try:
        engine = create_engine(url, pool_pre_ping=True)
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
    """Initialize database tables and schema migrations."""
    global HAS_NATIVE_VECTOR_EXTENSION

    if engine.dialect.name == "postgresql":
        try:
            with engine.connect() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                conn.commit()
            HAS_NATIVE_VECTOR_EXTENSION = True
            logger.info("PostgreSQL pgvector extension verified/enabled.")
        except Exception as e:
            HAS_NATIVE_VECTOR_EXTENSION = False
            logger.info(
                f"Note: PostgreSQL native pgvector extension not present on server ({e}). "
                "Using high-performance JSON vector storage."
            )

    try:
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        # If Vector type creation fails due to missing C extension, adapt column to JSON
        if 'type "vector" does not exist' in str(e):
            Episode.embedding.type = JSON()
            Base.metadata.create_all(bind=engine)
        else:
            raise

    # Safe schema migration for existing tables
    if engine.dialect.name == "postgresql":
        try:
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS user_id VARCHAR(128) DEFAULT 'default_user';"))
                conn.execute(text("ALTER TABLE conversation_summaries ADD COLUMN IF NOT EXISTS user_id VARCHAR(128) DEFAULT 'default_user';"))
                conn.execute(text("ALTER TABLE episodes ADD COLUMN IF NOT EXISTS user_id VARCHAR(128) DEFAULT 'default_user';"))
                conn.execute(text("ALTER TABLE episodes ADD COLUMN IF NOT EXISTS start_message_id INTEGER;"))
                conn.execute(text("ALTER TABLE episodes ADD COLUMN IF NOT EXISTS end_message_id INTEGER;"))
                conn.commit()
        except Exception as e:
            logger.warning(f"Schema column check notice: {e}")

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
