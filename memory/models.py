import json
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, Index, JSON
from sqlalchemy.types import TypeDecorator
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class VectorType(TypeDecorator):
    """
    Robust Vector column type that handles dense float vectors (e.g. 3072 dimensions)
    consistently across PostgreSQL (pgvector / JSON) and SQLite.
    """
    impl = JSON
    cache_ok = True

    def __init__(self, dim: int = 3072, *args, **kwargs):
        self.dim = dim
        super().__init__(*args, **kwargs)

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if isinstance(value, list):
            return value
        return list(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                return [float(x) for x in value.strip("[]").split(",") if x.strip()]
        return list(value)


class ChatMessage(Base):
    """
    Database model representing a single conversational message.
    Scoped by user_id for multi-tenant isolation and session_id for conversation threads.
    """
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(128), nullable=False, index=True, default="default_user")
    session_id = Column(String(128), nullable=False, index=True)
    role = Column(String(32), nullable=False)  # 'user', 'assistant', 'system'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False, index=True)

    __table_args__ = (
        Index("ix_chat_user_session", "user_id", "session_id", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<ChatMessage(id={self.id}, user_id='{self.user_id}', session_id='{self.session_id}', role='{self.role}')>"


class ConversationSummary(Base):
    """
    Database model storing the running condensed summary of earlier messages
    that have been evicted from the active sliding window buffer.
    """
    __tablename__ = "conversation_summaries"

    session_id = Column(String(128), primary_key=True)
    user_id = Column(String(128), nullable=False, index=True, default="default_user")
    summary = Column(Text, nullable=False, default="")
    last_summarized_message_id = Column(Integer, nullable=False, default=0)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_summary_user_session", "user_id", "session_id"),
    )

    def __repr__(self) -> str:
        return f"<ConversationSummary(user_id='{self.user_id}', session_id='{self.session_id}', last_id={self.last_summarized_message_id})>"


class Episode(Base):
    """
    Database model representing an extracted episodic memory record.
    Captures specific events, experiences, discussions, and topics bounded in time.
    
    Fields:
    - timestamp: when the conversation/event actually took place.
    - created_at: when this episodic record was extracted and created in DB.
    - start_message_id / end_message_id: message provenance tracking.
    - embedding: 3072-dimensional vector embedding for semantic similarity search.
    """
    __tablename__ = "episodes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    episode_id = Column(String(128), unique=True, nullable=False, index=True)
    user_id = Column(String(128), nullable=False, index=True, default="default_user")
    session_id = Column(String(128), nullable=False, index=True)

    # When the conversation/event occurred
    timestamp = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False, index=True)

    # Source provenance (message boundaries)
    start_message_id = Column(Integer, nullable=True)
    end_message_id = Column(Integer, nullable=True)

    summary = Column(Text, nullable=False)
    events = Column(JSON, nullable=False, default=list)  # List of string events
    topics = Column(JSON, nullable=False, default=list)  # List of topic keywords

    # 3072-dimensional vector embedding
    embedding = Column(VectorType(dim=3072), nullable=True)

    # When the episode record was saved to DB
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_episodes_user_session_time", "user_id", "session_id", "timestamp"),
    )

    def __repr__(self) -> str:
        return f"<Episode(id={self.id}, episode_id='{self.episode_id}', user_id='{self.user_id}', session_id='{self.session_id}')>"
