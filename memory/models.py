from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, Index
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class ChatMessage(Base):
    """
    Database model representing a single conversational message.
    """
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(128), nullable=False, index=True)
    role = Column(String(32), nullable=False)  # 'user', 'assistant', 'system'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    __table_args__ = (
        Index("ix_session_created", "session_id", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<ChatMessage(id={self.id}, session_id='{self.session_id}', role='{self.role}', created_at='{self.created_at}')>"


class ConversationSummary(Base):
    """
    Database model storing the running condensed summary of earlier messages
    that have been evicted from the active sliding window buffer.
    """
    __tablename__ = "conversation_summaries"

    session_id = Column(String(128), primary_key=True)
    summary = Column(Text, nullable=False, default="")
    last_summarized_message_id = Column(Integer, nullable=False, default=0)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        return f"<ConversationSummary(session_id='{self.session_id}', last_id={self.last_summarized_message_id}, updated_at='{self.updated_at}')>"
