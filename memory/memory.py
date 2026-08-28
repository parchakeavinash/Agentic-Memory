from typing import List, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from memory.database import get_db
from memory.models import ChatMessage, ConversationSummary
from memory.config import settings
import re
import logging

logger = logging.getLogger(__name__)


class ShortTermMemory:
    """
    Manages short-term conversation memory using a Sliding Window Buffer
    combined with an incremental Conversation Summary.
    Scoped by user_id for multi-tenant isolation.
    """

    def __init__(self, default_window_size: Optional[int] = None):
        self.default_window_size = default_window_size or settings.memory_window

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        user_id: Optional[str] = None,
    ) -> ChatMessage:
        """Saves a message to the database under the specified session and user."""
        uid = user_id or settings.user_id
        with get_db() as db:
            msg = ChatMessage(user_id=uid, session_id=session_id, role=role, content=content)
            db.add(msg)
            db.flush()
            db.refresh(msg)
            return msg

    def add_user_message(self, session_id: str, content: str, user_id: Optional[str] = None) -> ChatMessage:
        """Convenience helper to record user input."""
        return self.add_message(session_id=session_id, role="user", content=content, user_id=user_id)

    def add_ai_message(self, session_id: str, content: str, user_id: Optional[str] = None) -> ChatMessage:
        """Convenience helper to record AI assistant response."""
        return self.add_message(session_id=session_id, role="assistant", content=content, user_id=user_id)

    def add_system_message(self, session_id: str, content: str, user_id: Optional[str] = None) -> ChatMessage:
        """Convenience helper to record a system message."""
        return self.add_message(session_id=session_id, role="system", content=content, user_id=user_id)

    def get_messages(
        self,
        session_id: str,
        window_size: Optional[int] = None,
        user_id: Optional[str] = None,
    ) -> List[BaseMessage]:
        """
        Retrieves the last `window_size` messages for the given session and user from DB,
        and converts them into LangChain BaseMessage objects in chronological order.
        """
        limit = window_size if window_size is not None else self.default_window_size
        uid = user_id or settings.user_id

        with get_db() as db:
            records = (
                db.query(ChatMessage)
                .filter(ChatMessage.user_id == uid, ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
                .limit(limit)
                .all()
            )

        records.reverse()

        messages: List[BaseMessage] = []
        for r in records:
            if r.role == "user":
                messages.append(HumanMessage(content=r.content))
            elif r.role == "assistant":
                messages.append(AIMessage(content=r.content))
            elif r.role == "system":
                messages.append(SystemMessage(content=r.content))
            else:
                messages.append(HumanMessage(content=r.content))

        return messages

    def get_raw_messages(
        self,
        session_id: str,
        limit: int = 100,
        user_id: Optional[str] = None,
    ) -> List[ChatMessage]:
        """Retrieves raw database rows for inspecting session history."""
        uid = user_id or settings.user_id
        with get_db() as db:
            return (
                db.query(ChatMessage)
                .filter(ChatMessage.user_id == uid, ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc())
                .limit(limit)
                .all()
            )

    def get_summary(self, session_id: str, user_id: Optional[str] = None) -> Optional[str]:
        """Retrieves the running conversation summary for the session, if one exists."""
        uid = user_id or settings.user_id
        with get_db() as db:
            record = (
                db.query(ConversationSummary)
                .filter(ConversationSummary.user_id == uid, ConversationSummary.session_id == session_id)
                .first()
            )
            if record and record.summary:
                return record.summary
            return None

    def save_summary(
        self,
        session_id: str,
        summary_text: str,
        last_message_id: int,
        user_id: Optional[str] = None,
    ) -> ConversationSummary:
        """Saves or updates the running conversation summary in the database."""
        uid = user_id or settings.user_id
        with get_db() as db:
            record = (
                db.query(ConversationSummary)
                .filter(ConversationSummary.user_id == uid, ConversationSummary.session_id == session_id)
                .first()
            )
            if not record:
                record = ConversationSummary(
                    user_id=uid,
                    session_id=session_id,
                    summary=summary_text,
                    last_summarized_message_id=last_message_id,
                )
                db.add(record)
            else:
                record.summary = summary_text
                record.last_summarized_message_id = last_message_id

            db.flush()
            db.refresh(record)
            return record

    def update_summary_if_needed(
        self,
        session_id: str,
        llm,
        window_size: Optional[int] = None,
        user_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Progressively summarizes messages that have fallen outside the active sliding window.
        """
        limit = window_size if window_size is not None else self.default_window_size
        uid = user_id or settings.user_id

        with get_db() as db:
            all_messages = (
                db.query(ChatMessage)
                .filter(ChatMessage.user_id == uid, ChatMessage.session_id == session_id)
                .order_by(ChatMessage.id.asc())
                .all()
            )

            summary_record = (
                db.query(ConversationSummary)
                .filter(ConversationSummary.user_id == uid, ConversationSummary.session_id == session_id)
                .first()
            )

        total_count = len(all_messages)
        if total_count <= limit:
            return self.get_summary(session_id=session_id, user_id=uid)

        # Messages that are outside the active sliding window
        evicted_messages = all_messages[:-limit]
        last_evicted_id = evicted_messages[-1].id
        last_summarized_id = summary_record.last_summarized_message_id if summary_record else 0

        # Find only the evicted messages that haven't been summarized yet
        new_unsummarized = [m for m in evicted_messages if m.id > last_summarized_id]
        if not new_unsummarized:
            return summary_record.summary if summary_record else None

        existing_summary = summary_record.summary if (summary_record and summary_record.summary) else "No prior summary."

        lines = []
        for m in new_unsummarized:
            speaker = "User" if m.role == "user" else "Assistant"
            lines.append(f"{speaker}: {m.content}")
        conversation_chunk = "\n".join(lines)

        prompt = [
            SystemMessage(
                content=(
                    "You are a memory distillation system for an AI assistant.\n"
                    "Your job is to update a running factual summary of a conversation.\n\n"
                    "RULES:\n"
                    "- Capture ONLY facts, user goals, decisions, and project details.\n"
                    "- DO NOT describe or mention what the assistant said, explained, or provided.\n"
                    "- DO NOT write: 'The assistant explained X', 'The assistant provided Y', "
                    "'A detailed explanation was given', 'A comprehensive guide was shared'.\n"
                    "- Focus on the USER: what they are building, doing, deciding, or asking about.\n"
                    "- Keep it short, factual, and bullet-point style.\n"
                    "- Return ONLY the updated summary, nothing else."
                )
            ),
            HumanMessage(
                content=(
                    f"Current Summary:\n{existing_summary}\n\n"
                    f"New Conversation Turns:\n{conversation_chunk}\n\n"
                    "Updated Factual Summary (user facts/goals/decisions only):"
                )
            ),
        ]

        try:
            response = llm.invoke(prompt)
            raw_summary = str(response.content)
            clean_summary = re.sub(r"<think>.*?</think>\s*", "", raw_summary, flags=re.DOTALL).strip()
            if not clean_summary:
                clean_summary = raw_summary

            self.save_summary(
                session_id=session_id,
                summary_text=clean_summary,
                last_message_id=last_evicted_id,
                user_id=uid,
            )
            logger.info(f"Updated conversation summary for session '{session_id}' (User: '{uid}') up to message #{last_evicted_id}.")
            return clean_summary
        except Exception as e:
            logger.warning(f"Failed to generate conversation summary: {e}")
            return self.get_summary(session_id=session_id, user_id=uid)

    def clear_memory(self, session_id: str, user_id: Optional[str] = None) -> int:
        """Clears all conversation history and summary for a given session and user."""
        uid = user_id or settings.user_id
        with get_db() as db:
            deleted_count = (
                db.query(ChatMessage)
                .filter(ChatMessage.user_id == uid, ChatMessage.session_id == session_id)
                .delete()
            )
            db.query(ConversationSummary).filter(
                ConversationSummary.user_id == uid,
                ConversationSummary.session_id == session_id,
            ).delete()
            return deleted_count

    def get_memory_stats(self, session_id: str, user_id: Optional[str] = None) -> dict:
        """Returns statistics about total messages in DB, sliding window, and summary state."""
        uid = user_id or settings.user_id
        with get_db() as db:
            total_count = (
                db.query(ChatMessage)
                .filter(ChatMessage.user_id == uid, ChatMessage.session_id == session_id)
                .count()
            )
            summary_record = (
                db.query(ConversationSummary)
                .filter(ConversationSummary.user_id == uid, ConversationSummary.session_id == session_id)
                .first()
            )

        has_summary = bool(summary_record and summary_record.summary)
        return {
            "user_id": uid,
            "session_id": session_id,
            "total_stored_messages": total_count,
            "window_size": self.default_window_size,
            "active_in_prompt": min(total_count, self.default_window_size),
            "evicted_from_window": max(0, total_count - self.default_window_size),
            "has_summary": has_summary,
            "summary_length": len(summary_record.summary) if has_summary else 0,
        }
