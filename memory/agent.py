import logging
from typing import Optional, List, Dict, Any
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from memory.config import settings
from memory.memory import ShortTermMemory
from memory.episodic import EpisodicMemoryManager
from memory.semantic import SemanticMemoryManager
from memory.database import init_db

logger = logging.getLogger(__name__)



class MemoryAgent:
    """
    An AI conversational agent powered by Groq LLMs and equipped with:
    1. Short-Term Memory (Sliding Window Buffer)
    2. Summary Memory (Incremental eviction summarization)
    3. Episodic Memory (Experience distillation + Google Gemini vector search)
    Supports multi-tenant user_id isolation and multi-session tracking.
    """

    def __init__(
        self,
        system_prompt: Optional[str] = None,
      
        window_size: Optional[int] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        init_db()

        self.user_id = user_id or settings.user_id
        self.api_key = api_key or settings.groq_api_key
        self.model_name = model_name or settings.groq_model
        self.memory = ShortTermMemory(default_window_size=window_size)
        self.episodic = EpisodicMemoryManager()
        self.semantic = SemanticMemoryManager()

        self.system_prompt = system_prompt or (
            "You are a conversational AI assistant with multi-layered memory.\n\n"
            "## Conversation Style\n"
            "Respond naturally and conversationally.\n\n"
            "IMPORTANT:\n"
            "- Do not provide long explanations unless the user asks for them.\n"
            "- Match the user's level of detail.\n"
            "- If the user makes a simple statement, respond briefly and naturally.\n"
            "- If the user asks a simple question, give a concise answer.\n"
            "- Do not automatically provide tutorials, architectures, code, tables, or "
            "detailed explanations unless requested.\n"
            "- Ask a follow-up question when it would naturally continue the conversation.\n"
            "- Avoid unnecessary headings, bullet points, and sections.\n"
            "- Prefer 1-3 short paragraphs for normal conversation.\n"
            "- Expand the answer only when the user explicitly asks for more detail."
        )

        # temperature=0.7 — conversational, natural, slightly creative
        self.llm = ChatGroq(
            groq_api_key=self.api_key,
            model_name=self.model_name,
            temperature=0.7,
        )

        # temperature=0 — deterministic, strictly grounded, no hallucination
        # Used for: episode extraction, conversation summarization
        self.extraction_llm = ChatGroq(
            groq_api_key=self.api_key,
            model_name=self.model_name,
            temperature=0,
        )

        self.last_prompt_debug = None

    def chat(
        self,
        user_input: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """
        Processes a conversational turn with multi-layered memory:
        1. Short-Term Memory Buffer (scoped to user_id and session_id)
        2. Condensed Summary (scoped to user_id and session_id)
        3. Vector-Retrieved Episodic Memories (scoped to user_id)
        """
        session_id = session_id or settings.session_id
        uid = user_id or self.user_id

        # 1. Fetch short-term memory window & running summary for this user and session
        history: List[BaseMessage] = self.memory.get_messages(session_id=session_id, user_id=uid)
        summary: Optional[str] = self.memory.get_summary(session_id=session_id, user_id=uid)

        # 2. Vector search for relevant past episodic memories for this user
        relevant_episodes = self.episodic.search_episodes(
            query=user_input,
            user_id=uid,
            session_id=None,  # Searches across all past sessions of this user
            top_k=settings.episodic_top_k,
            min_similarity=settings.episodic_min_similarity,
        )

        # 3. Retrieve relevant semantic facts (persistent user knowledge)
        relevant_facts = self.semantic.search_facts(
            query=user_input,
            user_id=uid,
            top_k=5,
            min_similarity=0.45,
        )

        # 4. Assemble complete prompt payload
        messages: List[BaseMessage] = [SystemMessage(content=self.system_prompt)]

        if relevant_facts:
            facts_lines = "\n".join(
                f"- {f['key'].replace('_', ' ').title()}: {f['value']}"
                for f in relevant_facts
            )
            messages.append(SystemMessage(
                content=f"What I know about you:\n{facts_lines}"
            ))

        if summary:
            summary_prompt = (
                "Summary of earlier conversation in this session:\n"
                f"{summary}"
            )
            messages.append(SystemMessage(content=summary_prompt))

        if relevant_episodes:
            ep_parts = []
            for ep in relevant_episodes:
                date_str = ep["timestamp"][:10] if ep.get("timestamp") else "N/A"
                events_text = "\n".join(f"- {e}" for e in ep.get("events", [])) or "- (none)"
                topics_text = ", ".join(ep.get("topics", [])) or "(none)"
                ep_parts.append(
                    f"Episode [{date_str}]\n\n"
                    f"Summary:\n{ep['summary']}\n\n"
                    f"Events:\n{events_text}\n\n"
                    f"Topics:\n{topics_text}"
                )
            ep_prompt = (
                "Relevant Episodic Memory (use only if relevant to the current question):\n\n"
                + "\n\n---\n\n".join(ep_parts)
            )
            messages.append(SystemMessage(content=ep_prompt))

        messages.extend(history)
        messages.append(HumanMessage(content=user_input))

        # Record prompt debug metadata
        stats = self.memory.get_memory_stats(session_id=session_id, user_id=uid)
        self.last_prompt_debug = {
            "user_id": uid,
            "session_id": session_id,
            "total_stored_messages": stats["total_stored_messages"],
            "window_size": self.memory.default_window_size,
            "has_summary": bool(summary),
            "summary_text": summary,
            "semantic_facts_count": len(relevant_facts),
            "semantic_facts_injected": relevant_facts,
            "episodic_count": len(relevant_episodes),
            "episodes_injected": relevant_episodes,
            "history_injected_count": len(history),
            "total_prompt_items": len(messages),
            "messages": messages,
        }

        # 4. Invoke LLM
        response = self.llm.invoke(messages)
        raw_text = str(response.content)

        # Strip internal <think>...</think> reasoning traces if present (e.g. Qwen/DeepSeek)
        import re
        response_text = re.sub(r"<think>.*?</think>\s*", "", raw_text, flags=re.DOTALL).strip()
        if not response_text:
            response_text = raw_text

        # 5. Save this turn into short-term memory persistence
        self.memory.add_user_message(session_id=session_id, content=user_input, user_id=uid)
        self.memory.add_ai_message(session_id=session_id, content=response_text, user_id=uid)

        # 6. Detect and store semantic facts from user message (background, non-blocking)
        try:
            result = self.semantic.process_message(
                user_message=user_input,
                user_id=uid,
                llm=self.extraction_llm,
            )
            if result:
                action, key, value = result
                logger.info(f"Semantic memory [{action}]: {key} = {value}")
        except Exception as e:
            logger.warning(f"Semantic memory processing failed (non-critical): {e}")

        # 7. Incrementally update running summary — use extraction_llm (temperature=0, no hallucination)
        self.memory.update_summary_if_needed(
            session_id=session_id,
            llm=self.extraction_llm,
            window_size=self.memory.default_window_size,
            user_id=uid,
        )

        return response_text

    def create_episode(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Distills the conversation in the session into a structured episodic memory
        and saves it with a Google Gemini vector embedding.
        """
        session_id = session_id or settings.session_id
        uid = user_id or self.user_id

        messages = self.memory.get_raw_messages(session_id=session_id, user_id=uid, limit=50)
        if not messages:
            return None

        extracted = self.episodic.extract_episode_from_conversation(
            messages=messages,
            llm=self.extraction_llm,  # temperature=0 — strictly grounded
            session_id=session_id,
            user_id=uid,
        )
        if not extracted:
            return None

        summary = extracted.get("summary", "")
        events = extracted.get("events", [])
        topics = extracted.get("topics", [])
        start_id = extracted.get("start_message_id")
        end_id = extracted.get("end_message_id")
        convo_ts = extracted.get("timestamp")

        episode = self.episodic.store_episode(
            user_id=uid,
            session_id=session_id,
            summary=summary,
            events=events,
            topics=topics,
            start_message_id=start_id,
            end_message_id=end_id,
            timestamp=convo_ts,
        )
        return {
            "episode_id": episode.episode_id,
            "user_id": episode.user_id,
            "session_id": episode.session_id,
            "timestamp": episode.timestamp.isoformat() if episode.timestamp else None,
            "created_at": episode.created_at.isoformat() if episode.created_at else None,
            "start_message_id": episode.start_message_id,
            "end_message_id": episode.end_message_id,
            "summary": episode.summary,
            "events": episode.events,
            "topics": episode.topics,
        }

    def search_episodes(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Searches past episodic memories using vector similarity for the user."""
        uid = user_id or self.user_id
        return self.episodic.search_episodes(
            query=query,
            user_id=uid,
            session_id=session_id,
            top_k=top_k,
            min_similarity=min_similarity,
        )

    def list_episodes(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 50,
    ):
        """Lists stored episodes for the user."""
        uid = user_id or self.user_id
        return self.episodic.list_episodes(user_id=uid, session_id=session_id, limit=limit)

    def clear_episodes(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> int:
        """Deletes stored episodes for the user."""
        uid = user_id or self.user_id
        return self.episodic.clear_episodes(user_id=uid, session_id=session_id)

    def get_summary(self, session_id: Optional[str] = None, user_id: Optional[str] = None) -> Optional[str]:
        """Retrieves the running conversation summary for the session and user."""
        session_id = session_id or settings.session_id
        uid = user_id or self.user_id
        return self.memory.get_summary(session_id=session_id, user_id=uid)

    def get_last_prompt_debug(self) -> Optional[dict]:
        """Returns the debug info for the most recent prompt payload sent to the LLM."""
        return self.last_prompt_debug

    def get_history(self, session_id: Optional[str] = None, user_id: Optional[str] = None):
        """Retrieves raw message history for inspection."""
        session_id = session_id or settings.session_id
        uid = user_id or self.user_id
        return self.memory.get_raw_messages(session_id=session_id, user_id=uid)

    def clear_memory(self, session_id: Optional[str] = None, user_id: Optional[str] = None) -> int:
        """Resets the memory for the session and user."""
        session_id = session_id or settings.session_id
        uid = user_id or self.user_id
        return self.memory.clear_memory(session_id=session_id, user_id=uid)

    def get_stats(self, session_id: Optional[str] = None, user_id: Optional[str] = None) -> dict:
        """Returns memory statistics for the user and session."""
        session_id = session_id or settings.session_id
        uid = user_id or self.user_id
        return self.memory.get_memory_stats(session_id=session_id, user_id=uid)
