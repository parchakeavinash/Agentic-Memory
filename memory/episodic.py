import json
import logging
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

import numpy as np
from langchain_core.messages import SystemMessage, HumanMessage

from memory.config import settings
from memory.database import get_db
from memory.embeddings import EmbeddingProvider, GeminiEmbeddingProvider
from memory.models import Episode, ChatMessage

logger = logging.getLogger(__name__)


def compute_cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Computes cosine similarity between two float vectors."""
    a = np.array(v1, dtype=np.float32)
    b = np.array(v2, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class EpisodicMemoryManager:
    """
    Manages episodic memory:
    - Decoupled EmbeddingProvider (e.g. GeminiEmbeddingProvider)
    - Distills conversations into structured episodes with provenance (start_id, end_id, timestamp)
    - Stores episodes in PostgreSQL with pgvector / vector embeddings
    - Executes vector similarity search with strict multi-user data isolation
    """

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        default_top_k: Optional[int] = None,
        default_min_similarity: Optional[float] = None,
    ):
        self.embedding_provider = embedding_provider or GeminiEmbeddingProvider()
        self.default_top_k = default_top_k or settings.episodic_top_k
        self.default_min_similarity = default_min_similarity or settings.episodic_min_similarity

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generates embedding vector via the configured EmbeddingProvider."""
        try:
            return self.embedding_provider.embed(text)
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None

    def extract_episode_from_conversation(
        self,
        messages: List[ChatMessage],
        llm,
        session_id: str,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Uses an LLM to distill a sequence of conversation messages into a structured episode.
        Captures start_message_id, end_message_id, and original conversation timestamp.
        """
        if not messages:
            return None

        user_id = user_id or settings.user_id
        start_msg_id = messages[0].id
        end_msg_id = messages[-1].id
        convo_timestamp = messages[0].created_at

        # Format conversation transcript
        convo_lines = []
        for m in messages:
            speaker = "User" if m.role == "user" else "Assistant" if m.role == "assistant" else "System"
            convo_lines.append(f"{speaker}: {m.content}")
        transcript = "\n".join(convo_lines)

        prompt = [
            SystemMessage(
                content=(
                    "You are a memory extraction system. Your job is to extract a factual record "
                    "of what was discussed in the conversation transcript below.\n\n"
                    "STRICT RULES:\n"
                    "1. ONLY extract information that is explicitly present in the transcript.\n"
                    "2. DO NOT add, infer, expand, or use any knowledge from outside the transcript.\n"
                    "3. If something was not said in the transcript, do not include it.\n"
                    "4. Events must describe what the USER asked or said — not what the assistant answered.\n"
                    "5. DO NOT summarize assistant replies or add general knowledge the assistant mentioned.\n"
                    "6. Topics must only reflect subjects the user explicitly raised.\n\n"
                    "Good event example: 'User asked whether drinking too much water can be harmful.'\n"
                    "Bad event example:  'Hot or cold drinks can irritate the stomach or throat.' "
                    "(← this was not said by the user; do not add it)\n\n"
                    "Output MUST be valid JSON with this exact schema:\n"
                    "{\n"
                    '  "summary": "1-2 sentences describing only what the user asked or discussed.",\n'
                    '  "events": ["User asked X", "User mentioned Y"],\n'
                    '  "topics": ["tag1", "tag2"]\n'
                    "}\n"
                    "Do NOT include markdown backticks or extra explanation. Return ONLY raw JSON."
                )
            ),
            HumanMessage(
                content=(
                    f"Conversation transcript (User: {user_id}, Session: {session_id}):\n\n"
                    f"{transcript}\n\n"
                    "Extract the episode JSON using ONLY what appears in the transcript above:"
                )
            ),
        ]

        try:
            response = llm.invoke(prompt)
            raw_text = str(response.content).strip()

            # Clean markdown json fences if present
            if raw_text.startswith("```"):
                raw_text = raw_text.split("\n", 1)[-1]
            if raw_text.endswith("```"):
                raw_text = raw_text.rsplit("```", 1)[0]
            raw_text = raw_text.strip()

            data = json.loads(raw_text)
            data["start_message_id"] = start_msg_id
            data["end_message_id"] = end_msg_id
            data["timestamp"] = convo_timestamp
            return data
        except Exception as e:
            logger.warning(f"Failed to extract structured episode: {e}")
            return None

    def store_episode(
        self,
        session_id: str,
        summary: str,
        events: List[str],
        topics: List[str],
        user_id: Optional[str] = None,
        start_message_id: Optional[int] = None,
        end_message_id: Optional[int] = None,
        episode_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> Episode:
        """
        Generates embedding and persists an episode to PostgreSQL with user isolation.
        """
        uid = user_id or settings.user_id
        ep_id = episode_id or f"ep_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        ts = timestamp or datetime.utcnow()

        # Build embedding content representation
        embed_payload = f"Summary: {summary}\nEvents: {' | '.join(events)}\nTopics: {', '.join(topics)}"
        vector = self.generate_embedding(embed_payload)

        with get_db() as db:
            episode = Episode(
                episode_id=ep_id,
                user_id=uid,
                session_id=session_id,
                timestamp=ts,
                start_message_id=start_message_id,
                end_message_id=end_message_id,
                summary=summary,
                events=events,
                topics=topics,
                embedding=vector,
            )
            db.add(episode)
            db.flush()
            db.refresh(episode)
            logger.info(f"Saved episode '{ep_id}' (User: '{uid}') with {len(events)} events and {len(topics)} topics.")
            return episode

    def search_episodes(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Performs vector similarity search against stored episodic memories
        with strict multi-user data isolation.
        """
        uid = user_id or settings.user_id
        k = top_k if top_k is not None else self.default_top_k
        threshold = min_similarity if min_similarity is not None else self.default_min_similarity

        query_vector = self.generate_embedding(query)
        if not query_vector:
            logger.warning("Unable to generate query embedding for episodic search.")
            return []

        with get_db() as db:
            # Strictly filter by user_id to prevent cross-user data leaks
            query_obj = db.query(Episode).filter(Episode.user_id == uid)
            if session_id:
                query_obj = query_obj.filter(Episode.session_id == session_id)
            episodes = query_obj.order_by(Episode.timestamp.desc()).all()

        scored_results = []
        for ep in episodes:
            if not ep.embedding:
                continue
            sim = compute_cosine_similarity(query_vector, ep.embedding)
            if sim >= threshold:
                scored_results.append({
                    "episode_id": ep.episode_id,
                    "user_id": ep.user_id,
                    "session_id": ep.session_id,
                    "timestamp": ep.timestamp.isoformat() if ep.timestamp else None,
                    "created_at": ep.created_at.isoformat() if ep.created_at else None,
                    "start_message_id": ep.start_message_id,
                    "end_message_id": ep.end_message_id,
                    "summary": ep.summary,
                    "events": ep.events,
                    "topics": ep.topics,
                    "similarity": round(sim, 4),
                })

        # Rank by cosine similarity descending
        scored_results.sort(key=lambda x: x["similarity"], reverse=True)
        return scored_results[:k]

    def list_episodes(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Episode]:
        """Retrieves raw episode records for the user."""
        uid = user_id or settings.user_id
        with get_db() as db:
            q = db.query(Episode).filter(Episode.user_id == uid)
            if session_id:
                q = q.filter(Episode.session_id == session_id)
            return q.order_by(Episode.timestamp.desc()).limit(limit).all()

    def clear_episodes(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> int:
        """Deletes stored episodes for the user."""
        uid = user_id or settings.user_id
        with get_db() as db:
            q = db.query(Episode).filter(Episode.user_id == uid)
            if session_id:
                q = q.filter(Episode.session_id == session_id)
            return q.delete()
