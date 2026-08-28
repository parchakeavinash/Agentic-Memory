from typing import Optional, List
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from memory.config import settings
from memory.memory import ShortTermMemory
from memory.database import init_db


class MemoryAgent:
    """
    An AI conversational agent powered by Groq LLMs and equipped with Short-Term Memory.
    
    Workflow for each user turn:
    1. Retrieve the last K messages from DB (Sliding Window Short-Term Memory).
    2. Construct prompt payload: [System Prompt] + [Prior Conversation Window] + [Current User Message].
    3. Call the LLM to get a contextualized response.
    4. Save both the new user message and the LLM response back into the DB memory.
    5. Return the response to the caller.
    """

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        window_size: Optional[int] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        init_db()

        self.api_key = api_key or settings.groq_api_key
        self.model_name = model_name or settings.groq_model
        self.memory = ShortTermMemory(default_window_size=window_size)

        self.system_prompt = system_prompt or (
            "You are a helpful and intelligent AI assistant with short-term conversational memory. "
            "You remember facts, preferences, and context mentioned earlier in the active conversation window."
        )

        self.llm = ChatGroq(
            groq_api_key=self.api_key,
            model_name=self.model_name,
            temperature=0.7,
        )

        self.last_prompt_debug = None

    def chat(self, user_input: str, session_id: Optional[str] = None) -> str:
        """
        Processes a single conversational turn with short-term memory.
        """
        session_id = session_id or settings.session_id

        # 1. Fetch short-term memory window (last K messages) & summary
        history: List[BaseMessage] = self.memory.get_messages(session_id=session_id)
        summary: Optional[str] = self.memory.get_summary(session_id=session_id)

        # 2. Build the complete prompt sequence
        messages: List[BaseMessage] = [SystemMessage(content=self.system_prompt)]

        if summary:
            summary_prompt = (
                "Summary of earlier conversation (condensed context):\n"
                f"{summary}"
            )
            messages.append(SystemMessage(content=summary_prompt))

        messages.extend(history)
        messages.append(HumanMessage(content=user_input))

        # Record prompt debug metadata
        stats = self.memory.get_memory_stats(session_id=session_id)
        self.last_prompt_debug = {
            "session_id": session_id,
            "total_stored_messages": stats["total_stored_messages"],
            "window_size": self.memory.default_window_size,
            "has_summary": bool(summary),
            "summary_text": summary,
            "history_injected_count": len(history),
            "total_prompt_items": len(messages),
            "messages": messages,
        }

        # 3. Invoke LLM
        response = self.llm.invoke(messages)
        raw_text = str(response.content)

        # Strip internal <think>...</think> reasoning traces if present (e.g. Qwen/DeepSeek)
        import re
        response_text = re.sub(r"<think>.*?</think>\s*", "", raw_text, flags=re.DOTALL).strip()
        if not response_text:
            response_text = raw_text

        # 4. Save this turn into short-term memory persistence
        self.memory.add_user_message(session_id=session_id, content=user_input)
        self.memory.add_ai_message(session_id=session_id, content=response_text)

        # 5. Incrementally update running summary for messages outside the sliding window
        self.memory.update_summary_if_needed(
            session_id=session_id,
            llm=self.llm,
            window_size=self.memory.default_window_size,
        )

        return response_text

    def get_summary(self, session_id: Optional[str] = None) -> Optional[str]:
        """Retrieves the running conversation summary for the session."""
        session_id = session_id or settings.session_id
        return self.memory.get_summary(session_id=session_id)

    def get_last_prompt_debug(self) -> Optional[dict]:
        """Returns the debug info for the most recent prompt payload sent to the LLM."""
        return self.last_prompt_debug

    def get_history(self, session_id: Optional[str] = None):
        """Retrieves raw message history for inspection."""
        session_id = session_id or settings.session_id
        return self.memory.get_raw_messages(session_id=session_id)

    def clear_memory(self, session_id: Optional[str] = None) -> int:
        """Resets the memory for the session."""
        session_id = session_id or settings.session_id
        return self.memory.clear_memory(session_id=session_id)

    def get_stats(self, session_id: Optional[str] = None) -> dict:
        """Returns memory statistics."""
        session_id = session_id or settings.session_id
        return self.memory.get_memory_stats(session_id=session_id)
