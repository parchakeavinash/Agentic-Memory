# Agent Memory System — Architecture, Implementation & Integration Guide
### still working on to improve the accuracy and latency
> **Target Audience:** Future AI Agents, Developers, and System Architects integrating this memory subsystem into existing assistants (e.g., Voice Assistants, Chatbots, Autonomous Agents).

---

## 1. Executive Summary

This repository implements a **Production-Grade, 4-Tier Memory System** for AI assistants:

```
                          ┌───────────────────────────┐
                          │   Current User Message    │
                          └─────────────┬─────────────┘
                                        ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │                        PROMPT INJECTION ENGINE                         │
   │                                                                        │
   │  1. System Prompt     (Conversational rules, concise persona)          │
   │  2. Semantic Memory   ("What I know about you: - Fact: Value")         │
   │  3. Running Summary   ("Summary of earlier conversation...")           │
   │  4. Episodic Memory   ("Relevant Past Episodes: Summary/Events/Topics")│
   │  5. Short-Term Window (Recent N messages, sliding buffer)              │
   │  6. Current User Turn                                                  │
   └────────────────────────────────────┬───────────────────────────────────┘
                                        ▼
                          ┌───────────────────────────┐
                          │    LLM (Groq temp=0.7)    │
                          └─────────────┬─────────────┘
                                        ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │                        POST-TURN BACKGROUND PIPELINE                   │
   │                                                                        │
   │  • Save turn to ChatMessage table                                      │
   │  • Detect & Extract Semantic Facts (LLM temp=0) → pgvector / SQLite    │
   │  • Incrementally Update Running Summary if window evicted              │
   │  • Periodic / Triggered Episode Distillation (LLM temp=0)              │
   └────────────────────────────────────────────────────────────────────────┘
```

The system solves the fundamental limitations of standard conversational agents:
1. **Context Limits:** Solved via sliding window + running summarization.
2. **Long-Term Recall:** Solved via vector similarity search over historical episodes (Episodic Memory).
3. **Cross-Session Personalization:** Solved via user-scoped persistent facts (Semantic Memory).
4. **Hallucination Prevention:** Solved via dual-LLM temperature separation and strict negative-constraint extraction prompts.

---

## 2. The 4 Memory Layers Explained

### Tier 1: Short-Term Memory (Sliding Buffer)
- **Model:** `ChatMessage` (table: `chat_messages`)
- **Scope:** `(user_id, session_id)`
- **Behavior:** Stores raw user and assistant messages with exact UTC timestamps.
- **Buffer Size:** Configurable via `memory_window` (default: 10 messages).
- **Function:** Injects the last $N$ turns into the prompt for immediate conversational coherence.

### Tier 2: Summary Memory (Running Eviction Summary)
- **Model:** `ConversationSummary` (table: `conversation_summaries`)
- **Scope:** `(user_id, session_id)`
- **Behavior:** When the active messages in a session exceed `memory_window`, older messages are evicted. An incremental summary is generated or updated using `extraction_llm` (`temperature=0`).
- **Function:** Injects a condensed context of earlier turns in the current session without blowing up the context window.

### Tier 3: Episodic Memory (Experience & Event Distillation)
- **Model:** `Episode` (table: `episodes`)
- **Scope:** `user_id` (cross-session or session-scoped)
- **Provenance Tracking:** Records `start_message_id`, `end_message_id`, and original conversation `timestamp`.
- **Payload Schema:**
  - `summary`: 1–2 factual sentences describing what the user asked or discussed.
  - `events`: Array of user-centric events (e.g. `["User asked about hydration benefits", "User asked about daily intake"]`).
  - `topics`: Array of keyword tags (e.g. `["hydration", "wellness", "water_intake"]`).
  - `embedding`: 3072-dimensional vector embedding.
- **Extraction Rules (Strict Grounding):**
  - Uses `extraction_llm` (`temperature=0`).
  - Strict rule: **ONLY** extract what the user explicitly said or asked.
  - Assistant replies and external medical/general knowledge are strictly excluded from events and topics to prevent memory contamination.
- **Search & Retrieval:**
  - On every user message, the query is embedded via Gemini (`gemini-embedding-001`).
  - Vector similarity search retrieves top-$K$ episodes matching `user_id` with cosine similarity $\ge 0.50$.
  - Injected into the prompt as:
    ```
    Relevant Episodic Memory (use only if relevant to the current question):

    Episode [2026-08-27]
    Summary: The user discussed daily water consumption and hydration habits.
    Events:
    - User asked about benefits of drinking enough water
    - User asked about recommended daily intake
    Topics: hydration, wellness
    ```

### Tier 4: Semantic Memory (Persistent User Facts)
- **Model:** `SemanticFact` (table: `semantic_facts`)
- **Scope:** **Strictly `user_id` scoped** (NOT session-scoped). A fact known about a user in Session A persists in Session B, C, etc.
- **Examples:**
  - `favorite_programming_language: Python`
  - `name: Avi`
  - `current_project: AI voice agent with STT and TTS`
  - `preferred_database: PostgreSQL`
- **Two-Step Extraction Pipeline:**
  1. **Classification:** Binary `YES`/`NO` check: "Does the user message contain a personal fact about the user?" (ignores generic queries like "What is 2+2?" or "Explain Python lists").
  2. **Extraction:** Formats key-value pair with normalized key.
  3. **Idempotency:**
     - **CREATE:** If new key/fact.
     - **UPDATE:** If user updates value (e.g. "I switched to Rust").
     - **IGNORE:** If duplicate value already stored.
  4. **Embedding:** Embeds `"key: value"` into 3072 dimensions.
- **Retrieval & Injection:**
  - Retrieved via cosine similarity against the user query ($\ge 0.45$).
  - Injected at the top of the prompt:
    ```
    What I know about you:
    - Favorite Programming Language: Python
    - Current Project: AI voice agent
    ```

---

## 3. Database Architecture & Vector Strategy

### Tables
| Table Name | Primary Key | Scoping Keys | Vector Column | Purpose |
|---|---|---|---|---|
| `chat_messages` | `id` (int) | `user_id`, `session_id` | N/A | Raw message stream |
| `conversation_summaries` | `session_id` | `user_id` | N/A | Incremental evicted summary |
| `episodes` | `id` (int) | `user_id`, `session_id` | `embedding` (3072d) | Distilled experiences & events |
| `semantic_facts` | `id` (int) | `user_id`, `key` (unique) | `embedding` (3072d) | Long-term user traits & facts |

### Robust Vector Column (`VectorType`)
Defined in `memory/models.py`:
- Designed to work seamlessly with **PostgreSQL (`pgvector`)** when available.
- Automatically falls back to **JSON arrays** on SQLite or standard PostgreSQL without native C extension compilation.
- Handles lists, numpy arrays, and JSON serialization transparently.

### Engine Resilience
Defined in `memory/database.py`:
- Parses `postgresql://` and translates to `postgresql+psycopg://`.
- If the PostgreSQL instance is unreachable (e.g. during local offline development), it automatically logs a warning and falls back to `sqlite:///agent_memory.db`.

---

## 4. Temperature & Dual-LLM Separation

A critical design choice implemented in this project is the **split LLM architecture**:

```python
# 1. Conversational LLM: temperature = 0.7
# Used for: Human-like, engaging, concise conversational turns
self.llm = ChatGroq(
    groq_api_key=self.api_key,
    model_name=self.model_name,
    temperature=0.7,
)

# 2. Extraction LLM: temperature = 0.0
# Used for: Fact classification, fact extraction, running summary, episode distillation
# Guarantees: Deterministic outputs, strict JSON formatting, zero hallucination
self.extraction_llm = ChatGroq(
    groq_api_key=self.api_key,
    model_name=self.model_name,
    temperature=0.0,
)
```

---

## 5. File-by-File Codebase Map

| File Path | Role | Key Components |
|---|---|---|
| `memory/models.py` | Database Schema | `VectorType`, `ChatMessage`, `ConversationSummary`, `Episode`, `SemanticFact` |
| `memory/config.py` | Configuration | Pydantic `Settings` loading from `.env` (`GROQ_API_KEY`, `GEMINI_API_KEY`, `DATABASE_URL`) |
| `memory/database.py` | DB Engine & Migrations | `init_db()`, `get_db()` context manager, PostgreSQL/SQLite fallback, `CREATE EXTENSION vector` |
| `memory/embeddings.py` | Vector Embeddings | `EmbeddingProvider` interface, `GeminiEmbeddingProvider` (Google `gemini-embedding-001`, 3072d) |
| `memory/memory.py` | Short-Term & Summary | `ShortTermMemory` class: sliding window buffer, eviction management, incremental summarizer |
| `memory/episodic.py` | Episodic Memory Manager | `EpisodicMemoryManager`: episode distillation, vector similarity search, cosine similarity |
| `memory/semantic.py` | Semantic Memory Manager | `SemanticMemoryManager`: fact classification, extraction, CRUD idempotency, vector search |
| `memory/agent.py` | Master Orchestrator | `MemoryAgent`: manages prompt injection pipeline, invokes LLM, triggers post-turn updates |
| `memory/cli.py` | Interactive CLI | Rich CLI with commands: `/facts`, `/forget`, `/episodes`, `/save_episode`, `/summary`, `/stats` |
| `requirements.txt` | Python Dependencies | `SQLAlchemy`, `psycopg[binary]`, `pgvector`, `langchain-groq`, `langchain-google-genai`, `numpy` |

---
