import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AliasChoices


class Settings(BaseSettings):
    """Application configuration settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    groq_api_key: str = Field(default="", description="API key for Groq Cloud")
    groq_model: str = Field(
        default="openai/gpt-oss-20b",
        description="LLM model identifier to use on Groq",
    )

    # Google Gemini Embedding Settings
    gemini_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        description="API key for Google Gemini GenAI",
    )
    gemini_embedding_model: str = Field(
        default="models/gemini-embedding-001",
        validation_alias=AliasChoices("GEMINI_EMBEDDING_MODEL", "GOOGLE_EMBEDDING_MODEL"),
        description="Google embedding model identifier",
    )

    database_url: str = Field(
        default="sqlite:///agent_memory.db",
        description="Database connection URL (PostgreSQL or SQLite)",
    )
    memory_window: int = Field(
        default=10,
        description="Number of recent messages to keep in active short-term memory window",
    )

    # Multi-User & Session Identifiers
    user_id: str = Field(
        default="default_user",
        description="Current user identifier for multi-tenant data isolation",
    )
    session_id: str = Field(
        default="default",
        description="Default session identifier for conversation tracking",
    )

    # Episodic Memory Search Configuration
    episodic_top_k: int = Field(
        default=3,
        description="Number of top relevant episodes to retrieve",
    )
    episodic_min_similarity: float = Field(
        default=0.50,
        description="Minimum cosine similarity threshold for episodic retrieval",
    )


# Singleton settings instance
settings = Settings()
