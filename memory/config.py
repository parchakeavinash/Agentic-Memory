import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application configuration settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    groq_api_key: str = Field(default="", description="API key for Groq Cloud")
    groq_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="LLM model identifier to use on Groq",
    )
    database_url: str = Field(
        default="sqlite:///agent_memory.db",
        description="Database connection URL (PostgreSQL or SQLite)",
    )
    memory_window: int = Field(
        default=10,
        description="Number of recent messages to keep in active short-term memory window",
    )
    session_id: str = Field(
        default="default",
        description="Default session identifier for conversation tracking",
    )


# Singleton settings instance
settings = Settings()
