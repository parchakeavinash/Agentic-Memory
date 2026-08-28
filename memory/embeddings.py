from abc import ABC, abstractmethod
from typing import List, Optional
import logging
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from memory.config import settings

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract interface for text embedding providers."""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Generates a dense vector embedding for the input text."""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Returns the dimensionality of the embedding vector."""
        pass


class GeminiEmbeddingProvider(EmbeddingProvider):
    """
    Google Gemini embedding provider using GoogleGenerativeAIEmbeddings.
    Default model: models/gemini-embedding-001 (3072 dimensions).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.api_key = api_key or settings.gemini_api_key
        self.model_name = model_name or settings.gemini_embedding_model
        self._dimension: Optional[int] = None

        if not self.api_key:
            logger.warning("GEMINI_API_KEY is not configured for GeminiEmbeddingProvider.")
            self._client = None
        else:
            try:
                self._client = GoogleGenerativeAIEmbeddings(
                    model=self.model_name,
                    google_api_key=self.api_key,
                )
                logger.info(f"Initialized GeminiEmbeddingProvider with model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize GeminiEmbeddingProvider: {e}")
                self._client = None

    def embed(self, text: str) -> List[float]:
        if not self._client:
            raise RuntimeError("GeminiEmbeddingProvider client is not initialized or API key is missing.")
        try:
            vec = self._client.embed_query(text)
            if self._dimension is None and vec:
                self._dimension = len(vec)
            return vec
        except Exception as e:
            logger.error(f"Error generating Gemini embedding: {e}")
            raise

    def get_dimension(self) -> int:
        if self._dimension is None:
            # Probe dimension with sample query
            sample_vec = self.embed("probe")
            self._dimension = len(sample_vec)
        return self._dimension
