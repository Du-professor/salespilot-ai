import os
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, SecretStr

PROJECT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_INDEX_PATH = PROJECT_DIR / ".salespilot" / "rag-index.sqlite3"
DEFAULT_MODELS_DIR = PROJECT_DIR / ".salespilot" / "models"


class RagSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    index_path: Path = DEFAULT_INDEX_PATH
    store_backend: Literal["sqlite", "postgres"] = "sqlite"
    postgres_dsn: SecretStr | None = None
    embedding_provider: Literal["hash", "openai", "local"] = "hash"
    embedding_model: str = "text-embedding-3-small"
    embedding_model_id: str = "workspace-default"
    models_dir: Path = DEFAULT_MODELS_DIR
    embedding_device: Literal["cpu", "cuda"] = "cpu"
    embedding_dimensions: int = Field(default=256, ge=8, le=4096)
    embedding_base_url: str | None = None
    embedding_api_key: SecretStr | None = None
    chat_provider: Literal["template", "openai"] = "template"
    chat_model: str = "gpt-4o-mini"
    chat_base_url: str | None = None
    chat_api_key: SecretStr | None = None
    timeout_seconds: float = Field(default=20.0, gt=0, le=300)
    max_retries: int = Field(default=2, ge=0, le=5)
    top_k: int = Field(default=8, ge=1, le=32)
    docling_artifacts: Path | None = None
    retrieval_strategy: Literal["hybrid", "vector", "bm25"] = "hybrid"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_base_url: str | None = None
    rerank_api_key: SecretStr | None = None

    @classmethod
    def from_environment(cls) -> Self:
        timeout = float(os.getenv("SALESPILOT_RAG_TIMEOUT_SECONDS", "20"))
        retries = int(os.getenv("SALESPILOT_RAG_MAX_RETRIES", "2"))
        top_k = int(os.getenv("SALESPILOT_RAG_TOP_K", "8"))
        artifacts = os.getenv("SALESPILOT_DOCLING_ARTIFACTS") or None
        models_dir = os.getenv("SALESPILOT_RAG_MODELS_DIR") or str(DEFAULT_MODELS_DIR)
        raw_dimensions = os.getenv("SALESPILOT_RAG_EMBEDDING_DIMENSIONS")
        extra: dict[str, int] = {}
        if raw_dimensions:
            extra["embedding_dimensions"] = int(raw_dimensions)
        return cls(
            index_path=os.getenv("SALESPILOT_RAG_INDEX_PATH", str(DEFAULT_INDEX_PATH)),
            store_backend=os.getenv("SALESPILOT_RAG_STORE_BACKEND", "sqlite"),
            postgres_dsn=os.getenv("SALESPILOT_POSTGRES_DSN") or None,
            embedding_provider=os.getenv("SALESPILOT_RAG_EMBEDDING_PROVIDER", "hash"),
            embedding_model=os.getenv("SALESPILOT_RAG_EMBEDDING_MODEL", "text-embedding-3-small"),
            embedding_model_id=os.getenv("SALESPILOT_RAG_EMBEDDING_MODEL_ID", "workspace-default"),
            models_dir=Path(models_dir),
            embedding_device=os.getenv("SALESPILOT_RAG_EMBEDDING_DEVICE", "cpu"),
            embedding_base_url=os.getenv("SALESPILOT_RAG_EMBEDDING_BASE_URL") or None,
            embedding_api_key=os.getenv("SALESPILOT_RAG_EMBEDDING_API_KEY") or None,
            chat_provider=os.getenv("SALESPILOT_RAG_CHAT_PROVIDER", "template"),
            chat_model=os.getenv("SALESPILOT_RAG_CHAT_MODEL", "gpt-4o-mini"),
            chat_base_url=os.getenv("SALESPILOT_RAG_CHAT_BASE_URL") or None,
            chat_api_key=os.getenv("SALESPILOT_RAG_CHAT_API_KEY") or None,
            timeout_seconds=timeout,
            max_retries=retries,
            top_k=top_k,
            docling_artifacts=Path(artifacts) if artifacts else None,
            retrieval_strategy=os.getenv("SALESPILOT_RAG_RETRIEVAL_STRATEGY", "hybrid"),
            rerank_model=os.getenv("SALESPILOT_RAG_RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
            rerank_base_url=os.getenv("SALESPILOT_RAG_RERANK_BASE_URL") or None,
            rerank_api_key=os.getenv("SALESPILOT_RAG_RERANK_API_KEY") or None,
            **extra,
        )