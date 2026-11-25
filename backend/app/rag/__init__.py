from .chat import ChatModel, OpenAICompatibleChat
from .config import RagSettings
from .embeddings import HashingEmbeddings, OpenAICompatibleEmbeddings
from .engine import SqliteRagEngine, build_rag_engine, build_rag_store
from .local_embeddings import LocalTransformerEmbeddings
from .model_catalog import BGE_M3, BGE_SMALL_ZH, LOCAL_MODELS, ModelSpec, model_options, resolve_model_spec
from .model_registry import RagModelRegistry
from .pg_store import PgVectorRagStore

__all__ = [
    "BGE_M3",
    "BGE_SMALL_ZH",
    "ChatModel",
    "HashingEmbeddings",
    "LOCAL_MODELS",
    "LocalTransformerEmbeddings",
    "ModelSpec",
    "OpenAICompatibleChat",
    "OpenAICompatibleEmbeddings",
    "PgVectorRagStore",
    "RagModelRegistry",
    "RagSettings",
    "SqliteRagEngine",
    "build_rag_engine",
    "build_rag_store",
    "model_options",
    "resolve_model_spec",
]