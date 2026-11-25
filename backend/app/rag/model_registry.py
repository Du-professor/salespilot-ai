import threading
import time
from collections.abc import Callable
from pathlib import Path

from langchain_core.embeddings import Embeddings

from app.contracts.rag_workspace import EmbeddingModelDescriptor, RagModelInvalid, RagModelUnavailable

from .config import RagSettings
from .local_embeddings import LocalTransformerEmbeddings
from .model_catalog import (
    LOCAL_MODELS,
    WORKSPACE_MODEL_ID,
    ModelSpec,
    manifest_path,
    model_availability,
    model_directory,
    resolve_model_spec,
)

RemoteEmbeddingResolver = Callable[[str], tuple[EmbeddingModelDescriptor, Embeddings] | None]


class RagModelRegistry:
    """Resolve one embedding model per knowledge base without mixing vectors."""

    def __init__(
        self,
        settings: RagSettings,
        default_embedder: Embeddings,
        *,
        models_dir: Path | None = None,
        local_factory=None,
        error_ttl_seconds: float = 30.0,
        remote_resolver: RemoteEmbeddingResolver | None = None,
    ):
        self.settings = settings
        self.default_embedder = default_embedder
        self.models_dir = models_dir
        self._local_factory = local_factory
        self._remote_resolver = remote_resolver
        self.error_ttl_seconds = max(0.0, float(error_ttl_seconds))
        self._embedders: dict[str, Embeddings] = {WORKSPACE_MODEL_ID: default_embedder}
        if settings.embedding_provider == "local" and settings.embedding_model_id in LOCAL_MODELS:
            self._embedders[settings.embedding_model_id] = default_embedder
        self._checked: dict[str, EmbeddingModelDescriptor] = {}
        self._errors: dict[str, tuple[float, tuple[bool, int | None, int | None], RagModelUnavailable]] = {}
        self._lock = threading.Lock()

    def _resolve_remote(self, model_id: str) -> tuple[EmbeddingModelDescriptor, Embeddings] | None:
        if self._remote_resolver is None or not model_id.startswith("cfg-"):
            return None
        return self._remote_resolver(model_id)

    def _model_signature(self, model_id: str) -> tuple[bool, int | None, int | None]:
        try:
            spec = self.spec(model_id)
        except RagModelUnavailable:
            return (False, None, None)
        if spec.provider == "workspace":
            return (True, None, None)
        directory = model_directory(spec, self.settings, self.models_dir)
        manifest = manifest_path(spec, self.settings, self.models_dir)
        dir_mtime = directory.stat().st_mtime_ns if directory.is_dir() else None
        manifest_mtime = manifest.stat().st_mtime_ns if manifest.is_file() else None
        return (directory.is_dir(), dir_mtime, manifest_mtime)

    def spec(self, model_id: str) -> ModelSpec:
        return resolve_model_spec(model_id, self.settings)

    def embedder(self, model_id: str) -> Embeddings:
        remote = self._resolve_remote(model_id)
        if remote is not None:
            return remote[1]
        if model_id.startswith("cfg-"):
            raise RagModelUnavailable(
                f"远程 Embedding 配置不可用：{model_id}。", code="embedding_model_unavailable", retryable=True,
            )
        with self._lock:
            existing = self._embedders.get(model_id)
            if existing is not None:
                return existing
            spec = self.spec(model_id)
            if spec.provider == "workspace":
                return self.default_embedder
            embedder = self._build_local(spec)
            self._embedders[model_id] = embedder
            return embedder

    def _build_local(self, spec: ModelSpec) -> Embeddings:
        if self._local_factory is not None:
            return self._local_factory(spec)
        return LocalTransformerEmbeddings(spec, self.settings, models_dir=self.models_dir)

    def descriptor(self, model_id: str) -> EmbeddingModelDescriptor:
        remote = self._resolve_remote(model_id)
        if remote is not None:
            return remote[0]
        if model_id.startswith("cfg-"):
            raise RagModelUnavailable(
                f"远程 Embedding 配置不可用：{model_id}。", code="embedding_model_unavailable", retryable=True,
            )
        spec = self.spec(model_id)
        if spec.provider == "workspace":
            dimensions = getattr(self.default_embedder, "dimensions", None) or self.settings.embedding_dimensions
            embedder_spec = getattr(self.default_embedder, "spec", None)
            max_tokens = int(getattr(embedder_spec, "max_tokens", 0) or 0)
            return EmbeddingModelDescriptor(
                model_id=WORKSPACE_MODEL_ID,
                revision=spec.revision,
                dimensions=int(dimensions),
                vector_config={
                    "provider": self.settings.embedding_provider,
                    "normalize": True,
                    "max_tokens": max_tokens,
                    "query_prefix": "",
                    "document_prefix": "",
                    "dimensions": int(dimensions),
                },
                runtime={"provider": self.settings.embedding_provider},
                display_name=spec.display_name,
            )
        embedder = self.embedder(model_id)
        describe = getattr(embedder, "descriptor", None)
        if describe is None:
            return spec.descriptor()
        return describe()

    def availability(self, model_id: str) -> tuple[bool, str | None, str | None]:
        if model_id.startswith("cfg-"):
            remote = self._resolve_remote(model_id)
            if remote is not None:
                return True, None, None
            return False, "远程 Embedding 配置未通过测试或已移除。", "embedding_model_unavailable"
        try:
            spec = self.spec(model_id)
        except RagModelUnavailable as exc:
            return False, str(exc), exc.code
        if spec.provider == "workspace":
            return True, None, None
        ready, reason, code = model_availability(spec, self.settings, self.models_dir)
        if not ready:
            return False, reason, code
        try:
            self.check(model_id)
        except RagModelUnavailable as exc:
            return False, str(exc), exc.code
        return True, None, None

    def check(self, model_id: str) -> EmbeddingModelDescriptor:
        if model_id.startswith("cfg-"):
            remote = self._resolve_remote(model_id)
            if remote is None:
                raise RagModelUnavailable(
                    "远程 Embedding 配置未通过测试或已移除。", code="embedding_model_unavailable", retryable=True,
                )
            return remote[0]
        signature = self._model_signature(model_id)
        with self._lock:
            cached = self._checked.get(model_id)
            if cached is not None:
                return cached
            failure_entry = self._errors.get(model_id)
            if failure_entry is not None:
                recorded_at, recorded_sig, failure = failure_entry
                if (time.monotonic() - recorded_at) < self.error_ttl_seconds and recorded_sig == signature:
                    raise failure
                self._errors.pop(model_id, None)
        try:
            descriptor = self.descriptor(model_id)
            spec = self.spec(model_id)
            if spec.provider == "local":
                check = getattr(self.embedder(model_id), "check", None)
                if check is None:
                    raise RagModelInvalid("本地模型缺少自检入口。", code="embedding_model_invalid")
                descriptor = check()
        except RagModelUnavailable as exc:
            with self._lock:
                self._errors[model_id] = (time.monotonic(), self._model_signature(model_id), exc)
            raise
        with self._lock:
            self._errors.pop(model_id, None)
            self._checked[model_id] = descriptor
        return descriptor

    def clear_check(self, model_id: str | None = None) -> None:
        with self._lock:
            if model_id is None:
                self._checked.clear()
                self._errors.clear()
            else:
                self._checked.pop(model_id, None)
                self._errors.pop(model_id, None)

    def check_workspace_default(self) -> EmbeddingModelDescriptor:
        descriptor = self.descriptor(WORKSPACE_MODEL_ID)
        if self.settings.embedding_provider == "hash":
            check = getattr(self.default_embedder, "embed_query", None)
            if check is None:
                raise RagModelInvalid("默认 Embedding 缺少向量入口。", code="embedding_model_invalid")
            vector = check("SalesPilot workspace embedding check")
            if len(vector) != descriptor.dimensions or any(not isinstance(value, (int, float)) for value in vector):
                raise RagModelInvalid("默认 Embedding 自检失败。", code="embedding_model_invalid")
        with self._lock:
            self._checked[WORKSPACE_MODEL_ID] = descriptor
        return descriptor