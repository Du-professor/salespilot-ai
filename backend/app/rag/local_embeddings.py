import importlib.metadata
import math
import threading
from pathlib import Path

from langchain_core.embeddings import Embeddings

from app.contracts.rag_workspace import EmbeddingModelDescriptor, RagModelInvalid, RagModelUnavailable

from .config import RagSettings
from .model_catalog import ModelSpec, model_availability, model_directory, read_manifest

CHECK_TEXTS = (
    "模型检查：需要一台 GigE 工业相机，至少 8 MP，100 fps。",
    "SalesPilot local embedding check.",
)


def _dependency_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


class LocalTransformerEmbeddings(Embeddings):
    """Load a BGE-style dense encoder from a pinned local directory only."""

    def __init__(
        self,
        spec: ModelSpec,
        settings: RagSettings,
        *,
        models_dir: Path | None = None,
        device: str | None = None,
    ):
        self.spec = spec
        self.settings = settings
        self.directory = model_directory(spec, settings, models_dir)
        self.device = device or settings.embedding_device
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._manifest: dict | None = None
        self._lock = threading.Lock()

    def _require_local(self) -> None:
        ready, reason, code = model_availability(self.spec, self.settings, self.directory.parent)
        if not ready:
            raise RagModelUnavailable(reason or "本地模型不可用。", code=code or "embedding_model_unavailable")

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            self._require_local()
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer
            except ImportError as exc:
                raise RagModelUnavailable(
                    "本地 Embedding 依赖未安装。请安装 backend[local]。",
                    code="embedding_model_dependency_missing",
                ) from exc
            self._manifest = read_manifest(self.spec, self.settings, self.directory.parent)
            try:
                tokenizer = AutoTokenizer.from_pretrained(str(self.directory), local_files_only=True)
                model = AutoModel.from_pretrained(str(self.directory), local_files_only=True)
            except (OSError, ValueError, RuntimeError) as exc:
                raise RagModelUnavailable(
                    f"本地模型加载失败：{exc.__class__.__name__}。",
                    code="embedding_model_load_failed",
                ) from exc
            model.eval()
            if self.device != "cpu":
                try:
                    model.to(self.device)
                except (RuntimeError, AssertionError) as exc:
                    raise RagModelUnavailable(
                        f"模型无法加载到 {self.device}。", code="embedding_model_device_unavailable",
                    ) from exc
            self._torch = torch
            self._tokenizer = tokenizer
            self._model = model

    def _prepare_documents(self, texts: list[str]) -> list[str]:
        return [f"{self.spec.document_prefix}{text.strip() or ' '}" for text in texts]

    def _prepare_query(self, text: str) -> str:
        return f"{self.spec.query_prefix}{text.strip() or ' '}"

    def token_counts(self, texts: list[str]) -> list[int]:
        self._load()
        return [len(self._tokenizer(text, truncation=False)["input_ids"]) for text in texts]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        self._load()
        torch = self._torch
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.spec.batch_size):
            batch = texts[start:start + self.spec.batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.spec.max_tokens,
                return_tensors="pt",
            )
            if self.device != "cpu":
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                output = self._model(**encoded)
                hidden = output.last_hidden_state
                if self.spec.pooling == "mean":
                    mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                else:
                    pooled = hidden[:, 0]
                if self.spec.normalize:
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.extend(row.tolist() for row in pooled)
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(self._prepare_documents(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self._prepare_query(text)])[0]

    def check(self) -> EmbeddingModelDescriptor:
        vectors = self.embed_documents(list(CHECK_TEXTS))
        if len(vectors) != len(CHECK_TEXTS):
            raise RagModelInvalid("模型检查返回的向量数量不完整。", code="embedding_model_invalid")
        dimensions = {len(vector) for vector in vectors}
        if dimensions != {self.spec.dimensions}:
            raise RagModelInvalid(
                f"模型检查维度不一致：期望 {self.spec.dimensions}，实际 {sorted(dimensions)}。",
                code="embedding_dimension_mismatch",
            )
        if any(not math.isfinite(value) for vector in vectors for value in vector):
            raise RagModelInvalid("模型检查出现非有限向量值。", code="embedding_model_invalid")
        repeat = self.embed_documents([CHECK_TEXTS[0]])[0]
        if any(abs(left - right) > 1e-6 for left, right in zip(vectors[0], repeat, strict=True)):
            raise RagModelInvalid("模型检查结果不确定。", code="embedding_model_invalid")
        return self.descriptor()

    def descriptor(self) -> EmbeddingModelDescriptor:
        manifest = read_manifest(self.spec, self.settings, self.directory.parent) or {}
        files = manifest.get("files") if isinstance(manifest.get("files"), dict) else {}
        return EmbeddingModelDescriptor(
            model_id=self.spec.model_id,
            revision=self.spec.revision,
            dimensions=self.spec.dimensions,
            vector_config=self.spec.vector_config(),
            files={str(name): str(digest) for name, digest in files.items()},
            runtime={
                "transformers": _dependency_version("transformers"),
                "torch": _dependency_version("torch"),
                "pooling": self.spec.pooling,
                "device": self.device,
            },
            display_name=self.spec.display_name,
        )