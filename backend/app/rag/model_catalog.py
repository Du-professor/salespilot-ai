import hashlib
import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from app.contracts.rag_workspace import EmbeddingModelDescriptor, RagModelUnavailable

from .config import DEFAULT_MODELS_DIR, RagSettings

MANIFEST_NAME = "model_manifest.json"
WORKSPACE_MODEL_ID = "workspace-default"
WORKSPACE_REVISION = "workspace"


class ModelSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str
    display_name: str
    provider: Literal["workspace", "local"]
    revision: str
    dimensions: int
    max_tokens: int
    query_prefix: str = ""
    document_prefix: str = ""
    normalize: bool = True
    pooling: Literal["cls", "mean"] = "cls"
    batch_size: int = 8
    directory_name: str | None = None
    repo_id: str | None = None

    def vector_config(self) -> dict[str, str | int | float | bool]:
        return {
            "provider": self.provider,
            "pooling": self.pooling,
            "normalize": self.normalize,
            "max_tokens": self.max_tokens,
            "query_prefix": self.query_prefix,
            "document_prefix": self.document_prefix,
            "dimensions": self.dimensions,
        }

    def descriptor(self) -> EmbeddingModelDescriptor:
        return EmbeddingModelDescriptor(
            model_id=self.model_id,
            revision=self.revision,
            dimensions=self.dimensions,
            vector_config=self.vector_config(),
            display_name=self.display_name,
        )


BGE_M3 = ModelSpec(
    model_id="bge-m3",
    display_name="BGE-M3",
    provider="local",
    revision="5617a9f61b028005a4858fdac845db406aefb181",
    dimensions=1024,
    max_tokens=8192,
    query_prefix="",
    document_prefix="",
    normalize=True,
    pooling="cls",
    batch_size=8,
    directory_name="bge-m3",
    repo_id="BAAI/bge-m3",
)

BGE_SMALL_ZH = ModelSpec(
    model_id="bge-small-zh-v1.5",
    display_name="BGE Small 中文",
    provider="local",
    revision="7999e1d3359715c523056ef9478215996d62a620",
    dimensions=512,
    max_tokens=512,
    query_prefix="为这个句子生成表示以用于检索相关文章：",
    document_prefix="",
    normalize=True,
    pooling="cls",
    batch_size=32,
    directory_name="bge-small-zh-v1.5",
    repo_id="BAAI/bge-small-zh-v1.5",
)

LOCAL_MODELS: dict[str, ModelSpec] = {
    BGE_M3.model_id: BGE_M3,
    BGE_SMALL_ZH.model_id: BGE_SMALL_ZH,
}


def models_directory(settings: RagSettings, override: Path | None = None) -> Path:
    if override is not None:
        return override
    configured = os.getenv("SALESPILOT_RAG_MODELS_DIR")
    if configured:
        return Path(configured)
    return settings.models_dir if settings.models_dir != DEFAULT_MODELS_DIR else DEFAULT_MODELS_DIR


def workspace_spec(settings: RagSettings) -> ModelSpec:
    return ModelSpec(
        model_id=WORKSPACE_MODEL_ID,
        display_name="工作区默认配置",
        provider="workspace",
        revision=WORKSPACE_REVISION,
        dimensions=settings.embedding_dimensions,
        max_tokens=0,
        normalize=True,
        pooling="cls",
        batch_size=32,
    )


def resolve_model_spec(model_id: str, settings: RagSettings) -> ModelSpec:
    if model_id == WORKSPACE_MODEL_ID:
        return workspace_spec(settings)
    spec = LOCAL_MODELS.get(model_id)
    if spec is None:
        raise RagModelUnavailable(
            f"未登记的 Embedding 模型：{model_id}。", code="embedding_model_unavailable", retryable=False,
        )
    return spec


def model_directory(spec: ModelSpec, settings: RagSettings, override: Path | None = None) -> Path:
    root = models_directory(settings, override)
    return root / (spec.directory_name or spec.model_id)


def manifest_path(spec: ModelSpec, settings: RagSettings, override: Path | None = None) -> Path:
    return model_directory(spec, settings, override) / MANIFEST_NAME


def read_manifest(spec: ModelSpec, settings: RagSettings, override: Path | None = None) -> dict | None:
    path = manifest_path(spec, settings, override)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def required_model_files(manifest: dict | None = None) -> tuple[str, ...]:
    files = {"config.json", "tokenizer_config.json"}
    if manifest and isinstance(manifest.get("files"), dict):
        weight_files = [
            name for name in manifest["files"]
            if name.endswith(".safetensors") or name.endswith(".bin")
        ]
        if weight_files:
            files.update(weight_files)
        else:
            files.add("model.safetensors")
    else:
        files.update({"model.safetensors", "tokenizer.json"})
    return tuple(sorted(files))


def model_availability(
    spec: ModelSpec,
    settings: RagSettings,
    override: Path | None = None,
) -> tuple[bool, str | None, str | None]:
    """Return (ready, reason, error_code) without importing the inference stack."""
    if spec.provider == "workspace":
        return True, None, None
    directory = model_directory(spec, settings, override)
    if not directory.is_dir():
        return False, f"本地模型目录缺失：{directory}", "embedding_model_directory_missing"
    manifest = read_manifest(spec, settings, override)
    if manifest is None:
        return False, f"本地模型清单缺失：{directory / MANIFEST_NAME}", "embedding_model_incomplete"
    if manifest.get("revision") != spec.revision:
        return (
            False,
            f"模型权重 revision 不匹配：期望 {spec.revision}，实际 {manifest.get('revision')}",
            "embedding_model_invalid",
        )
    missing = [name for name in required_model_files(manifest) if not (directory / name).is_file()]
    if missing:
        return False, f"本地模型文件不完整：{', '.join(missing)}", "embedding_model_incomplete"
    return True, None, None


def model_options(settings: RagSettings, override: Path | None = None) -> list[tuple[str, str, bool]]:
    options = [(WORKSPACE_MODEL_ID, "工作区默认配置", True)]
    for spec in LOCAL_MODELS.values():
        ready, _, _ = model_availability(spec, settings, override)
        options.append((spec.model_id, spec.display_name, ready))
    return options


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(
    spec: ModelSpec,
    directory: Path,
    *,
    files: list[str],
) -> Path:
    payload = {
        "model_id": spec.model_id,
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "files": {name: file_digest(directory / name) for name in sorted(files) if (directory / name).is_file()},
        "dimensions": spec.dimensions,
        "vector_config": spec.vector_config(),
    }
    path = directory / MANIFEST_NAME
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path