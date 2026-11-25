import hashlib
import math
import re
import time

import httpx
from langchain_core.embeddings import Embeddings
from pydantic import SecretStr

from app.contracts.rag_workspace import RagUnavailable

from .config import RagSettings

_WORD = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
_CJK = re.compile(r"[\u4e00-\u9fff]+")


def _tokens(text: str) -> list[str]:
    lowered = text.casefold()
    tokens = _WORD.findall(lowered)
    for block in _CJK.findall(lowered):
        tokens.extend(block)
        tokens.extend(block[index:index + 2] for index in range(len(block) - 1))
    return tokens or ["<empty>"]


class HashingEmbeddings(Embeddings):
    def __init__(self, dimensions: int = 256):
        if dimensions < 8:
            raise ValueError("dimensions must be at least 8")
        self.dimensions = dimensions

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in _tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class OpenAICompatibleEmbeddings(Embeddings):
    def __init__(
        self,
        *,
        model: str,
        api_key: SecretStr | None,
        base_url: str | None,
        timeout: float,
        max_retries: int,
        dimensions: int | None = None,
        requires_api_key: bool = True,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.dimensions = dimensions
        self.requires_api_key = requires_api_key

    def _request(self, texts: list[str]) -> list[list[float]]:
        if self.api_key is None and self.requires_api_key:
            raise RagUnavailable("Embedding 密钥未配置。")
        headers = {}
        if self.api_key is not None:
            headers["Authorization"] = f"Bearer {self.api_key.get_secret_value()}"
        payload = {"model": self.model, "input": texts}
        last_error = "Embedding 服务不可用。"
        for attempt in range(self.max_retries + 1):
            try:
                response = httpx.post(
                    f"{self.base_url}/embeddings", json=payload, headers=headers, timeout=self.timeout,
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = f"Embedding 服务不可用：{exc.__class__.__name__}"
            else:
                if response.status_code == 429 or response.status_code >= 500:
                    last_error = f"Embedding 服务返回 {response.status_code}。"
                elif response.status_code >= 400:
                    raise RagUnavailable(f"Embedding 配置无效：HTTP {response.status_code}。")
                else:
                    try:
                        data = response.json()["data"]
                        ordered = sorted(data, key=lambda item: item["index"])
                        vectors = [[float(value) for value in item["embedding"]] for item in ordered]
                    except (KeyError, TypeError, ValueError):
                        raise RagUnavailable("Embedding 服务返回了无效数据。") from None
                    if len(vectors) != len(texts) or any(not vector for vector in vectors):
                        raise RagUnavailable("Embedding 服务返回的向量数量或内容不完整。")
                    if self.dimensions is not None and any(len(vector) != self.dimensions for vector in vectors):
                        raise RagUnavailable("Embedding 向量维度与配置不一致，请重新索引。")
                    return vectors
            if attempt < self.max_retries:
                time.sleep(min(0.2 * (2 ** attempt), 1.0))
        raise RagUnavailable(last_error)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 64):
            vectors.extend(self._request(texts[start:start + 64]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._request([text])[0]


def build_embeddings(settings: RagSettings) -> Embeddings:
    if settings.embedding_provider == "openai":
        expected_dims = settings.embedding_dimensions if "embedding_dimensions" in settings.model_fields_set else None
        return OpenAICompatibleEmbeddings(
            model=settings.embedding_model,
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url,
            timeout=settings.timeout_seconds,
            max_retries=settings.max_retries,
            dimensions=expected_dims,
        )
    if settings.embedding_provider == "local":
        from .local_embeddings import LocalTransformerEmbeddings
        from .model_catalog import resolve_model_spec

        spec = resolve_model_spec(settings.embedding_model_id, settings)
        if spec.provider != "local":
            raise RagUnavailable("本地 Embedding 需要选择已登记的 BGE 模型。")
        return LocalTransformerEmbeddings(spec, settings)
    return HashingEmbeddings(settings.embedding_dimensions)