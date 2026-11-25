import time
from typing import Protocol, runtime_checkable

import httpx
from pydantic import SecretStr

from app.contracts.rag_workspace import RagUnavailable


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, texts: list[str]) -> list[float]: ...


class HttpReranker:
    """OpenAI / BGE-Reranker / Jina / Xinference compatible HTTP /rerank adapter (0 local RAM)."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: SecretStr | None = None,
        timeout: float = 20.0,
        max_retries: int = 1,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        headers: dict[str, str] = {}
        if self.api_key is not None:
            headers["Authorization"] = f"Bearer {self.api_key.get_secret_value()}"
        url = self.base_url if self.base_url.endswith("/rerank") else f"{self.base_url}/rerank"
        payload = {"model": self.model, "query": query, "documents": texts}
        last_error = "Reranker 服务暂不可用。"
        for attempt in range(self.max_retries + 1):
            try:
                response = httpx.post(url, json=payload, headers=headers, timeout=self.timeout)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = f"Reranker 服务不可用：{exc.__class__.__name__}"
            else:
                if response.status_code == 429 or response.status_code >= 500:
                    last_error = f"Reranker 服务返回 HTTP {response.status_code}。"
                elif response.status_code >= 400:
                    raise RagUnavailable(f"Reranker 请求无效：HTTP {response.status_code}。")
                else:
                    try:
                        body = response.json()
                        results = body.get("results") or body.get("data") or []
                        scores = [0.0] * len(texts)
                        for item in results:
                            idx = int(item["index"])
                            raw_score = float(item.get("relevance_score", item.get("score", 0.0)))
                            if 0 <= idx < len(scores):
                                scores[idx] = max(0.0, min(1.0, raw_score))
                        return scores
                    except (KeyError, TypeError, ValueError):
                        raise RagUnavailable("Reranker 返回了无效数据结构。") from None
            if attempt < self.max_retries:
                time.sleep(min(0.2 * (2 ** attempt), 1.0))
        raise RagUnavailable(last_error)


def build_reranker(settings) -> Reranker | None:
    if not settings.rerank_base_url:
        return None
    return HttpReranker(
        base_url=settings.rerank_base_url,
        model=settings.rerank_model,
        api_key=settings.rerank_api_key,
        timeout=settings.timeout_seconds,
        max_retries=settings.max_retries,
    )
