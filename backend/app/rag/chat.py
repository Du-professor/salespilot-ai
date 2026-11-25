import json
import time
from typing import Protocol

import httpx
from pydantic import SecretStr

from app.contracts.rag_workspace import RagUnavailable

from .config import RagSettings
from .models import ModelClaim, ModelDraftPayload


class ChatModel(Protocol):
    def compose(self, *, system: str, user: str) -> str: ...

    def compose_draft(self, *, system: str, user: str, evidence: list[dict]) -> ModelDraftPayload: ...


class ChatServiceError(RagUnavailable):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


class OpenAICompatibleChat:
    def __init__(
        self,
        *,
        model: str,
        api_key: SecretStr | None,
        base_url: str | None,
        timeout: float,
        max_retries: int,
        requires_api_key: bool = True,
        disable_thinking: bool = False,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.requires_api_key = requires_api_key
        self.disable_thinking = disable_thinking

    def _request(self, payload: dict) -> str:
        if self.api_key is None and self.requires_api_key:
            raise ChatServiceError("model_invalid_credentials", "模型密钥未配置。")
        headers = {}
        if self.api_key is not None:
            headers["Authorization"] = f"Bearer {self.api_key.get_secret_value()}"
        if self.disable_thinking:
            payload["thinking"] = {"type": "disabled"}
        last_error = ChatServiceError("model_service_unavailable", "模型服务暂不可用。")
        for attempt in range(self.max_retries + 1):
            try:
                response = httpx.post(
                    f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=self.timeout,
                )
            except httpx.TimeoutException:
                last_error = ChatServiceError("model_timeout", "连接模型服务超时，请检查服务状态后重试。")
            except httpx.TransportError:
                last_error = ChatServiceError("model_connection_failed", "无法连接模型服务，请检查 API 地址和网络。")
            else:
                if response.status_code == 429 or response.status_code >= 500:
                    last_error = ChatServiceError("model_service_unavailable", f"模型服务暂不可用（HTTP {response.status_code}）。")
                elif response.status_code in {401, 403}:
                    raise ChatServiceError("model_invalid_credentials", "模型服务拒绝认证，请检查 API 密钥与权限。")
                elif response.status_code == 404:
                    raise ChatServiceError("model_not_found", "模型或接口不存在，请检查模型 ID 和 API 地址。")
                elif response.status_code == 402:
                    raise ChatServiceError("model_payment_required", "模型服务拒绝计费，请检查账号余额或额度。")
                elif response.status_code >= 300:
                    raise ChatServiceError("invalid_model_request", "模型服务拒绝请求，请检查模型 ID 和 API 地址。")
                else:
                    try:
                        content = response.json()["choices"][0]["message"]["content"]
                    except (KeyError, IndexError, TypeError, ValueError):
                        raise ChatServiceError("invalid_model_output", "模型服务返回了无效数据。") from None
                    if not isinstance(content, str) or not content.strip():
                        raise ChatServiceError("invalid_model_output", "模型服务返回了空回复。")
                    return content.strip()
            if attempt < self.max_retries:
                time.sleep(min(0.2 * (2 ** attempt), 1.0))
        raise last_error

    def compose(self, *, system: str, user: str) -> str:
        return self._request({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        })

    def compose_draft(self, *, system: str, user: str, evidence: list[dict]) -> ModelDraftPayload:
        evidence_json = json.dumps(
            [{key: value for key, value in item.items() if key in ModelClaim.model_fields} for item in evidence],
            ensure_ascii=False,
        )
        instruction = (
            f"客户需求：{user}\n"
            "可用证据如下。每条声明必须逐项复制证据中的 chunk_id、document_id、version、model、"
            "assembly、page、section、attribute、value、upper、unit、polarity 和 excerpt，"
            "不得新增证据之外的事实。\n"
            f"{evidence_json}\n"
            "只返回 JSON 对象，格式为 {\"claims\": [...]}，不要输出 Markdown 代码块。"
        )
        content = self._request({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": instruction},
            ],
        })
        if content.startswith("```"):
            content = re_fence(content)
        try:
            payload = json.loads(content)
            return ModelDraftPayload.model_validate(payload)
        except (ValueError, TypeError):
            raise ChatServiceError("invalid_model_output", "模型服务未返回可解析的结构化声明。") from None


def re_fence(content: str) -> str:
    lines = content.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def build_chat(settings: RagSettings) -> ChatModel | None:
    if settings.chat_provider == "openai":
        return OpenAICompatibleChat(
            model=settings.chat_model,
            api_key=settings.chat_api_key,
            base_url=settings.chat_base_url,
            timeout=settings.timeout_seconds,
            max_retries=settings.max_retries,
        )
    return None
