from __future__ import annotations

import json
import math
import sqlite3
import time
from urllib.parse import urlsplit
from uuid import uuid4

from cryptography.fernet import Fernet, InvalidToken
from pydantic import SecretStr

from app.api.errors import ApiProblem
from app.auth.sessions import Principal
from app.contracts.model_configs import (
    Capability, ModelConfigInput, ModelConfigView, ModelProvider, UpdateModelConfig,
)
from app.contracts.rag_workspace import EmbeddingModelDescriptor, RagUnavailable
from app.infrastructure.sqlite import Database, record_audit
from app.rag.chat import ChatServiceError, OpenAICompatibleChat
from app.rag.embeddings import OpenAICompatibleEmbeddings
from app.rag.graph import SYSTEM_PROMPT
from app.rag.models import ModelClaim
from app.rag.parsing import call_external_parser
from app.rag.rerank import HttpReranker

CONFIG_MODEL_PREFIX = "cfg-"

PROVIDERS = (
    ModelProvider(provider="deepseek", label="DeepSeek", base_urls=("https://api.deepseek.com/v1",), requires_key=True),
    ModelProvider(provider="kimi", label="Kimi", base_urls=("https://api.moonshot.cn/v1",), requires_key=True),
    ModelProvider(provider="glm", label="智谱 GLM", base_urls=("https://open.bigmodel.cn/api/paas/v4",), requires_key=True),
    ModelProvider(provider="openai", label="OpenAI（ChatGPT）", base_urls=("https://api.openai.com/v1",), requires_key=True),
    ModelProvider(provider="minimax", label="MiniMax", base_urls=("https://api.minimaxi.com/v1", "https://api.minimax.io/v1"), requires_key=True),
    ModelProvider(provider="ollama", label="Ollama", base_urls=("http://127.0.0.1:11434/v1", "http://localhost:11434/v1"), requires_key=False),
    ModelProvider(provider="custom", label="自定义兼容接口", base_urls=(), requires_key=False),
)
PROBE = ModelClaim(
    chunk_id="demo-claim", model="DEMO-CAMERA", assembly="标准", document_id="demo-manual",
    version="v1", page=1, section=None, attribute="fps", value="120", upper=None,
    unit="fps", polarity="affirmed", excerpt="DEMO-CAMERA 的帧率为 120 fps。",
)


def canonical_base_url(value: str) -> str:
    try:
        parts = urlsplit(value.strip())
        if (
            parts.scheme not in {"http", "https"} or not parts.hostname
            or parts.username is not None or parts.password is not None
            or parts.query or parts.fragment or "?" in value or "#" in value
            or any(ch.isspace() or ch in "\\%" for ch in value)
            or any(item in {".", ".."} for item in parts.path.split("/"))
            or parts.port == 0
        ):
            raise ValueError
        return parts._replace(netloc=parts.netloc.lower(), path=parts.path.rstrip("/")).geturl()
    except (ValueError, UnicodeError):
        raise ApiProblem(422, "invalid_model_base_url", "请输入完整的模型 API 基础地址。") from None


def config_view(row: sqlite3.Row) -> ModelConfigView:
    keys = row.keys()
    return ModelConfigView(
        config_id=row["config_id"],
        capability=row["capability"] if "capability" in keys else "chat",
        name=row["name"],
        provider=row["provider"],
        model_id=row["model_id"],
        base_url=row["base_url"],
        has_key=row["encrypted_key"] is not None,
        dimensions=row["dimensions"] if "dimensions" in keys else None,
        revision=row["revision"],
        test_status=row["test_status"],
        test_message=row["test_message"],
        test_at=row["test_at"],
        active=bool(row["active"]),
    )


class ModelConfigs:
    def __init__(self, database: Database, key: SecretStr | None, allowed_base_urls: tuple[str, ...]):
        self.database = database
        self.allowed_base_urls = tuple(dict.fromkeys(map(canonical_base_url, allowed_base_urls)))
        try:
            self.cipher = Fernet(key.get_secret_value()) if key is not None else None
        except (ValueError, TypeError):
            raise ValueError("SALESPILOT_MODEL_CONFIG_KEY is not a valid Fernet key") from None

    @property
    def providers(self) -> tuple[ModelProvider, ...]:
        return tuple(
            provider.model_copy(update={"base_urls": tuple(dict.fromkeys((*provider.base_urls, *self.allowed_base_urls)))})
            if not provider.requires_key else provider
            for provider in PROVIDERS
        )

    def endpoint(self, provider_id: str, url: str | None) -> tuple[ModelProvider, str]:
        provider = next(item for item in self.providers if item.provider == provider_id)
        if url is None and not provider.base_urls:
            raise ApiProblem(422, "model_base_url_required", "请填写部署环境允许的模型 API 基础地址。")
        base_url = canonical_base_url(url) if url is not None else provider.base_urls[0]
        if base_url not in provider.base_urls:
            raise ApiProblem(422, "model_base_url_not_allowed", "该地址不在此供应商允许的 API 地址范围内。")
        return provider, base_url

    def list(self, org_id: str, capability: Capability | None = None) -> list[ModelConfigView]:
        with self.database.connect() as connection:
            if capability is not None:
                rows = connection.execute(
                    """SELECT * FROM model_configs WHERE org_id=? AND capability=?
                    ORDER BY active DESC, name, config_id""",
                    (org_id, capability),
                )
            else:
                rows = connection.execute(
                    """SELECT * FROM model_configs WHERE org_id=?
                    ORDER BY capability, active DESC, name, config_id""",
                    (org_id,),
                )
            return [config_view(row) for row in rows]

    def active(self, org_id: str, capability: Capability = "chat") -> sqlite3.Row | None:
        with self.database.connect() as connection:
            return connection.execute(
                "SELECT * FROM model_configs WHERE org_id=? AND capability=? AND active=1",
                (org_id, capability),
            ).fetchone()

    def _one(self, connection: sqlite3.Connection, org_id: str, config_id: str) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM model_configs WHERE org_id=? AND config_id=?", (org_id, config_id),
        ).fetchone()
        if row is None:
            raise ApiProblem(404, "model_config_not_found", "找不到该模型配置。")
        return row

    def _require_revision(self, row: sqlite3.Row, revision: int) -> None:
        if row["revision"] != revision:
            raise ApiProblem(409, "model_config_changed", "模型配置已变更，请刷新后再操作。")

    def _encrypt(self, api_key: SecretStr, binding: tuple[str, ...]) -> bytes:
        if self.cipher is None:
            raise ApiProblem(503, "model_key_storage_unavailable", "服务器尚未配置模型密钥加密主密钥。")
        key = api_key.get_secret_value().strip()
        if not key or any(not 33 <= ord(character) <= 126 for character in key):
            raise ApiProblem(422, "invalid_model_api_key", "请输入有效的 API 密钥。")
        return self.cipher.encrypt(json.dumps([*binding, key]).encode())

    def _decrypt(self, row: sqlite3.Row) -> SecretStr | None:
        if row["encrypted_key"] is None:
            return None
        if self.cipher is None:
            raise ApiProblem(503, "model_key_storage_unavailable", "服务器尚未配置模型密钥加密主密钥。")
        try:
            *binding, key = json.loads(self.cipher.decrypt(row["encrypted_key"]))
            if binding != [row[field] for field in ("org_id", "config_id", "provider", "base_url")]:
                raise ValueError
            return SecretStr(key)
        except (InvalidToken, ValueError, TypeError):
            raise ApiProblem(503, "model_key_unreadable", "无法读取已保存的模型密钥，请重新配置。") from None

    def save(
        self, actor: Principal, request_id: str, body: ModelConfigInput | UpdateModelConfig,
        config_id: str | None = None,
    ) -> ModelConfigView:
        provider, base_url = self.endpoint(body.provider, body.base_url)
        if body.clear_key and body.api_key:
            raise ApiProblem(422, "conflicting_model_key", "请仅选择填写或清除密钥的一种操作。")
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            previous = self._one(connection, actor.org_id, config_id) if config_id is not None else None
            if previous is not None:
                self._require_revision(previous, body.revision)
                if previous["active"]:
                    raise ApiProblem(409, "active_model_config", "请先停用或切换此模型，再编辑配置。")
            key_id = config_id or uuid4().hex
            encrypted = None
            if body.api_key is not None:
                encrypted = self._encrypt(body.api_key, (actor.org_id, key_id, body.provider, base_url))
            elif (previous is not None and not body.clear_key and previous["provider"] == body.provider
                  and previous["base_url"] == base_url):
                encrypted = previous["encrypted_key"]
            if provider.requires_key and encrypted is None:
                raise ApiProblem(422, "model_api_key_required", "此供应商需要 API 密钥。")
            connection.execute(
                """INSERT INTO model_configs
                (config_id, org_id, capability, name, provider, model_id, base_url, encrypted_key, dimensions, revision)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, 1)
                ON CONFLICT(config_id) DO UPDATE SET capability=excluded.capability, name=excluded.name,
                provider=excluded.provider, model_id=excluded.model_id, base_url=excluded.base_url,
                encrypted_key=excluded.encrypted_key, dimensions=NULL,
                revision=model_configs.revision+1, test_status='untested', test_message=NULL, test_at=NULL""",
                (key_id, actor.org_id, body.capability, body.name, body.provider, body.model_id, base_url, encrypted),
            )
            self._audit(connection, actor, request_id, "model_config_saved", key_id)
            return config_view(self._one(connection, actor.org_id, key_id))

    def _chat(self, row: sqlite3.Row, *, testing: bool = False) -> OpenAICompatibleChat:
        provider, base_url = self.endpoint(row["provider"], row["base_url"])
        return OpenAICompatibleChat(
            model=row["model_id"], api_key=self._decrypt(row), base_url=base_url,
            timeout=12 if testing else 25, max_retries=0 if testing else 1,
            requires_api_key=provider.requires_key,
            disable_thinking=provider.provider == "deepseek",
        )

    def _embedding(self, row: sqlite3.Row, *, testing: bool = False) -> OpenAICompatibleEmbeddings:
        provider, base_url = self.endpoint(row["provider"], row["base_url"])
        expected_dims = None if testing else row["dimensions"]
        return OpenAICompatibleEmbeddings(
            model=row["model_id"],
            api_key=self._decrypt(row),
            base_url=base_url,
            timeout=15 if testing else 30,
            max_retries=0 if testing else 2,
            dimensions=expected_dims,
            requires_api_key=provider.requires_key,
        )

    def _reranker(self, row: sqlite3.Row, *, testing: bool = False) -> HttpReranker:
        _, base_url = self.endpoint(row["provider"], row["base_url"])
        return HttpReranker(
            base_url=base_url,
            model=row["model_id"],
            api_key=self._decrypt(row),
            timeout=12 if testing else 20,
            max_retries=0 if testing else 1,
        )

    def _parser_endpoint(self, row: sqlite3.Row) -> tuple[str, str | None]:
        _, base_url = self.endpoint(row["provider"], row["base_url"])
        secret = self._decrypt(row)
        return base_url, (secret.get_secret_value() if secret is not None else None)

    def chat_for_org(self, org_id: str) -> OpenAICompatibleChat | None:
        row = self.active(org_id, "chat")
        if row is None:
            return None
        try:
            return self._chat(row)
        except ApiProblem as exc:
            raise RagUnavailable(str(exc)) from None

    def reranker_for_org(self, org_id: str) -> HttpReranker | None:
        row = self.active(org_id, "rerank")
        if row is None:
            return None
        try:
            return self._reranker(row)
        except ApiProblem as exc:
            raise RagUnavailable(str(exc)) from None

    def parser_for_org(self, org_id: str) -> tuple[str, str | None] | None:
        row = self.active(org_id, "parser")
        if row is None:
            return None
        try:
            return self._parser_endpoint(row)
        except ApiProblem as exc:
            raise RagUnavailable(str(exc)) from None

    def embedding_options_for_org(self, org_id: str) -> list[tuple[str, str, bool]]:
        with self.database.connect() as connection:
            rows = connection.execute(
                """SELECT config_id, name, dimensions, test_status FROM model_configs
                WHERE org_id=? AND capability='embedding' AND test_status='passed' AND dimensions IS NOT NULL
                ORDER BY active DESC, name, config_id""",
                (org_id,),
            ).fetchall()
        return [
            (f"{CONFIG_MODEL_PREFIX}{row['config_id']}", f"{row['name']}（远程 API · {row['dimensions']} 维）", True)
            for row in rows
        ]

    def resolve_remote_embedding(
        self, model_id: str,
    ) -> tuple[EmbeddingModelDescriptor, OpenAICompatibleEmbeddings] | None:
        if not model_id.startswith(CONFIG_MODEL_PREFIX):
            return None
        config_id = model_id[len(CONFIG_MODEL_PREFIX):]
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM model_configs WHERE config_id=? AND capability='embedding'",
                (config_id,),
            ).fetchone()
        if row is None or row["test_status"] != "passed" or not row["dimensions"]:
            return None
        embedder = self._embedding(row)
        dims = int(row["dimensions"])
        descriptor = EmbeddingModelDescriptor(
            model_id=model_id,
            revision=f"cfg-rev-{row['revision']}",
            dimensions=dims,
            vector_config={
                "provider": "openai",
                "model": row["model_id"],
                "base_url": row["base_url"],
                "normalize": True,
                "max_tokens": 0,
                "dimensions": dims,
            },
            runtime={"provider": row["provider"], "remote": "true"},
            display_name=row["name"],
        )
        return descriptor, embedder

    def test(self, actor: Principal, request_id: str, config_id: str, revision: int) -> ModelConfigView:
        with self.database.connect() as connection:
            row = self._one(connection, actor.org_id, config_id)
            self._require_revision(row, revision)
        capability = row["capability"]
        detected_dimensions: int | None = row["dimensions"]
        try:
            if capability == "chat":
                chat = self._chat(row, testing=True)
                draft = chat.compose_draft(
                    system=SYSTEM_PROMPT,
                    user="连接测试。请按所给合成资料返回结构化声明。",
                    evidence=[PROBE.model_dump(mode="json")],
                )
                if draft.claims != (PROBE,):
                    raise ChatServiceError("invalid_model_output", "模型未按样例返回有效结构化声明。")
                status, message = "passed", "连接成功，结构化输出验证通过。"
            elif capability == "embedding":
                embedder = self._embedding(row, testing=True)
                vectors = embedder.embed_documents(["SalesPilot 向量化连接探针测试。"])
                if not vectors or not vectors[0] or any(not math.isfinite(val) for val in vectors[0]):
                    raise RagUnavailable("向量接口未返回有效浮点向量。")
                detected_dimensions = len(vectors[0])
                status, message = "passed", f"连接成功，向量维度 {detected_dimensions} 维验证通过。"
            elif capability == "rerank":
                reranker = self._reranker(row, testing=True)
                scores = reranker.rerank("GigE 工业相机", ["SP-VC-120 GigE 相机 120 fps", "普通电源线"])
                if len(scores) != 2 or any(not math.isfinite(val) for val in scores):
                    raise RagUnavailable("重排接口未返回有效分数列表。")
                status, message = "passed", "连接成功，重排接口验证通过。"
            else:
                base_url, api_key = self._parser_endpoint(row)
                sections = call_external_parser(
                    "probe.pdf",
                    b"%PDF-1.4\n% SalesPilot parser probe\n",
                    base_url=base_url,
                    api_key=api_key,
                    timeout=15.0,
                )
                if not sections:
                    raise RagUnavailable("文档解析服务未返回有效分段。")
                status, message = "passed", "连接成功，文档解析服务验证通过。"
        except (ChatServiceError, RagUnavailable, ApiProblem, ValueError) as exc:
            status, message = "failed", str(exc)
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._require_revision(self._one(connection, actor.org_id, config_id), revision)
            connection.execute(
                """UPDATE model_configs
                SET test_status=?, test_message=?, test_at=?, dimensions=?
                WHERE org_id=? AND config_id=?""",
                (status, message, int(time.time()), detected_dimensions, actor.org_id, config_id),
            )
            self._audit(connection, actor, request_id, "model_config_tested", config_id)
            return config_view(self._one(connection, actor.org_id, config_id))

    def activate(self, actor: Principal, request_id: str, config_id: str, revision: int) -> ModelConfigView:
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._one(connection, actor.org_id, config_id)
            self._require_revision(row, revision)
            if row["test_status"] != "passed":
                raise ApiProblem(409, "model_test_required", "请先完成此配置的连接与能力探针测试。")
            capability = row["capability"]
            if capability == "chat":
                self._chat(row)
            elif capability == "embedding":
                self._embedding(row)
            elif capability == "rerank":
                self._reranker(row)
            else:
                self._parser_endpoint(row)
            connection.execute(
                "UPDATE model_configs SET active=0 WHERE org_id=? AND capability=? AND active=1",
                (actor.org_id, capability),
            )
            connection.execute("UPDATE model_configs SET active=1 WHERE config_id=?", (config_id,))
            self._audit(connection, actor, request_id, "model_config_activated", config_id)
            return config_view(self._one(connection, actor.org_id, config_id))

    def deactivate(self, actor: Principal, request_id: str, config_id: str) -> ModelConfigView:
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._one(connection, actor.org_id, config_id)
            connection.execute("UPDATE model_configs SET active=0 WHERE config_id=?", (config_id,))
            self._audit(connection, actor, request_id, "model_config_deactivated", config_id)
            return config_view(self._one(connection, actor.org_id, config_id))

    def delete(self, actor: Principal, request_id: str, config_id: str) -> None:
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if self._one(connection, actor.org_id, config_id)["active"]:
                raise ApiProblem(409, "active_model_config", "请先停用或切换此模型，再删除配置。")
            connection.execute("DELETE FROM model_configs WHERE org_id=? AND config_id=?", (actor.org_id, config_id))
            self._audit(connection, actor, request_id, "model_config_deleted", config_id)

    @staticmethod
    def _audit(connection: sqlite3.Connection, actor: Principal, request_id: str, action: str, config_id: str) -> None:
        record_audit(connection, org_id=actor.org_id, user_id=actor.user_id, action=action,
                     request_id=request_id, object_id=config_id)
