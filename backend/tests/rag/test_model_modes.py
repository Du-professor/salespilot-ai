import json
from json import dumps as json_dumps

import httpx
import pytest
from pydantic import SecretStr

from app.contracts.rag_workspace import RagContext, RagUnavailable
from app.rag import HashingEmbeddings, RagSettings, SqliteRagEngine
from app.rag.chat import OpenAICompatibleChat
from app.rag.embeddings import OpenAICompatibleEmbeddings
from app.rag.models import ModelClaim, ModelDraftPayload

from conftest import ask


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self.payload = payload

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


def _model_claim(item, **overrides):
    values = {
        key: item.get(key)
        for key in (
            "chunk_id", "model", "assembly", "document_id", "version", "page", "section",
            "attribute", "value", "upper", "unit", "polarity", "excerpt",
        )
    }
    values.update(overrides)
    return ModelClaim(**values)


def test_openai_and_local_chat_urls_and_headers(monkeypatch):
    calls = []

    def post(url, json, headers, timeout):
        calls.append((url, json, headers, timeout))
        return FakeResponse(200, {"choices": [{"message": {"content": "推荐 SP-VC-120。"}}]})

    monkeypatch.setattr("app.rag.chat.httpx.post", post)
    chat = OpenAICompatibleChat(
        model="demo-model", api_key=SecretStr("secret"), base_url=None, timeout=5, max_retries=0,
    )
    assert chat.compose(system="system", user="user") == "推荐 SP-VC-120。"
    assert calls[-1][0] == "https://api.openai.com/v1/chat/completions"
    assert calls[-1][2]["Authorization"] == "Bearer secret"

    local = OpenAICompatibleChat(
        model="local-model", api_key=SecretStr("local"), base_url="http://127.0.0.1:8000/v1",
        timeout=5, max_retries=0,
    )
    assert local.compose(system="system", user="user") == "推荐 SP-VC-120。"
    assert calls[-1][0] == "http://127.0.0.1:8000/v1/chat/completions"


def test_structured_chat_response_is_parsed(monkeypatch):
    payload = {
        "claims": [{
            "chunk_id": "chk_1", "model": "AX-920", "assembly": "标准装配",
            "document_id": "ax", "version": "v1", "page": None, "section": "1. AX-920 标准装配",
            "attribute": "fps", "value": "120", "unit": "fps", "polarity": "affirmed",
            "excerpt": "AX-920 的帧率为 120 fps。",
        }],
    }

    def post(url, json, headers, timeout):
        return FakeResponse(200, {"choices": [{"message": {"content": json_dumps(payload)}}]})

    monkeypatch.setattr("app.rag.chat.httpx.post", post)
    chat = OpenAICompatibleChat(model="m", api_key=SecretStr("k"), base_url=None, timeout=5, max_retries=0)
    result = chat.compose_draft(system="system", user="user", evidence=[{"chunk_id": "chk_1"}])
    assert result.claims[0].model == "AX-920"
    assert result.claims[0].value == "120"


def test_structured_chat_invalid_json_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        "app.rag.chat.httpx.post",
        lambda *args, **kwargs: FakeResponse(200, {"choices": [{"message": {"content": "不是 JSON"}}]}),
    )
    chat = OpenAICompatibleChat(model="m", api_key=SecretStr("k"), base_url=None, timeout=5, max_retries=0)
    with pytest.raises(RagUnavailable):
        chat.compose_draft(system="system", user="user", evidence=[])


def test_chat_without_key_is_unavailable(monkeypatch):
    def post(*args, **kwargs):
        raise AssertionError("network must not be called")

    monkeypatch.setattr("app.rag.chat.httpx.post", post)
    chat = OpenAICompatibleChat(model="m", api_key=None, base_url=None, timeout=5, max_retries=0)
    with pytest.raises(RagUnavailable):
        chat.compose(system="system", user="user")


def test_chat_timeout_retries_then_unavailable(monkeypatch):
    calls = []

    def post(*args, **kwargs):
        calls.append(1)
        raise httpx.ConnectTimeout("timeout")

    monkeypatch.setattr("app.rag.chat.httpx.post", post)
    chat = OpenAICompatibleChat(model="m", api_key=SecretStr("k"), base_url=None, timeout=1, max_retries=1)
    with pytest.raises(RagUnavailable):
        chat.compose(system="system", user="user")
    assert len(calls) == 2


def test_invalid_http_and_invalid_payload_are_unavailable(monkeypatch):
    monkeypatch.setattr("app.rag.chat.httpx.post", lambda *args, **kwargs: FakeResponse(400, {}))
    chat = OpenAICompatibleChat(model="m", api_key=SecretStr("k"), base_url=None, timeout=1, max_retries=2)
    with pytest.raises(RagUnavailable):
        chat.compose(system="system", user="user")

    monkeypatch.setattr("app.rag.chat.httpx.post", lambda *args, **kwargs: FakeResponse(200, {"choices": []}))
    with pytest.raises(RagUnavailable):
        chat.compose(system="system", user="user")


def test_embedding_response_is_ordered_and_validated(monkeypatch):
    def post(url, json, headers, timeout):
        assert url.endswith("/embeddings")
        return FakeResponse(200, {"data": [
            {"index": index, "embedding": [0.1 + 0.2 * index, 0.2 + 0.2 * index]}
            for index, _ in enumerate(json["input"])
        ]})

    monkeypatch.setattr("app.rag.embeddings.httpx.post", post)
    embeddings = OpenAICompatibleEmbeddings(
        model="embed", api_key=SecretStr("k"), base_url=None, timeout=1, max_retries=0, dimensions=2,
    )
    vectors = embeddings.embed_documents(["first", "second"])
    assert vectors[0] == pytest.approx([0.1, 0.2])
    assert vectors[1] == pytest.approx([0.3, 0.4])
    assert embeddings.embed_query("first")[0] == pytest.approx(0.1)

    broken = OpenAICompatibleEmbeddings(
        model="embed", api_key=SecretStr("k"), base_url=None, timeout=1, max_retries=0, dimensions=4,
    )
    with pytest.raises(RagUnavailable):
        broken.embed_query("first")


def test_settings_read_provider_and_secret_from_environment(monkeypatch):
    monkeypatch.setenv("SALESPILOT_RAG_CHAT_PROVIDER", "openai")
    monkeypatch.setenv("SALESPILOT_RAG_CHAT_MODEL", "local-chat")
    monkeypatch.setenv("SALESPILOT_RAG_CHAT_BASE_URL", "http://127.0.0.1:9000/v1")
    monkeypatch.setenv("SALESPILOT_RAG_CHAT_API_KEY", "secret")
    settings = RagSettings.from_environment()
    assert settings.chat_provider == "openai"
    assert settings.chat_model == "local-chat"
    assert settings.chat_base_url == "http://127.0.0.1:9000/v1"
    assert settings.chat_api_key is not None
    assert settings.chat_api_key.get_secret_value() == "secret"


def test_model_failure_is_unavailable(engine, settings, vision, cases):
    class FailingChat:
        def compose(self, *, system, user):
            raise RagUnavailable("模型超时。")

        def compose_draft(self, *, system, user, evidence):
            raise RagUnavailable("模型超时。")

    failing = SqliteRagEngine(
        settings, embedder=HashingEmbeddings(settings.embedding_dimensions), chat=FailingChat(),
    )
    with pytest.raises(RagUnavailable):
        failing.recommend(engine_context(vision, cases))


@pytest.mark.parametrize("incorrect_fps", [9999, 20])
def test_hallucinated_structured_claim_falls_back_to_verified_text(engine, settings, vision, cases, incorrect_fps):
    class HallucinatingChat:
        def compose(self, *, system, user):
            return f"推荐 SP-VC-120，帧率 {incorrect_fps} fps。"

        def compose_draft(self, *, system, user, evidence):
            source = next(item for item in evidence if item["attribute"] == "fps")
            return ModelDraftPayload(claims=(_model_claim(source, value=str(incorrect_fps)),))

    injected = SqliteRagEngine(
        settings, embedder=HashingEmbeddings(settings.embedding_dimensions), chat=HallucinatingChat(),
    )
    result = injected.recommend(engine_context(vision, cases))
    assert result.status == "recommended"
    assert "9999" not in result.answer
    assert f"帧率 {incorrect_fps} fps" not in result.answer
    assert "SP-VC-120" in result.answer and "120 fps" in result.answer


def test_grounded_structured_claims_are_used(engine, settings, vision, cases):
    class GroundedChat:
        def __init__(self):
            self.calls = 0

        def compose(self, *, system, user):
            return "推荐 SP-VC-120。"

        def compose_draft(self, *, system, user, evidence):
            self.calls += 1
            return ModelDraftPayload(claims=tuple(_model_claim(item) for item in evidence))

    chat = GroundedChat()
    polished = SqliteRagEngine(
        settings, embedder=HashingEmbeddings(settings.embedding_dimensions), chat=chat,
    )
    result = polished.recommend(engine_context(vision, cases))
    assert chat.calls == 1
    assert result.status == "recommended"
    assert "12 MP" in result.answer and "120 fps" in result.answer
    assert result.citations


def engine_context(document, cases):
    return RagContext(
        org_id="org-a", user_id="user-a", text=cases["vision"]["text"],
        previous_turns=(), documents=(document,),
    )