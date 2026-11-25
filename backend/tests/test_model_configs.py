import asyncio
import json as jsonlib
import sqlite3

import httpx
from cryptography.fernet import Fernet

from app.main import create_app
from app.rag.models import ModelClaim
from app.settings import Settings

from workspace_helpers import ORIGIN, PASSWORD_A, PASSWORD_B, PREFIX, browser, login
from test_rag_workspace import CASES, FIXTURES, HISTORY, LIBRARY, TURNS, upload_body

CONFIGS = f"{PREFIX}/model-configs"
FAKE_KEY = "synthetic-provider-credential"


def echo_evidence(url, *, json, headers, timeout):
    prompt = json["messages"][-1]["content"]
    raw = prompt[prompt.index("\n[") + 1:prompt.rindex("\n只返回")]
    claims = [{key: item[key] for key in ModelClaim.model_fields} for item in jsonlib.loads(raw)]
    response = jsonlib.dumps({"claims": claims}, ensure_ascii=False)
    return httpx.Response(200, json={"choices": [{"message": {"content": response}}]})


def app_for(path, key=None, **values):
    return create_app(Settings(
        database_path=path, rag_mode="demo", demo_password_a=PASSWORD_A, demo_password_b=PASSWORD_B,
        web_origins=(ORIGIN,), model_config_key=key, **values,
    ))


async def add_config(client, **values):
    body = {"name": "云端售前", "provider": "deepseek", "model_id": "test-model", "api_key": FAKE_KEY, **values}
    response = await client.post(CONFIGS, json=body)
    assert response.status_code == 201, response.text
    return response.json()


async def apply(client, item, action):
    return await client.post(f"{CONFIGS}/{item['config_id']}/{action}", json={"revision": item["revision"]})


def test_workspace_can_test_activate_switch_and_reopen_model_configs(tmp_path, monkeypatch):
    requests = []

    def fake_provider(url, *, json, headers, timeout):
        requests.append((url, json["model"], headers.get("Authorization"), json["messages"][-1]["content"]))
        return echo_evidence(url, json=json, headers=headers, timeout=timeout)

    monkeypatch.setattr("app.rag.chat.httpx.post", fake_provider)
    path = tmp_path / "models.sqlite3"
    key = Fernet.generate_key().decode()

    async def check():
        app = app_for(path, key)
        async with app.router.lifespan_context(app), browser(app) as manager, browser(app) as sales:
            await login(manager)
            await login(sales, "sales-b", PASSWORD_B)
            assert (await sales.get(CONFIGS)).status_code == 403
            assert (await sales.get(CONFIGS + "/active")).json()["mode"] == "offline"
            providers = (await manager.get(CONFIGS)).json()["providers"]
            assert {item["provider"] for item in providers} == {"deepseek", "kimi", "glm", "openai", "minimax", "ollama", "custom"}
            cloud = await add_config(manager)
            ollama = await add_config(manager, provider="ollama", name="Ollama 内部选型", model_id="local-model", api_key=None)
            assert cloud["has_key"] and not ollama["has_key"]
            assert (await apply(manager, cloud, "activate")).status_code == 409
            assert (await apply(sales, cloud, "test")).status_code == 403
            assert not requests
            assert (await apply(manager, cloud, "test")).json()["test_status"] == "passed"
            assert (await apply(manager, cloud, "activate")).json()["active"]
            assert (await sales.get(CONFIGS + "/active")).json()["model_id"] == "test-model"

            manual = await manager.post(LIBRARY, json=upload_body(FIXTURES / "specs/sp-vision-cameras.md"))
            doc_id = manual.json()["document_id"]
            assert (await manager.post(f"{LIBRARY}/{doc_id}/v1/publish")).status_code == 200
            turn = {"conversation_id": "models", "client_request_id": "cloud-turn", "text": CASES["vision"]["text"]}
            first = await sales.post(TURNS, json=turn)
            assert first.status_code == 200 and first.json()["models"] == ["SP-VC-120"]
            assert first.json()["has_report"] and (await sales.get(HISTORY)).json()
            assert len(requests) == 2
            assert requests[0][:3] == ("https://api.deepseek.com/v1/chat/completions", "test-model", f"Bearer {FAKE_KEY}")
            assert "DEMO-CAMERA" in requests[0][3] and "SP-VC-120" not in requests[0][3]
            assert "SP-VC-120" in requests[1][3]

            assert (await apply(manager, ollama, "test")).json()["test_status"] == "passed"
            assert (await apply(manager, ollama, "activate")).json()["active"]
            second = await sales.post(TURNS, json={**turn, "client_request_id": "local-turn"})
            assert second.status_code == 200
            assert requests[-1][:3] == ("http://127.0.0.1:11434/v1/chat/completions", "local-model", None)
            listing = await manager.get(CONFIGS)
            assert FAKE_KEY not in listing.text
            assert [item["model_id"] for item in listing.json()["configs"] if item["active"]] == ["local-model"]

        with sqlite3.connect(path) as db:
            encrypted = db.execute("SELECT encrypted_key FROM model_configs WHERE config_id=?", (cloud["config_id"],)).fetchone()[0]
            assert FAKE_KEY.encode() not in encrypted
            assert jsonlib.loads(Fernet(key).decrypt(encrypted))[-1] == FAKE_KEY

        reopened = app_for(path, key)
        async with reopened.router.lifespan_context(reopened), browser(reopened) as manager:
            await login(manager)
            assert (await manager.get(CONFIGS + "/active")).json()["model_id"] == "local-model"
            assert not (await apply(manager, ollama, "deactivate")).json()["active"]
            assert (await manager.get(CONFIGS + "/active")).json()["mode"] == "offline"
            assert (await manager.delete(f"{CONFIGS}/{cloud['config_id']}")).status_code == 204

    asyncio.run(check())


def test_config_rejects_unapproved_endpoint_and_stale_or_failed_activation(tmp_path, monkeypatch):
    monkeypatch.setattr("app.rag.chat.httpx.post", lambda *args, **kwargs: httpx.Response(401))
    app = app_for(tmp_path / "validation.sqlite3", Fernet.generate_key().decode())

    async def check():
        async with app.router.lifespan_context(app), browser(app) as client:
            await login(client)
            rejected = await client.post(CONFIGS, json={
                "provider": "custom", "name": "错误地址", "model_id": "unknown", "base_url": "http://127.0.0.1:9080/v1",
            })
            assert (rejected.status_code, rejected.json()["code"]) == (422, "model_base_url_not_allowed")
            item = await add_config(client)
            outcome = (await apply(client, item, "test")).json()
            assert outcome["test_status"] == "failed"
            assert "密钥" in outcome["test_message"] and FAKE_KEY not in str(outcome)
            assert (await apply(client, item, "activate")).status_code == 409
            changed = await client.patch(f"{CONFIGS}/{item['config_id']}", json={
                "provider": "deepseek", "name": "更新连接", "model_id": "replacement", "revision": item["revision"],
            })
            assert changed.status_code == 200 and changed.json()["has_key"]
            assert changed.json()["test_status"] == "untested"
            assert (await apply(client, item, "test")).status_code == 409
            assert (await client.get(CONFIGS + "/active")).json()["mode"] == "offline"

    asyncio.run(check())
