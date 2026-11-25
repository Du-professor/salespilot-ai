import asyncio
import json

from app.rag import HashingEmbeddings, RagSettings, SqliteRagEngine
from app.infrastructure.library import library_index_write

from workspace_helpers import PREFIX, browser, login
from test_rag_workspace import FIXTURES, LIBRARY, TURNS, stub_application, upload_body

BASES = f"{PREFIX}/library/knowledge-bases"
RETRIEVAL = f"{PREFIX}/library/retrieval-test"
CASES = json.loads((FIXTURES / "rag/cases.json").read_text(encoding="utf-8"))
SAMPLE = FIXTURES / "specs/sp-vision-cameras.md"


def _engine(tmp_path):
    settings = RagSettings(
        index_path=tmp_path / "index.sqlite3",
        embedding_provider="hash",
        chat_provider="template",
        max_retries=0,
    )
    return SqliteRagEngine(settings, embedder=HashingEmbeddings(settings.embedding_dimensions), chat=None)


def test_new_base_publishes_to_active_index_and_serves_retrieval(tmp_path):
    async def check():
        database = tmp_path / "admin.sqlite3"
        engine = _engine(tmp_path)
        app = stub_application(database, engine, models_dir=tmp_path / "models")
        async with app.router.lifespan_context(app), browser(app) as client:
            await login(client)
            created = await client.post(BASES, json={
                "name": "相机资料", "description": "", "embedding_model_id": "workspace-default",
            })
            assert created.status_code == 201, created.text
            base_id = created.json()["knowledge_base_id"]

            availability = await client.get(f"{BASES}/{base_id}/index")
            assert availability.status_code == 200, availability.text
            assert availability.json()["state"] == "awaiting_adapter"

            uploaded = await client.post(LIBRARY, json=upload_body(SAMPLE, knowledge_base_id=base_id))
            assert uploaded.status_code == 201, uploaded.text
            document = uploaded.json()
            path = f"{LIBRARY}/{document['document_id']}/{document['version']}"
            published = await client.post(f"{path}/publish")
            assert published.status_code == 200, published.text
            assert published.json()["status"] == "published"

            availability = (await client.get(f"{BASES}/{base_id}/index")).json()
            assert availability["state"] == "available"
            active = availability["active_index_version"]
            assert active != "legacy"

            diagnostics = await client.post(RETRIEVAL, json={
                "text": CASES["vision"]["text"], "knowledge_base_id": base_id,
            })
            assert diagnostics.status_code == 200, diagnostics.text
            assert diagnostics.json()["status"] == "retrieved"
            assert diagnostics.json()["hits"][0]["knowledge_base_id"] == base_id
            assert diagnostics.json()["hits"][0]["index_version"] == active

            chunks = await client.get(f"{path}/chunks")
            assert chunks.status_code == 200 and chunks.json()["total"] > 0

            reply = await client.post(TURNS, json={
                "conversation_id": "kb-chat",
                "client_request_id": "kb-request",
                "text": CASES["vision"]["text"],
                "knowledge_base_id": base_id,
            })
            assert reply.status_code == 200, reply.text
            assert reply.json()["models"] == CASES["vision"]["expected_models"]
            payload = {
                "conversation_id": "kb-chat", "client_request_id": "kb-request",
                "text": CASES["vision"]["text"], "knowledge_base_id": base_id,
            }
            assert (await client.post(TURNS, json=payload)).json() == reply.json()
            wrong_scope = await client.post(TURNS, json={**payload, "knowledge_base_id": "default"})
            assert wrong_scope.status_code == 409
            assert wrong_scope.json()["code"] == "recommendation_request_conflict"
            moved_thread = await client.post(TURNS, json={
                **payload, "client_request_id": "next-request", "knowledge_base_id": "default",
            })
            assert moved_thread.status_code == 409
            assert moved_thread.json()["code"] == "conversation_knowledge_base_conflict"
            thread = await client.get(f"{PREFIX}/recommendations/conversations/kb-chat")
            assert thread.json()["knowledge_base_id"] == base_id

            rebuild = await client.post(f"{BASES}/{base_id}/rebuild", json={"background": False})
            assert rebuild.status_code == 200, rebuild.text
            task = rebuild.json()
            assert task["state"] == "succeeded"
            assert task["target_index_version"] != active
            switch_path = f"{BASES}/{base_id}/indexes/{task['target_index_version']}/activate"
            with library_index_write(database):
                busy = await client.post(switch_path, json={"allow_stale": False})
            assert busy.status_code == 409
            assert busy.json()["code"] == "library_index_busy"

            switched = await client.post(switch_path, json={"allow_stale": False})
            assert switched.status_code == 200, switched.text
            assert switched.json()["scope_changed"] is False
            availability = (await client.get(f"{BASES}/{base_id}/index")).json()
            assert availability["active_index_version"] == task["target_index_version"]

            rollback = await client.post(
                f"{BASES}/{base_id}/indexes/{active}/activate",
                json={"allow_stale": False},
            )
            assert rollback.status_code == 200, rollback.text
            availability = (await client.get(f"{BASES}/{base_id}/index")).json()
            assert availability["active_index_version"] == active

            versions = (await client.get(f"{BASES}/{base_id}/indexes")).json()
            assert {item["index_version"] for item in versions} >= {active, task["target_index_version"]}
            historical = await client.post(f"{BASES}/default/indexes/legacy/activate", json={"allow_stale": False})
            assert historical.status_code == 200, historical.text
            assert historical.json()["scope_changed"] is False
            default_index = (await client.get(f"{BASES}/default/index")).json()
            assert default_index["active_index_version"] == "legacy"
            assert default_index["state"] == "legacy"

    asyncio.run(check())
