import asyncio
import base64
import json
from io import BytesIO
from pathlib import Path
from threading import Event
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

from app.contracts.rag_workspace import Citation, DocumentVersionKey, RagDocumentError, RagRecommendation, RagUnavailable
from app.infrastructure.library import library_index_write
from app.main import create_app
from app.settings import Settings

from workspace_helpers import ORIGIN, PASSWORD_A, PASSWORD_B, PREFIX, application, browser, login

LIBRARY = f"{PREFIX}/library/documents"
TURNS = f"{PREFIX}/recommendations/turns"
HISTORY = f"{PREFIX}/recommendations/history"
CONVERSATIONS = f"{PREFIX}/recommendations/conversations"
FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
CASES = json.loads((FIXTURES / "rag/cases.json").read_text(encoding="utf-8"))


class WorkspaceStub:
    """Predetermined responses to exercise the API adapter boundary."""

    def __init__(self):
        self.indexed = {}
        self.contexts = []
        self.fail_index = False
        self.fail_model = False
        self.forge_citation = False

    def index_document(self, org_id, doc, payload):
        if self.fail_index:
            raise RagDocumentError()
        self.indexed[(org_id, doc.document_id, doc.version)] = payload

    def recommend(self, context):
        self.contexts.append(context)
        if self.fail_model:
            raise RagUnavailable()
        if "补充" in context.text:
            return RagRecommendation(status="needs_input", answer="请提供相机接口。")
        if not context.documents or context.text == CASES["no_match"]["text"]:
            return RagRecommendation(status="insufficient_evidence", answer="当前资料中没有符合要求的产品。")
        doc = context.documents[0]
        assert (context.org_id, doc.document_id, doc.version) in self.indexed
        key = DocumentVersionKey(document_id=doc.document_id, version="forged" if self.forge_citation else doc.version)
        return RagRecommendation(
            status="recommended", models=("SP-VC-120",),
            answer="推荐 SP-VC-120。手册第 2 页标注 GigE 接口和 12 MP 下 120 fps。",
            citations=(Citation(**key.model_dump(), page=2, excerpt="12 MP resolution at 120 fps"),),
            manuals=(key,), include_report=True,
        )


def stub_application(database_path, engine, *, models_dir=None):
    fields = {
        "database_path": database_path,
        "web_origins": (ORIGIN,),
        "demo_password_a": PASSWORD_A,
        "demo_password_b": PASSWORD_B,
    }
    if models_dir is not None:
        fields["rag_models_dir"] = models_dir
    return create_app(
        Settings(**fields), rag_engine=engine,
    )


def upload_body(path, **fields):
    return {"file": {
        "filename": path.name, "content_base64": base64.b64encode(path.read_bytes()).decode(),
    }, **fields}


def test_unwired_app_keeps_user_file_and_reports_core_unavailable(tmp_path):
    async def check():
        path = FIXTURES / "rag/demo-vision-manual.pdf"
        async with application(tmp_path / "rag.sqlite3") as app, browser(app) as client:
            assert (await client.get(LIBRARY)).status_code == 401
            await login(client)
            assert (await client.get(LIBRARY)).json() == []
            upload = await client.post(LIBRARY, json=upload_body(path))
            assert upload.status_code == 201, upload.text
            doc = upload.json()
            key = f"{LIBRARY}/{doc['document_id']}/{doc['version']}"
            assert (await client.get(f"{key}/file")).content == path.read_bytes()
            result = await client.post(f"{key}/publish")
            assert (result.status_code, result.json()["code"]) == (503, "rag_unavailable")
            assert (await client.get(LIBRARY)).json() == [doc]
            result = await client.post(TURNS, json={
                "conversation_id": "conversation-1", "client_request_id": "request-1", "text": "需要相机",
            })
            assert (result.status_code, result.json()["code"]) == (503, "rag_unavailable")
            assert (await client.get(HISTORY)).json() == []
    asyncio.run(check())


def test_standard_startup_enables_offline_rag_when_demo_mode_is_set(tmp_path, monkeypatch):
    database = tmp_path / "workspace.sqlite3"
    monkeypatch.setenv("SALESPILOT_DB_PATH", str(database))
    monkeypatch.setenv("SALESPILOT_DEMO_PASSWORD_A", PASSWORD_A)
    monkeypatch.setenv("SALESPILOT_WEB_ORIGINS", ORIGIN)
    monkeypatch.setenv("SALESPILOT_RAG_MODE", "demo")
    monkeypatch.delenv("SALESPILOT_RAG_INDEX_PATH", raising=False)
    monkeypatch.setenv("SALESPILOT_RAG_CHAT_PROVIDER", "openai")
    monkeypatch.setenv("SALESPILOT_RAG_EMBEDDING_PROVIDER", "openai")
    monkeypatch.delenv("SALESPILOT_RAG_CHAT_API_KEY", raising=False)
    monkeypatch.delenv("SALESPILOT_RAG_EMBEDDING_API_KEY", raising=False)

    async def check():
        app = create_app()
        async with app.router.lifespan_context(app), browser(app) as client:
            await login(client)
            uploaded = await client.post(LIBRARY, json=upload_body(FIXTURES / "specs/sp-vision-cameras.md"))
            assert uploaded.status_code == 201, uploaded.text
            document_id = uploaded.json()["document_id"]
            published = await client.post(f"{LIBRARY}/{document_id}/v1/publish")
            assert published.status_code == 200, published.text
            reply = await client.post(TURNS, json={
                "conversation_id": "demo-chat", "client_request_id": "demo-request", "text": CASES["vision"]["text"],
            })
            assert reply.status_code == 200, reply.text
            assert reply.json()["models"] == CASES["vision"]["expected_models"]
            assert reply.json()["manuals"][0]["document_id"] == document_id
            assert reply.json()["has_report"] is True
            assert (await client.get(HISTORY)).json()[0]["reply"]["turn_id"] == reply.json()["turn_id"]

    asyncio.run(check())
    assert database.with_name("workspace-rag.sqlite3").is_file()


def test_library_docx_supports_larger_upload_and_expanded_word_archive(tmp_path):
    buffer = BytesIO()
    with ZipFile(buffer, mode="w") as archive:
        archive.writestr("word/document.xml", b"<document/>")
        archive.writestr("word/media/image.bin", b"a" * 2_100_000, compress_type=ZIP_STORED)
        compact = ZipInfo("word/media/compressed.bin")
        compact.compress_type = ZIP_DEFLATED
        with archive.open(compact, "w") as member:
            member.write(b"b" * 10_100_000)
    file = tmp_path / "large.docx"
    file.write_bytes(buffer.getvalue())

    async def check():
        app = stub_application(tmp_path / "large-library.sqlite3", WorkspaceStub())
        async with app.router.lifespan_context(app), browser(app) as owner:
            await login(owner)
            result = await owner.post(LIBRARY, json=upload_body(file))
            assert result.status_code == 201, result.text
            row = result.json()
            assert row["size_bytes"] == file.stat().st_size
            url = f"{LIBRARY}/{row['document_id']}/v1"
            assert (await owner.post(url + "/publish")).status_code == 200
            assert (await owner.get(url + "/file")).content == file.read_bytes()

    asyncio.run(check())


def test_published_manual_recommendation_archives_once_and_survives_restart(tmp_path):
    async def check():
        path = tmp_path / "persist.sqlite3"
        core = WorkspaceStub()
        app = stub_application(path, core)
        sample = FIXTURES / CASES["vision"]["files"][0]
        question = {"conversation_id": "camera-chat", "client_request_id": "camera-request", "text": CASES["vision"]["text"]}
        async with app.router.lifespan_context(app), browser(app) as owner, browser(app) as reader:
            await login(owner)
            await login(reader, "sales-b", PASSWORD_B)
            assert (await owner.get(f"{PREFIX}/catalog")).status_code == 404
            assert (await reader.post(LIBRARY, json=upload_body(sample))).status_code == 403
            uploaded = await owner.post(LIBRARY, json=upload_body(sample))
            assert uploaded.status_code == 201, uploaded.text
            doc = uploaded.json()
            manual_path = f"{LIBRARY}/{doc['document_id']}/{doc['version']}"
            assert (await reader.get(LIBRARY)).json() == []
            assert (await reader.get(f"{manual_path}/file")).status_code == 404
            assert (await owner.post(LIBRARY, json=upload_body(sample))).json() == doc
            published = await owner.post(f"{manual_path}/publish")
            assert published.json() == {**doc, "status": "published", "index_status": "ready"}
            assert (await owner.post(f"{manual_path}/publish")).json() == published.json()
            assert next(iter(core.indexed.values())) == sample.read_bytes()
            assert (await reader.get(f"{manual_path}/file")).content == sample.read_bytes()

            answers = await asyncio.gather(*(owner.post(TURNS, json=question) for _ in range(2)))
            assert all(answer.status_code == 200 for answer in answers)
            reply = answers[0].json()
            assert reply == answers[1].json()
            assert reply["models"] == CASES["vision"]["expected_models"]
            assert reply["citations"][0]["page"] == CASES["vision"]["citation"]["page"]
            assert reply["citations"][0]["filename"] == sample.name
            assert (await owner.post(TURNS, json={**question, "text": "改为其他需求"})).status_code == 409
            records = (await owner.get(HISTORY)).json()
            assert len(records) == 1 and records[0]["reply"] == reply
            assert (await reader.get(HISTORY)).json() == []
            report = f"{HISTORY}/{reply['turn_id']}/report.md"
            assert (await reader.get(report)).status_code == 404
            assert "第 2 页" in (await owner.get(report)).text

        app = stub_application(path, core)
        async with app.router.lifespan_context(app), browser(app) as owner, browser(app) as reader:
            thread_url = f"{CONVERSATIONS}/{question['conversation_id']}"
            assert (await owner.get(CONVERSATIONS)).status_code == 401
            await login(owner)
            await login(reader, "sales-b", PASSWORD_B)
            assert (await reader.get(CONVERSATIONS)).json() == {"items": [], "next_offset": None}
            assert (await reader.get(thread_url)).status_code == 404
            assert (await owner.get(CONVERSATIONS)).json() == {"items": [{
                "conversation_id": question["conversation_id"],
                "title": question["text"],
                "updated_at": reply["created_at"],
                "pinned": False,
            }], "next_offset": None}
            assert [item["conversation_id"] for item in (await owner.get(
                CONVERSATIONS, params={"q": "手册第 2 页标注"},
            )).json()["items"]] == [question["conversation_id"]]
            initial_turn = {"turn_id": reply["turn_id"], "text": question["text"], "reply": reply}
            assert (await owner.get(thread_url)).json()["turns"] == [initial_turn]
            assert (await owner.get(HISTORY)).json() == records
            clarification = await owner.post(TURNS, json={
                **question, "text": "请补充确认条件。", "client_request_id": "followup",
            })
            assert clarification.json()["status"] == "needs_input"
            assert core.contexts[-1].previous_turns[0].text == question["text"]
            assert (await owner.get(thread_url)).json()["turns"] == [initial_turn, {
                "turn_id": clarification.json()["turn_id"],
                "text": "请补充确认条件。",
                "reply": clarification.json(),
            }]
            assert (await owner.get(HISTORY)).json() == records
            assert (await owner.post(f"{manual_path}/withdraw")).json()["status"] == "withdrawn"
            assert (await owner.get(f"{manual_path}/file")).status_code == 404
            assert (await owner.get(report)).status_code == 404
            assert (await owner.get(CONVERSATIONS, params={"q": "手册第 2 页标注"})).json()["items"] == []
            record = (await owner.get(HISTORY)).json()[0]
            assert record["available"] is False and record["reply"] is None
            thread = (await owner.get(thread_url)).json()["turns"]
            assert thread[0] == {**initial_turn, "reply": None}
            assert thread[1]["reply"] == clarification.json()
            replay = await owner.post(TURNS, json=question)
            assert (replay.status_code, replay.json()["code"]) == (409, "recommendation_sources_changed")
            after = await owner.post(TURNS, json={**question, "client_request_id": "after-withdrawal"})
            assert after.json()["status"] == "insufficient_evidence"
            assert core.contexts[-1].documents == ()
            assert all(turn.text != question["text"] for turn in core.contexts[-1].previous_turns)

    asyncio.run(check())


def test_conversation_list_and_turn_pages_preserve_order_and_scope(tmp_path, monkeypatch):
    monkeypatch.setattr("app.api.recommendations.CONVERSATION_PAGE_SIZE", 2)

    async def check():
        app = stub_application(tmp_path / "conversations.sqlite3", WorkspaceStub())
        async with app.router.lifespan_context(app), browser(app) as owner, browser(app) as other:
            await login(owner)
            await login(other, "sales-b", PASSWORD_B)

            async def send(client, thread, index):
                response = await client.post(TURNS, json={
                    "conversation_id": thread, "client_request_id": f"turn-{index}",
                    "text": f"请补充选型条件 {index}。",
                })
                assert response.status_code == 200, response.text
                return response.json()["turn_id"]

            first = await send(owner, "first", 1)
            await send(owner, "second", 2)
            unrelated = await send(owner, "third", 3)
            fourth = await send(owner, "first", 4)
            fifth = await send(owner, "first", 5)
            other_turn = await send(other, "first", 1)
            result = (await owner.get(CONVERSATIONS)).json()
            assert [(item["conversation_id"], item["title"]) for item in result["items"]] == [
                ("first", "请补充选型条件 1。"), ("third", "请补充选型条件 3。"),
            ]
            assert result["next_offset"] == 2
            result = (await owner.get(CONVERSATIONS, params={"offset": result["next_offset"]})).json()
            assert [item["conversation_id"] for item in result["items"]] == ["second"]
            assert result["next_offset"] is None

            url = f"{CONVERSATIONS}/first"
            recent = (await owner.get(url)).json()
            assert [turn["turn_id"] for turn in recent["turns"]] == [fourth, fifth]
            assert recent["before"] == fourth
            await send(owner, "first", 6)
            older = (await owner.get(url, params={"before": recent["before"]})).json()
            assert [turn["turn_id"] for turn in older["turns"]] == [first]
            assert older["before"] is None
            for cursor in (unrelated, other_turn):
                assert (await owner.get(url, params={"before": cursor})).status_code == 404
            assert (await other.get(url, params={"before": fourth})).status_code == 404
            assert [turn["turn_id"] for turn in (await other.get(url)).json()["turns"]] == [other_turn]

            pinned = f"{CONVERSATIONS}/second"
            assert (await other.patch(pinned, json={"pinned": True})).status_code == 404
            assert (await owner.patch(pinned, json={})).status_code == 422
            assert (await owner.patch(pinned, json={"pinned": True, "title": "视觉项目"})).status_code == 204
            assert (await owner.patch(pinned, json={"title": "视觉项目 Beta"})).status_code == 204
            items = (await owner.get(CONVERSATIONS)).json()["items"]
            assert (items[0]["conversation_id"], items[0]["title"], items[0]["pinned"]) == (
                "second", "视觉项目 Beta", True,
            )
            assert (await owner.get(CONVERSATIONS, params={"q": "BETA"})).json()["items"] == [items[0]]
            assert (await owner.get(CONVERSATIONS, params={"q": "条件 2。"})).json()["items"] == [items[0]]
            assert (await other.get(CONVERSATIONS, params={"q": "beta"})).json()["items"] == []
            assert (await owner.patch(pinned, json={"pinned": False})).status_code == 204
            assert (await owner.get(CONVERSATIONS)).json()["items"][0]["conversation_id"] == "first"

    asyncio.run(check())


def test_delete_conversation_clears_memory_rejects_inflight_write_and_keeps_report(tmp_path):
    async def check():
        core = WorkspaceStub()
        original_recommend = core.recommend
        started, release = Event(), Event()

        def controlled_reply(context):
            if context.text == "补充，请等待此轮生成。":
                started.set()
                assert release.wait(10)
            return original_recommend(context)

        core.recommend = controlled_reply
        app = stub_application(tmp_path / "deleted-conversation.sqlite3", core)
        async with app.router.lifespan_context(app), browser(app) as owner, browser(app) as other:
            await login(owner)
            await login(other, "sales-b", PASSWORD_B)
            doc = (await owner.post(LIBRARY, json=upload_body(FIXTURES / "specs/sp-vision-cameras.md"))).json()
            assert (await owner.post(f"{LIBRARY}/{doc['document_id']}/v1/publish")).status_code == 200
            body = {"conversation_id": "removed", "client_request_id": "archive", "text": CASES["vision"]["text"]}
            reply = await owner.post(TURNS, json=body)
            assert reply.status_code == 200
            followup = {**body, "client_request_id": "clarify", "text": "补充，临时草稿词。"}
            assert (await owner.post(TURNS, json=followup)).status_code == 200
            assert (await owner.patch(f"{CONVERSATIONS}/removed", json={"title": "临时会话标题", "pinned": True})).status_code == 204
            assert (await other.delete(f"{CONVERSATIONS}/removed")).status_code == 404
            pending = {**followup, "client_request_id": "pending", "text": "补充，请等待此轮生成。"}
            background = asyncio.create_task(owner.post(TURNS, json=pending))
            try:
                assert await asyncio.to_thread(started.wait, 5)
                assert (await owner.delete(f"{CONVERSATIONS}/removed")).status_code == 204
            finally:
                release.set()
            assert (await background).status_code == 404
            assert (await owner.delete(f"{CONVERSATIONS}/removed")).status_code == 204
            assert (await owner.get(f"{CONVERSATIONS}/removed")).status_code == 404
            assert (await owner.get(CONVERSATIONS, params={"q": "临时"})).json()["items"] == []
            assert (await owner.post(TURNS, json=body)).status_code == 404
            archive = (await owner.get(HISTORY)).json()
            assert len(archive) == 1 and archive[0]["reply"] == reply.json()
            assert (await owner.get(f"{HISTORY}/{reply.json()['turn_id']}/report.md")).status_code == 200
            with app.state.database.connect() as connection:
                assert connection.execute(
                    "SELECT client_request_id FROM rag_recommendation_turns WHERE conversation_id='removed'",
                ).fetchall()[0][0] == "archive"
                assert connection.execute(
                    "SELECT count(*) FROM rag_recommendation_turns WHERE conversation_id='removed'",
                ).fetchone()[0] == 1
                state = connection.execute(
                    "SELECT title, pinned, deleted FROM rag_conversation_settings WHERE conversation_id='removed'",
                ).fetchone()
                assert tuple(state) == (None, 0, 1)
            assert (await owner.post(TURNS, json={**pending, "conversation_id": "fresh", "text": "补充其他需求。"})).status_code == 200
            assert core.contexts[-1].previous_turns == ()

    asyncio.run(check())


def test_delete_library_version_disables_reads_during_retry_and_purges_exact_version(tmp_path):
    class RemovableStub(WorkspaceStub):
        fail_removal = True

        def remove_document(self, org_id, key):
            if self.fail_removal:
                raise RagUnavailable()
            self.indexed.pop((org_id, key.document_id, key.version), None)

    async def check():
        core = RemovableStub()
        app = stub_application(tmp_path / "delete-manual.sqlite3", core)
        file = FIXTURES / "specs/sp-vision-cameras.md"
        async with app.router.lifespan_context(app), browser(app) as owner, browser(app) as reader:
            await login(owner)
            await login(reader, "sales-b", PASSWORD_B)
            original = (await owner.post(LIBRARY, json=upload_body(file))).json()
            doc_id = original["document_id"]
            base = f"{LIBRARY}/{doc_id}"
            assert (await owner.post(LIBRARY, json=upload_body(file, document_id=doc_id, version="v2"))).status_code == 201
            for version in ("v1", "v2"):
                assert (await owner.post(f"{base}/{version}/publish")).status_code == 200
            assert (await reader.delete(f"{base}/v1")).status_code == 403
            untrusted = await owner.delete(f"{base}/v1", headers={"Origin": "https://other.example"})
            assert untrusted.status_code == 403
            with library_index_write(app.state.database.path):
                busy = await owner.delete(f"{base}/v1")
                assert (busy.status_code, busy.json()["code"]) == (409, "library_index_busy")
            assert (await owner.get(f"{base}/v1/file")).status_code == 200
            failed = await owner.delete(f"{base}/v1")
            assert (failed.status_code, failed.json()["code"]) == (503, "rag_index_removal_failed")
            assert (await owner.get(f"{base}/v1/file")).status_code == 404
            assert next(doc for doc in (await owner.get(LIBRARY)).json() if doc["version"] == "v1")["status"] == "deleting"
            assert [doc["version"] for doc in (await reader.get(LIBRARY)).json()] == ["v2"]
            assert (await owner.post(f"{base}/v1/publish")).status_code == 409
            checked = await owner.post(TURNS, json={
                "conversation_id": "after-delete", "client_request_id": "verify", "text": CASES["vision"]["text"],
            })
            assert checked.json()["manuals"][0]["version"] == "v2"
            core.fail_removal = False
            assert (await owner.delete(f"{base}/v1")).status_code == 204
            assert (await owner.delete(f"{base}/v1")).status_code == 204
            assert [doc["version"] for doc in (await owner.get(LIBRARY)).json()] == ["v2"]
            assert [key[2] for key in core.indexed] == ["v2"]
            assert (await owner.get(f"{base}/v2/file")).content == file.read_bytes()
            assert (await owner.post(LIBRARY, json=upload_body(file, document_id=doc_id))).status_code == 409
            uploaded = (await owner.post(LIBRARY, json=upload_body(file))).json()
            assert uploaded["document_id"] != doc_id
            assert (await owner.post(LIBRARY, json=upload_body(file))).json() == uploaded
            assert (await owner.get(f"{base}/v1/file")).status_code == 404

    asyncio.run(check())


def test_failed_index_model_and_ungranted_citation_are_retryable_without_archive(tmp_path):
    async def check():
        core = WorkspaceStub()
        app = stub_application(tmp_path / "errors.sqlite3", core)
        async with app.router.lifespan_context(app), browser(app) as client:
            await login(client)
            sample = FIXTURES / "rag/demo-vision-manual.pdf"
            uploaded = (await client.post(LIBRARY, json=upload_body(sample))).json()
            doc_path = f"{LIBRARY}/{uploaded['document_id']}/v1"
            incompatible = upload_body(FIXTURES / "specs/sp-vision-cameras.md", document_id=uploaded["document_id"])
            assert (await client.post(LIBRARY, json=incompatible)).status_code == 409
            version = await client.post(LIBRARY, json={**incompatible, "version": "v2"})
            assert version.json()["document_id"] == uploaded["document_id"]
            assert version.json()["version"] == "v2"
            core.fail_index = True
            failed = await client.post(f"{doc_path}/publish")
            assert (failed.status_code, failed.json()["code"]) == (422, "rag_document_invalid")
            assert uploaded in (await client.get(LIBRARY)).json()
            core.fail_index = False
            assert (await client.post(f"{doc_path}/publish")).status_code == 200
            unavailable = await client.delete(doc_path)
            assert (unavailable.status_code, unavailable.json()["code"]) == (503, "rag_index_removal_unavailable")
            assert (await client.get(f"{doc_path}/file")).status_code == 200
            question = {"conversation_id": "camera-chat", "client_request_id": "camera-request", "text": CASES["vision"]["text"]}
            core.fail_model = True
            failed = await client.post(TURNS, json=question)
            assert (failed.status_code, failed.json()["code"]) == (503, "rag_unavailable")
            core.fail_model = False
            core.forge_citation = True
            failed = await client.post(TURNS, json=question)
            assert (failed.status_code, failed.json()["code"]) == (502, "rag_invalid_response")
            assert (await client.get(HISTORY)).json() == []
            core.forge_citation = False
            assert (await client.post(TURNS, json=question)).status_code == 200
            assert len((await client.get(HISTORY)).json()) == 1

    asyncio.run(check())
