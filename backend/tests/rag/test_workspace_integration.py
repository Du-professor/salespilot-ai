import asyncio
import base64
import os
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.contracts.rag_workspace import RagUnavailable
from app.main import create_app
from app.rag import HashingEmbeddings, RagSettings, SqliteRagEngine
from app.rag.models import ModelClaim, ModelDraftPayload
from app.settings import Settings

from conftest import FIXTURES

ORIGIN = "http://test"
PREFIX = "/api/v1"
PASSWORD_A = "demo-a-test-password"
PASSWORD_B = "demo-b-test-password"


@asynccontextmanager
async def application(db_path, engine):
    app = create_app(
        Settings(
            database_path=db_path,
            web_origins=(ORIGIN,),
            demo_password_a=PASSWORD_A,
            demo_password_b=PASSWORD_B,
        ),
        rag_engine=engine,
    )
    async with app.router.lifespan_context(app):
        yield app


def browser(app):
    return AsyncClient(transport=ASGITransport(app=app), base_url=ORIGIN, headers={"Origin": ORIGIN})


async def login(client, username="sales-a", password=PASSWORD_A):
    response = await client.post(f"{PREFIX}/session", json={"username": username, "password": password})
    assert response.status_code == 200
    return response


def upload_body(path: Path, *, document_id=None, version="v1"):
    body = {
        "file": {"filename": path.name, "content_base64": base64.b64encode(path.read_bytes()).decode()},
        "version": version,
    }
    if document_id is not None:
        body["document_id"] = document_id
    return body


async def upload_and_publish(client, path: Path, *, document_id=None, version="v1"):
    response = await client.post(f"{PREFIX}/library/documents", json=upload_body(path, document_id=document_id, version=version))
    assert response.status_code == 201, response.text
    document = response.json()
    published = await client.post(
        f"{PREFIX}/library/documents/{document['document_id']}/{document['version']}/publish"
    )
    assert published.status_code == 200, published.text
    return published.json()


def make_engine(path: Path, *, artifacts: Path | None = None, chat=None):
    settings = RagSettings(index_path=path, embedding_provider="hash", chat_provider="template", docling_artifacts=artifacts)
    return SqliteRagEngine(settings, embedder=HashingEmbeddings(settings.embedding_dimensions), chat=chat)


async def ask_turn(client, text, *, conversation_id, request_id):
    response = await client.post(f"{PREFIX}/recommendations/turns", json={
        "conversation_id": conversation_id,
        "client_request_id": request_id,
        "text": text,
    })
    assert response.status_code == 200, response.text
    return response.json()


class DialogueChat:
    def __init__(self, reply=None):
        self.reply = reply
        self.compose_calls = 0
        self.draft_calls = 0

    def compose(self, *, system, user):
        self.compose_calls += 1
        if isinstance(self.reply, Exception):
            raise self.reply
        if callable(self.reply):
            return self.reply(user)
        return self.reply or "我是产品选型助手，请说明具体型号和需求。"

    def compose_draft(self, *, system, user, evidence):
        self.draft_calls += 1
        claims = tuple(
            ModelClaim(**{
                key: item.get(key)
                for key in (
                    "chunk_id", "model", "assembly", "document_id", "version", "page",
                    "section", "attribute", "value", "upper", "unit", "polarity", "excerpt",
                )
            })
            for item in evidence
        )
        return ModelDraftPayload(claims=claims)


def test_publish_recommend_archive_replay_restart_and_withdraw(tmp_path):
    async def check():
        database = tmp_path / "workspace.sqlite3"
        index = tmp_path / "index.sqlite3"
        engine = make_engine(index)
        sample = FIXTURES / "specs" / "sp-vision-cameras.md"

        async with application(database, engine) as app, browser(app) as client:
            await login(client)
            document = await upload_and_publish(client, sample)
            manual_path = f"{PREFIX}/library/documents/{document['document_id']}/{document['version']}"
            assert (await client.get(f"{manual_path}/file")).content == sample.read_bytes()

            question = {
                "conversation_id": "camera-chat",
                "client_request_id": "camera-request",
                "text": "需要一款 GigE 相机，分辨率至少 8 MP，标称分辨率下至少 100 fps。确认匹配后请附手册和选型报告。",
            }
            first = await ask_turn(client, question["text"], conversation_id="camera-chat", request_id="camera-request")
            assert first["status"] == "recommended"
            assert first["models"] == ["SP-VC-120"]
            assert first["has_report"] is True
            assert first["manuals"][0]["document_id"] == document["document_id"]
            assert first["citations"][0]["section"]

            history = (await client.get(f"{PREFIX}/recommendations/history")).json()
            assert len(history) == 1 and history[0]["reply"] == first
            report = await client.get(f"{PREFIX}/recommendations/history/{first['turn_id']}/report.md")
            assert report.status_code == 200
            assert "SP-VC-120 标准装配" in report.text

            replay = await client.post(f"{PREFIX}/recommendations/turns", json=question)
            assert replay.json() == first
            assert len((await client.get(f"{PREFIX}/recommendations/history")).json()) == 1

        restarted = make_engine(index)
        async with application(database, restarted) as app, browser(app) as client:
            await login(client)
            assert len((await client.get(f"{PREFIX}/recommendations/history")).json()) == 1
            replay = await client.post(f"{PREFIX}/recommendations/turns", json=question)
            assert replay.json() == first
            withdraw = await client.post(f"{manual_path}/withdraw")
            assert withdraw.status_code == 200 and withdraw.json()["status"] == "withdrawn"
            history = (await client.get(f"{PREFIX}/recommendations/history")).json()
            assert history[0]["available"] is False and history[0]["reply"] is None
            assert (await client.get(f"{PREFIX}/recommendations/history/{first['turn_id']}/report.md")).status_code == 404
            replay = await client.post(f"{PREFIX}/recommendations/turns", json=question)
            assert replay.status_code == 409
            assert replay.json()["code"] == "recommendation_sources_changed"
            after = await ask_turn(
                client, question["text"], conversation_id="camera-chat", request_id="after-withdrawal",
            )
            assert after["status"] == "insufficient_evidence"
            assert after["models"] == [] and after["manuals"] == []
            assert len((await client.get(f"{PREFIX}/recommendations/history")).json()) == 1

    asyncio.run(check())


def test_ipc_recommendation_without_attachment_is_not_archived(tmp_path):
    async def check():
        engine = make_engine(tmp_path / "index.sqlite3")
        async with application(tmp_path / "ipc.sqlite3", engine) as app, browser(app) as client:
            await login(client)
            for name in ("sp-ipc-620.md", "sp-ipc-670.md", "sp-ipc-680.md", "rtx-a2000-12gb.md"):
                await upload_and_publish(client, FIXTURES / "specs" / name)
            result = await ask_turn(
                client,
                "需要一台工控机，至少 6 个有外部供电的 GigE 相机直连接口，每路 300 Mbps，"
                "同时独立 Modbus TCP 连 PLC。安装 RTX A2000 12GB，GPU 电气 x16，"
                "GPU 供电余量至少 15 W，并预留一个电气 x4 插槽，带卡 0 至 45 °C。给出推荐和依据。",
                conversation_id="ipc-chat", request_id="ipc-request",
            )
            assert result["status"] == "recommended"
            assert result["models"] == ["SP-IPC-670"]
            assert result["manuals"] == [] and result["has_report"] is False
            assert (await client.get(f"{PREFIX}/recommendations/history")).json() == []

    asyncio.run(check())


def test_followup_and_no_match_do_not_archive(tmp_path):
    async def check():
        engine = make_engine(tmp_path / "index.sqlite3")
        async with application(tmp_path / "followup.sqlite3", engine) as app, browser(app) as client:
            await login(client)
            await upload_and_publish(client, FIXTURES / "specs" / "sp-vision-cameras.md")

            first = await ask_turn(
                client, "需要一款 12 MP 的工业相机，请帮我确认还需要哪些条件。",
                conversation_id="clarify", request_id="clarify-1",
            )
            assert first["status"] == "needs_input"
            assert first["models"] == [] and first["manuals"] == []
            assert (await client.get(f"{PREFIX}/recommendations/history")).json() == []

            second = await ask_turn(
                client, "GigE 接口，至少 100 fps。", conversation_id="clarify", request_id="clarify-2",
            )
            assert second["status"] == "recommended"
            assert second["models"] == ["SP-VC-120"]
            assert (await client.get(f"{PREFIX}/recommendations/history")).json() == []

            third = await ask_turn(
                client, "需要 GigE、至少 8 MP、标称分辨率下至少 1000 fps 的相机。",
                conversation_id="clarify", request_id="clarify-3",
            )
            assert third["status"] == "insufficient_evidence"
            assert third["models"] == []
            assert (await client.get(f"{PREFIX}/recommendations/history")).json() == []

    asyncio.run(check())


def test_ax920_and_robot_through_library(tmp_path):
    async def check():
        engine = make_engine(tmp_path / "index.sqlite3")
        async with application(tmp_path / "generic.sqlite3", engine) as app, browser(app) as client:
            await login(client)
            await upload_and_publish(client, FIXTURES / "rag" / "ax-920.md")
            camera = await ask_turn(
                client,
                "需要至少 8 MP、至少 100 fps 的 GigE 相机。",
                conversation_id="ax-chat", request_id="ax-request",
            )
            assert camera["status"] == "recommended"
            assert camera["models"] == ["AX-920"]
            assert camera["manuals"] == [] and camera["has_report"] is False

            await upload_and_publish(client, FIXTURES / "rag" / "arm-600.md")
            await upload_and_publish(client, FIXTURES / "rag" / "arm-300.md")
            robot = await ask_turn(
                client,
                "需要一台额定负载至少 8 kg、工作半径至少 1200 mm 的工业机器人。",
                conversation_id="robot-chat", request_id="robot-request",
            )
            assert robot["status"] == "recommended"
            assert robot["models"] == ["ARM-600-STD"]
            assert (await client.get(f"{PREFIX}/recommendations/history")).json() == []

    asyncio.run(check())


def test_hard_poe_requirement_is_not_archived_without_evidence(tmp_path):
    async def check():
        engine = make_engine(tmp_path / "index.sqlite3")
        async with application(tmp_path / "poe.sqlite3", engine) as app, browser(app) as client:
            await login(client)
            await upload_and_publish(client, FIXTURES / "rag" / "ax-920.md")
            result = await ask_turn(
                client,
                "需要 GigE、至少 8 MP、至少 100 fps 的相机，必须支持 PoE 供电，请附报告。",
                conversation_id="poe-chat", request_id="poe-request",
            )
            assert result["status"] == "insufficient_evidence"
            assert result["models"] == []
            assert result["manuals"] == [] and result["has_report"] is False
            assert (await client.get(f"{PREFIX}/recommendations/history")).json() == []

    asyncio.run(check())


def test_model_added_poe_is_absent_from_report(tmp_path):
    class AddingPoeChat:
        def compose(self, *, system, user):
            return "推荐 AX-920，支持 PoE 供电。"

        def compose_draft(self, *, system, user, evidence):
            claims = [
                ModelClaim(**{
                    key: item.get(key)
                    for key in (
                        "chunk_id", "model", "assembly", "document_id", "version", "page",
                        "section", "attribute", "value", "upper", "unit", "polarity", "excerpt",
                    )
                })
                for item in evidence
            ]
            first = evidence[0]
            claims.append(ModelClaim(
                chunk_id=first["chunk_id"], model=first["model"], assembly=first["assembly"],
                document_id=first["document_id"], version=first["version"], page=first["page"],
                section=first["section"], attribute="poe", value="True", unit=None,
                polarity="affirmed", excerpt=first["excerpt"],
            ))
            return ModelDraftPayload(claims=tuple(claims))

    async def check():
        engine = make_engine(tmp_path / "index.sqlite3", chat=AddingPoeChat())
        async with application(tmp_path / "report.sqlite3", engine) as app, browser(app) as client:
            await login(client)
            await upload_and_publish(client, FIXTURES / "rag" / "ax-920.md")
            result = await ask_turn(
                client,
                "需要至少 8 MP、至少 100 fps 的 GigE 相机，请附选型报告。",
                conversation_id="report-chat", request_id="report-request",
            )
            assert result["status"] == "recommended"
            assert result["has_report"] is True
            assert "PoE" not in result["answer"]
            report = await client.get(f"{PREFIX}/recommendations/history/{result['turn_id']}/report.md")
            assert report.status_code == 200
            assert "PoE" not in report.text

    asyncio.run(check())


def test_dialogue_spec_query_and_domain_out_do_not_archive(tmp_path):
    chat = DialogueChat(lambda user: "今天天气晴朗，气温 25 度。" if "天气" in user else "请说明具体型号。")

    async def check():
        engine = make_engine(tmp_path / "index.sqlite3", chat=chat)
        async with application(tmp_path / "dialogue.sqlite3", engine) as app, browser(app) as client:
            await login(client)
            document = await upload_and_publish(client, FIXTURES / "specs" / "sp-vision-cameras.md")
            manual_path = f"{PREFIX}/library/documents/{document['document_id']}/{document['version']}"
            first = await ask_turn(
                client,
                "需要一款 GigE 相机，分辨率至少 8 MP，标称分辨率下至少 100 fps。确认匹配后请附手册和选型报告。",
                conversation_id="dialogue", request_id="dialogue-1",
            )
            assert first["status"] == "recommended" and first["has_report"] is True
            assert chat.draft_calls == 1
            assert len((await client.get(f"{PREFIX}/recommendations/history")).json()) == 1

            second = await ask_turn(
                client, "刚才那款是什么网口？", conversation_id="dialogue", request_id="dialogue-2",
            )
            assert second["status"] == "recommended"
            assert second["models"] == ["SP-VC-120"]
            assert second["has_report"] is False and second["manuals"] == []
            assert second["citations"][0]["section"]
            assert len((await client.get(f"{PREFIX}/recommendations/history")).json()) == 1
            assert chat.compose_calls == 0

            third = await ask_turn(
                client, "今天天气怎么样？", conversation_id="dialogue", request_id="dialogue-3",
            )
            assert third["status"] == "needs_input"
            assert "天气" not in third["answer"] and "气温" not in third["answer"]
            assert len((await client.get(f"{PREFIX}/recommendations/history")).json()) == 1
            assert chat.compose_calls == 1

            assert (await client.post(f"{manual_path}/withdraw")).status_code == 200
            after = await ask_turn(
                client, "刚才那款是什么网口？", conversation_id="dialogue", request_id="dialogue-4",
            )
            assert after["status"] in {"needs_input", "insufficient_evidence"}
            assert "已撤回" in after["answer"] and "型号" in after["answer"]
            assert "SP-VC-120" not in after["answer"] and not after["citations"]
            assert chat.compose_calls == 1
            history = (await client.get(f"{PREFIX}/recommendations/history")).json()
            assert len(history) == 1 and history[0]["available"] is False

    asyncio.run(check())


def test_dialogue_timeout_returns_retryable_error(tmp_path):
    chat = DialogueChat(RagUnavailable("模型超时。"))

    async def check():
        engine = make_engine(tmp_path / "index.sqlite3", chat=chat)
        async with application(tmp_path / "timeout.sqlite3", engine) as app, browser(app) as client:
            await login(client)
            result = await client.post(f"{PREFIX}/recommendations/turns", json={
                "conversation_id": "timeout", "client_request_id": "timeout-1", "text": "你能做什么？",
            })
            assert (result.status_code, result.json()["code"]) == (503, "rag_unavailable")
            assert (await client.get(f"{PREFIX}/recommendations/history")).json() == []

    asyncio.run(check())


def test_pdf_page_citation_and_report_when_docling_is_ready(tmp_path):
    artifacts = os.getenv("SALESPILOT_DOCLING_ARTIFACTS")
    if not artifacts or not Path(artifacts).is_dir():
        pytest.skip("SALESPILOT_DOCLING_ARTIFACTS is not configured")
    pytest.importorskip("docling")

    async def check():
        engine = make_engine(tmp_path / "index.sqlite3", artifacts=Path(artifacts))
        async with application(tmp_path / "pdf.sqlite3", engine) as app, browser(app) as client:
            await login(client)
            sample = FIXTURES / "rag" / "demo-vision-manual.pdf"
            document = await upload_and_publish(client, sample)
            result = await ask_turn(
                client,
                "需要一款 GigE 相机，分辨率至少 8 MP，标称分辨率下至少 100 fps。确认匹配后请附手册和选型报告。",
                conversation_id="pdf-chat", request_id="pdf-request",
            )
            assert result["status"] == "recommended"
            assert result["models"] == ["SP-VC-120"]
            assert result["citations"][0]["page"] == 2
            assert result["manuals"][0]["filename"] == sample.name
            history = (await client.get(f"{PREFIX}/recommendations/history")).json()
            assert len(history) == 1
            report = await client.get(f"{PREFIX}/recommendations/history/{result['turn_id']}/report.md")
            assert report.status_code == 200 and "第 2 页" in report.text

    asyncio.run(check())
