import asyncio

from app.main import create_app
from app.settings import Settings
from workspace_helpers import ORIGIN, PASSWORD_A, PASSWORD_B, PREFIX, browser, login
from test_rag_workspace import CASES, FIXTURES, HISTORY, LIBRARY, upload_body


def test_library_inspection_reads_real_index_and_never_archives(tmp_path):
    async def check():
        app = create_app(Settings(
            database_path=tmp_path / "knowledge.sqlite3", web_origins=(ORIGIN,),
            demo_password_a=PASSWORD_A, demo_password_b=PASSWORD_B, rag_mode="demo",
        ))
        async with app.router.lifespan_context(app), browser(app) as admin, browser(app) as sales:
            await login(admin)
            await login(sales, "sales-b", PASSWORD_B)
            fixture = FIXTURES / "specs/sp-vision-cameras.md"
            upload = await admin.post(LIBRARY, json=upload_body(fixture))
            assert upload.status_code == 201, upload.text
            base = f"{LIBRARY}/{upload.json()['document_id']}"
            assert (await sales.get(base + "/versions")).status_code == 404
            assert (await admin.get(base + "/v1/chunks")).json()["total"] == 0
            published = await admin.post(base + "/v1/publish")
            assert published.status_code == 200, published.text

            versions = (await sales.get(base + "/versions")).json()
            assert versions == [published.json()]
            first_page = (await sales.get(base + "/v1/chunks", params={"limit": 1})).json()
            assert first_page["total"] > 1
            assert len(first_page["chunks"]) == 1
            assert first_page["chunks"][0]["section"]
            next_page = (await sales.get(base + "/v1/chunks", params={"offset": 1, "limit": 1})).json()
            assert next_page["chunks"][0]["chunk_id"] != first_page["chunks"][0]["chunk_id"]

            diagnostic = f"{PREFIX}/library/retrieval-test"
            assert (await sales.post(diagnostic, json={"text": CASES["vision"]["text"]})).status_code == 403
            result = await admin.post(diagnostic, json={"text": CASES["vision"]["text"]})
            assert result.status_code == 200, result.text
            detail = result.json()
            assert detail["status"] == "retrieved"
            assert detail["hits"][0]["filename"] == fixture.name
            assert detail["hits"][0]["version"] == "v1"
            assert -1.00001 <= detail["hits"][0]["score"] <= 1.00001
            assert (await admin.get(HISTORY)).json() == []

            query = {"text": "IPC410-DEMO1 的接口是什么？"}
            assert (await admin.post(diagnostic, json=query)).json()["status"] == "spec_query"
            assert (await admin.post(base + "/v1/withdraw")).status_code == 200
            assert (await sales.get(base + "/v1/chunks")).status_code == 404
            assert (await admin.get(base + "/v1/chunks")).json()["total"] == 0
            unavailable = await admin.post(diagnostic, json={"text": CASES["vision"]["text"]})
            assert unavailable.json()["status"] == "insufficient_evidence"
            assert unavailable.json()["hits"] == []

    asyncio.run(check())
