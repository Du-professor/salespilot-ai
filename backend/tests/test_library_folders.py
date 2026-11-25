import asyncio
from io import BytesIO
from zipfile import ZipFile

from app.infrastructure.sqlite import DEMO_ORG
from app.main import create_app
from app.settings import Settings

from workspace_helpers import ORIGIN, PASSWORD_A, PASSWORD_B, PREFIX, browser, login
from test_rag_workspace import CASES, FIXTURES, HISTORY, LIBRARY, TURNS, WorkspaceStub, stub_application, upload_body

FOLDERS = f"{PREFIX}/library/folders"
BROWSER = f"{PREFIX}/library/browser"


async def folder(client, name, parent_id=None):
    response = await client.post(FOLDERS, json={"name": name, "parent_id": parent_id})
    assert response.status_code == 201, response.text
    return response.json()["folder_id"]


def test_nested_folders_support_mixed_uploads_and_role_filtered_navigation(tmp_path):
    sample = FIXTURES / "specs/sp-vision-cameras.md"
    pdf = FIXTURES / "rag/demo-vision-manual.pdf"
    document = tmp_path / "manual.docx"
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", "<document/>")
    document.write_bytes(buffer.getvalue())

    async def check():
        app = stub_application(tmp_path / "folders.sqlite3", WorkspaceStub())
        async with app.router.lifespan_context(app), browser(app) as owner, browser(app) as sales:
            assert (await owner.get(BROWSER)).status_code == 401
            await login(owner)
            await login(sales, "sales-b", PASSWORD_B)
            root = await folder(owner, "产品资料")
            target = await folder(owner, "工业相机", root)
            private = await folder(owner, "草稿", root)
            assert (await sales.post(FOLDERS, json={"name": "无权限"})).status_code == 403
            assert (await owner.post(FOLDERS, json={"name": "工业相机", "parent_id": root})).status_code == 409
            assert (await owner.post(FOLDERS, json={"name": "缺少父级", "parent_id": "missing"})).status_code == 404
            assert (await owner.post(LIBRARY, json=upload_body(sample, folder_id="missing"))).status_code == 404
            assert (await sales.get(BROWSER)).json() == {"folders": [], "documents": []}

            uploaded = []
            for path in (sample, pdf, document):
                response = await owner.post(LIBRARY, json=upload_body(path, folder_id=target))
                assert response.status_code == 201, response.text
                uploaded.append(response.json())
            tree = (await owner.get(BROWSER)).json()
            assert {entry["folder_id"] for entry in tree["folders"]} == {root, target, private}
            assert {entry["filename"] for entry in tree["documents"]} == {sample.name, pdf.name, document.name}
            assert {entry["folder_id"] for entry in tree["documents"]} == {target}

            key = f"{LIBRARY}/{uploaded[0]['document_id']}/v1"
            assert (await owner.post(key + "/publish")).status_code == 200
            visible = (await sales.get(BROWSER)).json()
            assert {entry["folder_id"] for entry in visible["folders"]} == {root, target}
            assert [entry["filename"] for entry in visible["documents"]] == [sample.name]

    asyncio.run(check())


def test_moving_all_versions_preserves_the_index_and_existing_report(tmp_path):
    sample = FIXTURES / "specs/sp-vision-cameras.md"
    database = tmp_path / "move.sqlite3"
    settings = Settings(database_path=database, web_origins=(ORIGIN,), demo_password_a=PASSWORD_A,
                        demo_password_b=PASSWORD_B, rag_mode="demo")

    async def check():
        app = create_app(settings)
        async with app.router.lifespan_context(app), browser(app) as owner, browser(app) as sales:
            await login(owner)
            await login(sales, "sales-b", PASSWORD_B)
            first = (await owner.post(LIBRARY, json=upload_body(sample))).json()
            assert (await owner.get(BROWSER)).json()["documents"][0]["folder_id"] is None
            doc_id = first["document_id"]
            path = f"{LIBRARY}/{doc_id}"
            assert (await owner.post(LIBRARY, json=upload_body(sample, document_id=doc_id, version="v2"))).status_code == 201
            published = await owner.post(path + "/v1/publish")
            assert published.status_code == 200, published.text
            core = app.state.rag_engine
            chunk_ids = core.store.chunk_ids(DEMO_ORG, doc_id, "v1")
            assert chunk_ids
            question = {"conversation_id": "existing", "client_request_id": "before-move", "text": CASES["vision"]["text"]}
            selection = await owner.post(TURNS, json=question)
            assert selection.status_code == 200, selection.text
            reply = selection.json()
            assert reply["models"] == CASES["vision"]["expected_models"]
            report = f"{HISTORY}/{reply['turn_id']}/report.md"
            before = await owner.get(report)
            assert before.status_code == 200
            target = await folder(owner, "相机")
            assert (await sales.patch(path + "/location", json={"folder_id": target})).status_code == 403
            assert (await owner.patch(path + "/location", json={"folder_id": "missing"})).status_code == 404
            assert (await owner.patch(path + "/location", json={"folder_id": target})).status_code == 204
            assert (await owner.post(LIBRARY, json=upload_body(sample, folder_id=None))).status_code == 409
            assert (await owner.post(TURNS, json=question)).json() == reply
            assert (await owner.get(report)).content == before.content
            assert (await owner.get(path + "/v1/file")).content == sample.read_bytes()
            assert core.store.chunk_ids(DEMO_ORG, doc_id, "v1") == chunk_ids
            fresh = (await owner.post(TURNS, json={**question, "client_request_id": "after-move"})).json()
            assert fresh["citations"] == reply["citations"]

        app = create_app(settings)
        async with app.router.lifespan_context(app), browser(app) as owner:
            await login(owner)
            items = (await owner.get(BROWSER)).json()["documents"]
            assert {(item["version"], item["folder_id"]) for item in items} == {("v1", target), ("v2", target)}
            assert (await owner.patch(path + "/location", json={"folder_id": None})).status_code == 204
            assert all(item["folder_id"] is None for item in (await owner.get(BROWSER)).json()["documents"])

    asyncio.run(check())
