import asyncio
import sqlite3

from app.api.knowledge_bases import EmbeddingModelOption, embedding_model_options

from workspace_helpers import PASSWORD_B, PREFIX, browser, login
from test_rag_workspace import FIXTURES, LIBRARY, WorkspaceStub, stub_application, upload_body

BASES = f"{PREFIX}/library/knowledge-bases"
MODELS = f"{PREFIX}/library/embedding-models"
FOLDERS = f"{PREFIX}/library/folders"
BROWSER = f"{PREFIX}/library/browser"
SAMPLE = FIXTURES / "specs/sp-vision-cameras.md"


def test_base_scopes_folders_uploads_and_index_availability(tmp_path):
    async def check():
        engine = WorkspaceStub()
        app = stub_application(
            tmp_path / "bases.sqlite3", engine, models_dir=tmp_path / "empty-models",
        )
        async with app.router.lifespan_context(app), browser(app) as manager, browser(app) as reader:
            assert (await manager.get(BASES)).status_code == 401
            await login(manager)
            await login(reader, "sales-b", PASSWORD_B)
            assert [item["knowledge_base_id"] for item in (await manager.get(BASES)).json()] == ["default"]
            models = {item["model_id"]: item for item in (await manager.get(MODELS)).json()}
            assert models["workspace-default"]["selectable"] is True
            assert models["bge-m3"]["selectable"] is False
            assert (await reader.get(MODELS)).status_code == 403
            data = {"name": "相机手册", "description": "规格与安装手册", "embedding_model_id": "workspace-default"}
            assert (await reader.post(BASES, json=data)).status_code == 403
            assert (await manager.post(BASES, json={**data, "embedding_model_id": "bge-m3"})).status_code == 409
            created = await manager.post(BASES, json=data)
            assert created.status_code == 201, created.text
            base = created.json()
            assert base["index_state"] == "awaiting_adapter"
            scope = {"knowledge_base_id": base["knowledge_base_id"]}
            assert (await manager.post(BASES, json=data)).status_code == 409

            original = await manager.post(FOLDERS, json={"name": "产品资料"})
            local = await manager.post(FOLDERS, json={"name": "产品资料", **scope})
            assert original.status_code == local.status_code == 201, local.text
            original_folder = original.json()["folder_id"]
            local_folder = local.json()["folder_id"]
            assert (await manager.post(FOLDERS, json={"name": "子目录", "parent_id": original_folder, **scope})).status_code == 404
            legacy = await manager.post(LIBRARY, json=upload_body(SAMPLE, folder_id=original_folder))
            scoped = await manager.post(LIBRARY, json=upload_body(SAMPLE, folder_id=local_folder, **scope))
            assert legacy.status_code == scoped.status_code == 201, scoped.text
            legacy_id = legacy.json()["document_id"]
            scoped_id = scoped.json()["document_id"]
            assert legacy_id != scoped_id
            assert [item["document_id"] for item in (await manager.get(BROWSER)).json()["documents"]] == [legacy_id]
            assert [item["document_id"] for item in (await manager.get(BROWSER, params=scope)).json()["documents"]] == [scoped_id]
            assert (await manager.post(LIBRARY, json=upload_body(SAMPLE, document_id=legacy_id, version="v2", **scope))).status_code == 404
            assert (await manager.post(LIBRARY, json=upload_body(SAMPLE, document_id=scoped_id, version="v2", **scope))).status_code == 201
            assert (await manager.patch(f"{LIBRARY}/{scoped_id}/location", json={"folder_id": original_folder})).status_code == 404
            assert (await manager.patch(f"{LIBRARY}/{scoped_id}/location", json={"folder_id": None})).status_code == 204
            assert {item["folder_id"] for item in (await manager.get(BROWSER, params=scope)).json()["documents"]} == {None}
            assert (await reader.get(BROWSER, params=scope)).status_code == 404
            assert (await reader.get(f"{LIBRARY}/{scoped_id}/v1/file")).status_code == 404
            assert (await manager.get(f"{LIBRARY}/{scoped_id}/v1/file")).content == SAMPLE.read_bytes()
            assert (await manager.post(f"{LIBRARY}/{scoped_id}/v1/publish")).json()["code"] == "knowledge_base_index_pending"
            assert (await manager.post(f"{PREFIX}/library/retrieval-test", json={"text": "GigE 相机", **scope})).json()["code"] == "knowledge_base_index_pending"
            assert engine.indexed == {}
            assert (await manager.delete(f"{LIBRARY}/{scoped_id}/v1")).status_code == 204
            assert (await manager.delete(f"{LIBRARY}/{scoped_id}/v1")).status_code == 204
            assert (await manager.get(f"{LIBRARY}/{scoped_id}/v1/file")).status_code == 404

            with app.state.database.connect() as connection:
                connection.execute(
                    """INSERT INTO library_knowledge_bases
                    (org_id, knowledge_base_id, name, name_key, description, embedding_model_id, created_at)
                    VALUES ('external', 'other', 'Other', 'other', '', 'workspace-default', '2026-11-23')"""
                )
            assert (await manager.get(BROWSER, params={"knowledge_base_id": "other"})).status_code == 404
            assert (await manager.post(FOLDERS, json={"name": "绕过", "knowledge_base_id": "other"})).status_code == 404
            assert (await manager.post(LIBRARY, json=upload_body(SAMPLE, knowledge_base_id="other"))).status_code == 404

            app.dependency_overrides[embedding_model_options] = lambda: (
                EmbeddingModelOption(model_id="bge-m3", name="BGE-M3", selectable=True),
            )
            selected = await manager.post(BASES, json={**data, "name": "工控手册", "embedding_model_id": "bge-m3"})
            assert selected.status_code == 201, selected.text
            assert selected.json()["embedding_model_id"] == "bge-m3"
            saved = (await manager.get(BASES)).json()
            assert {item["embedding_model_id"] for item in saved} == {"workspace-default", "bge-m3"}
            assert [item["knowledge_base_id"] for item in (await reader.get(BASES)).json()] == ["default"]

            renamed_default = await manager.patch(
                f"{BASES}/default",
                json={"name": "工控机行业产品手册", "description": "工业计算机及面板电脑手册"},
            )
            assert renamed_default.status_code == 200, renamed_default.text
            assert renamed_default.json()["knowledge_base_id"] == "default"
            assert renamed_default.json()["name"] == "工控机行业产品手册"
            assert (await reader.patch(f"{BASES}/default", json={"name": "越权改名"})).status_code == 403
            dup_rename = await manager.patch(f"{BASES}/default", json={"name": "相机手册"})
            assert dup_rename.status_code == 409
            listed_after = (await manager.get(BASES)).json()
            assert [item["name"] for item in listed_after if item["knowledge_base_id"] == "default"] == ["工控机行业产品手册"]
            assert len([item for item in listed_after if item["knowledge_base_id"] == "default"]) == 1

    asyncio.run(check())


def test_preexisting_sqlite_schema_keeps_document_identity_on_restart(tmp_path):
    database = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.executescript("""
        CREATE TABLE library_folders (
            org_id TEXT NOT NULL, folder_id TEXT NOT NULL, parent_id TEXT,
            name TEXT NOT NULL, name_key TEXT NOT NULL, PRIMARY KEY (org_id, folder_id)
        );
        CREATE UNIQUE INDEX idx_library_folder_name
            ON library_folders (org_id, coalesce(parent_id, ''), name_key);
        CREATE TABLE library_document_versions (
            org_id TEXT NOT NULL, document_id TEXT NOT NULL, version TEXT NOT NULL, filename TEXT NOT NULL,
            size_bytes INTEGER NOT NULL, digest TEXT NOT NULL, content BLOB NOT NULL, status TEXT NOT NULL,
            uploaded_at TEXT NOT NULL, PRIMARY KEY (org_id, document_id, version)
        );
        CREATE TABLE library_document_locations (
            org_id TEXT NOT NULL, document_id TEXT NOT NULL, folder_id TEXT NOT NULL,
            PRIMARY KEY (org_id, document_id)
        );
        INSERT INTO library_folders VALUES ('demo', 'previous-folder', NULL, '相机', '相机');
        INSERT INTO library_document_versions
            VALUES ('demo', 'previous-document', 'original', 'camera.md', 3, 'hash', X'616263',
                'published', '2026-11-19T00:00:00+00:00');
        INSERT INTO library_document_locations VALUES ('demo', 'previous-document', 'previous-folder');
        """)

    async def check():
        for _ in range(2):
            app = stub_application(database, WorkspaceStub())
            async with app.router.lifespan_context(app), browser(app) as manager:
                await login(manager)
                assert (await manager.get(BASES)).json()[0]["knowledge_base_id"] == "default"
                tree = (await manager.get(BROWSER)).json()
                assert tree["documents"][0]["document_id"] == "previous-document"
                assert tree["documents"][0]["folder_id"] == "previous-folder"
                assert (await manager.get(f"{LIBRARY}/previous-document/original/file")).content == b"abc"

    asyncio.run(check())
