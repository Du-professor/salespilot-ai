import asyncio
import hashlib
import threading
import time
import tracemalloc
from io import BytesIO
from pathlib import Path

from app.infrastructure.object_store import LocalCasObjectStore, S3ObjectStore
from app.rag import HashingEmbeddings, RagSettings, SqliteRagEngine

from test_rag_workspace import FIXTURES, LIBRARY, stub_application, upload_body
from workspace_helpers import PREFIX, browser, login

BASES = f"{PREFIX}/library/knowledge-bases"
UPLOAD = f"{PREFIX}/library/documents/upload"
PUBLISH_TASKS = f"{PREFIX}/library/publish-tasks"


def _engine(tmp_path):
    settings = RagSettings(
        index_path=tmp_path / "b2-index.sqlite3",
        embedding_provider="hash",
        chat_provider="template",
        max_retries=0,
    )
    return SqliteRagEngine(settings, embedder=HashingEmbeddings(settings.embedding_dimensions), chat=None)


def test_cas_object_store_deduplication_refcount_gc_and_rebuild_from_object_store(tmp_path):
    sample = FIXTURES / "specs/sp-vision-cameras.md"
    raw = sample.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()

    async def check():
        database = tmp_path / "cas.sqlite3"
        engine = _engine(tmp_path)
        app = stub_application(database, engine, models_dir=tmp_path / "models")
        async with app.router.lifespan_context(app), browser(app) as client:
            await login(client)
            kb_id = (await client.post(BASES, json={
                "name": "对象存储库", "description": "", "embedding_model_id": "workspace-default",
            })).json()["knowledge_base_id"]

            # Upload identical file to default KB (via JSON) and custom KB (via octet-stream).
            doc_default = (await client.post(LIBRARY, json=upload_body(sample))).json()
            doc_custom_resp = await client.post(
                UPLOAD,
                params={"filename": sample.name, "version": "v1", "knowledge_base_id": kb_id},
                content=raw,
                headers={"Content-Type": "application/octet-stream"},
            )
            assert doc_custom_resp.status_code == 201, doc_custom_resp.text
            doc_custom = doc_custom_resp.json()

            # Verify SQLite stores empty BLOB and CAS URI instead of duplicating file bytes in DB.
            with app.state.database.connect() as conn:
                rows = conn.execute(
                    "SELECT document_id, length(content) AS blob_len, storage_uri FROM library_document_versions",
                ).fetchall()
            assert len(rows) == 2
            assert all(row["blob_len"] == 0 for row in rows)
            assert {row["storage_uri"] for row in rows} == {f"cas://{digest}"}
            assert app.state.object_store.exists(f"cas://{digest}")

            # Publish & rebuild custom KB reading payload directly from ObjectStore.
            assert (await client.post(f"{LIBRARY}/{doc_custom['document_id']}/v1/publish")).status_code == 200
            rebuild = await client.post(f"{BASES}/{kb_id}/rebuild", json={"background": False})
            assert rebuild.status_code == 200 and rebuild.json()["state"] == "succeeded"

            # List rebuilds endpoint returns the completed rebuild task for cross-tab recovery.
            rebuilds = await client.get(f"{BASES}/{kb_id}/rebuilds")
            assert rebuilds.status_code == 200
            assert rebuilds.json()[0]["task_id"] == rebuild.json()["task_id"]

            # Deleting one reference keeps CAS object alive for the other KB.
            assert (await client.delete(f"{LIBRARY}/{doc_default['document_id']}/v1")).status_code == 204
            assert app.state.object_store.exists(f"cas://{digest}")
            dl = await client.get(f"{LIBRARY}/{doc_custom['document_id']}/v1/file")
            assert dl.status_code == 200
            assert dl.headers.get("content-length") == str(len(raw))
            assert dl.content == raw

            # Deleting the last reference GCs the CAS object from disk.
            assert (await client.delete(f"{LIBRARY}/{doc_custom['document_id']}/v1")).status_code == 204
            assert not app.state.object_store.exists(f"cas://{digest}")

    asyncio.run(check())


def test_streaming_multipart_and_large_octet_stream_memory_bounds(tmp_path):
    """Verify both multipart/form-data and large octet-stream uploads work with bounded peak memory."""
    # Synthesize a ~12 MB valid PDF payload to validate streaming memory efficiency.
    large_pdf = b"%PDF-1.4\n" + (b"% SalesPilot streaming test line 0123456789abcdef\n" * 240_000)

    async def check():
        database = tmp_path / "stream.sqlite3"
        engine = _engine(tmp_path)
        app = stub_application(database, engine, models_dir=tmp_path / "models")
        async with app.router.lifespan_context(app), browser(app) as client:
            await login(client)
            folder_id = (await client.post(f"{PREFIX}/library/folders", json={"name": "大文件目录"})).json()["folder_id"]

            # 1) Streaming multipart/form-data upload with form fields.
            boundary = "----SalesPilotBoundary7MA4YWxkTrZu0gW"
            sample = FIXTURES / "specs/sp-vision-cameras.md"
            sample_bytes = sample.read_bytes()
            body = (
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"version\"\r\n\r\nv2\r\n"
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"folder_id\"\r\n\r\n{folder_id}\r\n"
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{sample.name}\"\r\n"
                f"Content-Type: text/markdown\r\n\r\n"
            ).encode("utf-8") + sample_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")
            mp = await client.post(
                UPLOAD,
                content=body,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            )
            assert mp.status_code == 201, mp.text
            assert mp.json()["version"] == "v2"
            assert mp.json()["size_bytes"] == len(sample_bytes)
            browser_docs = (await client.get(f"{PREFIX}/library/browser")).json()["documents"]
            assert browser_docs[0]["folder_id"] == folder_id

            # 2) Large octet-stream upload and streaming download.
            tracemalloc.start()
            before_cur, _ = tracemalloc.get_traced_memory()
            resp = await client.post(
                UPLOAD,
                params={"filename": "large-manual.pdf", "version": "v1"},
                content=large_pdf,
                headers={"Content-Type": "application/octet-stream"},
            )
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            assert resp.status_code == 201, resp.text
            assert resp.json()["size_bytes"] == len(large_pdf)
            # Peak Python allocation overhead stays well below Base64 (which would be > 5x file size).
            assert (peak - before_cur) < len(large_pdf) * 3.5

            dl = await client.get(f"{LIBRARY}/{resp.json()['document_id']}/v1/file")
            assert dl.status_code == 200
            assert dl.content == large_pdf

    asyncio.run(check())


def test_async_publish_queue_single_concurrency_idempotency_and_recovery(tmp_path):
    """Async publish enforces concurrency=1 (2C8G safe), is idempotent, and recovers interrupted tasks."""
    cam = FIXTURES / "specs/sp-vision-cameras.md"
    ipc = FIXTURES / "specs/sp-ipc-670.md"

    async def check():
        database = tmp_path / "async-pub.sqlite3"
        engine = _engine(tmp_path)
        original_index = engine.index_document
        state = {"active": 0, "max_active": 0}
        guard = threading.Lock()

        def tracked_index(org_id, document, payload):
            with guard:
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
            try:
                time.sleep(0.08)
                return original_index(org_id, document, payload)
            finally:
                with guard:
                    state["active"] -= 1

        engine.index_document = tracked_index
        app = stub_application(database, engine, models_dir=tmp_path / "models")
        async with app.router.lifespan_context(app), browser(app) as client:
            await login(client)
            d1 = (await client.post(LIBRARY, json=upload_body(cam))).json()
            d2 = (await client.post(LIBRARY, json=upload_body(ipc))).json()

            # Trigger async publish for both documents concurrently + duplicate call for d1.
            t1 = await client.post(f"{LIBRARY}/{d1['document_id']}/v1/publish-async")
            t1_dup = await client.post(f"{LIBRARY}/{d1['document_id']}/v1/publish-async")
            t2 = await client.post(f"{LIBRARY}/{d2['document_id']}/v1/publish-async")
            assert t1.status_code == t1_dup.status_code == t2.status_code == 202
            assert t1.json()["task_id"] == t1_dup.json()["task_id"]

            # Poll until both succeed.
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                s1 = (await client.get(f"{PUBLISH_TASKS}/{t1.json()['task_id']}")).json()
                s2 = (await client.get(f"{PUBLISH_TASKS}/{t2.json()['task_id']}")).json()
                if s1["stage"] in {"succeeded", "failed"} and s2["stage"] in {"succeeded", "failed"}:
                    break
                await asyncio.sleep(0.05)

            assert s1["stage"] == "succeeded", s1
            assert s2["stage"] == "succeeded", s2
            assert state["max_active"] == 1

            # List tasks returns both completed tasks.
            listed = (await client.get(PUBLISH_TASKS)).json()
            assert {item["task_id"] for item in listed} >= {s1["task_id"], s2["task_id"]}

            # Verify interrupted task recovery on restart.
            app.state.publish_queue.update_stage(s1["task_id"], "indexing")
            recovered = app.state.publish_queue.recover_interrupted()
            assert recovered == 1
            reloaded = (await client.get(f"{PUBLISH_TASKS}/{s1['task_id']}")).json()
            assert reloaded["stage"] == "failed"
            assert reloaded["error_code"] == "publish_interrupted"

    asyncio.run(check())


def test_s3_object_store_adapter_contract(tmp_path):
    """Verify S3/MinIO object store adapter put_file/put_bytes/open_stream/exists/delete contract."""
    buckets: dict[tuple[str, str], bytes] = {}

    class FakeBody:
        def __init__(self, data: bytes):
            self._data = data

        def iter_chunks(self, chunk_size=1024):
            for i in range(0, len(self._data), chunk_size):
                yield self._data[i:i + chunk_size]

        def close(self):
            pass

    class FakeS3Client:
        def head_object(self, Bucket, Key):
            if (Bucket, Key) not in buckets:
                raise KeyError("not found")
            return {"ContentLength": len(buckets[(Bucket, Key)])}

        def put_object(self, Bucket, Key, Body):
            buckets[(Bucket, Key)] = Body

        def upload_file(self, Filename, Bucket, Key):
            buckets[(Bucket, Key)] = Path(Filename).read_bytes()

        def get_object(self, Bucket, Key):
            return {"Body": FakeBody(buckets[(Bucket, Key)])}

        def delete_object(self, Bucket, Key):
            buckets.pop((Bucket, Key), None)

    store = S3ObjectStore(bucket="salespilot", prefix="manuals", client=FakeS3Client())
    payload = b"hello minio object store"
    digest = hashlib.sha256(payload).hexdigest()
    src = tmp_path / "src.bin"
    src.write_bytes(payload)

    uri = store.put_file(src, digest=digest)
    assert uri.startswith("s3://salespilot/manuals/")
    assert store.exists(uri)
    assert b"".join(store.open_stream(uri, chunk_size=5)) == payload
    store.delete(uri)
    assert not store.exists(uri)
