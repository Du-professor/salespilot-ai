import hashlib

from app.contracts.rag_workspace import DocumentVersionKey, RagDocumentError

from conftest import FIXTURES, ORG, ask, library_document


class FakeSource:
    def __init__(self, documents, payloads):
        self.documents = tuple(documents)
        self.payloads = dict(payloads)

    def snapshot(self, org_id, knowledge_base_id):
        return self.documents

    def read_payload(self, org_id, key: DocumentVersionKey):
        return self.payloads[(key.document_id, key.version)]


def _published(document_id, payload, *, filename):
    return library_document(
        document_id, filename=filename, size=len(payload),
    ).model_copy(update={
        "knowledge_base_id": "kb-rebuild",
        "digest": hashlib.sha256(payload).hexdigest(),
    })


def _source():
    camera = (FIXTURES / "specs" / "sp-vision-cameras.md").read_bytes()
    ipc = (FIXTURES / "specs" / "sp-ipc-670.md").read_bytes()
    documents = (
        _published("camera", camera, filename="camera.md"),
        _published("ipc", ipc, filename="ipc.md"),
    )
    payloads = {("camera", "v1"): camera, ("ipc", "v1"): ipc}
    return FakeSource(documents, payloads)


def test_rebuild_creates_independent_ready_version(engine, cases):
    source = _source()
    task = engine.start_rebuild(ORG, "kb-rebuild", "workspace-default", source, background=False)
    assert task.state == "succeeded", task.error_message
    assert task.completed == 2 and task.failed == 0

    descriptor = engine.index_descriptor(ORG, "kb-rebuild", task.target_index_version)
    assert descriptor.state == "ready"
    assert descriptor.document_count == 2
    assert descriptor.source_snapshot_id == task.task_id

    result = ask(
        engine, cases["vision"]["text"], source.documents,
        knowledge_base_id="kb-rebuild", index_version=task.target_index_version,
        embedding_model_id="workspace-default",
    )
    assert result.status == "recommended"
    assert result.models == ("SP-VC-120",)


def test_rebuild_resume_is_idempotent(engine):
    source = _source()
    task = engine.start_rebuild(ORG, "kb-rebuild", "workspace-default", source, background=False)
    before = engine.store.index_version_counts(ORG, "kb-rebuild", task.target_index_version)
    resumed = engine.resume_rebuild(task.task_id, source, background=False)
    assert resumed.state == "succeeded"
    assert engine.store.index_version_counts(ORG, "kb-rebuild", task.target_index_version) == before


def test_partial_rebuild_can_retry_failed_items(engine, monkeypatch):
    source = _source()
    original = engine.index_document
    failed_once = {"done": False}

    def flaky(org_id, document, payload):
        if document.document_id == "ipc" and not failed_once["done"]:
            failed_once["done"] = True
            raise RagDocumentError("synthetic rebuild failure")
        return original(org_id, document, payload)

    monkeypatch.setattr(engine, "index_document", flaky)
    task = engine.start_rebuild(ORG, "kb-retry", "workspace-default", source, background=False)
    assert task.state == "partial_failed"
    assert task.failed == 1
    failed_item = next(item for item in task.items if item.state == "failed")
    assert failed_item.document_id == "ipc"
    assert engine.index_descriptor(ORG, "kb-retry", task.target_index_version).state == "failed"

    monkeypatch.setattr(engine, "index_document", original)
    resumed = engine.resume_rebuild(task.task_id, source, background=False)
    assert resumed.state == "succeeded"
    assert resumed.failed == 0
    assert engine.index_descriptor(ORG, "kb-retry", task.target_index_version).state == "ready"
    documents, chunks = engine.store.index_version_counts(ORG, "kb-retry", task.target_index_version)
    assert documents == 2 and chunks > 0


def test_recover_marks_running_task_interrupted_and_resumes(engine):
    source = _source()
    task = engine.start_rebuild(ORG, "kb-recover", "workspace-default", source, background=False)
    assert task.state == "succeeded"

    engine.store.set_rebuild_task_state(task.task_id, "running")
    engine.store.set_rebuild_item_state(
        task.task_id, task.items[0].document_id, task.items[0].version, "running",
    )
    engine.recover_rebuilds()
    interrupted = engine.rebuild_status(task.task_id)
    assert interrupted.state == "interrupted"
    assert all(item.state in {"pending", "succeeded"} for item in interrupted.items)

    resumed = engine.resume_rebuild(task.task_id, source, background=False)
    assert resumed.state == "succeeded"
    assert resumed.failed == 0