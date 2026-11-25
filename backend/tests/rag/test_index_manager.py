import pytest

from app.contracts.rag_workspace import (
    DocumentVersionKey,

    RagContext,
    RagIndexConflict,
    RagModelUnavailable,
)
from app.rag import HashingEmbeddings, SqliteRagEngine

from conftest import FIXTURES, ORG, USER, ask, index_scoped, scoped_document


def _camera():
    return (FIXTURES / "specs" / "sp-vision-cameras.md").read_bytes()


def test_default_base_reports_legacy_availability(engine):
    availability = engine.availability(ORG, "default")
    assert availability.state == "legacy"
    assert availability.active_index_version == "legacy"
    assert availability.ready is True
    plan = engine.mark_active(ORG, "default", "legacy")
    assert plan.ready is True and plan.scope_changed is False


def test_index_manager_lifecycle_and_diagnostics(engine, cases):
    assert engine.availability(ORG, "kb-new").state == "awaiting_adapter"
    descriptor = engine.ensure_index_version(ORG, "kb-new", "workspace-default")
    assert descriptor.state == "ready"
    pending = engine.availability(ORG, "kb-new")
    assert pending.state == "ready" and pending.ready is False

    document = index_scoped(engine, descriptor, _camera(), "camera", filename="camera.md")
    plan = engine.mark_active(ORG, "kb-new", descriptor.index_version, documents=(document,))
    assert plan.ready is True and plan.scope_changed is False
    available = engine.availability(ORG, "kb-new")
    assert available.state == "available" and available.ready is True
    assert available.active_index_version == descriptor.index_version

    page = engine.read_chunks(
        ORG, "kb-new", descriptor.index_version, DocumentVersionKey(document_id="camera", version="v1"),
    )
    assert page.total > 0 and page.chunks
    assert all(chunk.page is not None or chunk.section for chunk in page.chunks)

    result = ask(
        engine, cases["vision"]["text"], (document,),
        knowledge_base_id="kb-new", index_version=descriptor.index_version,
        embedding_model_id="workspace-default",
    )
    assert result.status == "recommended"
    assert result.models == ("SP-VC-120",)

    diagnostics = engine.diagnose(RagContext(
        org_id=ORG, user_id=USER, text=cases["vision"]["text"], previous_turns=(),
        documents=(document,), knowledge_base_id="kb-new", index_version=descriptor.index_version,
    ))
    assert diagnostics.status == "retrieved"
    assert diagnostics.knowledge_base_id == "kb-new"
    assert diagnostics.index_version == descriptor.index_version
    assert diagnostics.total > 0
    assert diagnostics.hits
    assert all(hit.index_version == descriptor.index_version for hit in diagnostics.hits)
    assert all(hit.embedding_model_id == "workspace-default" for hit in diagnostics.hits)
    assert len({hit.chunk_id for hit in diagnostics.hits}) == len(diagnostics.hits)


def test_missing_local_model_directory_is_retryable(tmp_path, settings):
    isolated = settings.model_copy(update={
        "index_path": tmp_path / "isolated-index.sqlite3",
        "models_dir": tmp_path / "missing-models",
    })
    engine = SqliteRagEngine(
        isolated, embedder=HashingEmbeddings(isolated.embedding_dimensions), chat=None,
    )
    with pytest.raises(RagModelUnavailable) as raised:
        engine.ensure_index_version(ORG, "kb-bge", "bge-m3")
    assert raised.value.code == "embedding_model_directory_missing"
    assert raised.value.retryable is True


def test_validate_scope_reports_missing_extra_and_changed(engine):
    descriptor = engine.ensure_index_version(ORG, "kb-scope", "workspace-default")
    document = index_scoped(engine, descriptor, _camera(), "camera", filename="camera.md")
    plan = engine.validate_scope(ORG, "kb-scope", descriptor.index_version, (document,))
    assert plan.ready is True and plan.scope_changed is False

    changed = document.model_copy(update={"digest": "different"})
    plan = engine.validate_scope(ORG, "kb-scope", descriptor.index_version, (changed,))
    assert plan.scope_changed is True
    assert [(item.kind, item.document_id) for item in plan.differences] == [("changed", "camera")]

    plan = engine.validate_scope(ORG, "kb-scope", descriptor.index_version, ())
    assert [(item.kind, item.document_id) for item in plan.differences] == [("extra", "camera")]

    with pytest.raises(RagIndexConflict):
        engine.mark_active(ORG, "kb-scope", descriptor.index_version, documents=(changed,))
    stale = engine.mark_active(
        ORG, "kb-scope", descriptor.index_version, documents=(changed,), allow_stale=True,
    )
    assert stale.scope_changed is True

    other = scoped_document(descriptor, "other", filename="other.md")
    plan = engine.validate_scope(ORG, "kb-scope", descriptor.index_version, (document, other))
    assert ("missing", "other") in {(item.kind, item.document_id) for item in plan.differences}


def test_remove_document_clears_every_index_version(engine):
    first = engine.ensure_index_version(ORG, "kb-delete", "workspace-default")
    second = engine.ensure_index_version(ORG, "kb-delete", "workspace-default")
    payload = _camera()
    index_scoped(engine, first, payload, "shared", filename="camera.md")
    index_scoped(engine, second, payload, "shared", filename="camera.md")

    engine.remove_document(ORG, DocumentVersionKey(document_id="shared", version="v1"))

    for descriptor in (first, second):
        assert engine.store.count_chunks(
            ORG, "shared", "v1",
            knowledge_base_id=descriptor.knowledge_base_id,
            index_version=descriptor.index_version,
        ) == 0
        assert engine.store.document_count(
            ORG, "shared", "v1",
            knowledge_base_id=descriptor.knowledge_base_id,
            index_version=descriptor.index_version,
        ) == 0
    engine.remove_document(ORG, DocumentVersionKey(document_id="shared", version="v1"))


def test_active_version_cannot_be_retired_until_switched(engine):
    first = engine.ensure_index_version(ORG, "kb-retire", "workspace-default")
    document = index_scoped(engine, first, _camera(), "camera", filename="camera.md")
    engine.mark_active(ORG, "kb-retire", first.index_version, documents=(document,))
    with pytest.raises(RagIndexConflict):
        engine.retire(ORG, "kb-retire", first.index_version)

    second = engine.ensure_index_version(ORG, "kb-retire", "workspace-default")
    engine.mark_active(ORG, "kb-retire", second.index_version, documents=(document,), allow_stale=True)
    engine.retire(ORG, "kb-retire", first.index_version)
    assert engine.index_descriptor(ORG, "kb-retire", first.index_version).state == "retired"
    assert engine.read_chunks(
        ORG, "kb-retire", first.index_version, DocumentVersionKey(document_id="camera", version="v1"),
    ).total > 0