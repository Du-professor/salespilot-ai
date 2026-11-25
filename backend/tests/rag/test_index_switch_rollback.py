import hashlib

import pytest

from app.contracts.rag_workspace import DocumentVersionKey, RagContext, RagIndexConflict

from conftest import FIXTURES, ORG, USER, index_scoped, library_document, scoped_document


def _camera():
    return (FIXTURES / "specs" / "sp-vision-cameras.md").read_bytes()


def test_switch_and_rollback_keep_one_active_version(engine, cases):
    first = engine.ensure_index_version(ORG, "kb-switch", "workspace-default")
    camera = index_scoped(engine, first, _camera(), "camera", filename="camera.md")
    engine.mark_active(ORG, "kb-switch", first.index_version, documents=(camera,))
    assert engine.index_descriptor(ORG, "kb-switch", first.index_version).state == "active"

    second = engine.ensure_index_version(ORG, "kb-switch", "workspace-default")
    camera_again = index_scoped(engine, second, _camera(), "camera", filename="camera.md")
    plan = engine.mark_active(ORG, "kb-switch", second.index_version, documents=(camera_again,))
    assert plan.ready is True and plan.scope_changed is False
    assert engine.index_descriptor(ORG, "kb-switch", first.index_version).state == "superseded"
    assert engine.index_descriptor(ORG, "kb-switch", second.index_version).state == "active"

    rollback = engine.mark_active(ORG, "kb-switch", first.index_version, documents=(camera,))
    assert rollback.ready is True
    assert engine.index_descriptor(ORG, "kb-switch", first.index_version).state == "active"
    assert engine.index_descriptor(ORG, "kb-switch", second.index_version).state == "superseded"
    assert engine.retire(ORG, "kb-switch", second.index_version) is None
    assert engine.index_descriptor(ORG, "kb-switch", second.index_version).state == "retired"


def test_request_binds_the_version_resolved_at_start(engine, cases):
    first = engine.ensure_index_version(ORG, "kb-pin", "workspace-default")
    camera = index_scoped(engine, first, _camera(), "camera", filename="camera.md")
    engine.mark_active(ORG, "kb-pin", first.index_version, documents=(camera,))
    second = engine.ensure_index_version(ORG, "kb-pin", "workspace-default")
    camera_again = index_scoped(engine, second, _camera(), "camera", filename="camera.md")
    engine.mark_active(ORG, "kb-pin", second.index_version, documents=(camera_again,))
    assert engine.index_descriptor(ORG, "kb-pin", second.index_version).state == "active"

    diagnostics = engine.diagnose(RagContext(
        org_id=ORG,
        user_id=USER,
        text=cases["vision"]["text"],
        previous_turns=(),
        documents=(camera,),
        knowledge_base_id="kb-pin",
        index_version=first.index_version,
    ))
    assert diagnostics.status == "retrieved"
    assert diagnostics.index_version == first.index_version


def test_rollback_after_scope_change_can_require_explicit_stale(engine, cases):
    first = engine.ensure_index_version(ORG, "kb-stale", "workspace-default")
    camera = index_scoped(engine, first, _camera(), "camera", filename="camera.md")
    engine.mark_active(ORG, "kb-stale", first.index_version, documents=(camera,))

    second = engine.ensure_index_version(ORG, "kb-stale", "workspace-default")
    camera_again = index_scoped(engine, second, _camera(), "camera", filename="camera.md")
    added = index_scoped(
        engine, second, (FIXTURES / "specs" / "sp-ipc-670.md").read_bytes(),
        "ipc", filename="ipc.md",
    )
    engine.mark_active(ORG, "kb-stale", second.index_version, documents=(camera_again, added))

    plan = engine.validate_scope(ORG, "kb-stale", first.index_version, (camera, added))
    assert plan.scope_changed is True
    assert {(item.kind, item.document_id) for item in plan.differences} == {("missing", "ipc")}

    stale = engine.mark_active(
        ORG, "kb-stale", first.index_version, documents=(camera, added), allow_stale=True,
    )
    assert stale.scope_changed is True
    assert engine.availability(ORG, "kb-stale").active_index_version == first.index_version

def test_legacy_rollback_validates_scope_and_syncs_active_state(engine, cases):
    payload = _camera()
    digest = hashlib.sha256(payload).hexdigest()
    legacy_document = library_document(
        "camera", filename="camera.md", size=len(payload),
    ).model_copy(update={"digest": digest})
    engine.index_document(ORG, legacy_document, payload)

    target = engine.ensure_index_version(
        ORG, "default", "workspace-default", index_version="idx-default-versioned",
    )
    scoped_camera = scoped_document(
        target, "camera", filename="camera.md", size=len(payload), digest=digest,
    )
    engine.index_document(ORG, scoped_camera, payload)
    engine.mark_active(ORG, "default", target.index_version, documents=(scoped_camera,))
    assert engine.availability(ORG, "default").active_index_version == target.index_version

    ipc_payload = (FIXTURES / "specs" / "sp-ipc-670.md").read_bytes()
    ipc_digest = hashlib.sha256(ipc_payload).hexdigest()
    scoped_ipc = scoped_document(
        target, "ipc", filename="ipc.md", size=len(ipc_payload), digest=ipc_digest,
    )
    engine.index_document(ORG, scoped_ipc, ipc_payload)

    plan = engine.validate_scope(ORG, "default", "legacy", (scoped_camera, scoped_ipc))
    assert plan.scope_changed is True
    assert {(item.kind, item.document_id) for item in plan.differences} == {("missing", "ipc")}

    with pytest.raises(RagIndexConflict) as raised:
        engine.mark_active(
            ORG, "default", "legacy", documents=(scoped_camera, scoped_ipc),
        )
    assert raised.value.code == "index_scope_changed"
    assert engine.availability(ORG, "default").active_index_version == target.index_version

    stale = engine.mark_active(
        ORG, "default", "legacy", documents=(scoped_camera, scoped_ipc), allow_stale=True,
    )
    assert stale.scope_changed is True
    availability = engine.availability(ORG, "default")
    assert availability.state == "legacy"
    assert availability.active_index_version == "legacy"
    assert engine.index_descriptor(ORG, "default", "legacy").state == "active"
    assert engine.index_descriptor(ORG, "default", target.index_version).state == "superseded"

    versions = [item.index_version for item in engine.list_index_versions(ORG, "default")]
    assert versions.count("legacy") == 1
    assert engine.read_chunks(
        ORG, "default", "legacy", DocumentVersionKey(document_id="ipc", version="v1"),
    ).total == 0
    assert engine.read_chunks(
        ORG, "default", "legacy", DocumentVersionKey(document_id="camera", version="v1"),
    ).total > 0

    diagnostics = engine.diagnose(RagContext(
        org_id=ORG,
        user_id=USER,
        text=cases["vision"]["text"],
        previous_turns=(),
        documents=(legacy_document,),
        knowledge_base_id="default",
    ))
    assert diagnostics.status == "retrieved"
    assert diagnostics.index_version == "legacy"