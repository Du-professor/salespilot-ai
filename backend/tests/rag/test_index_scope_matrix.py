import pytest

from app.contracts.rag_workspace import EmbeddingModelDescriptor, RagContext, RagIndexConflict
from app.rag import HashingEmbeddings, RagSettings, SqliteRagEngine

from conftest import FIXTURES, ORG, USER, ask, index_scoped


class FakeRegistry:
    def __init__(self, embedders, descriptors):
        self._embedders = embedders
        self._descriptors = descriptors

    def availability(self, model_id):
        return True, None, None

    def check(self, model_id):
        return self._descriptors[model_id]

    def descriptor(self, model_id):
        return self._descriptors[model_id]

    def embedder(self, model_id):
        return self._embedders[model_id]


def _camera():
    return (FIXTURES / "specs" / "sp-vision-cameras.md").read_bytes()


def _ipc():
    return {
        name: (FIXTURES / "specs" / name).read_bytes()
        for name in ("sp-ipc-620.md", "sp-ipc-670.md", "sp-ipc-680.md", "rtx-a2000-12gb.md")
    }


def _index_ipc(engine, descriptor):
    documents = tuple(
        index_scoped(engine, descriptor, payload, name.removesuffix(".md"), filename=name)
        for name, payload in _ipc().items()
    )
    return documents


def test_knowledge_base_and_index_version_are_hard_scope(engine, cases):
    a = engine.ensure_index_version(ORG, "kb-a", "workspace-default")
    b = engine.ensure_index_version(ORG, "kb-b", "workspace-default")
    camera = index_scoped(engine, a, _camera(), "document", filename="camera.md")
    ipc_documents = _index_ipc(engine, b)

    assert engine.store.count_chunks(
        ORG, "document", "v1", knowledge_base_id="kb-a", index_version=a.index_version,
    ) > 0
    assert engine.store.count_chunks(
        ORG, "document", "v1", knowledge_base_id="kb-b", index_version=b.index_version,
    ) == 0

    result = ask(
        engine, cases["vision"]["text"], (camera,),
        knowledge_base_id="kb-a", index_version=a.index_version, embedding_model_id="workspace-default",
    )
    assert result.status == "recommended"
    assert result.models == ("SP-VC-120",)

    cross_base = ask(
        engine, cases["vision"]["text"], (camera,),
        knowledge_base_id="kb-b", index_version=b.index_version, embedding_model_id="workspace-default",
    )
    assert cross_base.status == "insufficient_evidence"
    assert cross_base.citations == ()

    other = ask(
        engine, cases["ipc"]["text"], ipc_documents,
        knowledge_base_id="kb-b", index_version=b.index_version, embedding_model_id="workspace-default",
    )
    assert other.status == "recommended"
    assert other.models == ("SP-IPC-670",)


def test_same_knowledge_base_versions_do_not_mix(engine, cases):
    first = engine.ensure_index_version(ORG, "kb-versions", "workspace-default")
    second = engine.ensure_index_version(ORG, "kb-versions", "workspace-default")
    camera = index_scoped(engine, first, _camera(), "camera", filename="camera.md")
    ipc_documents = _index_ipc(engine, second)

    result = ask(
        engine, cases["vision"]["text"], (camera, *ipc_documents),
        knowledge_base_id="kb-versions", index_version=first.index_version,
    )
    assert result.status == "recommended"
    assert result.models == ("SP-VC-120",)
    assert all(citation.document_id == "camera" for citation in result.citations)

    result = ask(
        engine, cases["vision"]["text"], (camera, *ipc_documents),
        knowledge_base_id="kb-versions", index_version=second.index_version,
    )
    assert result.status == "insufficient_evidence"
    assert engine.store.count_chunks(
        ORG, "camera", "v1", knowledge_base_id="kb-versions", index_version=second.index_version,
    ) == 0


def test_cross_org_same_knowledge_base_and_document_never_leak(engine):
    a = engine.ensure_index_version(ORG, "kb-org", "workspace-default")
    b = engine.ensure_index_version("org-b", "kb-org", "workspace-default")
    index_scoped(engine, a, _camera(), "shared", filename="camera.md")
    index_scoped(engine, b, _ipc()["sp-ipc-670.md"], "shared", filename="sp-ipc-670.md")

    org_a = engine.store.load_chunks(
        ORG, {("shared", "v1")}, knowledge_base_id="kb-org", index_version=a.index_version,
    )
    org_b = engine.store.load_chunks(
        "org-b", {("shared", "v1")}, knowledge_base_id="kb-org", index_version=b.index_version,
    )
    assert org_a and org_b
    assert all("SP-IPC-670" not in chunk.text for chunk, _ in org_a)
    assert all("SP-VC-120" not in chunk.text for chunk, _ in org_b)


def test_same_dimension_models_do_not_mix(settings, cases):
    descriptors = {
        name: EmbeddingModelDescriptor(
            model_id=name,
            revision=f"rev-{name}",
            dimensions=64,
            vector_config={"provider": "test", "dimensions": 64},
        )
        for name in ("model-a", "model-b")
    }
    registry = FakeRegistry(
        {name: HashingEmbeddings(64) for name in descriptors},
        descriptors,
    )
    engine = SqliteRagEngine(
        settings, embedder=HashingEmbeddings(64), model_registry=registry, chat=None,
    )
    a = engine.ensure_index_version(ORG, "kb-models", "model-a")
    b = engine.ensure_index_version(ORG, "kb-models", "model-b")
    camera = index_scoped(engine, a, _camera(), "camera", filename="camera.md")
    ipc = index_scoped(engine, b, _ipc()["sp-ipc-670.md"], "ipc", filename="sp-ipc-670.md")

    assert engine.store.count_chunks(
        ORG, "camera", "v1", knowledge_base_id="kb-models", index_version=a.index_version,
    ) > 0
    assert engine.store.count_chunks(
        ORG, "camera", "v1", knowledge_base_id="kb-models", index_version=b.index_version,
    ) == 0

    result = ask(
        engine, cases["vision"]["text"], (camera,),
        knowledge_base_id="kb-models", index_version=a.index_version, embedding_model_id="model-a",
    )
    assert result.status == "recommended"
    assert result.models == ("SP-VC-120",)

    with pytest.raises(RagIndexConflict):
        engine.recommend(RagContext(
            org_id=ORG, user_id=USER, text=cases["vision"]["text"], previous_turns=(),
            documents=(camera,), knowledge_base_id="kb-models",
            index_version=b.index_version, embedding_model_id="model-a",
        ))