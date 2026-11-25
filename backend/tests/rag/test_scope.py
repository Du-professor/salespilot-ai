from app.contracts.rag_workspace import RagContext

from conftest import FIXTURES, ORG, USER, ask, index_file


def test_cross_org_same_keys_never_leak(engine, cases):
    document_a = index_file(engine, FIXTURES / "specs" / "sp-vision-cameras.md", "shared", org_id="org-a")
    document_b = index_file(engine, FIXTURES / "specs" / "sp-ipc-670.md", "shared", org_id="org-b")

    org_a = [chunk for chunk, _ in engine.store.load_chunks("org-a", {("shared", "v1")})]
    org_b = [chunk for chunk, _ in engine.store.load_chunks("org-b", {("shared", "v1")})]
    assert org_a and org_b
    assert all("SP-IPC-670" not in chunk.text for chunk in org_a)
    assert all("SP-VC-120" not in chunk.text for chunk in org_b)

    result_a = ask(engine, cases["vision"]["text"], (document_a,), org_id="org-a")
    assert result_a.status == "recommended"
    assert result_a.models == ("SP-VC-120",)
    assert all("SP-IPC-670" not in citation.excerpt for citation in result_a.citations)

    card_b = index_file(engine, FIXTURES / "specs" / "rtx-a2000-12gb.md", "rtx-b", org_id="org-b")
    result_b = ask(engine, cases["ipc"]["text"], (document_b, card_b), org_id="org-b")
    assert result_b.status == "recommended"
    assert result_b.models == ("SP-IPC-670",)
    assert all("SP-VC-120" not in citation.excerpt for citation in result_b.citations)
    assert engine.store.load_chunks("org-a", {("rtx-b", "v1")}) == []


def test_empty_documents_return_insufficient_evidence(engine):
    result = ask(engine, "需要 GigE、至少 8 MP、至少 100 fps 的相机。", ())
    assert result.status == "insufficient_evidence"
    assert result.models == ()
    assert result.citations == ()
    assert result.manuals == ()


def test_withdrawn_version_is_not_retrieved_but_remains_indexed(engine):
    old = index_file(engine, FIXTURES / "specs" / "sp-vision-cameras.md", "versioned", version="v1")
    new = index_file(engine, FIXTURES / "specs" / "sp-ipc-670.md", "versioned", version="v2")

    result = ask(engine, "需要 GigE、至少 8 MP、至少 100 fps 的相机。", (new,))
    assert result.status == "insufficient_evidence"
    assert result.citations == ()
    assert engine.store.count_chunks(ORG, "versioned", "v1") > 0

    result = ask(engine, "需要 GigE、至少 8 MP、至少 100 fps 的相机。", (old,))
    assert result.status == "recommended"
    assert result.models == ("SP-VC-120",)
    assert all(citation.version == "v1" for citation in result.citations)


def test_unindexed_or_out_of_scope_document_returns_insufficient(engine):
    index_file(engine, FIXTURES / "specs" / "sp-vision-cameras.md", "indexed")
    context = RagContext(
        org_id=ORG,
        user_id=USER,
        text="需要 GigE、至少 8 MP、至少 100 fps 的相机。",
        previous_turns=(),
        documents=(),
    )
    assert engine.recommend(context).status == "insufficient_evidence"