from conftest import FIXTURES, ask, index_file


def test_versions_are_not_stitched(engine, cases):
    v1 = index_file(engine, FIXTURES / "rag" / "ax-920-v1.md", "ax-versioned", version="v1")
    v2 = index_file(engine, FIXTURES / "rag" / "ax-920-v2.md", "ax-versioned", version="v2")
    result = ask(engine, cases["version_split"]["text"], (v1, v2))
    assert result.status == "insufficient_evidence"
    assert result.models == ()
    assert result.citations == ()
    assert result.manuals == ()
    assert result.include_report is False
    assert "v1" in result.answer and "v2" in result.answer


def test_one_version_with_all_facts_is_used(engine, cases):
    v1 = index_file(engine, FIXTURES / "rag" / "ax-920-v1.md", "ax-versioned", version="v1")
    v2 = index_file(engine, FIXTURES / "rag" / "ax-920.md", "ax-versioned", version="v2")
    result = ask(engine, cases["version_split"]["text"], (v1, v2))
    assert result.status == "recommended"
    assert result.models == ("AX-920",)
    assert result.citations
    assert all(citation.version == "v2" for citation in result.citations)
    assert "12 MP" in result.answer and "120 fps" in result.answer


def test_assemblies_are_not_stitched(engine):
    document = index_file(engine, FIXTURES / "rag" / "ax-920-assemblies.md", "ax-assemblies")
    result = ask(
        engine,
        "需要至少 8 MP、至少 100 fps 的 GigE AX-920 相机。",
        (document,),
    )
    assert result.status == "insufficient_evidence"
    assert result.models == ()
    assert result.citations == ()
    assert "标准装配" in result.answer or "高速装配" in result.answer


def test_ipc_component_evidence_keeps_source_identity(engine, ipc_documents, cases):
    result = ask(engine, cases["ipc"]["text"], ipc_documents)
    assert result.status == "recommended"
    assert result.models == ("SP-IPC-670",)
    cited_documents = {citation.document_id for citation in result.citations}
    assert "sp-ipc-670" in cited_documents
    assert "rtx-a2000-12gb" in cited_documents
    assert "sp-ipc-620" not in cited_documents
    assert "sp-ipc-680" not in cited_documents
    assert "最大功耗 70 W" in result.answer
    assert all(
        (citation.section or "").startswith("1.")
        for citation in result.citations
        if citation.document_id == "sp-ipc-670"
    )