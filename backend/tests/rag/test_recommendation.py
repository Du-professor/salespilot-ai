from app.contracts.rag_workspace import PreviousTurn

from conftest import ask


def test_vision_case_recommends_sp_vc_120_with_manuals_and_report(engine, vision, cases):
    result = ask(engine, cases["vision"]["text"], (vision,))
    assert result.status == "recommended"
    assert result.models == ("SP-VC-120",)
    assert [(manual.document_id, manual.version) for manual in result.manuals] == [
        (vision.document_id, vision.version)
    ]
    assert result.include_report is True
    assert result.citations
    assert all(citation.document_id == vision.document_id for citation in result.citations)
    assert any("SP-VC-120 标准装配" in (citation.section or "") for citation in result.citations)
    assert "SP-VC-120" in result.answer
    assert "12 MP" in result.answer and "120 fps" in result.answer


def test_ipc_case_respects_assembly_and_unknown_power(engine, ipc_documents, cases):
    result = ask(engine, cases["ipc"]["text"], ipc_documents)
    assert result.status == "recommended"
    assert result.models == ("SP-IPC-670",)
    assert result.manuals == ()
    assert result.include_report is False
    cited_documents = {citation.document_id for citation in result.citations}
    assert "sp-ipc-670" in cited_documents
    assert "rtx-a2000-12gb" in cited_documents
    assert "sp-ipc-620" not in cited_documents
    assert "sp-ipc-680" not in cited_documents
    assert all((citation.section or "").startswith("1.") for citation in result.citations if citation.document_id == "sp-ipc-670")
    assert "SP-IPC-620" not in result.answer
    assert "SP-IPC-680" not in result.answer


def test_followup_inherits_missing_resolution_and_answers_with_input(engine, vision, cases):
    first_text = cases["followup"]["messages"][0]
    first = ask(engine, first_text, (vision,))
    assert first.status == "needs_input"
    assert first.models == ()
    assert first.citations == ()
    assert first.manuals == () and first.include_report is False
    assert "接口" in first.answer and "帧率" in first.answer

    second = ask(
        engine,
        cases["followup"]["messages"][1],
        (vision,),
        previous_turns=(PreviousTurn(text=first_text, answer=first.answer),),
    )
    assert second.status == "recommended"
    assert second.models == ("SP-VC-120",)


def test_no_match_is_insufficient_and_carries_no_archive(engine, vision, cases):
    result = ask(engine, cases["no_match"]["text"], (vision,))
    assert result.status == "insufficient_evidence"
    assert result.models == ()
    assert result.citations == ()
    assert result.manuals == ()
    assert result.include_report is False
    assert "1000" not in result.answer


def test_recommendation_is_deterministic(engine, vision, cases):
    first = ask(engine, cases["vision"]["text"], (vision,))
    second = ask(engine, cases["vision"]["text"], (vision,))
    assert first == second


def test_manual_keys_are_backed_by_citations(engine, vision, cases):
    result = ask(engine, cases["vision"]["text"], (vision,))
    cited = {(citation.document_id, citation.version) for citation in result.citations}
    assert all((manual.document_id, manual.version) in cited for manual in result.manuals)