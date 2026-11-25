from app.contracts.rag_workspace import PreviousTurn
from app.rag.facts import build_scopes

from conftest import FIXTURES, ORG, ask, index_bytes, index_file


def test_extracted_facts_keep_model_spec_value_unit_and_location(engine):
    document = index_file(engine, FIXTURES / "rag" / "ax-920.md", "ax-facts")
    chunks = [
        chunk
        for chunk, _ in engine.store.load_chunks(ORG, {(document.document_id, document.version)})
    ]
    facts = [fact for scope in build_scopes(chunks) for fact in scope.facts]
    resolution = next(fact for fact in facts if fact.attribute == "resolution_mp")
    assert resolution.model == "AX-920"
    assert resolution.label == "分辨率"
    assert resolution.value == 12
    assert resolution.unit == "MP"
    assert resolution.chunk_id
    assert resolution.section == "1. AX-920 标准装配"
    assert "12 MP" in resolution.excerpt


def test_non_demo_model_is_recommended(engine, cases):
    document = index_file(engine, FIXTURES / "rag" / "ax-920.md", "ax-920")
    result = ask(engine, cases["ax920"]["text"], (document,))
    assert result.status == "recommended"
    assert result.models == ("AX-920",)
    assert result.citations
    assert any("12 MP" in citation.excerpt and "120 fps" in citation.excerpt for citation in result.citations)
    assert all(citation.document_id == "ax-920" for citation in result.citations)


def test_robot_category_uses_payload_and_reach(engine, cases):
    arm600 = index_file(engine, FIXTURES / "rag" / "arm-600.md", "arm-600")
    arm300 = index_file(engine, FIXTURES / "rag" / "arm-300.md", "arm-300")
    result = ask(engine, cases["robot"]["text"], (arm600, arm300))
    assert result.status == "recommended"
    assert result.models == ("ARM-600-STD",)
    assert "ARM-300-STD" not in result.answer
    assert any("1300 mm" in citation.excerpt for citation in result.citations)
    assert any("10 kg" in citation.excerpt for citation in result.citations)


def test_hard_poe_without_manual_evidence_is_insufficient(engine, cases):
    document = index_file(engine, FIXTURES / "rag" / "ax-920.md", "ax-no-poe")
    result = ask(engine, cases["poe"]["text"], (document,))
    assert result.status == "insufficient_evidence"
    assert result.models == ()
    assert result.citations == ()
    assert result.manuals == ()
    assert result.include_report is False
    assert "PoE" in result.answer and "缺少依据" in result.answer


def test_explicit_negative_poe_is_failure(engine, cases):
    document = index_file(engine, FIXTURES / "rag" / "ax-920-no-poe.md", "ax-negative-poe")
    result = ask(engine, cases["poe"]["text"], (document,))
    assert result.status == "insufficient_evidence"
    assert result.models == ()
    assert "PoE" in result.answer and "不满足" in result.answer


def test_explicit_positive_poe_is_recommended_with_quote(engine, cases):
    document = index_file(engine, FIXTURES / "rag" / "ax-920-poe.md", "ax-positive-poe")
    result = ask(engine, cases["poe"]["text"], (document,))
    assert result.status == "recommended"
    assert result.models == ("AX-920",)
    assert any("支持 PoE 供电" in citation.excerpt for citation in result.citations)
    assert "支持 PoE 供电" in result.answer


def test_followup_retains_extra_hard_requirement(engine):
    document = index_file(engine, FIXTURES / "rag" / "ax-920.md", "ax-followup")
    initial_text = "需要至少 8 MP 的工业相机，必须支持 PoE 供电。"
    initial = ask(engine, initial_text, (document,))
    assert initial.status == "needs_input"
    reply = ask(
        engine,
        "GigE 接口，至少 100 fps。请附报告。",
        (document,),
        previous_turns=(PreviousTurn(text=initial_text, answer=initial.answer),),
    )
    assert reply.status == "insufficient_evidence"
    assert "PoE" in reply.answer
    assert reply.models == () and not reply.include_report


def test_frame_rate_upper_bound_is_preserved_with_a_lower_bound(engine):
    document = index_file(engine, FIXTURES / "rag" / "ax-920.md", "ax-fps-upper")
    question = "需要 GigE 工业相机，至少 8 MP，帧率至少 80 fps，帧率不超过 {} fps。"
    rejected = ask(engine, question.format(100), (document,))
    assert rejected.status == "insufficient_evidence"
    accepted = ask(engine, question.format(120), (document,))
    assert accepted.models == ("AX-920",)


def test_ambiguous_multi_model_section_does_not_assign_all_facts_to_first_model(engine):
    manual = "# 产品对照\n\nAX-920 使用 GigE 数据接口，分辨率为 12 MP，BX-930 的帧率为 120 fps。"
    document = index_bytes(engine, manual.encode(), "two-cameras")
    result = ask(engine, "需要至少 8 MP、至少 100 fps 的 GigE 相机。", (document,))
    assert result.status == "insufficient_evidence"
    assert result.models == ()


def test_preference_does_not_block_recommendation(engine):
    document = index_file(engine, FIXTURES / "rag" / "ax-920.md", "ax-preference")
    result = ask(
        engine,
        "需要至少 8 MP、至少 100 fps 的 GigE 相机，最好支持 PoE 供电。",
        (document,),
    )
    assert result.status == "recommended"
    assert result.models == ("AX-920",)
    assert "PoE" not in result.answer


def test_missing_camera_conditions_are_questions(engine):
    document = index_file(engine, FIXTURES / "rag" / "ax-920.md", "ax-clarify")
    result = ask(engine, "需要一款 12 MP 的工业相机，请帮我确认还需要哪些条件。", (document,))
    assert result.status == "needs_input"
    assert "接口" in result.answer and "帧率" in result.answer
    assert result.models == ()


def test_missing_robot_reach_is_a_question(engine):
    document = index_file(engine, FIXTURES / "rag" / "arm-600.md", "arm-clarify")
    result = ask(engine, "需要一台额定负载至少 8 kg 的工业机器人。", (document,))
    assert result.status == "needs_input"
    assert "工作半径" in result.answer or "臂展" in result.answer
    assert result.models == ()
