import pytest
from pydantic import ValidationError

from app.contracts.rag_workspace import Citation, RagContext, RagRecommendation

from conftest import ORG, USER, ask


def _citation():
    return Citation(document_id="manual", version="v1", page=2, excerpt="12 MP resolution at 120 fps")


def test_recommended_requires_model_and_citation():
    with pytest.raises(ValidationError):
        RagRecommendation(status="recommended", answer="推荐 SP-VC-120。")
    with pytest.raises(ValidationError):
        RagRecommendation(status="recommended", answer="推荐 SP-VC-120。", models=("SP-VC-120",))


def test_non_recommendation_rejects_models_manuals_and_reports():
    for extra in (
        {"models": ("SP-VC-120",)},
        {"manuals": ({"document_id": "manual", "version": "v1"},)},
        {"include_report": True},
    ):
        with pytest.raises(ValidationError):
            RagRecommendation(status="insufficient_evidence", answer="证据不足。", **extra)


def test_manual_must_be_backed_by_citation():
    with pytest.raises(ValidationError):
        RagRecommendation(
            status="recommended",
            answer="推荐 SP-VC-120。",
            models=("SP-VC-120",),
            citations=(_citation(),),
            manuals=({"document_id": "other", "version": "v1"},),
        )


def test_citation_requires_page_or_section():
    with pytest.raises(ValidationError):
        Citation(document_id="manual", version="v1", excerpt="12 MP")


def test_engine_outputs_obey_contract(engine, vision, cases):
    result = ask(engine, cases["vision"]["text"], (vision,))
    assert result.status == "recommended"
    assert result.models and result.citations
    assert all(item.page is not None or item.section is not None for item in result.citations)

    no_match = ask(engine, cases["no_match"]["text"], (vision,))
    assert no_match.models == ()
    assert no_match.citations == ()
    assert no_match.manuals == ()
    assert no_match.include_report is False


def test_engine_accepts_dict_context(engine, vision, cases):
    context = RagContext(
        org_id=ORG,
        user_id=USER,
        text=cases["vision"]["text"],
        previous_turns=(),
        documents=(vision,),
    )
    result = engine.recommend(context.model_dump(mode="json"))
    assert result.status == "recommended"
    assert result.models == ("SP-VC-120",)