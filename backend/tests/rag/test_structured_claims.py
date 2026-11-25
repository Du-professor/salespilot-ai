import pytest

from app.rag import HashingEmbeddings, SqliteRagEngine
from app.rag.models import ModelClaim, ModelDraftPayload

from conftest import FIXTURES, ask, index_file


def _claim(item, **overrides):
    values = {
        key: item.get(key)
        for key in (
            "chunk_id", "model", "assembly", "document_id", "version", "page", "section",
            "attribute", "value", "upper", "unit", "polarity", "excerpt",
        )
    }
    values.update(overrides)
    return ModelClaim(**values)


class StructuredChat:
    def __init__(self, extra=None, keep=None):
        self.extra = extra
        self.keep = keep
        self.calls = 0

    def compose(self, *, system, user):
        return ""

    def compose_draft(self, *, system, user, evidence):
        self.calls += 1
        selected = evidence if self.keep is None else evidence[: self.keep]
        claims = [_claim(item) for item in selected]
        if self.extra is not None:
            claims.append(self.extra)
        return ModelDraftPayload(claims=tuple(claims))


def _engine(engine, chat):
    settings = engine.settings
    return SqliteRagEngine(
        settings, embedder=HashingEmbeddings(settings.embedding_dimensions), chat=chat,
    )


def test_model_added_poe_is_removed_when_original_lacks_it(engine):
    document = index_file(engine, FIXTURES / "rag" / "ax-920.md", "ax-model-poe")
    source = ask(engine, "需要至少 8 MP、至少 100 fps 的 GigE 相机。", (document,))
    assert source.status == "recommended"
    chunk = _chunk_for(engine, document, "12 MP")
    fake = ModelClaim(
        chunk_id=chunk.chunk_id, model="AX-920", assembly="标准装配",
        document_id=document.document_id, version=document.version,
        page=None, section=chunk.section, attribute="poe", value="True",
        unit=None, polarity="affirmed", excerpt=chunk.text,
    )
    chat = StructuredChat(extra=fake)
    result = _engine(engine, chat).recommend(_context(document))
    assert result.status == "recommended"
    assert "PoE" not in result.answer
    assert all("PoE" not in citation.excerpt for citation in result.citations)


@pytest.mark.parametrize("polarity", ["affirmed", "negated"])
def test_model_positive_claim_from_negative_text_is_removed(engine, polarity):
    document = index_file(engine, FIXTURES / "rag" / "ax-920-no-poe.md", "ax-model-negative")
    source = ask(engine, "需要至少 8 MP、至少 100 fps 的 GigE 相机。", (document,))
    assert source.status == "recommended"
    real_chunk = _chunk_for(engine, document, "不支持 PoE")
    fake = ModelClaim(
        chunk_id=real_chunk.chunk_id, model="AX-920", assembly="标准装配",
        document_id=document.document_id, version=document.version, page=None,
        section=real_chunk.section, attribute="poe", value="True", unit=None,
        polarity=polarity, excerpt=real_chunk.text,
    )
    result = _engine(engine, StructuredChat(extra=fake)).recommend(_context(document))
    assert result.status == "recommended"
    assert "支持 PoE" not in result.answer


def test_model_fact_for_another_model_is_removed(engine):
    document = index_file(engine, FIXTURES / "rag" / "ax-920.md", "ax-model-cross")
    source = ask(engine, "需要至少 8 MP、至少 100 fps 的 GigE 相机。", (document,))
    chunk_id = _chunk_for(engine, document, "12 MP").chunk_id
    fake = ModelClaim(
        chunk_id=chunk_id, model="SP-VC-120", assembly="标准装配",
        document_id=document.document_id, version=document.version, page=None,
        section="1. AX-920 标准装配", attribute="resolution_mp", value="12", unit="MP",
        polarity="affirmed", excerpt=_chunk_for(engine, document, "12 MP").text,
    )
    result = _engine(engine, StructuredChat(extra=fake)).recommend(_context(document))
    assert result.status == "recommended"
    assert result.models == ("AX-920",)
    assert "SP-VC-120" not in result.answer


def test_model_missing_required_claims_falls_back_to_deterministic(engine):
    document = index_file(engine, FIXTURES / "rag" / "ax-920.md", "ax-model-partial")
    result = _engine(engine, StructuredChat(keep=1)).recommend(_context(document))
    assert result.status == "recommended"
    assert "12 MP" in result.answer and "120 fps" in result.answer


def test_model_wrong_version_claim_is_removed(engine):
    document = index_file(engine, FIXTURES / "rag" / "ax-920.md", "ax-model-version")
    source = ask(engine, "需要至少 8 MP、至少 100 fps 的 GigE 相机。", (document,))
    chunk = _chunk_for(engine, document, "120 fps")
    fake = ModelClaim(
        chunk_id=chunk.chunk_id, model="AX-920", assembly="标准装配",
        document_id=document.document_id, version="v2", page=None, section=chunk.section,
        attribute="fps", value="120", unit="fps", polarity="affirmed", excerpt=chunk.text,
    )
    result = _engine(engine, StructuredChat(extra=fake)).recommend(_context(document))
    assert result.status == "recommended"
    assert all(citation.version == "v1" for citation in result.citations)


def test_structured_component_claim_keeps_component_source(engine, ipc_documents, cases):
    chat = StructuredChat()
    result = _engine(engine, chat).recommend(_context_for(cases["ipc"]["text"], ipc_documents))
    assert result.status == "recommended"
    assert result.models == ("SP-IPC-670",)
    assert "RTX A2000 12GB 最大功耗 70 W" in result.answer
    assert any(citation.document_id == "rtx-a2000-12gb" for citation in result.citations)
    assert chat.calls == 1


def _context_for(text, documents):
    from app.contracts.rag_workspace import RagContext

    return RagContext(
        org_id="org-a", user_id="user-a", text=text, previous_turns=(), documents=tuple(documents),
    )


def _chunk_for(engine, document, needle):
    return next(
        chunk
        for chunk, _ in engine.store.load_chunks("org-a", {(document.document_id, document.version)})
        if needle in chunk.text
    )


def _context(document):
    from app.contracts.rag_workspace import RagContext

    return RagContext(
        org_id="org-a",
        user_id="user-a",
        text="需要至少 8 MP、至少 100 fps 的 GigE 相机。",
        previous_turns=(),
        documents=(document,),
    )
