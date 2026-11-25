from typing import Literal

from app.contracts.common import ContractModel
from app.contracts.rag_workspace import RagContext, RagDiagnostics
from app.rag.models import Chunk

DIAGNOSTIC_LIMIT = 100


class ChunkPreview(ContractModel):
    chunk_id: str
    page: int | None
    section: str | None
    text: str
    truncated: bool = False

    @classmethod
    def from_chunk(cls, chunk: Chunk):
        return cls(
            chunk_id=chunk.chunk_id,
            page=chunk.page,
            section=chunk.section,
            text=chunk.text,
            truncated=chunk.truncated,
        )


class ChunkPage(ContractModel):
    total: int
    chunks: list[ChunkPreview]


class RetrievalHit(ChunkPreview):
    document_id: str
    version: str
    filename: str
    score: float
    knowledge_base_id: str
    index_version: str
    embedding_model_id: str
    vector_score: float | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None


class RetrievalTest(ContractModel):
    status: Literal["retrieved", "needs_input", "insufficient_evidence", "spec_query"]
    message: str
    total: int = 0
    hits: tuple[RetrievalHit, ...] = ()


def inspect_retrieval(
    engine: RagDiagnostics,
    context: RagContext,
    *,
    mode: Literal["selection", "raw_retrieval"] = "selection",
    strategy: Literal["hybrid", "vector", "bm25"] | None = None,
    top_k: int | None = None,
) -> tuple[RetrievalTest, set[tuple[str, str]]]:
    # Reuse the recall entry of the recommendation graph (or direct hybrid recall in raw_retrieval mode).
    # The diagnostic stops before candidate checks or generation and never calls a chat adapter.
    detailed = getattr(engine, "diagnose_detailed", None)
    if callable(detailed):
        try:
            diagnostics, used = detailed(context, mode=mode, strategy=strategy, top_k=top_k)
        except TypeError:
            diagnostics, used = detailed(context)
    else:
        diagnostics = engine.diagnose(context)
        used = {(item.document_id, item.version) for item in diagnostics.hits}
    files = {(doc.document_id, doc.version): doc.filename for doc in context.documents}
    hits = tuple(
        RetrievalHit(
            chunk_id=item.chunk_id,
            page=item.page,
            section=item.section,
            text=item.text,
            truncated=item.truncated,
            document_id=item.document_id,
            version=item.version,
            filename=files.get((item.document_id, item.version), ""),
            score=item.score,
            knowledge_base_id=item.knowledge_base_id,
            index_version=item.index_version,
            embedding_model_id=item.embedding_model_id,
            vector_score=item.vector_score,
            bm25_score=item.bm25_score,
            rerank_score=item.rerank_score,
        )
        for item in diagnostics.hits[:DIAGNOSTIC_LIMIT]
    )
    return RetrievalTest(
        status=diagnostics.status,
        message=diagnostics.message,
        total=diagnostics.total,
        hits=hits,
    ), used