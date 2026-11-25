from datetime import datetime
from typing import Annotated, Literal, Protocol, runtime_checkable

from pydantic import Field, StrictBool, model_validator

from app.contracts.common import ContractModel, Identifier, PositiveInt, Text

RecommendationStatus = Literal["recommended", "needs_input", "insufficient_evidence"]
IndexState = Literal[
    "legacy", "awaiting_adapter", "building", "ready", "available", "active", "superseded", "retired", "failed",
]
RebuildState = Literal["pending", "running", "succeeded", "partial_failed", "failed", "interrupted"]
RebuildItemState = Literal["pending", "running", "succeeded", "failed", "skipped"]


class DocumentVersionKey(ContractModel):
    document_id: Identifier
    version: Identifier


class LibraryDocument(DocumentVersionKey):
    filename: str
    size_bytes: PositiveInt
    uploaded_at: datetime
    status: Literal["draft", "published", "withdrawn", "deleting"]
    index_status: Literal["pending", "ready"]
    knowledge_base_id: Identifier = "default"
    index_version: Identifier | None = None
    embedding_model_id: Identifier | None = None
    embedding_model_revision: str | None = None
    dimensions: PositiveInt | None = None
    digest: str | None = None


class EmbeddingModelDescriptor(ContractModel):
    model_id: Identifier
    revision: str
    dimensions: PositiveInt
    vector_config: dict[str, str | int | float | bool] = Field(default_factory=dict)
    files: dict[str, str] = Field(default_factory=dict)
    runtime: dict[str, str] = Field(default_factory=dict)
    display_name: str = ""


class IndexDescriptor(ContractModel):
    org_id: str
    knowledge_base_id: Identifier
    index_version: Identifier
    embedding_model_id: Identifier
    embedding_model_revision: str
    dimensions: PositiveInt
    vector_config: dict[str, str | int | float | bool] = Field(default_factory=dict)
    state: IndexState
    document_count: int = 0
    chunk_count: int = 0
    source_snapshot_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class IndexAvailability(ContractModel):
    knowledge_base_id: Identifier
    state: IndexState
    active_index_version: Identifier | None = None
    embedding_model_id: Identifier
    embedding_model_revision: str | None = None
    dimensions: PositiveInt | None = None
    ready: bool
    reason: str | None = None
    retryable: bool = False


class DiagnosticHit(ContractModel):
    chunk_id: str
    document_id: Identifier
    version: Identifier
    knowledge_base_id: Identifier
    index_version: Identifier
    embedding_model_id: Identifier
    page: PositiveInt | None = None
    section: str | None = None
    text: str
    score: float
    truncated: bool = False
    vector_score: float | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None


class RetrievalDiagnostics(ContractModel):
    status: Literal["retrieved", "needs_input", "insufficient_evidence", "spec_query"]
    message: str
    knowledge_base_id: Identifier
    index_version: Identifier
    total: int = 0
    hits: tuple[DiagnosticHit, ...] = ()


class ReadChunk(ContractModel):
    chunk_id: str
    page: PositiveInt | None = None
    section: str | None = None
    text: str
    truncated: bool = False


class ChunkReadPage(ContractModel):
    total: int
    chunks: tuple[ReadChunk, ...] = ()


class RebuildItem(ContractModel):
    document_id: Identifier
    version: Identifier
    state: RebuildItemState
    attempts: int = 0
    last_error: str | None = None


class RebuildTask(ContractModel):
    task_id: Identifier
    org_id: str
    knowledge_base_id: Identifier
    target_index_version: Identifier
    embedding_model_id: Identifier
    embedding_model_revision: str
    state: RebuildState
    total: int = 0
    completed: int = 0
    failed: int = 0
    items: tuple[RebuildItem, ...] = ()
    error_code: str | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime


class IndexScopeDifference(ContractModel):
    kind: Literal["missing", "extra", "changed"]
    document_id: Identifier
    version: Identifier


class IndexSwitchPlan(ContractModel):
    org_id: str
    knowledge_base_id: Identifier
    target_index_version: Identifier
    embedding_model_id: Identifier
    embedding_model_revision: str
    dimensions: PositiveInt
    ready: bool
    scope_changed: bool = False
    differences: tuple[IndexScopeDifference, ...] = ()


class IndexGcResult(ContractModel):
    org_id: str
    knowledge_base_id: Identifier
    purged_versions: tuple[Identifier, ...] = ()
    purged_documents: int = 0
    purged_chunks: int = 0


class RecommendationInput(ContractModel):
    conversation_id: Identifier
    client_request_id: Identifier
    text: Annotated[Text, Field(max_length=4000)]
    knowledge_base_id: Identifier = "default"


class Citation(DocumentVersionKey):
    page: PositiveInt | None = None
    section: Annotated[Text, Field(max_length=180)] | None = None
    excerpt: Annotated[Text, Field(max_length=2000)]

    @model_validator(mode="after")
    def require_locator(self):
        if self.page is None and self.section is None:
            raise ValueError("A citation requires a page or section")
        return self


class RagRecommendation(ContractModel):
    status: RecommendationStatus
    answer: Annotated[Text, Field(max_length=12000)]
    models: tuple[Annotated[Text, Field(max_length=160)], ...] = Field(default=(), max_length=12)
    citations: tuple[Citation, ...] = Field(default=(), max_length=32)
    manuals: tuple[DocumentVersionKey, ...] = Field(default=(), max_length=12)
    include_report: bool = False

    @model_validator(mode="after")
    def require_grounded_outputs(self):
        if self.status == "recommended":
            if not self.models or not self.citations:
                raise ValueError("Recommendations require a model and verified citations")
        elif self.models or self.manuals or self.include_report:
            raise ValueError("Only recommendations can include models, manuals or reports")
        cited = {(item.document_id, item.version) for item in self.citations}
        if any((item.document_id, item.version) not in cited for item in self.manuals):
            raise ValueError("A manual must be backed by a citation")
        return self


class PreviousTurn(ContractModel):
    text: str
    answer: str


class RagContext(ContractModel):
    org_id: str
    user_id: str
    text: str
    previous_turns: tuple[PreviousTurn, ...]
    documents: tuple[LibraryDocument, ...]
    knowledge_base_id: Identifier = "default"
    index_version: Identifier | None = None
    embedding_model_id: Identifier | None = None
    embedding_model_revision: str | None = None


class ManualAttachment(DocumentVersionKey):
    filename: str


class LocatedCitation(Citation):
    filename: str


class RecommendationTurn(ContractModel):
    turn_id: Identifier
    conversation_id: Identifier
    created_at: datetime
    status: RecommendationStatus
    answer: str
    models: tuple[str, ...]
    citations: tuple[LocatedCitation, ...]
    manuals: tuple[ManualAttachment, ...]
    has_report: bool


class HistoryEntry(ContractModel):
    turn_id: Identifier
    created_at: datetime
    request_summary: str
    available: bool
    reply: RecommendationTurn | None


class ConversationSummary(ContractModel):
    conversation_id: Identifier
    title: str
    updated_at: datetime
    pinned: bool


class ConversationUpdate(ContractModel):
    title: Annotated[Text, Field(max_length=160)] | None = None
    pinned: StrictBool | None = None

    @model_validator(mode="after")
    def require_change(self):
        if self.title is None and self.pinned is None:
            raise ValueError("A conversation update requires a title or pinned state")
        return self


class ConversationList(ContractModel):
    items: list[ConversationSummary]
    next_offset: int | None


class ConversationTurn(ContractModel):
    turn_id: Identifier
    text: str
    reply: RecommendationTurn | None


class ConversationHistory(ContractModel):
    conversation_id: Identifier
    knowledge_base_id: Identifier = "default"
    turns: list[ConversationTurn]
    before: Identifier | None


class RagUnavailable(RuntimeError):
    """A required RAG service is unavailable."""


class RagDocumentError(ValueError):
    """The uploaded document cannot be parsed or indexed."""


class RagModelUnavailable(RagUnavailable):
    """The configured embedding model cannot be loaded right now."""

    def __init__(self, message: str, *, code: str = "embedding_model_unavailable", retryable: bool = True):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class RagModelInvalid(RagDocumentError):
    """The embedding model failed validation or does not match the index."""

    def __init__(self, message: str, *, code: str = "embedding_model_invalid"):
        super().__init__(message)
        self.code = code


class RagIndexNotFound(RagUnavailable):
    """The requested index version does not exist or is not readable."""

    def __init__(self, message: str, *, code: str = "index_version_not_found", retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class RagIndexConflict(RuntimeError):
    """An index version is busy, inconsistent or cannot take over requests."""

    def __init__(self, message: str, *, code: str = "index_task_conflict", retryable: bool = True):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class RagEngine(Protocol):
    def index_document(self, org_id: str, document: LibraryDocument, payload: bytes) -> None: ...

    def recommend(self, context: RagContext) -> RagRecommendation: ...


@runtime_checkable
class RagDiagnostics(Protocol):
    def diagnose(self, context: RagContext) -> RetrievalDiagnostics: ...


@runtime_checkable
class RagChunkReader(Protocol):
    def read_chunks(
        self,
        org_id: str,
        knowledge_base_id: str,
        index_version: str,
        key: DocumentVersionKey,
        *,
        offset: int = 0,
        limit: int = 20,
    ) -> ChunkReadPage: ...


@runtime_checkable
class RebuildSource(Protocol):
    def snapshot(self, org_id: str, knowledge_base_id: str) -> tuple[LibraryDocument, ...]:
        """Return the currently published versions for the knowledge base."""
        ...

    def read_payload(self, org_id: str, key: DocumentVersionKey) -> bytes:
        """Return the source bytes for a document version."""
        ...


@runtime_checkable
class RagIndexManager(Protocol):
    def availability(self, org_id: str, knowledge_base_id: str) -> IndexAvailability: ...

    def model_availability(self, model_id: str) -> tuple[bool, str | None, str | None]: ...

    def ensure_index_version(
        self,
        org_id: str,
        knowledge_base_id: str,
        embedding_model_id: str,
        *,
        index_version: str | None = None,
        activate: bool = False,
    ) -> IndexDescriptor: ...

    def index_descriptor(self, org_id: str, knowledge_base_id: str, index_version: str) -> IndexDescriptor: ...

    def list_index_versions(self, org_id: str, knowledge_base_id: str) -> tuple[IndexDescriptor, ...]: ...

    def mark_ready(self, org_id: str, knowledge_base_id: str, index_version: str) -> IndexDescriptor: ...

    def start_rebuild(
        self,
        org_id: str,
        knowledge_base_id: str,
        embedding_model_id: str,
        source: RebuildSource,
        *,
        task_id: str | None = None,
        background: bool = True,
    ) -> RebuildTask: ...

    def rebuild_status(self, task_id: str) -> RebuildTask: ...

    def list_rebuilds(
        self, org_id: str, knowledge_base_id: str, *, limit: int = 10,
    ) -> tuple[RebuildTask, ...]: ...

    def resume_rebuild(
        self,
        task_id: str,
        source: RebuildSource,
        *,
        background: bool = True,
    ) -> RebuildTask: ...

    def recover_rebuilds(self) -> None: ...

    def validate_scope(
        self,
        org_id: str,
        knowledge_base_id: str,
        index_version: str,
        documents: tuple[LibraryDocument, ...],
    ) -> IndexSwitchPlan: ...

    def mark_active(
        self,
        org_id: str,
        knowledge_base_id: str,
        index_version: str,
        *,
        documents: tuple[LibraryDocument, ...] | None = None,
        allow_stale: bool = False,
    ) -> IndexSwitchPlan: ...

    def retire(self, org_id: str, knowledge_base_id: str, index_version: str) -> None: ...


@runtime_checkable
class RagIndexRemover(Protocol):
    def remove_document(self, org_id: str, key: DocumentVersionKey) -> None:
        """Remove the version from every knowledge base and index version."""
        ...


@runtime_checkable
class RagIndexGarbageCollector(Protocol):
    def purge_retired(
        self, org_id: str, knowledge_base_id: str, *, keep_superseded: int = 1,
    ) -> IndexGcResult: ...
