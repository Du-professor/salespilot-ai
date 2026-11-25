from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RagModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ParsedSection(RagModel):
    page: int | None = None
    section: str | None = None
    text: str
    model: str | None = None


class Chunk(RagModel):
    org_id: str
    document_id: str
    version: str
    chunk_id: str
    model: str | None = None
    page: int | None = None
    section: str | None = None
    ordinal: int
    text: str
    truncated: bool = False
    parent_id: str | None = None


class ScoredChunk(RagModel):
    chunk: Chunk
    score: float
    vector_score: float | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None


class Fact(RagModel):
    model: str
    assembly: str | None = None
    document_id: str
    version: str
    chunk_id: str
    page: int | None = None
    section: str | None = None
    attribute: str
    label: str
    value: float | str | bool | None = None
    upper: float | None = None
    unit: str | None = None
    polarity: Literal["affirmed", "negated"] = "affirmed"
    excerpt: str


class CandidateScope(RagModel):
    scope_id: str
    model: str
    assembly: str
    document_id: str
    version: str
    facts: tuple[Fact, ...] = ()


class Requirement(RagModel):
    attribute: str
    operator: Literal["min", "max", "eq", "range", "present", "absent"]
    value: float | str | bool | None = None
    upper: float | None = None
    unit: str | None = None
    label: str
    sentence: str
    hard: bool = True
    source: Literal["current", "previous", "profile"] = "current"
    known: bool = True


class RequirementSet(RagModel):
    category: str
    requirements: tuple[Requirement, ...] = ()
    missing: tuple[str, ...] = ()
    questions: tuple[str, ...] = ()
    wants_manuals: bool = False
    wants_report: bool = False


class Check(RagModel):
    attribute: str
    label: str
    status: Literal["pass", "fail", "unknown"]
    detail: str
    hard: bool = True
    requirement_sentence: str = ""
    evidence: tuple[Fact, ...] = ()


class CandidateEvaluation(RagModel):
    model: str
    assembly: str
    document_id: str
    version: str
    status: Literal["pass", "fail", "unknown"]
    checks: tuple[Check, ...]
    scope: CandidateScope


class Claim(RagModel):
    model: str
    assembly: str | None = None
    document_id: str
    version: str
    page: int | None = None
    section: str | None = None
    attribute: str
    value: str
    upper: str | None = None
    unit: str | None = None
    polarity: Literal["affirmed", "negated"] = "affirmed"
    excerpt: str
    chunk_id: str
    statement: str = ""
    supporting: bool = False


class StructuredDraft(RagModel):
    status: Literal["recommended", "needs_input", "insufficient_evidence"]
    answer: str
    models: tuple[str, ...] = ()
    claims: tuple[Claim, ...] = ()
    manuals: tuple[tuple[str, str], ...] = ()
    include_report: bool = False
    notes: tuple[str, ...] = ()


class VerifiedClaim(RagModel):
    claim: Claim
    chunk: Chunk


class ModelClaim(RagModel):
    chunk_id: str
    model: str
    assembly: str
    document_id: str
    version: str
    page: int | None = None
    section: str | None = None
    attribute: str
    value: str
    upper: str | None = None
    unit: str | None = None
    polarity: Literal["affirmed", "negated"] = "affirmed"
    excerpt: str


class ModelDraftPayload(RagModel):
    claims: tuple[ModelClaim, ...] = ()
    answer: str | None = None
