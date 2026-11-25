import re
from decimal import Decimal

from app.contracts.rag_workspace import (
    Citation,
    DocumentVersionKey,
    RagContext,
    RagRecommendation,
)

from .facts import assembly_for
from .industrial import FIELDS as INDUSTRIAL_FIELDS, category_for_model, read_specs
from .models import Chunk, Claim, ModelClaim, StructuredDraft, VerifiedClaim
from .specs import SPECS, normalize_phrase, spec_label
from .text import normalized

_NUMBER = r"[+-]?\d+(?:,\d{3})*(?:\.\d+)?"
_NUMBERS = re.compile(rf"(?<![\d.]){_NUMBER}(?![\d.])")
_NUMERIC_UNIT = re.compile(rf"(?<![\d.])({_NUMBER})\s*(Mbps|Gbps|MP|fps|W|°C)(?![A-Za-z])", re.I)
_LANES = re.compile(r"x(\d+)", re.I)
_NEGATION = ("不支持", "不具备", "不提供", "不含", "未提供", "未内置", "无", "非")


def _quantities(text: str) -> set[tuple[Decimal, str]]:
    quantities = {
        (Decimal(value.replace(",", "")), unit.casefold()) for value, unit in _NUMERIC_UNIT.findall(text)
    }
    quantities.update((Decimal(value), "lanes") for value in _LANES.findall(text))
    return quantities


def _number_present(excerpt: str, raw: str) -> bool:
    if not re.fullmatch(_NUMBER, raw):
        return False
    value = Decimal(raw.replace(",", ""))
    return any(Decimal(item.replace(",", "")) == value for item in _NUMBERS.findall(excerpt))


def _attribute_present(attribute: str, excerpt: str, value: str) -> bool:
    if attribute.startswith("capability:"):
        label = attribute.split(":", 1)[1]
        return normalize_phrase(label) in normalize_phrase(excerpt)
    if attribute.startswith("generic:"):
        label = attribute.split(":", 1)[1]
        return normalize_phrase(label) in normalize_phrase(excerpt)
    spec = SPECS.get(attribute)
    if spec is None:
        return False
    if attribute == "gpu_model":
        return normalize_phrase(value) in normalize_phrase(excerpt)
    return any(normalize_phrase(alias) in normalize_phrase(excerpt) for alias in spec.aliases)


def _policy_state(attribute: str, excerpt: str, value: str) -> str | None:
    compact = normalize_phrase(excerpt)
    if attribute.startswith("capability:"):
        target = normalize_phrase(attribute.split(":", 1)[1])
    elif attribute == "poe":
        target = "poe"
    elif attribute == "modbus_tcp":
        target = "modbustcp"
    elif attribute == "interface":
        target = normalize_phrase(value)
    elif attribute in {"gpu_model", "ingress_protection"}:
        target = normalize_phrase(value)
    elif attribute == "collaborative":
        target = "协作模式"
    else:
        target = normalize_phrase(spec_label(attribute))
    if not target or target not in compact:
        return None
    index = compact.find(target)
    prefix = compact[max(0, index - 10):index]
    return "negated" if any(cue in prefix for cue in _NEGATION) else "affirmed"


def _verified(claim: Claim, chunk: Chunk, allowed: set[tuple[str, str]]) -> bool:
    if (chunk.document_id, chunk.version) not in allowed:
        return False
    if claim.document_id != chunk.document_id or claim.version != chunk.version:
        return False
    if claim.assembly is not None and normalize_phrase(claim.assembly) != normalize_phrase(assembly_for(chunk)):
        return False
    if chunk.page is not None:
        if claim.page != chunk.page:
            return False
    elif claim.section != chunk.section:
        return False

    excerpt = normalized(claim.excerpt)
    text = normalized(chunk.text)
    if not excerpt or excerpt.casefold() not in text.casefold():
        return False
    model_in_excerpt = normalize_phrase(claim.model) in normalize_phrase(excerpt)
    model_on_chunk = bool(chunk.model) and normalize_phrase(claim.model) == normalize_phrase(chunk.model)
    if not model_in_excerpt and not model_on_chunk:
        return False
    if claim.attribute in INDUSTRIAL_FIELDS:
        return model_on_chunk and category_for_model(chunk.model) is not None and any(
            item.attribute == claim.attribute and str(item.value) == claim.value and item.unit == claim.unit
            and claim.polarity == ("negated" if item.value is False else "affirmed")
            for item in read_specs(claim.excerpt)
        )
    if not _attribute_present(claim.attribute, excerpt, claim.value):
        return False

    spec = SPECS.get(claim.attribute)
    if spec is not None and spec.kind in {"boolean", "enum"}:
        state = _policy_state(claim.attribute, excerpt, claim.value)
        if state is None:
            return False
        if claim.polarity != state:
            return False
        if spec.kind == "enum" and normalize_phrase(claim.value) not in normalize_phrase(excerpt):
            return False
        return True

    if claim.upper:
        if not _number_present(excerpt, claim.value) or not _number_present(excerpt, claim.upper):
            return False
    elif re.fullmatch(_NUMBER, claim.value):
        if not _number_present(excerpt, claim.value):
            return False
    elif normalize_phrase(claim.value) not in normalize_phrase(excerpt):
        return False
    if claim.unit:
        if claim.unit == "x":
            if not _LANES.search(excerpt):
                return False
        elif claim.unit.casefold() not in excerpt.casefold():
            return False
    return True


def _statement_for(claim: Claim) -> str:
    label = spec_label(claim.attribute)
    if claim.attribute == "interface":
        return f"数据接口 {claim.value}"
    if claim.attribute == "gpu_model":
        return f"支持安装 {claim.value}"
    if claim.attribute == "ingress_protection":
        return f"防护等级 {claim.value}"
    if claim.upper is not None:
        return f"{label} {claim.value} 至 {claim.upper} {claim.unit or ''}".strip()
    if claim.value in {"True", "False"}:
        return f"{'支持' if claim.value == 'True' else '不支持'} {label}"
    if claim.unit == "x":
        return f"{label} x{claim.value}"
    if claim.unit:
        return f"{label} {claim.value} {claim.unit}"
    return f"{label} {claim.value}".strip()


def _claim_identity(claim: Claim | ModelClaim) -> tuple:
    return (
        claim.chunk_id, claim.model, claim.assembly, claim.document_id, claim.version, claim.page,
        claim.section, claim.attribute, claim.value, claim.upper, claim.unit, claim.polarity, claim.excerpt,
    )


def compose_answer(models: tuple[str, ...], claims: list[VerifiedClaim], notes: tuple[str, ...] = ()) -> str:
    primary = [item.claim for item in claims if not item.claim.supporting]
    supporting = [item.claim for item in claims if item.claim.supporting]
    parts: list[str] = []
    for model in models:
        statements = list(dict.fromkeys(
            claim.statement for claim in primary
            if claim.model.casefold() == model.casefold() and claim.statement
        ))
        parts.append(f"推荐 {model}。" + ("；".join(statements) + "。" if statements else ""))

    supporting_statements = list(dict.fromkeys(
        f"{claim.model} {claim.statement}" if not claim.statement.startswith(claim.model) else claim.statement
        for claim in supporting if claim.statement
    ))
    if supporting_statements:
        parts.append("配套依据：" + "；".join(supporting_statements) + "。")
    parts.extend(notes)
    return "".join(part if part.endswith("。") else f"{part}。" for part in parts)


def build_recommendation(
    draft: StructuredDraft,
    context: RagContext,
    chunks: list[Chunk],
    model_claims: tuple[ModelClaim, ...] = (),
) -> RagRecommendation:
    if draft.status != "recommended":
        return RagRecommendation(status=draft.status, answer=draft.answer or "当前无法给出已验证的推荐。")

    allowed = {(document.document_id, document.version) for document in context.documents}
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    deterministic: list[VerifiedClaim] = []
    rejected: set[str] = set()
    for claim in draft.claims:
        chunk = chunk_by_id.get(claim.chunk_id)
        if chunk is None or not _verified(claim, chunk, allowed):
            rejected.add(claim.model)
            continue
        deterministic.append(VerifiedClaim(claim=claim, chunk=chunk))

    chosen = deterministic
    if model_claims:
        # The model can order verified facts but cannot introduce or modify them.
        verified_by_identity = {_claim_identity(item.claim): item for item in deterministic}
        ordered_identities = dict.fromkeys(map(_claim_identity, model_claims))
        if verified_by_identity.keys() <= ordered_identities.keys():
            chosen = [verified_by_identity[key] for key in ordered_identities if key in verified_by_identity]

    primary_models = {model.casefold() for model in draft.models}
    chosen = [
        item.model_copy(update={"claim": item.claim.model_copy(update={
            "supporting": item.claim.model.casefold() not in primary_models,
            "statement": item.claim.statement or _statement_for(item.claim),
        })})
        for item in chosen
    ]

    models = tuple(
        model for model in draft.models
        if model not in rejected
        and any(item.claim.model.casefold() == model.casefold() and not item.claim.supporting for item in chosen)
    )
    if not models:
        return RagRecommendation(
            status="insufficient_evidence",
            answer="关键声明未能通过引用校验，当前资料不足以给出已验证推荐。",
        )

    citations: list[Citation] = []
    seen: set[tuple[str, str, int | None, str | None, str]] = set()
    for item in chosen:
        if item.claim.model.casefold() not in {model.casefold() for model in models} and not item.claim.supporting:
            continue
        chunk = item.chunk
        citation = Citation(
            document_id=chunk.document_id,
            version=chunk.version,
            page=chunk.page,
            section=None if chunk.page is not None else chunk.section,
            excerpt=item.claim.excerpt,
        )
        key = (citation.document_id, citation.version, citation.page, citation.section, citation.excerpt)
        if key not in seen:
            seen.add(key)
            citations.append(citation)

    cited = {(item.document_id, item.version) for item in citations}
    manuals = tuple(
        DocumentVersionKey(document_id=document_id, version=version)
        for document_id, version in draft.manuals
        if (document_id, version) in cited and (document_id, version) in allowed
    )
    answer = compose_answer(models, chosen, draft.notes)
    return RagRecommendation(
        status="recommended",
        answer=answer,
        models=models,
        citations=tuple(citations),
        manuals=manuals,
        include_report=draft.include_report,
    )


def answer_is_grounded(answer: str, claims: tuple, models: tuple[str, ...] = ()) -> bool:
    excerpts = [normalized(claim.excerpt).casefold() for claim in claims]
    allowed = {claim.model.casefold() for claim in claims}
    allowed.update(model.casefold() for model in models)
    mentioned = {match.casefold() for match in re.findall(r"[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+", answer)}
    if not mentioned <= allowed:
        return False
    supported = set().union(*(_quantities(excerpt) for excerpt in excerpts))
    return _quantities(answer) <= supported
