import hashlib
import re

from .models import CandidateScope, Chunk, Claim, Fact
from .industrial import FIELDS, PANEL, category_for_model, read_specs
from .specs import MODEL_PATTERN, find_capability_mentions, find_numeric_mentions, spec_label
from .text import sentence_at


def _format(value: float | str | bool | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


_PAGE_SECTION_RE = re.compile(r"^第\s*\d+\s*页$")


def assembly_for(chunk: Chunk) -> str:
    if category_for_model(chunk.model):
        return "规格书所列系列"
    section = (chunk.section or "").strip()
    # Decouple PDF page numbers from product assembly identity so multi-page
    # specs for the same model and document version combine into one candidate
    # while citations still preserve exact page numbers.
    if _PAGE_SECTION_RE.fullmatch(section):
        return "默认装配"
    title = section.split("·")[0].strip()
    title = re.sub(r"^\d+(?:\.\d+)*[\.、\s]*", "", title).strip()
    title = re.sub(
        r"^(?:" + MODEL_PATTERN.pattern + r")\s*", "", title, flags=re.I,
    ).strip()
    return title or section or "默认装配"


def fact_statement(fact: Fact) -> str:
    label = fact.label or spec_label(fact.attribute)
    if fact.attribute == "industrial_kind":
        return "电阻触摸平板电脑" if fact.value == PANEL else "机架式电脑"
    if fact.attribute == "interface":
        return f"数据接口 {fact.value}"
    if fact.attribute == "gpu_model":
        return f"支持安装 {fact.value}"
    if fact.attribute == "ingress_protection":
        return f"防护等级 {fact.value}"
    if fact.upper is not None:
        return f"{label} {_format(fact.value)} 至 {_format(fact.upper)} {fact.unit or ''}".strip()
    if isinstance(fact.value, bool) or fact.polarity == "negated":
        verb = "支持" if fact.polarity == "affirmed" and fact.value is not False else "不支持"
        return f"{verb} {label}"
    if fact.unit == "x":
        return f"{label} x{_format(fact.value)}"
    if fact.unit:
        return f"{label} {_format(fact.value)} {fact.unit}"
    return f"{label} {_format(fact.value)}".strip()


def _chunk_model(chunk: Chunk) -> str | None:
    if chunk.model:
        return chunk.model
    models = set(MODEL_PATTERN.findall(chunk.section or "")) | set(MODEL_PATTERN.findall(chunk.text))
    return next(iter(models)) if len(models) == 1 else None


INSTRUCTION_CUES = (
    "忽略以上", "忽略上述", "忽略所有", "ignore previous", "系统指令", "系统提示", "提示语",
)


def _instruction_like(text: str) -> bool:
    lowered = text.casefold()
    return any(cue in lowered for cue in INSTRUCTION_CUES)


def extract_facts(chunk: Chunk) -> tuple[Fact, ...]:
    model = _chunk_model(chunk)
    if model is None:
        return ()
    assembly = assembly_for(chunk)
    if category_for_model(chunk.model):
        if _instruction_like(chunk.text):
            return ()
        return tuple(
            Fact(model=model, assembly=assembly, document_id=chunk.document_id, version=chunk.version,
                 chunk_id=chunk.chunk_id, page=chunk.page, section=chunk.section, attribute=item.attribute,
                 label=FIELDS[item.attribute], value=item.value, unit=item.unit,
                 polarity="negated" if item.value is False else "affirmed", excerpt=chunk.text)
            for item in read_specs(chunk.text)
        )
    facts: list[Fact] = []
    seen: set[tuple[str, str | None, str | None, str]] = set()

    def add(
        attribute: str,
        label: str,
        value: float | str | bool | None,
        unit: str | None,
        polarity: str,
        start: int,
        end: int,
        upper: float | None = None,
    ) -> None:
        excerpt = sentence_at(chunk.text, start, end)
        if _instruction_like(excerpt):
            return
        marker = (attribute, str(value), str(upper), polarity, excerpt)
        if marker in seen:
            return
        seen.add(marker)
        facts.append(Fact(
            model=model,
            assembly=assembly,
            document_id=chunk.document_id,
            version=chunk.version,
            chunk_id=chunk.chunk_id,
            page=chunk.page,
            section=chunk.section,
            attribute=attribute,
            label=label,
            value=value,
            upper=upper,
            unit=unit,
            polarity=polarity,
            excerpt=excerpt,
        ))

    for mention in find_numeric_mentions(chunk.text):
        add(
            mention.key, mention.label, mention.value, mention.unit,
            "affirmed", mention.start, mention.end, mention.upper,
        )

    for mention in find_capability_mentions(chunk.text):
        add(
            mention.key, mention.label, mention.value, None,
            mention.polarity, mention.start, mention.end,
        )

    facts.sort(key=lambda item: (item.attribute, item.page or 0, item.section or "", item.chunk_id, str(item.value)))
    return tuple(facts)


def build_scopes(chunks: list[Chunk]) -> list[CandidateScope]:
    grouped: dict[tuple[str, str, str, str], list[Fact]] = {}
    for chunk in chunks:
        for fact in extract_facts(chunk):
            key = (fact.model, fact.assembly or "", fact.document_id, fact.version)
            grouped.setdefault(key, []).append(fact)
    scopes: list[CandidateScope] = []
    for (model, assembly, document_id, version), facts in sorted(grouped.items()):
        identity = "\0".join((model, assembly, document_id, version))
        scopes.append(CandidateScope(
            scope_id="scope_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24],
            model=model,
            assembly=assembly,
            document_id=document_id,
            version=version,
            facts=tuple(facts),
        ))
    return scopes


def facts_for(scope: CandidateScope, attribute: str) -> tuple[Fact, ...]:
    return tuple(fact for fact in scope.facts if fact.attribute == attribute)

def claim_for_fact(fact: Fact, *, supporting: bool = False) -> Claim:
    return Claim(
        model=fact.model,
        assembly=fact.assembly,
        document_id=fact.document_id,
        version=fact.version,
        page=fact.page,
        section=fact.section,
        attribute=fact.attribute,
        value=_format(fact.value),
        upper=_format(fact.upper) if fact.upper is not None else None,
        unit=fact.unit,
        polarity=fact.polarity,
        excerpt=fact.excerpt,
        chunk_id=fact.chunk_id,
        statement=fact_statement(fact),
        supporting=supporting,
    )
