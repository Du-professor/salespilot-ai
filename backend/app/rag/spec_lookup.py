import re

from app.contracts.rag_workspace import Citation, RagContext, RagRecommendation

from .facts import build_scopes, claim_for_fact, fact_statement
from .models import Chunk, Claim, Fact, StructuredDraft
from .specs import SPECS, spec_label
from .verification import build_recommendation


def _targets(question: str) -> list[str]:
    lowered = question.casefold()
    keys: list[str] = []
    if re.search(r"双网口|两个网口|2\s*个网口", question):
        keys.append("gige_ports")
    if any(word in lowered for word in ("网口", "接口", "gige", "以太网")):
        keys.append("interface")
    if "poe" in lowered or "供电" in lowered:
        keys.append("poe")
    if "分辨率" in lowered or "像素" in lowered or re.search(r"\bmp\b", lowered):
        keys.append("resolution_mp")
    if "帧率" in lowered or "帧速" in lowered or "fps" in lowered:
        keys.append("fps")
    if "modbus" in lowered:
        keys.append("modbus_tcp")
    if "负载" in lowered or "公斤" in lowered or "kg" in lowered:
        keys.append("payload_kg")
    if "工作半径" in lowered or "臂展" in lowered or "工作范围" in lowered or "毫米" in lowered:
        keys.append("reach_mm")
    if "温度" in lowered:
        keys.append("operating_temp_c")
    if "功耗" in lowered or "功率" in lowered:
        keys.extend(("gpu_max_power_w", "gpu_power_w"))
    for key, spec in SPECS.items():
        if any(alias.casefold() in lowered for alias in spec.aliases):
            keys.append(key)
    return list(dict.fromkeys(keys))


def _citation(fact: Fact) -> Citation:
    return Citation(
        document_id=fact.document_id,
        version=fact.version,
        page=fact.page,
        section=None if fact.page is not None else fact.section,
        excerpt=fact.excerpt,
    )


def _unique_values(facts: list[Fact]) -> list[str]:
    values: list[str] = []
    for fact in facts:
        value = f"{fact.value}|{fact.upper}|{fact.polarity}"
        if value not in values:
            values.append(value)
    return values


def _scope_facts(scopes, key: str) -> list[Fact]:
    return [fact for scope in scopes for fact in scope.facts if fact.attribute == key]


def lookup_spec(context: RagContext, chunks: list[Chunk], model: str, question: str) -> RagRecommendation:
    targets = _targets(question)
    if not targets:
        return RagRecommendation(
            status="needs_input",
            answer="请说明要查证的规格，例如接口、分辨率、帧率、供电、负载或工作半径。",
        )

    version_match = re.search(r"\b(v\d+)\b", question, re.I)
    version = version_match.group(1) if version_match else None
    scopes = [scope for scope in build_scopes(chunks) if scope.model.casefold() == model.casefold()]
    if version is not None:
        scopes = [scope for scope in scopes if scope.version.casefold() == version.casefold()]
    if not scopes:
        suffix = f" 的 {version} 版本" if version else " 的已核验规格"
        return RagRecommendation(
            status="insufficient_evidence",
            answer=f"当前授权资料中没有找到型号 {model}{suffix}。",
        )

    confirmed: list[Fact] = []
    claims: list[Claim] = []
    gaps: list[str] = []
    for key in targets:
        facts = _scope_facts(scopes, key)
        if not facts:
            gaps.append(key)
            continue
        values = _unique_values(facts)
        if len(values) > 1:
            locations = "；".join(sorted({
                f"{fact.version} · {fact.assembly or '默认装配'} · {fact.value}{fact.unit or ''}"
                for fact in facts
            }))
            citations_list: list[Citation] = []
            citation_keys: set[tuple[str, str, int | None, str | None, str]] = set()
            for fact in facts:
                citation = _citation(fact)
                marker = (
                    citation.document_id, citation.version, citation.page,
                    citation.section, citation.excerpt,
                )
                if marker not in citation_keys:
                    citation_keys.add(marker)
                    citations_list.append(citation)
            return RagRecommendation(
                status="insufficient_evidence",
                answer=f"型号 {model} 的 {spec_label(key)}在不同版本或装配中记录不一致：{locations}。",
                citations=tuple(citations_list),
            )
        confirmed.extend(facts)
        claims.extend(claim_for_fact(fact) for fact in facts)

    if not confirmed:
        missing = "；".join(f"{spec_label(key)}未在资料中记录" for key in gaps)
        return RagRecommendation(
            status="insufficient_evidence",
            answer=f"当前资料无法确认型号 {model} 的规格：{missing}。",
        )

    draft = StructuredDraft(
        status="recommended",
        answer="",
        models=(model,),
        claims=tuple(claims),
        manuals=(),
        include_report=False,
    )
    result = build_recommendation(draft, context, chunks)
    if result.status != "recommended":
        return RagRecommendation(
            status="insufficient_evidence",
            answer=f"型号 {model} 的规格声明未能通过引用校验，当前资料不足以回答该问题。",
        )

    statements = list(dict.fromkeys(fact_statement(fact) for fact in confirmed))
    gap_statements = [f"{spec_label(key)}未在资料中记录" for key in gaps]
    prefix = f"{model}" + (f"（{version}）" if version else "")
    answer = prefix + "：" + "；".join(statements + gap_statements) + "。"
    return result.model_copy(update={"answer": answer})