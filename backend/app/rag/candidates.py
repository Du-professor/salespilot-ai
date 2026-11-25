from .facts import build_scopes, fact_statement, facts_for
from .industrial import FIELDS as INDUSTRIAL_FIELDS, NUMERIC_FIELDS as INDUSTRIAL_NUMERIC_FIELDS
from .models import (
    CandidateEvaluation,
    CandidateScope,
    Check,
    Chunk,
    Claim,
    Requirement,
    RequirementSet,
    StructuredDraft,
)
from .specs import SPECS, spec_label


def _number(value: float | str | bool | None) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format(value: float | str | bool | None) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _distinct_values(facts: tuple) -> list[float]:
    values = sorted({_number(fact.value) for fact in facts if _number(fact.value) is not None})
    return values


def _unknown(requirement: Requirement, detail: str, evidence: tuple = ()) -> Check:
    return Check(
        attribute=requirement.attribute, label=requirement.label, status="unknown", detail=detail,
        hard=requirement.hard, requirement_sentence=requirement.sentence, evidence=tuple(evidence),
    )


def _numeric_check(scope: CandidateScope, requirement: Requirement) -> Check:
    facts = facts_for(scope, requirement.attribute)
    if not facts:
        return _unknown(requirement, f"{requirement.label}在资料中未记录")
    expected = _number(requirement.value)
    if expected is None:
        return _unknown(requirement, f"{requirement.label}的需求值无法识别", facts)
    values = _distinct_values(facts)
    if not values:
        return _unknown(requirement, f"{requirement.label}在资料中没有可比较数值", facts)
    if len(values) > 1:
        return _unknown(requirement, f"{requirement.label}在资料中存在冲突数值", facts)
    actual = values[0]
    if requirement.operator == "min":
        ok = actual >= expected
    elif requirement.operator == "max":
        ok = actual <= expected
    else:
        ok = actual == expected
    detail = f"{requirement.label} {_format(actual)}{requirement.unit or ''}"
    return Check(
        attribute=requirement.attribute, label=requirement.label,
        status="pass" if ok else "fail", detail=detail, hard=requirement.hard,
        requirement_sentence=requirement.sentence, evidence=tuple(facts),
    )


def _range_check(scope: CandidateScope, requirement: Requirement) -> Check:
    facts = tuple(fact for fact in facts_for(scope, requirement.attribute) if fact.upper is not None)
    if not facts:
        return _unknown(requirement, f"{requirement.label}在资料中未记录")
    pairs = sorted({(float(fact.value), float(fact.upper)) for fact in facts if _number(fact.value) is not None})
    if len(pairs) > 1:
        return _unknown(requirement, f"{requirement.label}在资料中存在冲突范围", facts)
    low, high = pairs[0]
    expected_low = _number(requirement.value)
    expected_high = _number(requirement.upper)
    if expected_low is None or expected_high is None:
        return _unknown(requirement, f"{requirement.label}的需求范围无法识别", facts)
    ok = low <= expected_low and high >= expected_high
    detail = f"{requirement.label} {_format(low)} 至 {_format(high)} {requirement.unit or ''}".strip()
    return Check(
        attribute=requirement.attribute, label=requirement.label,
        status="pass" if ok else "fail", detail=detail, hard=requirement.hard,
        requirement_sentence=requirement.sentence, evidence=facts,
    )


def _present_check(scope: CandidateScope, requirement: Requirement) -> Check:
    facts = facts_for(scope, requirement.attribute)
    if not facts:
        return _unknown(requirement, f"{requirement.label}在资料中未记录")
    affirmed = any(
        fact.polarity == "affirmed" and fact.value is not False and str(fact.value).casefold() not in {"否", "无"}
        for fact in facts
    )
    negated = any(fact.polarity == "negated" or fact.value is False for fact in facts)
    if affirmed and negated:
        return _unknown(requirement, f"{requirement.label}在资料中存在正负冲突", facts)
    if negated:
        return Check(
            attribute=requirement.attribute, label=requirement.label, status="fail",
            detail=f"资料明确不支持{requirement.label}", hard=requirement.hard,
            requirement_sentence=requirement.sentence, evidence=facts,
        )
    if affirmed:
        return Check(
            attribute=requirement.attribute, label=requirement.label, status="pass",
            detail=f"支持{requirement.label}", hard=requirement.hard,
            requirement_sentence=requirement.sentence, evidence=facts,
        )
    return _unknown(requirement, f"{requirement.label}在资料中没有肯定依据", facts)


def _enum_check(scope: CandidateScope, requirement: Requirement) -> Check:
    facts = facts_for(scope, requirement.attribute)
    if not facts:
        return _unknown(requirement, f"{requirement.label}在资料中未记录")
    expected = str(requirement.value).casefold()
    actual_values = {str(fact.value).casefold() for fact in facts if fact.value is not None}
    if expected in actual_values:
        return Check(
            attribute=requirement.attribute, label=requirement.label, status="pass",
            detail=f"{requirement.label} {requirement.value}", hard=requirement.hard,
            requirement_sentence=requirement.sentence, evidence=facts,
        )
    return Check(
        attribute=requirement.attribute, label=requirement.label, status="fail",
        detail=f"{requirement.label} 为 {'、'.join(sorted(actual_values)) or '其他值'}",
        hard=requirement.hard, requirement_sentence=requirement.sentence, evidence=facts,
    )


def _power_check(
    scope: CandidateScope, requirement: Requirement, scopes: list[CandidateScope],
) -> Check:
    power_facts = facts_for(scope, "gpu_power_w")
    gpu_facts = facts_for(scope, "gpu_model")
    if not power_facts:
        return _unknown(requirement, "GPU 供电额度在资料中未记录")
    if not gpu_facts:
        return _unknown(requirement, "显卡兼容型号在资料中未记录")
    power_values = _distinct_values(power_facts)
    if len(power_values) != 1:
        return _unknown(requirement, "GPU 供电额度存在冲突数值", power_facts)
    gpu_value = str(gpu_facts[0].value).casefold()
    component_facts = [
        fact
        for candidate in scopes
        if candidate.model.casefold() == gpu_value
        for fact in facts_for(candidate, "gpu_max_power_w")
    ]
    if not component_facts:
        return _unknown(requirement, "独立显卡功耗资料未取得", (*power_facts, *gpu_facts))
    component_values = _distinct_values(tuple(component_facts))
    if len(component_values) != 1:
        return _unknown(requirement, "独立显卡功耗存在冲突数值", tuple(component_facts))
    expected = _number(requirement.value)
    if expected is None:
        return _unknown(requirement, "GPU 供电余量需求无法识别")
    headroom = power_values[0] - component_values[0]
    detail = (
        f"GPU 供电额度 {_format(power_values[0])} W，"
        f"{gpu_facts[0].value} 最大功耗 {_format(component_values[0])} W"
    )
    return Check(
        attribute=requirement.attribute, label=requirement.label,
        status="pass" if headroom >= expected else "fail", detail=detail, hard=requirement.hard,
        requirement_sentence=requirement.sentence,
        evidence=tuple((*power_facts, *gpu_facts, *component_facts)),
    )


def _check(scope: CandidateScope, requirement: Requirement, scopes: list[CandidateScope]) -> Check:
    if requirement.attribute in INDUSTRIAL_FIELDS:
        facts = facts_for(scope, requirement.attribute)
        if requirement.attribute in INDUSTRIAL_NUMERIC_FIELDS:
            check = _numeric_check(scope, requirement)
            return check.model_copy(update={"evidence": check.evidence[:1]}) if check.status == "pass" else check
        matching = tuple(fact for fact in facts if fact.value == requirement.value)
        if not facts:
            return _unknown(requirement, f"{requirement.label}在资料中未记录")
        if matching and isinstance(requirement.value, bool) and len(matching) != len(facts):
            return _unknown(requirement, f"{requirement.label}在资料中存在冲突", facts)
        return Check(
            attribute=requirement.attribute, label=requirement.label, status="pass" if matching else "fail",
            detail=requirement.label, hard=requirement.hard, requirement_sentence=requirement.sentence,
            evidence=matching[:1] or facts[:1],
        )
    if requirement.attribute == "gpu_power_headroom_w":
        return _power_check(scope, requirement, scopes)
    spec = SPECS.get(requirement.attribute)
    if spec is not None and spec.kind == "range":
        return _range_check(scope, requirement)
    if spec is not None and spec.kind == "boolean":
        return _present_check(scope, requirement)
    if spec is not None and spec.kind == "enum":
        return _enum_check(scope, requirement)
    if requirement.attribute.startswith("capability:"):
        return _present_check(scope, requirement)
    return _numeric_check(scope, requirement)


def _evaluate(scope: CandidateScope, requirements: RequirementSet, scopes: list[CandidateScope]) -> CandidateEvaluation:
    checks = tuple(_check(scope, requirement, scopes) for requirement in requirements.requirements)
    hard = [check for check in checks if check.hard]
    if any(check.status == "fail" for check in hard):
        status = "fail"
    elif any(check.status == "unknown" for check in hard):
        status = "unknown"
    else:
        status = "pass"
    return CandidateEvaluation(
        model=scope.model,
        assembly=scope.assembly,
        document_id=scope.document_id,
        version=scope.version,
        status=status,
        checks=checks,
        scope=scope,
    )


def evaluate_candidates(
    chunks: list[Chunk], requirements: RequirementSet,
) -> tuple[list[CandidateEvaluation], list[CandidateScope]]:
    scopes = build_scopes(chunks)
    return [_evaluate(scope, requirements, scopes) for scope in scopes], scopes


def _claim(fact, model: str, statement: str, supporting: bool) -> Claim:
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
        statement=statement,
        supporting=supporting,
    )


def _claims_for(evaluation: CandidateEvaluation) -> list[Claim]:
    claims: list[Claim] = []
    for check in evaluation.checks:
        if not check.hard or check.status != "pass":
            continue
        for fact in check.evidence:
            supporting = fact.model.casefold() != evaluation.model.casefold()
            claims.append(_claim(
                fact, evaluation.model,
                fact_statement(fact) if not supporting else fact_statement(fact),
                supporting,
            ))
    unique: dict[tuple[str, str, str, str, str], Claim] = {}
    for claim in claims:
        key = (claim.model, claim.attribute, claim.chunk_id, claim.value, claim.polarity)
        unique.setdefault(key, claim)
    return list(unique.values())


def _insufficient_answer(
    evaluations: list[CandidateEvaluation], requirements: RequirementSet,
) -> str:
    hard = [requirement for requirement in requirements.requirements if requirement.hard]
    fail_labels = sorted({
        check.label for evaluation in evaluations for check in evaluation.checks
        if check.hard and check.status == "fail"
    })
    unknown_labels = sorted({
        check.label for evaluation in evaluations for check in evaluation.checks
        if check.hard and check.status == "unknown"
    })
    parts: list[str] = []
    if fail_labels:
        parts.append("候选不满足：" + "、".join(fail_labels))
    if unknown_labels:
        parts.append("候选缺少依据：" + "、".join(unknown_labels))
    if not parts:
        parts.append("现有资料中没有满足全部硬性要求的候选")
    if not hard:
        parts.append("需求没有可核验的硬性条件")

    versions: dict[tuple[str, str], set[str]] = {}
    for evaluation in evaluations:
        versions.setdefault((evaluation.model, evaluation.assembly), set()).add(evaluation.version)
    conflicts = [key for key, values in versions.items() if len(values) > 1]
    if conflicts:
        parts.append("同一型号的多个版本或装配未能由任一单版同时满足全部硬条件，未合并版本证据")

    unresolved = [
        f"{evaluation.model} · {evaluation.assembly} · {evaluation.version}"
        for evaluation in evaluations if evaluation.status != "pass"
    ]
    if unresolved:
        parts.append("未纳入：" + "；".join(sorted(set(unresolved))[:6]))
    return "；".join(f"{part}。" for part in parts)


def build_draft(
    evaluations: list[CandidateEvaluation],
    requirements: RequirementSet,
    chunks: list[Chunk],
    scopes: list[CandidateScope],
) -> StructuredDraft:
    passing = [evaluation for evaluation in evaluations if evaluation.status == "pass"]
    if not passing:
        return StructuredDraft(
            status="insufficient_evidence",
            answer=_insufficient_answer(evaluations, requirements),
        )

    by_model: dict[str, list[CandidateEvaluation]] = {}
    for evaluation in passing:
        by_model.setdefault(evaluation.model, []).append(evaluation)
    selected = [
        sorted(items, key=lambda item: (item.version, item.document_id, item.assembly))[-1]
        for items in by_model.values()
    ]
    models = tuple(sorted(by_model))

    claims: list[Claim] = []
    for evaluation in selected:
        claims.extend(_claims_for(evaluation))
    unique: dict[tuple[str, str, str, str, str], Claim] = {}
    for claim in claims:
        key = (claim.model, claim.attribute, claim.chunk_id, claim.value, claim.polarity)
        unique.setdefault(key, claim)
    ordered = tuple(unique.values())

    manuals: list[tuple[str, str]] = []
    if requirements.wants_manuals:
        cited = {
            (chunk.document_id, chunk.version)
            for chunk in chunks
            if chunk.chunk_id in {claim.chunk_id for claim in ordered}
        }
        for chunk in chunks:
            if chunk.model in models and (chunk.document_id, chunk.version) in cited:
                key = (chunk.document_id, chunk.version)
                if key not in manuals:
                    manuals.append(key)

    notes: list[str] = []
    selected_scopes = {(item.model, item.assembly, item.version) for item in selected}
    if any(item.status != "pass" for item in evaluations):
        notes.append("其他候选缺少可核实的装配、版本或供电证据，未纳入推荐。")
    for evaluation in evaluations:
        if (
            evaluation.model in by_model
            and (evaluation.model, evaluation.assembly, evaluation.version) not in selected_scopes
            and evaluation.status != "pass"
        ):
            notes.append(f"{evaluation.model} 的其他版本或装配未同时满足全部硬条件，未合并其证据。")
            break

    return StructuredDraft(
        status="recommended",
        answer="",
        models=models,
        claims=ordered,
        manuals=tuple(manuals),
        include_report=requirements.wants_report,
        notes=tuple(dict.fromkeys(notes)),
    )
