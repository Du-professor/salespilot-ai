import re

from app.contracts.rag_workspace import PreviousTurn

from .industrial import industrial_requirements
from .models import Requirement, RequirementSet
from .specs import (
    PROFILES,
    PREFERENCE_CUES,
    SPECS,
    detect_profile,
    find_capability_mentions,
    find_numeric_mentions,
    question_for,
    spec_label,
)
from .text import normalized, sentence_at

_UPPER_BOUND_BEFORE = re.compile(r"(?:不超过|不得超过|不高于|不得高于|至多|最多|最高|上限(?:为)?|<=|≤)\s*$")
_UPPER_BOUND_AFTER = re.compile(r"\s*(?:及以下|以下|以内)")


def _hardness(text: str, start: int, end: int) -> bool:
    window = text[max(0, start - 24):min(len(text), end + 4)]
    return not any(cue in window for cue in PREFERENCE_CUES)


def _merge(requirements: list[Requirement], requirement: Requirement) -> None:
    existing = next((item for item in requirements
                     if (item.attribute, item.operator) == (requirement.attribute, requirement.operator)), None)
    if existing is None:
        requirements.append(requirement)
    elif requirement.hard and not existing.hard:
        requirements[requirements.index(existing)] = requirement


def _operator(key: str, text: str, start: int, end: int) -> str:
    spec = SPECS.get(key)
    if spec is None:
        return "present"
    if key == "operating_temp_c":
        return "range"
    if spec.kind == "boolean":
        return "present"
    if spec.kind == "enum":
        return "eq"
    if _UPPER_BOUND_BEFORE.search(text[max(0, start - 16):start]) or _UPPER_BOUND_AFTER.match(text, end):
        return "max"
    return "min"


def _parse(text: str, source: str) -> list[Requirement]:
    requirements: list[Requirement] = []

    for mention in find_numeric_mentions(text):
        _merge(requirements, Requirement(
            attribute=mention.key,
            operator=_operator(mention.key, text, mention.start, mention.end),
            value=mention.value,
            upper=mention.upper,
            unit=mention.unit,
            label=spec_label(mention.key),
            sentence=sentence_at(text, mention.start, mention.end),
            hard=_hardness(text, mention.start, mention.end),
            source=source,
            known=mention.key in SPECS,
        ))

    for mention in find_capability_mentions(text):
        _merge(requirements, Requirement(
            attribute=mention.key,
            operator="present" if mention.key not in {"interface", "gpu_model", "ingress_protection"} else "eq",
            value=mention.value,
            unit=None,
            label=spec_label(mention.key),
            sentence=sentence_at(text, mention.start, mention.end),
            hard=_hardness(text, mention.start, mention.end),
            source=source,
            known=mention.key in SPECS,
        ))

    interface = re.search(r"\bGigE\b", text, re.I)
    if interface is not None and not any(item.attribute == "interface" for item in requirements):
        _merge(requirements, Requirement(
            attribute="interface", operator="eq", value="GigE", label=spec_label("interface"),
            sentence=sentence_at(text, interface.start(), interface.end()),
            hard=_hardness(text, interface.start(), interface.end()), source=source, known=True,
        ))

    gpu = re.search(
        r"(?:安装|搭载|配置|使用|支持安装)\s*(" + r"[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+" + r"|[A-Z]{2,}\s+[A-Z]\d{3,}\s+\d+GB)",
        text, re.I,
    )
    if gpu is not None and not any(item.attribute == "gpu_model" for item in requirements):
        _merge(requirements, Requirement(
            attribute="gpu_model", operator="eq", value=gpu.group(1), label=spec_label("gpu_model"),
            sentence=sentence_at(text, gpu.start(), gpu.end()),
            hard=_hardness(text, gpu.start(), gpu.end()), source=source, known=True,
        ))

    modbus = re.search(r"Modbus\s*TCP", text, re.I)
    if modbus is not None and not any(item.attribute == "modbus_tcp" for item in requirements):
        _merge(requirements, Requirement(
            attribute="modbus_tcp", operator="present", value=True, label=spec_label("modbus_tcp"),
            sentence=sentence_at(text, modbus.start(), modbus.end()),
            hard=_hardness(text, modbus.start(), modbus.end()), source=source, known=True,
        ))

    poe = re.search(r"PoE", text, re.I)
    if poe is not None and not any(item.attribute == "poe" for item in requirements):
        _merge(requirements, Requirement(
            attribute="poe", operator="present", value=True, label=spec_label("poe"),
            sentence=sentence_at(text, poe.start(), poe.end()),
            hard=_hardness(text, poe.start(), poe.end()), source=source, known=True,
        ))
    return requirements


def extract_requirements(text: str, previous_turns: tuple[PreviousTurn, ...] = ()) -> RequirementSet:
    previous_text = previous_turns[-1].text if previous_turns and detect_profile(text) == "unknown" else ""
    if industrial := industrial_requirements(text, previous_text):
        return industrial
    current = normalized(text)
    requirements = _parse(current, "current")
    category = detect_profile(current)
    if category == "unknown":
        for turn in reversed(previous_turns):
            previous_category = detect_profile(normalized(turn.text))
            if previous_category != "unknown":
                category = previous_category
                break

    # The newest turn mentioning an attribute replaces its previous bounds.
    seen = {item.attribute for item in requirements}
    for turn in reversed(previous_turns):
        previous_text = normalized(turn.text)
        previous_category = detect_profile(previous_text)
        if previous_category != "unknown" and previous_category != category:
            break
        previous = _parse(previous_text, "previous")
        for requirement in previous:
            if requirement.attribute not in seen:
                _merge(requirements, requirement)
        seen.update(item.attribute for item in previous)

    if category in PROFILES:
        missing = tuple(
            key for key in PROFILES[category].required
            if not any(item.attribute == key and item.hard for item in requirements)
        )
    elif not requirements:
        missing = ("category",)
    else:
        missing = ()

    return RequirementSet(
        category=category,
        requirements=tuple(requirements),
        missing=missing,
        questions=tuple(question_for(key) for key in missing),
        wants_manuals="手册" in current,
        wants_report="报告" in current,
    )
