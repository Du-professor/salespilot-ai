import re
import unicodedata
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from xml.etree.ElementTree import ParseError
from zipfile import BadZipFile, ZipFile

from defusedxml.ElementTree import fromstring
from defusedxml.common import DefusedXmlException

from app.contracts.rag_workspace import RagDocumentError
from app.infrastructure.uploads import LIBRARY_DOCX_MAX_EXPANDED_BYTES, validate_archive

from .models import ParsedSection, Requirement, RequirementSet

PANEL = "resistive_panel_pc"
RACK = "rack_mount_pc"
INDUSTRIAL_CATEGORIES = frozenset({PANEL, RACK})
PANEL_SKU = re.compile(r"PL\d{3}RQ-([A-Z0-9]+)", re.I)
PANEL_SERIES = re.compile(r"PLxxxRQ-[A-Z0-9]+(?: B)?", re.I)
RACK_SKU = re.compile(r"IPC\d{3,4}-[A-Z0-9]+(?:-[A-Z0-9]+)*", re.I)
WORD = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
DOCX_MEMBER = "word/document.xml"
GENERATION = re.compile(r"(?<!\d)(\d{1,2}(?:\s*[/~–—-]\s*\d{1,2})*)\s*(?:代|(?:st|nd|rd|th)\b)", re.I)
FIELDS = {
    "industrial_kind": "产品类别",
    "industrial_cpu": "可选处理器",
    "industrial_generation": "支持的处理器代际",
    "industrial_mobile": "处理器平台",
    "industrial_chipset": "芯片组",
    "industrial_db9": "DB9 串口数量",
    "industrial_ip": "前面板防护",
    "industrial_fanless": "无风扇设计",
    "industrial_gigabit": "双千兆网口",
    "industrial_tray": "可抽拉 2.5 英寸硬盘",
    "industrial_height": "机架高度",
    "industrial_width": "机架宽度",
    "industrial_alarm": "非法开箱报警",
    "industrial_lock": "可锁前门",
}
NUMERIC_FIELDS = {"industrial_db9", "industrial_height", "industrial_width"}


@dataclass(frozen=True)
class SpecValue:
    attribute: str
    value: str | int | bool
    unit: str | None = None


def _normal(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def category_for_model(model: str | None) -> str | None:
    if model and PANEL_SERIES.fullmatch(model):
        return PANEL
    if model and RACK_SKU.fullmatch(model):
        return RACK
    return None


def _normalize_radicals(value: str) -> str:
    """Convert Kangxi radical code points (U+2F00..U+2FDF, e.g. 英⼨/前⾯板) to standard CJK ideographs."""
    return "".join(
        unicodedata.normalize("NFKC", ch) if "\u2e80" <= ch <= "\u2fdf" else ch
        for ch in value
    )


def _docx_lines(payload: bytes) -> list[tuple[str, str, tuple[str, ...]]]:
    try:
        with ZipFile(BytesIO(payload)) as archive:
            body = fromstring(archive.read(DOCX_MEMBER)).find(WORD + "body")
    except (BadZipFile, ParseError, DefusedXmlException, KeyError):
        raise RagDocumentError("DOCX 正文结构无效。") from None
    if body is None:
        raise RagDocumentError("DOCX 没有正文。")

    def paragraph_text(node) -> str:
        raw = "".join(element.text or "" for element in node.iter(WORD + "t")).strip()
        return _normalize_radicals(raw)

    def cell_text(cell) -> str:
        paragraphs = [value for p in cell.findall(WORD + "p") if (value := paragraph_text(p))]
        if paragraphs:
            return "\n".join(paragraphs)
        return paragraph_text(cell)

    lines = []
    for block in body:
        if block.tag == WORD + "p":
            if value := paragraph_text(block):
                lines.append(("产品说明", value, ()))
        elif block.tag == WORD + "tbl":
            for row in block.findall(WORD + "tr"):
                cells = tuple(value for cell in row.findall(WORD + "tc") if (value := cell_text(cell)))
                if not cells:
                    continue
                if len(cells) == 1 and cells[0].rstrip("：:") in {"特点", "产品规格", "产品尺寸", "产品I/O", "主板接口"}:
                    continue
                first_line = cells[0].splitlines()[0][:100]
                lines.append((f"产品规格表 · {first_line}", " ｜ ".join(cells), cells))
    return lines


def industrial_docx(filename: str, payload: bytes) -> tuple[ParsedSection, ...]:
    validate_archive(payload, DOCX_MEMBER, max_expanded_bytes=LIBRARY_DOCX_MAX_EXPANDED_BYTES)
    rows = _docx_lines(payload)
    identity_cells = {
        value
        for _, _, cells in rows
        if len(cells) > 1 and _normal(cells[0]) in {"model", "型号", "产品型号", "规格"}
        for value in cells[1:]
    }
    rack_ids = {cell.upper() for cell in identity_cells if RACK_SKU.fullmatch(cell)}
    panel_platforms = {match[1].upper() for cell in identity_cells if (match := PANEL_SKU.fullmatch(cell))}
    if len(rack_ids) == 1 and not panel_platforms:
        model = next(iter(rack_ids))
    elif len(panel_platforms) == 1 and not rack_ids:
        platform = next(iter(panel_platforms))
        suffix = " B" if re.match(rf"PLxxxRQ-{re.escape(platform)}\s+B(?=系列|_|\b)", Path(filename).stem, re.I) else ""
        model = f"PLxxxRQ-{platform}{suffix}"
    else:
        return ()
    if len(rows) > 200 or sum(len(text) for _, text, _ in rows) > 120_000:
        raise RagDocumentError("资料段落和正文超出可索引限制。")
    return tuple(ParsedSection(section=section, text=text, model=model) for section, text, _ in rows)


def _generations(text: str) -> set[int]:
    generations = set()
    for match in GENERATION.finditer(text):
        raw = match[1]
        numbers = [int(value) for value in re.findall(r"\d+", raw)]
        if len(numbers) == 2 and re.search(r"[~–—-]", raw) and 0 <= numbers[1] - numbers[0] <= 30:
            generations.update(range(numbers[0], numbers[1] + 1))
        else:
            generations.update(numbers)
    return generations


def read_specs(text: str) -> tuple[SpecValue, ...]:
    text = _normal(text)
    specs = []
    if re.search(
        r"电阻触摸|电阻式触摸|resistive touch|工业平板|触摸平板|触控平板"
        r"|戴手套[^。；\n]{0,20}(?:面板|点控|触控|屏幕|操作)"
        r"|(?:设备面板|面板上)[^。；\n]{0,12}(?:点控|触控|触摸)",
        text,
    ):
        specs.append(SpecValue("industrial_kind", PANEL))
    if re.search(
        r"机架式|上架式|rack[ -]?mount"
        r"|机柜[^。；\n]{0,24}(?:主机|工控|安装|装一台)"
        r"|扩展[^。；\n]{0,16}采集卡",
        text,
    ):
        specs.append(SpecValue("industrial_kind", RACK))
    for cpu in sorted(set(re.findall(r"(?<![a-z0-9])(?:n\d{2,4}|j\d{3,4})(?![a-z0-9])", text))):
        specs.append(SpecValue("industrial_cpu", cpu.upper()))
    if re.search(r"\bcpu\b|\bcore\b|酷睿|处理器|奔腾|赛扬|移动平台", text):
        for generation in sorted(_generations(text)):
            specs.append(SpecValue("industrial_generation", str(generation)))
    if re.search(r"11th\s*-?\s*u\b|(?:11th|第?\s*11\s*代).{0,80}(?:mobile[ -]?u|移动平台)", text):
        specs.append(SpecValue("industrial_mobile", "11th-U"))
    if "芯片组" in text or "chipset" in text:
        for chipset in sorted(set(re.findall(r"(?<![a-z0-9])(?:h|q|b|z)\d{2,3}(?![a-z0-9])", text))):
            specs.append(SpecValue("industrial_chipset", chipset.upper()))
    if match := re.search(r"(?<!\d)(\d{1,2})\s*(?:个|[x*])\s*db9", text):
        specs.append(SpecValue("industrial_db9", int(match[1]), "个"))
    for match in re.finditer(r"前面板([^。；\n]{0,24}?)ip\s*(\d{2})(?!\d)", text):
        if not re.search(r"未达到|不满足|不支持|不到", match[1]):
            specs.append(SpecValue("industrial_ip", "IP" + match[2]))
    if "rack" in text or any(word in text for word in ("机架", "上架", "机箱")):
        if match := re.search(r"(?<!\d)(\d{1,2})\s*u(?=\s*(?:的\s*)?(?:上架|机架|机箱|rack))", text):
            specs.append(SpecValue("industrial_height", int(match[1]), "U"))
        if match := re.search(r"(?<![\d.])(\d{1,2})\s*(?:英寸|inch|\")", text):
            specs.append(SpecValue("industrial_width", int(match[1]), "英寸"))
    flags = {
        "industrial_fanless": r"无风扇|\bfanless\b",
        "industrial_gigabit": r"(?:双|2\s*个)(?:intel®?\s*)?千兆(?:网口|网卡|网接口|网络接口)",
        "industrial_tray": r"2\.5\s*(?:英寸|寸|\")[^。；\n]{0,26}(?:抽拉|抽取)",
        "industrial_alarm": r"非法开箱报警|开箱(?:入侵)?(?:告警|报警)|防(?:止)?误开箱",
        "industrial_lock": r"可锁前门|(?:前门|前面板)可锁",
    }
    for attribute, pattern in flags.items():
        if match := re.search(pattern, text):
            prefix = text[max(0, match.start() - 6):match.start()]
            specs.append(SpecValue(attribute, not re.search(r"(?:不支持|不具备|未提供|不可)\s*$", prefix)))
    return tuple(dict.fromkeys(specs))


def industrial_requirements(text: str, previous_text: str = "") -> RequirementSet | None:
    current = list(read_specs(text))
    previous = list(read_specs(previous_text))
    current_kind = next((item.value for item in current if item.attribute == "industrial_kind"), None)
    previous_kind = next((item.value for item in previous if item.attribute == "industrial_kind"), None)
    kind = current_kind or previous_kind
    if kind is None:
        return None
    if kind == PANEL and re.search(r"粉尘|防尘|灰尘", _normal(text)):
        if not any(item.attribute == "industrial_ip" for item in current):
            current.append(SpecValue("industrial_ip", "IP65"))
    if current_kind and previous_kind and current_kind != previous_kind:
        previous, previous_text = [], ""
    current_fields = {item.attribute for item in current}
    merged = (*current, *(item for item in previous if item.attribute not in current_fields))
    requirements = tuple(
        Requirement(attribute=item.attribute, label=FIELDS[item.attribute], operator="eq",
                    value=item.value, unit=item.unit, sentence=(text if item in current else previous_text))
        for item in merged
    )
    unsupported = next((
        clause.strip()
        for source in (text, previous_text)
        for clause in re.split(r"[，,。；;\n]|并且|以及|同时|且", source)
        if re.search(r"必须|硬性|不得|不能|至少|不超过", clause) and not read_specs(clause)
    ), None)
    if unsupported:
        return RequirementSet(
            category=kind, requirements=requirements, missing=("industrial_specs",),
            questions=(f"目前无法从这批规格表核对「{unsupported[:100]}」，请补充可比较的规格或明确是否为硬性要求。",),
        )
    if len(requirements) == 1:
        return RequirementSet(
            category=kind, requirements=requirements, missing=("industrial_specs",),
            questions=("请补充处理器、芯片组、接口或机箱等具体要求。",),
        )
    return RequirementSet(
        category=kind, requirements=requirements, wants_manuals=True,
        wants_report="报告" in text,
    )
