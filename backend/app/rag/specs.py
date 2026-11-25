import re
from typing import Literal

from .models import RagModel

MODEL_PATTERN = re.compile(
    r"\b(?:[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+|[A-Z]{2,}\s+[A-Z]\d{3,}\s+\d+GB)\b"
)
NUMBER = r"(-?\d[\d,]*(?:\.\d+)?)"
NUMERIC_UNITS = ("MP", "fps", "Mbps", "Gbps", "W", "kg", "公斤", "千克", "mm", "毫米", "°C")


class SpecDefinition(RagModel):
    key: str
    label: str
    aliases: tuple[str, ...]
    kind: Literal["number", "range", "boolean", "enum"]
    unit: str | None = None
    question: str | None = None


class CategoryProfile(RagModel):
    key: str
    label: str
    aliases: tuple[str, ...]
    required: tuple[str, ...] = ()


class NumericMention(RagModel):
    key: str
    label: str
    value: float
    upper: float | None = None
    unit: str | None = None
    start: int
    end: int


class CapabilityMention(RagModel):
    key: str
    label: str
    value: bool | str = True
    polarity: Literal["affirmed", "negated"] = "affirmed"
    start: int
    end: int


SPECS: dict[str, SpecDefinition] = {
    "interface": SpecDefinition(
        key="interface", label="数据接口", aliases=("GigE", "数据接口", "以太网接口", "网口"),
        kind="enum", question="请确认相机的数据接口（例如 GigE）。",
    ),
    "resolution_mp": SpecDefinition(
        key="resolution_mp", label="分辨率", aliases=("分辨率", "像素", "resolution"), kind="number", unit="MP",
        question="请确认所需的最低分辨率（MP）。",
    ),
    "fps": SpecDefinition(
        key="fps", label="帧率", aliases=("帧率", "帧速", "frame rate", "fps"), kind="number", unit="fps",
        question="请确认标称分辨率下所需的最低帧率（fps）。",
    ),
    "poe": SpecDefinition(
        key="poe", label="PoE 供电", aliases=("PoE", "PoE 供电", "以太网供电"), kind="boolean",
    ),
    "gige_ports": SpecDefinition(
        key="gige_ports", label="GigE 直连接口", aliases=("相机端口", "相机接口", "直连接口", "相机直连接口"),
        kind="number", unit="个", question="请确认需要的外部供电 GigE 相机直连接口数量。",
    ),
    "port_mbps": SpecDefinition(
        key="port_mbps", label="单口带宽", aliases=("单口", "每路", "单口带宽", "每路带宽"),
        kind="number", unit="Mbps", question="请确认每路相机接口所需的最低带宽（Mbps）。",
    ),
    "total_mbps": SpecDefinition(
        key="total_mbps", label="总入口带宽", aliases=("总入口带宽", "总带宽", "可用总入口带宽"),
        kind="number", unit="Mbps",
    ),
    "modbus_tcp": SpecDefinition(
        key="modbus_tcp", label="独立 Modbus TCP", aliases=("Modbus TCP", "Modbus"),
        kind="boolean", question="请确认是否需要独立的 Modbus TCP 连接。",
    ),
    "gpu_model": SpecDefinition(
        key="gpu_model", label="显卡兼容", aliases=("显卡", "GPU", "支持安装"), kind="enum",
        question="请确认需要安装的显卡型号。",
    ),
    "gpu_electrical_lanes": SpecDefinition(
        key="gpu_electrical_lanes", label="GPU 电气通道", aliases=("GPU 电气", "电气通道", "GPU 插槽"),
        kind="number", unit="x", question="请确认 GPU 插槽需要的最低电气通道数（例如 x16）。",
    ),
    "gpu_power_w": SpecDefinition(
        key="gpu_power_w", label="GPU 供电额度", aliases=("供电额度", "GPU 供电", "显卡供电"),
        kind="number", unit="W",
    ),
    "gpu_power_headroom_w": SpecDefinition(
        key="gpu_power_headroom_w", label="GPU 供电余量", aliases=("供电余量",),
        kind="number", unit="W", question="请确认 GPU 供电余量要求（W）。",
    ),
    "free_electrical_lanes": SpecDefinition(
        key="free_electrical_lanes", label="预留 PCIe 电气通道", aliases=("预留电气", "剩余 PCIe", "电气 x"),
        kind="number", unit="x", question="请确认需要预留的 PCIe 插槽电气通道数。",
    ),
    "operating_temp_c": SpecDefinition(
        key="operating_temp_c", label="带卡工作温度范围", aliases=("工作温度", "带卡温度", "带卡工作温度"),
        kind="range", unit="°C", question="请确认带卡工作温度范围。",
    ),
    "payload_kg": SpecDefinition(
        key="payload_kg", label="额定负载", aliases=("额定负载", "有效负载", "负载"),
        kind="number", unit="kg", question="请确认机器人需要的最低额定负载（kg）。",
    ),
    "reach_mm": SpecDefinition(
        key="reach_mm", label="最大臂展", aliases=("最大臂展", "工作半径", "臂展", "工作范围"),
        kind="number", unit="mm", question="请确认机器人需要的最小工作半径或臂展（mm）。",
    ),
    "collaborative": SpecDefinition(
        key="collaborative", label="协作模式", aliases=("协作模式", "协作型"), kind="boolean",
    ),
    "ingress_protection": SpecDefinition(
        key="ingress_protection", label="防护等级", aliases=("防护等级", "IP 等级"), kind="enum",
    ),
    "gpu_max_power_w": SpecDefinition(
        key="gpu_max_power_w", label="最大功耗", aliases=("最大功耗", "额定功耗"), kind="number", unit="W",
    ),
}

PROFILES: dict[str, CategoryProfile] = {
    "vision_camera": CategoryProfile(
        key="vision_camera", label="机器视觉相机", aliases=("相机", "视觉", "机器视觉", "GigE 相机"),
        required=("interface", "resolution_mp", "fps"),
    ),
    "industrial_pc": CategoryProfile(
        key="industrial_pc",
        label="工业计算机",
        aliases=("工控机", "工业计算机", "工业电脑", "平板电脑", "工控一体机", "工控主机", "机柜", "采集卡", "IPC", "PLC", "工控"),
        required=(
            "gige_ports", "port_mbps", "modbus_tcp", "gpu_model", "gpu_electrical_lanes",
            "gpu_power_headroom_w", "free_electrical_lanes", "operating_temp_c",
        ),
    ),
    "robot_arm": CategoryProfile(
        key="robot_arm", label="工业机器人", aliases=("机器人", "机械臂", "协作机器人", "工业机器人"),
        required=("payload_kg", "reach_mm"),
    ),
}

PROFILE_ORDER = ("robot_arm", "industrial_pc", "vision_camera")
PREFERENCE_CUES = ("优先", "最好", "希望", "尽量", "倾向", "可选", "可接受", "不强制")


def normalize_unit(unit: str | None) -> str | None:
    if unit is None:
        return None
    lowered = unit.casefold()
    mapping = {
        "mp": "MP", "fps": "fps", "mbps": "Mbps", "gbps": "Gbps", "w": "W",
        "kg": "kg", "公斤": "kg", "千克": "kg", "mm": "mm", "毫米": "mm",
        "°c": "°C", "个": "个", "x": "x",
    }
    return mapping.get(lowered, unit)


def normalize_phrase(text: str) -> str:
    return re.sub(r"\s+", "", text).strip("，。；：:、的为是有").casefold()


def detect_profile(text: str) -> str:
    lowered = text.casefold()
    for key in PROFILE_ORDER:
        if any(alias.casefold() in lowered for alias in PROFILES[key].aliases):
            return key
    return "unknown"


def spec_label(key: str) -> str:
    spec = SPECS.get(key)
    if spec is not None:
        return spec.label
    if key.startswith("generic:") or key.startswith("capability:"):
        return key.split(":", 1)[1]
    return key


def question_for(key: str) -> str:
    if key == "category":
        return "请说明需要的产品类别和硬性条件。"
    spec = SPECS.get(key)
    if spec is not None and spec.question:
        return spec.question
    return f"请确认{spec_label(key)}的硬性要求。"


def canonical_spec(label: str, unit: str | None = None) -> str | None:
    text = normalize_phrase(label)
    normalized = normalize_unit(unit)
    if normalized is not None:
        candidates: list[tuple[int, str]] = []
        for key, spec in SPECS.items():
            if spec.unit is None or normalize_unit(spec.unit) != normalized:
                continue
            for alias in spec.aliases:
                token = normalize_phrase(alias)
                position = text.rfind(token)
                if position >= 0:
                    candidates.append((position + len(token), key))
        if candidates:
            return max(candidates)[1]
        fallback = {
            "MP": "resolution_mp",
            "fps": "fps",
            "Mbps": "port_mbps",
            "kg": "payload_kg",
            "mm": "reach_mm",
        }
        return fallback.get(normalized)

    if "poe" in text:
        return "poe"
    if "modbus" in text:
        return "modbus_tcp"
    if "协作" in text:
        return "collaborative"
    if re.search(r"\bip\s*\d{2}\b", text, re.I) or "防护等级" in text:
        return "ingress_protection"
    if "供电余量" in text:
        return "gpu_power_headroom_w"
    if "供电额度" in text or "gpu供电" in text or "显卡供电" in text:
        return "gpu_power_w"
    if "最大功耗" in text or "额定功耗" in text:
        return "gpu_max_power_w"
    if "剩余" in text and "pcie" in text and "电气" in text:
        return "free_electrical_lanes"
    if "电气通道" in text or ("gpu" in text and "电气" in text):
        return "gpu_electrical_lanes"
    if "相机端口" in text or "直连接口" in text or "相机接口" in text:
        return "gige_ports"
    if "单口" in text or "每路" in text:
        return "port_mbps"
    if "分辨率" in text or "像素" in text:
        return "resolution_mp"
    if "帧率" in text or "帧速" in text:
        return "fps"
    if "额定负载" in text or "有效负载" in text or "负载" in text:
        return "payload_kg"
    if "最大臂展" in text or "工作半径" in text or "臂展" in text or "工作范围" in text:
        return "reach_mm"
    if "工作温度" in text or "带卡温度" in text:
        return "operating_temp_c"
    if "接口" in text or "gige" in text:
        return "interface"
    if "显卡" in text or "gpu" in text or "支持安装" in text:
        return "gpu_model"
    return None


def _number(raw: str) -> float:
    return float(raw.replace(",", ""))


def _mention(
    mentions: list[NumericMention], seen: set[tuple[str, int, int]],
    key: str, value: float, unit: str | None, start: int, end: int, upper: float | None = None,
) -> None:
    marker = (key, start, end)
    if marker in seen:
        return
    seen.add(marker)
    mentions.append(NumericMention(
        key=key, label=spec_label(key), value=value, upper=upper, unit=normalize_unit(unit),
        start=start, end=end,
    ))


def find_numeric_mentions(text: str) -> list[NumericMention]:
    mentions: list[NumericMention] = []
    seen: set[tuple[str, int, int]] = set()
    range_spans: list[tuple[int, int]] = []

    for match in re.finditer(
        r"最低\s*" + NUMBER + r"\s*°C[，,]\s*最高\s*" + NUMBER + r"\s*°C", text, re.I,
    ):
        low, high = sorted((_number(match.group(1)), _number(match.group(2))))
        _mention(mentions, seen, "operating_temp_c", low, "°C", match.start(), match.end(), high)
        range_spans.append((match.start(), match.end()))

    for match in re.finditer(NUMBER + r"\s*(?:至|到|~|-)\s*" + NUMBER + r"\s*°C", text, re.I):
        if any(start <= match.start() < end for start, end in range_spans):
            continue
        low, high = sorted((_number(match.group(1)), _number(match.group(2))))
        _mention(mentions, seen, "operating_temp_c", low, "°C", match.start(), match.end(), high)
        range_spans.append((match.start(), match.end()))

    for match in re.finditer(r"电气通道为\s*x" + NUMBER, text, re.I):
        _mention(mentions, seen, "gpu_electrical_lanes", _number(match.group(1)), "x", match.start(), match.end())
    for match in re.finditer(r"GPU 插槽[^。；\n]{0,50}?电气[^。；\n]{0,10}?x" + NUMBER, text, re.I):
        _mention(mentions, seen, "gpu_electrical_lanes", _number(match.group(1)), "x", match.start(), match.end())
    for match in re.finditer(r"GPU 电气[^。；\n]{0,10}?x" + NUMBER, text, re.I):
        _mention(mentions, seen, "gpu_electrical_lanes", _number(match.group(1)), "x", match.start(), match.end())

    for match in re.finditer(
        r"(?:预留|剩余)[^。；\n]{0,24}?电气[^。；\n]{0,10}?x" + NUMBER, text, re.I,
    ):
        _mention(mentions, seen, "free_electrical_lanes", _number(match.group(1)), "x", match.start(), match.end())

    for match in re.finditer(
        r"(?:提供\s*)?" + NUMBER + r"\s*个[^。；\n]{0,24}?(?:相机端口|GigE[^。；\n]{0,12}?(?:直连|接口))",
        text, re.I,
    ):
        _mention(mentions, seen, "gige_ports", _number(match.group(1)), "个", match.start(), match.end())

    for match in re.finditer(NUMBER + r"\s*(MP|fps|Mbps|Gbps|W|kg|公斤|千克|mm|毫米)\b", text, re.I):
        if any(start <= match.start() < end for start, end in range_spans):
            continue
        unit = normalize_unit(match.group(2))
        window = text[max(0, match.start() - 36):match.start()]
        key = canonical_spec(window, unit)
        if key is None:
            continue
        _mention(mentions, seen, key, _number(match.group(1)), unit, match.start(), match.end())

    mentions.sort(key=lambda item: (item.start, item.end, item.key))
    return mentions


def _canonical_capability(label: str, sentence: str) -> tuple[str, bool | str] | None:
    compact = normalize_phrase(label)
    if not compact:
        return None
    if "poe" in compact:
        return "poe", True
    if "modbus" in compact:
        return "modbus_tcp", True
    if "协作" in compact:
        return "collaborative", True
    if "gige" in compact:
        return "interface", "GigE"
    protection = re.search(r"IP\s*\d{2}", sentence, re.I)
    if "防护等级" in compact or protection is not None:
        return "ingress_protection", (protection.group(0).upper().replace(" ", "") if protection else label)
    model = MODEL_PATTERN.search(label)
    if model is not None and ("安装" in compact or "显卡" in compact or "gpu" in sentence.casefold()):
        return "gpu_model", model.group(0)
    if re.match(r"^\d", compact) or "相机端口" in compact or "直连" in compact:
        return None
    return f"capability:{compact}", True


def find_capability_mentions(text: str) -> list[CapabilityMention]:
    mentions: list[CapabilityMention] = []
    cursor = 0
    for sentence in re.split(r"(?<=[。；!?])", text):
        sentence = sentence.strip()
        if not sentence:
            continue
        start = text.find(sentence, cursor)
        if start < 0:
            continue
        cursor = start + len(sentence)
        seen: set[str] = set()

        def add(key: str, value: bool | str, polarity: str, label: str, match: re.Match[str]) -> None:
            if key in seen:
                return
            seen.add(key)
            mentions.append(CapabilityMention(
                key=key, label=label, value=value, polarity=polarity,
                start=start + match.start(), end=start + match.end(),
            ))

        for match in re.finditer(
            r"(不支持|不具备|不提供|不含|未提供|未内置|无|非)\s*([^，。；\n]{0,30})", sentence,
        ):
            canonical = _canonical_capability(match.group(2), sentence)
            if canonical is not None:
                add(canonical[0], False if canonical[0] != "interface" else "未知", "negated", match.group(2), match)

        for match in re.finditer(
            r"(支持|具备|提供|内置|装有|包含|带有)\s*([^，。；\n]{0,30})", sentence,
        ):
            canonical = _canonical_capability(match.group(2), sentence)
            if canonical is not None:
                add(canonical[0], canonical[1], "affirmed", match.group(2), match)

        if "Modbus TCP" in sentence and ("网口" in sentence or "接口" in sentence or "可用于" in sentence):
            match = re.search(r"Modbus\s*TCP", sentence, re.I)
            if match is not None:
                add("modbus_tcp", True, "affirmed", "Modbus TCP", match)
        if re.search(r"GigE", sentence, re.I):
            match = re.search(r"GigE", sentence, re.I)
            if match is not None:
                add("interface", "GigE", "affirmed", "GigE", match)
        protection = re.search(r"IP\s*\d{2}", sentence, re.I)
        if protection is not None and "防护等级" in sentence:
            add(
                "ingress_protection", protection.group(0).upper().replace(" ", ""),
                "affirmed", "防护等级", protection,
            )
    return mentions
