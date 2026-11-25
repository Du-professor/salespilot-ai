import json
import re
from typing import Literal

from app.contracts.rag_workspace import PreviousTurn

from .chat import re_fence
from .models import RequirementSet
from .specs import MODEL_PATTERN, PROFILES, detect_profile

DialogueIntent = Literal[
    "capability",
    "clarification",
    "domain_out",
    "general_qa",
    "greeting",
    "scenario_probe",
    "spec_query",
    "selection",
]

DIALOGUE_SYSTEM_PROMPT = """你是 SalesPilot AI 的企业产品选型助手，同时提供售前技术问答与办公沟通协助，面向销售、售前工程师和提出采购需求的用户。你可以帮助梳理应用场景、明确硬性要求和偏好、解答行业与通用技术概念、协助撰写售前沟通文案，以及引导用户查找产品手册中的依据。

先回应本轮实际问题：
1. 礼貌寒暄时（如问好、致谢），自然简短回应一两句，不要生硬背诵长篇功能清单。
2. 遇到行业/技术概念对比、原理科普或售前办公协作（如“工控机和家用电脑的区别”、撰写客户跟进邮件等），先直接给出清晰、简洁、有用的回答，末尾可自然附一句如需具体型号选型可告知场景。用通用行业表述，避免给出特定商品的未核验参数或承诺。
3. 询问功能时，简要说明能提供的选型帮助，并给一个贴近用户语境的开始方式。
4. 需要澄清时，利用同一会话中和当前问题相关、已经确认的信息。先用自然第二人称说明你理解的需求（不要用“用户想了解…”等第三人称腔调），再提出一到两个最影响选型的具体问题。
5. 涉及具体型号、参数、供电、接口、装配或兼容性时，等待当前已授权资料的查证结果。不得补写型号、数值、报价、产品能力、出处或附件。请勿把常识、用户猜测和旧回复当作产品证据。
6. 遇到纯娱乐或非办公域外问题（天气、写诗、笑话、股票、菜谱、旅游、星座等），用一两句礼貌说明职责范围，并邀请用户提供应用场景、技术问题或型号。

使用与用户一致的语言，默认简洁中文。按本轮要求的格式输出准备展示给用户的简短答复。不要输出内部判断、工具参数、网址、资料 ID、引用、附件或报告标记。"""

DOMAIN_CUES = (
    "天气", "气温", "股票", "新闻", "笑话", "汇率", "旅游", "电影", "音乐",
    "五言绝句", "七言绝句", "菜谱", "星座", "算命", "讲个故事",
)
DOMAIN_REQUEST = re.compile(
    r"(?:^(?:(?:请|帮我|给我|你能|能不能|可以)\s*)*"
    r"(?:计算(?!机)|翻译|(?:现在|当前|今天)(?:的)?(?:时间|日期|几点|几号)))"
    r"|(?:写一首[^。！？\n]{0,12}诗|(?:中午|晚上|今天)吃什么)"
)
CAPABILITY_CUES = (
    "你能做什么", "能帮我做什么", "可以做什么", "怎么使用", "如何使用",
    "有什么功能", "介绍一下", "你是谁", "帮助我",
)
ASSISTANT_CAPABILITY_QUESTION = re.compile(
    r"(?:请问[\s，,]*)?(?:你(?:们)?|(?:这个|该)?(?:选型助手|助手|智能体|模型|agent))\s*"
    r"(?:(?:都)?(?:能(?:够)?|可以)(?:帮我|为我)?(?:做(?:些)?什么|提供(?:哪些|什么)帮助)"
    r"|(?:有|具备)(?:哪些|什么)(?:功能|能力))",
    re.I,
)
GREETING_PATTERN = re.compile(
    r"^(?:你好|您好|哈喽|嗨|早上好|上午好|中午好|下午好|晚上好|谢谢|多谢|感谢|辛苦了|在吗|hi|hello|hey|thanks|thank you)"
    r"(?:[!！。.~～，,\s]+(?:你好|您好|辛苦了|谢谢|请问|在吗))?"
    r"[!！。.~～，,\s]*$",
    re.I,
)
GENERAL_QA_PATTERN = re.compile(
    r"(?:有什么区别|有何区别|的区别|区别在哪|区别是什么|是什么(?:意思|东西|作用)?"
    r"|工作原理|工作机制|优缺点|有什么优势|为什么需要|注意事项"
    r"|帮我写(?:一封|一份|一段)?[^。！？\n]{0,18}(?:邮件|话术|回复|纪要|总结|方案框架)"
    r"|跟进话术|沟通话术|销售话术)",
    re.I,
)
RECOMMEND_CUES = re.compile(
    r"(?:推荐|选型|买一台|买一款|采购|配一台|配一款|配几台|买几台|哪款|哪种|哪几款"
    r"|有没有合适|找一款|找一台|需要(?:一台|一款|一套|一个|购买|配置)|求购|想买|适合(?:用|做|玩))"
)
REFERENCE_CUES = (
    "刚才那款", "刚才的", "这款", "那款", "该型号", "这个型号", "上一款", "同款",
)
SPEC_CUES = (
    "网口", "接口", "供电", "poe", "分辨率", "像素", "帧率", "帧速", "参数",
    "规格", "兼容", "双网口", "千兆", "负载", "臂展", "工作半径", "功耗",
    "功率", "温度", "modbus", "gige",
)
ATTACHMENT_CUES = ("报告", "附件", "引用", "资料 ID", "资料ID", "第 ", "下载")
MANUAL_OFFER = re.compile(r"(?:附上|附有|已附|附带|发送)[^。！？\n]{0,16}手册")
QUESTION_MARK = re.compile(r"[?？]")
NUMBER_TOKEN = re.compile(
    r"(?<!\d)(\d[\d,]*(?:\.\d+)?)\s*(MP|fps|Mbps|Gbps|W|kg|公斤|千克|mm|毫米|°C|x\d+)",
    re.I,
)


def resolve_target_model(text: str, previous_turns: tuple[PreviousTurn, ...] = ()) -> str | None:
    current = list(dict.fromkeys(MODEL_PATTERN.findall(text)))
    if len(current) == 1:
        return current[0]
    if len(current) > 1:
        return None
    for turn in reversed(previous_turns):
        found = list(dict.fromkeys(MODEL_PATTERN.findall(f"{turn.text} {turn.answer}")))
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            return None
    return None


def _out_of_domain(text: str) -> bool:
    return any(cue in text for cue in DOMAIN_CUES) or DOMAIN_REQUEST.search(text.strip()) is not None


def is_assistant_capability_question(text: str) -> bool:
    return ASSISTANT_CAPABILITY_QUESTION.fullmatch(text.strip().rstrip("。！？!?. ")) is not None


def _is_greeting(text: str) -> bool:
    return GREETING_PATTERN.fullmatch(text.strip()) is not None


def _is_general_qa(text: str) -> bool:
    if MODEL_PATTERN.search(text) is not None:
        return False
    if RECOMMEND_CUES.search(text) is not None:
        return False
    return GENERAL_QA_PATTERN.search(text) is not None


def classify_intent(
    text: str,
    requirements: RequirementSet,
    previous_turns: tuple[PreviousTurn, ...] = (),
) -> DialogueIntent:
    lowered = text.casefold()
    asks_spec = any(cue in lowered for cue in SPEC_CUES)
    names_reference = any(cue in lowered for cue in REFERENCE_CUES)
    current_product = detect_profile(text) != "unknown" or MODEL_PATTERN.search(text) is not None
    # Prior turns cannot turn a new off-topic request into an inherited selection.
    if not current_product and not (asks_spec and names_reference) and _out_of_domain(lowered):
        return "domain_out"
    if is_assistant_capability_question(text) or any(cue in lowered for cue in CAPABILITY_CUES):
        return "capability"
    if _is_greeting(text):
        return "greeting"
    target = resolve_target_model(text, previous_turns)
    if names_reference and asks_spec:
        return "spec_query"
    if _is_general_qa(text):
        return "general_qa"
    if requirements.requirements and not requirements.missing:
        return "selection"
    if target is not None and asks_spec:
        return "spec_query"
    if requirements.category != "unknown" and requirements.missing:
        return "clarification"
    if requirements.requirements:
        return "selection"
    if RECOMMEND_CUES.search(text) is not None or MODEL_PATTERN.search(text) is not None:
        return "scenario_probe"
    return "capability"


def dialogue_user_prompt(
    intent: DialogueIntent,
    text: str,
    requirements: RequirementSet,
    previous_turns: tuple[PreviousTurn, ...] = (),
) -> str:
    lines = [f"当前消息：{text}"]
    if intent == "clarification":
        profile = PROFILES.get(requirements.category)
        if profile is not None:
            lines.append(f"已识别品类：{profile.label}")
        if requirements.requirements:
            conditions = []
            for requirement in requirements.requirements:
                value = f"{requirement.value}{requirement.unit or ''}"
                conditions.append(
                    f"{requirement.label} {value}（{'硬性' if requirement.hard else '偏好'}）"
                )
            lines.append("已识别条件：" + "；".join(conditions))
        if requirements.questions:
            lines.append("候选缺项（按场景选一到两项）：" + "；".join(requirements.questions))
        if previous_turns and requirements.category != "unknown":
            for turn in reversed(previous_turns):
                if detect_profile(turn.text) == requirements.category:
                    lines.append(f"同会话相关上一轮：{turn.text}")
                    break
        lines.extend((
            '只输出 JSON 对象，格式为 {"summary":"一句自然第二人称的需求理解。","questions":["一个完整问句？"]}。',
            "保留用户已给出的条件和数量。问题按重要性排序，共一到两项，每项只问一个条件并以问号结尾。"
            "不要重复询问已知条件。工件重量已知时，可以追问夹具重量或工作半径。不要在 JSON 外添加内容。",
        ))
    elif intent == "greeting":
        lines.append("请自然、礼貌地简短回应问候或致谢（1-2 句），并温和询问有什么产品选型或技术问题需要协助。")
    elif intent == "general_qa":
        lines.append(
            "这是一条通用技术概念或办公协作问题。请直接、清晰、简洁地回答用户的问题（3-5 点以内），"
            "不要生硬反问或使用第三人称；结尾可用一句话自然提示如需针对具体场景选型可随时告知。"
        )
    elif intent == "capability":
        lines.append("请简要说明你能提供的选型与技术问答帮助，并告诉用户可以从描述客户的使用场景开始。")
    elif intent == "domain_out":
        lines.append("请仅围绕产品选型与售前办公范围用一两句简短礼貌回应，并邀请用户提供应用场景、技术问题或型号。")
    else:
        lines.append("请用简洁中文回应本轮问题。")
    lines.append("不要给出具体商品型号事实、报价、引用、附件或报告。")
    return "\n".join(lines)


def scenario_probe_prompt(
    text: str,
    catalog_titles: tuple[str, ...],
    excerpts: tuple[str, ...],
) -> str:
    lines = [
        f"当前客户需求：{text}",
        "当前知识库已发布资料标题：" + ("；".join(catalog_titles[:12]) if catalog_titles else "（空）"),
    ]
    if excerpts:
        lines.append("知识库召回的最相关片段摘要：")
        for idx, excerpt in enumerate(excerpts[:3], start=1):
            compact = " ".join(excerpt.split())[:220]
            lines.append(f"{idx}. {compact}")
    lines.extend((
        '请判断当前知识库是否包含与客户所求品类相关的产品资料，并只返回 JSON 对象：{"in_catalog": true或false, "answer": "给用户的中文回复"}。',
        "判断规则：",
        "1. 若客户明确求购的产品品类与上述资料完全无关（例如资料库只有工控机/工业平板，而客户要游戏笔记本、激光雷达、消费手机等），"
        "必须令 in_catalog=false，并在 answer 中直接、礼貌地说明当前知识库暂未收录该类产品资料，简述库内现有产品方向（不编造具体型号参数或价格），不要继续追问库外产品的参数。",
        "2. 若客户描述的是自动化/工业监控/产线控制等模糊应用场景，且库内工业产品可能适用，则令 in_catalog=true，"
        "在 answer 中简要说明可结合现场形态匹配的方向，并提出 1-2 个最关键的澄清问题以缩小选型范围。",
        "回复中请使用产品品类名称（如工业工控机、工业触摸平板电脑），不要直接输出任何具体型号编号、数值、报价、附件或引用编号，不要在 JSON 外输出其他文字。",
    ))
    return "\n".join(lines)


def _unsafe(text: str, user_text: str, intent: DialogueIntent) -> bool:
    lowered = text.casefold()
    if any(cue.casefold() in lowered for cue in ATTACHMENT_CUES) or MANUAL_OFFER.search(text):
        return True
    if re.search(r"https?://|www\.", text, re.I):
        return True
    user_lower = user_text.casefold()
    for model in MODEL_PATTERN.findall(text):
        if model.casefold() not in user_lower:
            return True
    if intent != "general_qa":
        for match in NUMBER_TOKEN.finditer(text):
            raw = match.group(1).replace(",", "")
            if raw in user_text:
                continue
            try:
                if str(int(float(raw))) in user_text:
                    continue
            except ValueError:
                return True
            return True
    if intent == "domain_out" and _out_of_domain(lowered):
        return True
    return False


def _fallback(intent: DialogueIntent, requirements: RequirementSet) -> str:
    if intent == "clarification" and requirements.questions:
        return "\n".join(question.rstrip("。！？!? ") + "？" for question in requirements.questions[:2])
    if intent == "greeting":
        return "您好！我是 SalesPilot AI 选型与售前助手，请问今天有什么应用场景、技术问题或产品选型需要协助？"
    if intent == "general_qa":
        return "我可以帮您解答通用工业计算与产品选型相关的概念问题；如需结合具体现场场景推荐产品型号，欢迎补充您的需求。"
    if intent == "domain_out":
        return "我主要负责产品选型与售前技术支持，可以帮你梳理应用场景、解答技术概念或查证手册依据。请提供具体产品或需求。"
    if intent == "capability":
        return "我是产品选型助手。我可以帮你梳理客户需求、解答常见技术问题，并根据资料库中的手册核对规格。你可以先描述使用场景。"
    return "我是产品选型助手，可以帮你梳理应用场景、硬性条件和资料依据。请告诉我产品类别和关键要求。"


def _render_clarification(answer: str) -> str:
    try:
        payload = json.loads(re_fence(answer) if answer.startswith("```") else answer)
    except ValueError:
        return ""
    if not isinstance(payload, dict):
        return ""
    summary, questions = payload.get("summary"), payload.get("questions")
    if not isinstance(summary, str) or not summary.strip() or QUESTION_MARK.search(summary):
        return ""
    if not isinstance(questions, list):
        return ""
    selected = [
        item.strip() for item in questions[:2]
        if isinstance(item, str) and item.strip().endswith(("？", "?"))
        and len(QUESTION_MARK.findall(item)) == 1
    ]
    return "\n".join((summary.strip(), *selected)) if selected else ""


def render_scenario_probe(
    model_text: str,
    user_text: str,
    *,
    fallback_in_catalog: bool,
) -> tuple[Literal["needs_input", "insufficient_evidence"], str]:
    raw = (model_text or "").strip()
    in_catalog: bool | None = None
    answer = ""
    if raw:
        try:
            payload = json.loads(re_fence(raw) if raw.startswith("```") else raw)
            if isinstance(payload, dict):
                if isinstance(payload.get("in_catalog"), bool):
                    in_catalog = payload["in_catalog"]
                if isinstance(payload.get("answer"), str):
                    answer = payload["answer"].strip()
        except ValueError:
            pass
    effective_in_catalog = fallback_in_catalog if in_catalog is None else in_catalog
    if not answer or _unsafe(answer, user_text, "scenario_probe"):
        if effective_in_catalog:
            answer = "请补充您的具体使用场景（如安装形态、接口扩展或环境要求），我将结合当前资料库中的产品手册为您匹配合适方案。"
        else:
            answer = "当前知识库中暂未收录与该需求匹配的产品资料，无法为您推荐对应型号。如需库内工业计算产品的选型支持，欢迎告知具体场景。"
    status: Literal["needs_input", "insufficient_evidence"] = (
        "needs_input" if effective_in_catalog else "insufficient_evidence"
    )
    return status, answer


def safe_dialogue_answer(
    intent: DialogueIntent,
    text: str,
    user_text: str,
    requirements: RequirementSet,
) -> str:
    answer = (text or "").strip()
    if intent == "clarification":
        answer = _render_clarification(answer)
    if not answer or _unsafe(answer, user_text, intent):
        return _fallback(intent, requirements)
    return answer
