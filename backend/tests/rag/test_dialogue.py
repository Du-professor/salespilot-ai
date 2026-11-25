import pytest

from app.contracts.rag_workspace import PreviousTurn, RagUnavailable
from app.rag import HashingEmbeddings, SqliteRagEngine
from app.rag.dialogue import DIALOGUE_SYSTEM_PROMPT, classify_intent
from app.rag.models import ModelClaim, ModelDraftPayload
from app.rag.requirements import extract_requirements
from app.rag.store import RagChunkStore

from conftest import FIXTURES, ask, index_file


class CountingStore(RagChunkStore):
    def __init__(self, path):
        super().__init__(path)
        self.loads = 0

    def load_chunks(self, org_id, keys, **kwargs):
        self.loads += 1
        return super().load_chunks(org_id, keys, **kwargs)

    def load_text_chunks(self, org_id, keys, **kwargs):
        self.loads += 1
        return super().load_text_chunks(org_id, keys, **kwargs)


class DialogueChat:
    def __init__(self, reply=None):
        self.reply = reply
        self.compose_calls = []
        self.draft_calls = []

    def compose(self, *, system, user):
        self.compose_calls.append((system, user))
        if isinstance(self.reply, Exception):
            raise self.reply
        if callable(self.reply):
            return self.reply(user)
        return self.reply or "我是产品选型助手，请告诉我产品类别和关键要求。"

    def compose_draft(self, *, system, user, evidence):
        self.draft_calls.append((system, user, evidence))
        claims = tuple(
            ModelClaim(**{
                key: item.get(key)
                for key in (
                    "chunk_id", "model", "assembly", "document_id", "version", "page",
                    "section", "attribute", "value", "upper", "unit", "polarity", "excerpt",
                )
            })
            for item in evidence
        )
        return ModelDraftPayload(claims=claims)


def _engine(settings, chat, store=None):
    return SqliteRagEngine(
        settings,
        store=store,
        embedder=HashingEmbeddings(settings.embedding_dimensions),
        chat=chat,
    )


def test_capability_consultation_uses_dialogue_prompt_without_retrieval(settings):
    store = CountingStore(settings.index_path)
    chat = DialogueChat("我可以帮你梳理应用场景，并查证手册中的规格依据。")
    result = ask(_engine(settings, chat, store), "你能做什么？", ())

    assert result.status == "needs_input"
    assert result.models == () and result.citations == () and result.manuals == ()
    assert result.include_report is False
    assert "查证手册中的规格依据" in result.answer
    assert store.loads == 0
    assert len(chat.compose_calls) == 1
    assert chat.compose_calls[0][0].startswith("你是 SalesPilot AI 的企业产品选型助手")
    assert not chat.draft_calls


@pytest.mark.parametrize(("question", "use_model"), (
    ("你能够做什么", True),
    ("请问，你具备哪些能力？", True),
    ("你能够做什么", False),
))
def test_capability_question_starts_a_new_intent_after_a_recommendation(settings, cases, question, use_model):
    store = CountingStore(settings.index_path)
    chat = DialogueChat("我可以帮你梳理应用场景，并查证手册中的规格依据。") if use_model else None
    engine = _engine(settings, chat, store)
    vision = index_file(engine, FIXTURES / "specs" / "sp-vision-cameras.md", "vision")
    first_text = cases["vision"]["text"]
    first = ask(engine, first_text, (vision,))
    assert first.status == "recommended" and first.models == ("SP-VC-120",)
    loads_before = store.loads
    previous = (PreviousTurn(text=first_text, answer=first.answer),)

    result = ask(engine, question, (vision,), previous_turns=previous)

    assert result.status == "needs_input"
    assert "选型" in result.answer or "应用场景" in result.answer
    assert "SP-VC-120" not in result.answer
    assert result.models == result.citations == result.manuals == ()
    assert result.include_report is False
    assert store.loads == loads_before
    if chat is not None:
        assert len(chat.draft_calls) == 1
        assert len(chat.compose_calls) == 1
        assert first_text not in chat.compose_calls[0][1]


def test_camera_and_robot_clarification_use_current_missing_conditions(settings):
    def reply(user):
        if "机器人" in user:
            return "你已要求搬运 8 公斤工件，请确认工作半径和搬运节拍。"
        return "请确认相机的数据接口、分辨率和帧率。"

    chat = DialogueChat(reply)
    engine = _engine(settings, chat)
    camera = ask(engine, "给高速产线配相机", ())
    robot = ask(engine, "想采购机器人搬运 8 公斤工件", ())

    assert camera.status == robot.status == "needs_input"
    assert len(chat.compose_calls) == 2
    assert "分辨率" in chat.compose_calls[0][1] and "帧率" in chat.compose_calls[0][1]
    assert "工作半径" in chat.compose_calls[1][1] or "臂展" in chat.compose_calls[1][1]
    assert "8" in chat.compose_calls[1][1]
    assert camera.citations == () and robot.citations == ()


def test_clarification_displays_at_most_two_model_questions(settings):
    chat = DialogueChat(
        '{"summary":"你要为高速产线选相机。","questions":'
        '["最低帧率需要多少？","需要什么分辨率？","要用哪种接口？"]}'
    )
    result = ask(_engine(settings, chat), "给高速产线配相机", ())

    assert result.answer == "你要为高速产线选相机。\n最低帧率需要多少？\n需要什么分辨率？"


@pytest.mark.parametrize("text", (
    "需要工业计算机，19 英寸 4U，Q270 芯片组。",
    "需要在高温天气下运行的工业计算机，19 英寸 4U，Q270 芯片组。",
))
def test_domain_words_do_not_override_a_current_product_request(text):
    requirements = extract_requirements(text)
    assert classify_intent(text, requirements) in {"selection", "clarification"}


def test_spec_query_uses_previous_model_without_dialogue_model(settings, cases):
    store = CountingStore(settings.index_path)
    chat = DialogueChat()
    engine = _engine(settings, chat, store)
    vision = index_file(engine, FIXTURES / "specs" / "sp-vision-cameras.md", "vision")
    first_text = cases["vision"]["text"]
    first = ask(engine, first_text, (vision,))
    assert first.status == "recommended" and first.models == ("SP-VC-120",)
    assert len(chat.draft_calls) == 1
    assert chat.compose_calls == []

    previous = PreviousTurn(text=first_text, answer=first.answer)
    loads_before = store.loads
    second = ask(engine, "刚才那款是什么网口？", (vision,), previous_turns=(previous,))
    assert second.status == "recommended"
    assert second.models == ("SP-VC-120",)
    assert second.manuals == () and second.include_report is False
    assert any("SP-VC-120 标准装配" in (citation.section or "") for citation in second.citations)
    assert "GigE" in second.answer
    assert store.loads > loads_before
    assert chat.compose_calls == []


def test_domain_out_does_not_retrieve_or_recommend(settings, cases):
    store = CountingStore(settings.index_path)
    chat = DialogueChat(lambda user: "今天天气晴朗，气温 25 度。" if "天气" in user else "请说明型号。")
    engine = _engine(settings, chat, store)
    vision = index_file(engine, FIXTURES / "specs" / "sp-vision-cameras.md", "vision")
    first = ask(engine, cases["vision"]["text"], (vision,))
    previous = (
        PreviousTurn(text=cases["vision"]["text"], answer=first.answer),
    )
    loads_before = store.loads
    result = ask(engine, "今天天气怎么样？", (vision,), previous_turns=previous)

    assert result.status == "needs_input"
    assert result.models == () and result.citations == () and result.manuals == ()
    assert store.loads == loads_before
    assert "天气" not in result.answer and "气温" not in result.answer
    assert "SP-VC-120" not in result.answer
    assert len(chat.compose_calls) == 1


def test_partial_spec_query_recommends_with_gap(settings, cases):
    chat = DialogueChat()
    engine = _engine(settings, chat)
    vision = index_file(engine, FIXTURES / "specs" / "sp-vision-cameras.md", "vision")
    first = ask(engine, cases["vision"]["text"], (vision,))
    previous = PreviousTurn(text=cases["vision"]["text"], answer=first.answer)
    result = ask(engine, "刚才那款是不是千兆双网口？", (vision,), previous_turns=(previous,))

    assert result.status == "recommended"
    assert result.models == ("SP-VC-120",)
    assert result.manuals == () and result.include_report is False
    assert "GigE" in result.answer
    assert "未在资料中记录" in result.answer
    assert chat.compose_calls == []


def test_spec_query_after_withdrawal_reports_gap(settings, cases):
    chat = DialogueChat()
    engine = _engine(settings, chat)
    vision = index_file(engine, FIXTURES / "specs" / "sp-vision-cameras.md", "vision")
    first = ask(engine, cases["vision"]["text"], (vision,))
    previous = PreviousTurn(text=cases["vision"]["text"], answer=first.answer)
    result = ask(engine, "刚才那款是什么网口？", (), previous_turns=(previous,))

    assert result.status == "insufficient_evidence"
    assert result.models == () and result.citations == ()
    assert "当前授权资料" in result.answer and "SP-VC-120" in result.answer
    assert chat.compose_calls == []


def test_unsafe_dialogue_output_is_replaced(settings):
    chat = DialogueChat("推荐虚构型号，9999 fps，附报告。")
    result = ask(_engine(settings, chat), "你能做什么？", ())
    assert result.status == "needs_input"
    assert "9999" not in result.answer and "虚构" not in result.answer and "报告" not in result.answer
    assert result.citations == () and result.manuals == ()
    assert len(chat.compose_calls) == 1


def test_dialogue_model_timeout_is_unavailable(settings):
    chat = DialogueChat(RagUnavailable("模型超时。"))
    with pytest.raises(RagUnavailable):
        ask(_engine(settings, chat), "你能做什么？", ())


def test_office_friendly_greeting_and_general_qa_work_naturally(settings):
    store = CountingStore(settings.index_path)

    def reply(user):
        if "工控机和普通家用电脑有什么区别" in user:
            return "工控机与普通家用电脑主要区别在于：1）支持 7x24 小时稳定运行与宽温防尘；2）提供丰富串口与扩展槽位；3）元器件供货周期更长。如需具体场景选型欢迎告知。"
        return "您好！请问今天有什么技术问题或选型需求需要帮忙？"

    chat = DialogueChat(reply)
    engine = _engine(settings, chat, store)

    greet = ask(engine, "你好，辛苦了！", ())
    assert greet.status == "needs_input" and "您好" in greet.answer
    assert store.loads == 0

    qa = ask(engine, "请问工控机和普通家用电脑有什么区别？", ())
    assert qa.status == "needs_input"
    assert "7x24" in qa.answer and "宽温防尘" in qa.answer
    assert store.loads == 0


def test_scenario_probe_rejects_out_of_catalog_products_with_insufficient_evidence(settings):
    store = CountingStore(settings.index_path)
    chat = DialogueChat(
        '{"in_catalog": false, "answer": "当前知识库暂未收录游戏笔记本电脑或激光雷达相关产品资料，无法为您推荐对应型号。"}'
    )
    engine = _engine(settings, chat, store)
    ipc = index_file(engine, FIXTURES / "specs" / "sp-ipc-670.md", "ipc-670")

    result = ask(engine, "帮我推荐一款适合玩3A大作的游戏笔记本电脑，或者推荐一台激光雷达", (ipc,))
    assert result.status == "insufficient_evidence"
    assert result.models == () and result.citations == ()
    assert "暂未收录" in result.answer
    assert store.loads > 0


def test_scenario_language_routes_to_industrial_panel_and_rack_not_camera():
    panel_text = "工厂产线车间环境粉尘比较大、需要操作员戴手套直接在设备面板上点控，有没有合适的工业电脑推荐？"
    panel_req = extract_requirements(panel_text)
    assert panel_req.category == "resistive_panel_pc"
    assert classify_intent(panel_text, panel_req) == "selection"

    rack_text = "我们想在机房机柜里装一台做多路工业视觉检测和扩展多张采集卡的主机，平常维护希望方便一些、还要防误开箱，推荐哪款？"
    rack_req = extract_requirements(rack_text)
    assert rack_req.category == "rack_mount_pc"
    assert any(item.attribute == "industrial_alarm" and item.value is True for item in rack_req.requirements)
    assert classify_intent(rack_text, rack_req) == "selection"
