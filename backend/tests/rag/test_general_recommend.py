import json
import re

from app.rag import HashingEmbeddings, SqliteRagEngine
from app.rag.store import RagChunkStore

from conftest import ask, index_bytes


class ScriptedGeneralChat:
    def __init__(self, handler):
        self.handler = handler
        self.calls = 0

    def compose(self, *, system: str, user: str) -> str:
        self.calls += 1
        return self.handler(user)


def _engine(settings, chat):
    return SqliteRagEngine(
        settings,
        store=RagChunkStore(settings.index_path),
        embedder=HashingEmbeddings(settings.embedding_dimensions),
        chat=chat,
    )


def _extract_chunk_id(prompt: str) -> str:
    match = re.search(r"chunk_id=([^\s|]+)", prompt)
    assert match is not None
    return match.group(1)


def test_commercial_pc_recommendation_is_grounded_with_citations_and_manuals(settings):
    laptop_md = (
        "# Lenovo ThinkPad X1 Carbon Gen 12 商用轻薄本规格书\n\n"
        "## 1. 核心规格参数\n\n"
        "- 型号：Lenovo ThinkPad X1 Carbon Gen 12\n"
        "- 重量：起始约 1.08 kg，适合高频差旅办公\n"
        "- 处理器：Intel Core Ultra 7 155U\n"
        "- 接口：2x Thunderbolt 4 / USB4、1x HDMI 2.1\n"
    ).encode("utf-8")

    def handler(user: str) -> str:
        cid = _extract_chunk_id(user)
        return json.dumps({
            "status": "recommended",
            "recommended_models": ["Lenovo ThinkPad X1 Carbon Gen 12"],
            "answer": "推荐 Lenovo ThinkPad X1 Carbon Gen 12，整机起始重量约 1.08 kg，配备双 Thunderbolt 4 接口，适合高频出差。",
            "citations": [{"chunk_id": cid, "excerpt": "起始约 1.08 kg，适合高频差旅办公"}],
        }, ensure_ascii=False)

    chat = ScriptedGeneralChat(handler)
    engine = _engine(settings, chat)
    doc = index_bytes(engine, laptop_md, "tp-x1", filename="ThinkPad-X1.md")

    # Tests numeric kg constraint (triggers selection -> unknown category -> general fallback)
    res_numeric = ask(engine, "需要一台重量不超过 1.2 kg、适合出差用的商用笔记本电脑，请附选型报告。", (doc,))
    assert res_numeric.status == "recommended"
    assert res_numeric.models == ("Lenovo ThinkPad X1 Carbon Gen 12",)
    assert len(res_numeric.citations) == 1
    assert res_numeric.citations[0].document_id == doc.document_id
    assert [(m.document_id, m.version) for m in res_numeric.manuals] == [(doc.document_id, doc.version)]
    assert res_numeric.include_report is True

    # Tests scenario query without numeric specs
    res_scenario = ask(engine, "帮我推荐一款适合经常出差的轻薄商务笔记本电脑", (doc,))
    assert res_scenario.status == "recommended"
    assert res_scenario.models == ("Lenovo ThinkPad X1 Carbon Gen 12",)
    assert res_scenario.include_report is False


def test_hallucinated_model_or_excerpt_is_rejected(settings):
    switch_md = (
        "# Huawei CloudEngine S5735-L-V2 交换机\n\n"
        "## 1. 端口规格\n\n"
        "- 型号：CloudEngine S5735-L24P4X-A-V2\n"
        "- 下行端口：24 x 10/100/1000BASE-T PoE+\n"
        "- 上行端口：4 x 10GE SFP+\n"
    ).encode("utf-8")

    def fake_excerpt_handler(user: str) -> str:
        cid = _extract_chunk_id(user)
        return json.dumps({
            "status": "recommended",
            "recommended_models": ["CloudEngine S5735-L24P4X-A-V2"],
            "answer": "推荐 CloudEngine S5735-L24P4X-A-V2。",
            "citations": [{"chunk_id": cid, "excerpt": "不存在的 400G QSFP28 端口"}],
        }, ensure_ascii=False)

    engine = _engine(settings, ScriptedGeneralChat(fake_excerpt_handler))
    doc = index_bytes(engine, switch_md, "hw-sw", filename="S5735.md")
    res_bad_excerpt = ask(engine, "需要一款 24 口千兆 PoE+ 交换机", (doc,))
    assert res_bad_excerpt.status == "insufficient_evidence"
    assert res_bad_excerpt.models == () and res_bad_excerpt.citations == ()

    def fake_model_handler(user: str) -> str:
        cid = _extract_chunk_id(user)
        return json.dumps({
            "status": "recommended",
            "recommended_models": ["FAKE-SWITCH-9999"],
            "answer": "推荐 FAKE-SWITCH-9999。",
            "citations": [{"chunk_id": cid, "excerpt": "24 x 10/100/1000BASE-T PoE+"}],
        }, ensure_ascii=False)

    engine2 = _engine(settings, ScriptedGeneralChat(fake_model_handler))
    doc2 = index_bytes(engine2, switch_md, "hw-sw", filename="S5735.md")
    res_bad_model = ask(engine2, "需要一款 24 口千兆 PoE+ 交换机", (doc2,))
    assert res_bad_model.status == "insufficient_evidence"
    assert res_bad_model.models == () and res_bad_model.citations == ()


def test_cross_industry_out_of_catalog_query_returns_insufficient_evidence(settings):
    printer_md = (
        "# HP LaserJet Pro 4003dn 产品手册\n\n"
        "## 1. 打印性能\n\n"
        "- 黑白激光打印速度最高 40 ppm，标配自动双面与千兆网口。\n"
    ).encode("utf-8")

    def out_handler(user: str) -> str:
        return json.dumps({
            "status": "insufficient_evidence",
            "recommended_models": [],
            "answer": "当前知识库仅收录了办公打印设备资料，暂未收录交换机或 PoE 网络设备，无法为您推荐对应型号。",
            "citations": [],
        }, ensure_ascii=False)

    engine = _engine(settings, ScriptedGeneralChat(out_handler))
    doc = index_bytes(engine, printer_md, "hp-4003", filename="HP-4003dn.md")
    res = ask(engine, "帮我配一台 24 口千兆 PoE+ 交换机", (doc,))
    assert res.status == "insufficient_evidence"
    assert res.models == () and res.citations == () and res.manuals == ()
    assert "暂未收录" in res.answer


def test_non_industrial_poe_and_capability_query_prefers_general_recommend(settings):
    switch_md = (
        "# H3C S5130S-EI 系列千兆以太网交换机规格书\n\n"
        "## 1. 产品概述与代表型号\n\n"
        "- H3C S5130S-28S-HPWR-EI：24 个千兆 PoE+ 电口 + 4 个 1G/10G SFP+ 万兆光口；整机 PoE+ 总功率最高达 370 W，支持 IRF2 智能弹性架构堆叠。\n\n"
        "## 2. 核心技术规格参数表\n\n"
        "| 特性 | 说明 |\n"
        "| --- | --- |\n"
        "| PoE 节能 | S5130S-EI 支持按时间段动态下发 PoE 节能供电策略 |\n"
    ).encode("utf-8")

    def handler(user: str) -> str:
        cids = re.findall(r"chunk_id=([^\s|]+)", user)
        assert cids
        return json.dumps({
            "status": "recommended",
            "recommended_models": ["S5130S-28S-HPWR-EI"],
            "answer": "推荐 H3C S5130S-28S-HPWR-EI，具备 24 个千兆 PoE+ 电口、370 W 整机供电预算、4 个 10G SFP+ 万兆光口及 IRF2 堆叠能力。",
            "citations": [{"chunk_id": cids[0], "excerpt": "H3C S5130S-28S-HPWR-EI：24 个千兆 PoE+ 电口 + 4 个 1G/10G SFP+ 万兆光口；整机 PoE+ 总功率最高达 370 W，支持 IRF2 智能弹性架构堆叠。"}],
        }, ensure_ascii=False)

    chat = ScriptedGeneralChat(handler)
    engine = _engine(settings, chat)
    doc = index_bytes(engine, switch_md, "h3c-switch", filename="H3C-S5130S-EI-SpecSheet.md")

    res = ask(
        engine,
        "需要支持 IRF2 智能弹性架构堆叠、24 口千兆 PoE+（370W 供电预算）以及万兆 SFP+ 上行的接入交换机，请推荐具体型号",
        (doc,),
    )
    assert chat.calls == 1
    assert res.status == "recommended"
    assert res.models == ("S5130S-28S-HPWR-EI",)
    assert len(res.citations) == 1
    assert [(m.document_id, m.version) for m in res.manuals] == [(doc.document_id, doc.version)]
