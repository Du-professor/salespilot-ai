from io import BytesIO
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

from app.rag.industrial import WORD

from conftest import ask, index_bytes


def manual(description: str, rows: tuple[tuple[str, ...], ...]) -> bytes:
    namespace = WORD.strip("{}")

    def paragraph(text):
        return f"<w:p><w:r><w:t>{escape(text)}</w:t></w:r></w:p>"

    table = "".join(
        "<w:tr>" + "".join(f"<w:tc>{paragraph(cell)}</w:tc>" for cell in row) + "</w:tr>" for row in rows
    )
    document = (
        f'<w:document xmlns:w="{namespace}"><w:body>{paragraph(description)}'
        f"<w:tbl>{table}</w:tbl></w:body></w:document>"
    )
    stream = BytesIO()
    with ZipFile(stream, "w", ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", document.encode())
    return stream.getvalue()


def test_panel_word_specs_use_model_row_and_return_manual_by_default(engine):
    document = index_bytes(engine, manual(
        "全屏电阻触摸屏设计，前面板满足IP65要求，集成双Intel®千兆网卡，无风扇设计，支持 DVI-D。",
        (("Model", "PL121RQ-Z9K", "PL150RQ-Z9K"), ("处理系统", "CPU", "Intel® Alder Lake N305")),
    ), "panel", filename="panel.docx")
    reply = ask(engine, "需要支持 N305、无风扇和双千兆网口的电阻触摸平板。", (document,))
    assert reply.status == "recommended"
    assert reply.models == ("PLxxxRQ-Z9K",)
    assert [item.document_id for item in reply.manuals] == ["panel"]
    assert any("Alder Lake N305" in item.excerpt and item.section.startswith("产品规格表") for item in reply.citations)


def test_rack_specs_are_scoped_and_unrecognized_hard_constraints_require_clarification(engine):
    description = "标准19英⼨4U上架式机箱，支持非法开箱报警，可锁前门。"
    supported = index_bytes(engine, manual(
        description, (("规格", "IPC410-H610"), ("芯片组", "H610"), ("CPU", "支持 Intel® 12/13/14代酷睿")),
    ), "rack-a", filename="a.docx")
    different = index_bytes(engine, manual(
        "标准19英寸4U上架式机箱。", (("规格", "IPC410-H610B"), ("芯片组", "H610"), ("CPU", "支持 Intel® 12/13/14代酷睿")),
    ), "rack-b", filename="b.docx")
    request = "需要 12/13/14 代酷睿、非法开箱报警和可锁前门的 19 英寸 4U 机架式电脑。"
    result = ask(engine, request, (supported, different))
    assert result.models == ("IPC410-H610",)
    assert all(citation.document_id == "rack-a" for citation in result.citations)
    unknown = ask(engine, request + "必须具备光纤接口。", (supported, different))
    assert unknown.status == "needs_input" and not unknown.manuals
