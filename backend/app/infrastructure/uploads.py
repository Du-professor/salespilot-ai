import base64
import binascii
import codecs
import hashlib
import re
import threading
from io import BytesIO
from pathlib import Path
from typing import Annotated, ClassVar
from zipfile import BadZipFile, ZipFile

from pydantic import Field

from app.api.errors import ApiProblem
from app.contracts.common import ContractModel, Text

MAX_BYTES = 2_000_000
LIBRARY_MAX_BYTES = 200_000_000
MAX_ARCHIVE_EXPANDED_BYTES = 12_000_000
LIBRARY_DOCX_MAX_EXPANDED_BYTES = 200_000_000
LIBRARY_PPTX_MAX_EXPANDED_BYTES = 200_000_000
LIBRARY_MAX_PAGES = 500
LIBRARY_MAX_SECTIONS = 4000
LIBRARY_MAX_TEXT_CHARS = 2_000_000
PDF_BATCH_PAGES = 25
_CONVERTER_CACHE: dict[tuple[str, str], object] = {}
_CONVERTER_LOCK = threading.Lock()
_BOUNDARY_RE = re.compile(r'boundary=(?:"([^"]+)"|([^;\s]+))', re.I)
_DISPOSITION_NAME_RE = re.compile(r'name="([^"]+)"')
_DISPOSITION_FILENAME_RE = re.compile(r'filename="([^"]*)"')


class FilePayload(ContractModel):
    max_bytes: ClassVar[int] = MAX_BYTES
    filename: Annotated[Text, Field(max_length=180)]
    content_base64: Annotated[str, Field(max_length=2_666_668)]

    def bytes(self) -> bytes:
        try:
            payload = base64.b64decode(self.content_base64, validate=True)
        except (ValueError, binascii.Error):
            raise ApiProblem(422, "invalid_file", "文件内容格式不正确。") from None
        if not payload or len(payload) > self.max_bytes:
            raise ApiProblem(422, "invalid_file_size", f"请提供不超过 {self.max_bytes // 1_000_000} MB 的非空文件。")
        return payload

    @property
    def suffix(self) -> str:
        return Path(self.filename).suffix.lower()


class LibraryFilePayload(FilePayload):
    max_bytes: ClassVar[int] = LIBRARY_MAX_BYTES
    content_base64: Annotated[str, Field(max_length=((LIBRARY_MAX_BYTES + 2) // 3) * 4)]


def _verify_zip_archive(archive: ZipFile, member: str, max_expanded_bytes: int) -> None:
    entries = archive.infolist()
    if (
        len(entries) > 1000
        or sum(item.file_size for item in entries) > max_expanded_bytes
        or member not in archive.namelist()
    ):
        raise ApiProblem(422, "invalid_archive", "文件结构或解压大小超出限制。")
    # Stream-verify actual decompressed byte count so forged ZIP headers cannot bypass the limit.
    expanded = 0
    for entry in entries:
        if entry.is_dir():
            continue
        with archive.open(entry, "r") as stream:
            while True:
                remaining = max_expanded_bytes - expanded
                block = stream.read(min(65536, remaining + 1))
                if not block:
                    break
                expanded += len(block)
                if expanded > max_expanded_bytes:
                    raise ApiProblem(422, "invalid_archive", "文件结构或解压大小超出限制。")


def validate_archive(
    payload: bytes, member: str, *, max_expanded_bytes: int = MAX_ARCHIVE_EXPANDED_BYTES,
) -> None:
    try:
        with ZipFile(BytesIO(payload)) as archive:
            _verify_zip_archive(archive, member, max_expanded_bytes)
    except (BadZipFile, RuntimeError):
        raise ApiProblem(422, "invalid_archive", "文件结构不正确。") from None


def validate_archive_file(
    path: Path, member: str, *, max_expanded_bytes: int = MAX_ARCHIVE_EXPANDED_BYTES,
) -> None:
    try:
        with ZipFile(path) as archive:
            _verify_zip_archive(archive, member, max_expanded_bytes)
    except (BadZipFile, RuntimeError):
        raise ApiProblem(422, "invalid_archive", "文件结构不正确。") from None


def validate_text_file(path: Path) -> None:
    decoder = codecs.getincrementaldecoder("utf-8-sig")()
    has_text = False
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(65536)
                if not chunk:
                    break
                text = decoder.decode(chunk, final=False)
                if not has_text and text.strip():
                    has_text = True
            tail = decoder.decode(b"", final=True)
            if not has_text and tail.strip():
                has_text = True
    except UnicodeDecodeError:
        raise ApiProblem(422, "invalid_text_encoding", "请上传 UTF-8 文本。") from None
    if not has_text:
        raise ApiProblem(422, "invalid_document_text", "文件没有可检索的文字。")


def validate_pdf_file(path: Path) -> None:
    with path.open("rb") as handle:
        head = handle.read(5)
    if head != b"%PDF-":
        raise ApiProblem(422, "invalid_document", "文件不是有效的 PDF。")


async def stream_raw_upload(request, destination: Path, *, max_bytes: int = LIBRARY_MAX_BYTES) -> tuple[int, str]:
    hasher = hashlib.sha256()
    total = 0
    with destination.open("wb") as handle:
        async for chunk in request.stream():
            if not chunk:
                continue
            total += len(chunk)
            if total > max_bytes:
                raise ApiProblem(422, "invalid_file_size", f"请提供不超过 {max_bytes // 1_000_000} MB 的非空文件。")
            hasher.update(chunk)
            handle.write(chunk)
    if total == 0:
        raise ApiProblem(422, "invalid_file_size", f"请提供不超过 {max_bytes // 1_000_000} MB 的非空文件。")
    return total, hasher.hexdigest()


async def stream_multipart_upload(
    request, destination: Path, *, max_bytes: int = LIBRARY_MAX_BYTES,
) -> tuple[dict[str, str], str | None, int, str]:
    """Zero-dependency streaming multipart/form-data parser that writes the file part directly to disk."""
    content_type = request.headers.get("content-type", "")
    match = _BOUNDARY_RE.search(content_type)
    if not match:
        raise ApiProblem(422, "invalid_file", "无效的 multipart 请求头。")
    boundary = (match.group(1) or match.group(2)).encode("latin-1")
    delimiter = b"--" + boundary
    sep = b"\r\n" + delimiter

    fields: dict[str, str] = {}
    filename: str | None = None
    hasher = hashlib.sha256()
    total = 0
    saw_file = False

    state = "preamble"
    buf = bytearray()
    current_name: str | None = None
    current_is_file = False
    field_buf = bytearray()

    with destination.open("wb") as handle:
        async for raw_chunk in request.stream():
            if not raw_chunk:
                continue
            buf.extend(raw_chunk)
            while True:
                if state == "preamble":
                    idx = buf.find(delimiter)
                    if idx < 0:
                        if len(buf) > len(delimiter) + 4:
                            del buf[:-(len(delimiter) + 4)]
                        break
                    del buf[:idx + len(delimiter)]
                    state = "after_boundary"
                elif state == "after_boundary":
                    if len(buf) < 2:
                        break
                    if buf[:2] == b"--":
                        state = "done"
                        break
                    if buf[:2] == b"\r\n":
                        del buf[:2]
                        state = "headers"
                    else:
                        raise ApiProblem(422, "invalid_file", "无效的 multipart 分隔符。")
                elif state == "headers":
                    idx = buf.find(b"\r\n\r\n")
                    if idx < 0:
                        if len(buf) > 16_384:
                            raise ApiProblem(422, "invalid_file", "表单头过大。")
                        break
                    header_text = bytes(buf[:idx]).decode("utf-8", errors="replace")
                    del buf[:idx + 4]
                    current_name = None
                    current_is_file = False
                    for line in header_text.split("\r\n"):
                        if line.lower().startswith("content-disposition:"):
                            name_match = _DISPOSITION_NAME_RE.search(line)
                            if name_match:
                                current_name = name_match.group(1)
                            file_match = _DISPOSITION_FILENAME_RE.search(line)
                            if file_match is not None:
                                current_is_file = True
                                filename = file_match.group(1)
                                saw_file = True
                    field_buf.clear()
                    state = "body"
                elif state == "body":
                    idx = buf.find(sep)
                    if idx >= 0:
                        piece = bytes(buf[:idx])
                        del buf[:idx + len(sep)]
                        if current_is_file:
                            total += len(piece)
                            if total > max_bytes:
                                raise ApiProblem(
                                    422, "invalid_file_size",
                                    f"请提供不超过 {max_bytes // 1_000_000} MB 的非空文件。",
                                )
                            hasher.update(piece)
                            handle.write(piece)
                        elif current_name is not None:
                            field_buf.extend(piece)
                            if len(field_buf) > 4096:
                                raise ApiProblem(422, "invalid_request", "表单字段过长。")
                            fields[current_name] = field_buf.decode("utf-8", errors="strict")
                        state = "after_boundary"
                    else:
                        keep = len(sep)
                        if len(buf) > keep:
                            flush = bytes(buf[:-keep])
                            del buf[:-keep]
                            if current_is_file:
                                total += len(flush)
                                if total > max_bytes:
                                    raise ApiProblem(
                                        422, "invalid_file_size",
                                        f"请提供不超过 {max_bytes // 1_000_000} MB 的非空文件。",
                                    )
                                hasher.update(flush)
                                handle.write(flush)
                            elif current_name is not None:
                                field_buf.extend(flush)
                                if len(field_buf) > 4096:
                                    raise ApiProblem(422, "invalid_request", "表单字段过长。")
                        break
                else:
                    break
            if state == "done":
                break

    if not saw_file or total == 0:
        raise ApiProblem(422, "invalid_file_size", f"请提供不超过 {max_bytes // 1_000_000} MB 的非空文件。")
    return fields, filename, total, hasher.hexdigest()


def document_sections(
    file: FilePayload, payload: bytes, artifacts: Path | None
) -> list[tuple[int | None, str, str]]:
    return parse_sections(file.filename, payload, artifacts)


def parse_sections(
    filename: str, payload: bytes, artifacts: Path | None,
    *,
    max_file_size: int = MAX_BYTES,
    max_docx_expanded: int = MAX_ARCHIVE_EXPANDED_BYTES,
    max_pptx_expanded: int = MAX_ARCHIVE_EXPANDED_BYTES,
    max_pages: int = 50,
    max_sections: int = 200,
    max_text_chars: int = 120_000,
) -> list[tuple[int | None, str, str]]:
    suffix = Path(filename).suffix.lower()
    if suffix in {".txt", ".md"}:
        try:
            text = payload.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise ApiProblem(422, "invalid_text_encoding", "请上传 UTF-8 文本。") from None
        parts = re.split(r"(?m)^#{1,6}[ \t]+(.+?)[ \t]*$", text)
        sections = [(None, "正文", parts[0].strip())]
        sections.extend((None, parts[i].strip()[:180], parts[i + 1].strip()) for i in range(1, len(parts), 2))
    elif suffix in {".pdf", ".docx", ".pptx"}:
        sections = _convert_document(
            suffix, payload, artifacts, max_file_size, max_docx_expanded, max_pptx_expanded, max_pages,
        )
    else:
        raise ApiProblem(422, "unsupported_document", "支持 TXT、Markdown、PDF、DOCX 和 PPTX。")
    sections = [item for item in sections if item[2]]
    if not sections or len(sections) > max_sections or sum(len(item[2]) for item in sections) > max_text_chars:
        raise ApiProblem(422, "invalid_document_text", "资料没有可引用文字，或段落和正文超出限制。")
    return sections


def parse_library_sections(filename: str, payload: bytes, artifacts: Path | None) -> list[tuple[int | None, str, str]]:
    return parse_sections(
        filename,
        payload,
        artifacts,
        max_file_size=LIBRARY_MAX_BYTES,
        max_docx_expanded=LIBRARY_DOCX_MAX_EXPANDED_BYTES,
        max_pptx_expanded=LIBRARY_PPTX_MAX_EXPANDED_BYTES,
        max_pages=LIBRARY_MAX_PAGES,
        max_sections=LIBRARY_MAX_SECTIONS,
        max_text_chars=LIBRARY_MAX_TEXT_CHARS,
    )


def _get_converter(suffix: str, artifacts: Path | None):
    key = (suffix, str(artifacts.resolve()) if artifacts is not None else "")
    with _CONVERTER_LOCK:
        cached = _CONVERTER_CACHE.get(key)
        if cached is not None:
            return cached
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption
        except ImportError:
            raise ApiProblem(503, "document_parser_unavailable", "Docling 解析组件尚未安装。") from None
        formats = {".pdf": InputFormat.PDF, ".docx": InputFormat.DOCX, ".pptx": InputFormat.PPTX}
        converter = DocumentConverter(
            allowed_formats=[formats[suffix]],
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=PdfPipelineOptions(
                artifacts_path=artifacts, document_timeout=180, enable_remote_services=False,
            ))} if suffix == ".pdf" else {},
        )
        _CONVERTER_CACHE[key] = converter
        return converter


def _export_document_pages(document) -> list[tuple[int | None, str, str]]:
    sections: list[tuple[int | None, str, str]] = []
    pages = sorted(document.pages)
    if not pages:
        text = document.export_to_markdown().strip()
        return [(None, "正文", text)] if text else []

    # Collect per-page supplemental text for items whose provenance spans multiple pages,
    # avoiding O(N^2) full-document re-conversions per spanning page.
    extra_by_page: dict[int, list[str]] = {}
    iterate = getattr(document, "iterate_items", None)
    if callable(iterate):
        for item, _ in iterate():
            prov = getattr(item, "prov", ()) or ()
            item_pages = sorted({location.page_no for location in prov if getattr(location, "page_no", None)})
            if len(item_pages) > 1:
                export_md = getattr(item, "export_to_markdown", None)
                snippet = ""
                if callable(export_md):
                    try:
                        snippet = export_md(doc=document).strip()
                    except TypeError:
                        snippet = export_md().strip()
                elif hasattr(item, "text") and isinstance(item.text, str):
                    snippet = item.text.strip()
                if snippet:
                    for page_no in item_pages[1:]:
                        extra_by_page.setdefault(page_no, []).append(snippet)

    for page in pages:
        text = document.export_to_markdown(page_no=page).strip()
        extras = [item for item in extra_by_page.get(page, ()) if item not in text]
        if extras:
            text = "\n\n".join([part for part in (text, *extras) if part]).strip()
        if text:
            sections.append((page, f"第 {page} 页", text))
    return sections


def _convert_document(
    suffix: str,
    payload: bytes,
    artifacts: Path | None,
    max_file_size: int,
    max_docx_expanded: int,
    max_pptx_expanded: int,
    max_pages: int,
) -> list[tuple[int | None, str, str]]:
    if suffix == ".pdf":
        if not payload.startswith(b"%PDF-"):
            raise ApiProblem(422, "invalid_document", "文件不是有效的 PDF。")
        if artifacts is None or not artifacts.is_dir():
            raise ApiProblem(503, "document_parser_unavailable", "请先配置本地 Docling 模型目录。")
    elif suffix == ".docx":
        validate_archive(payload, "word/document.xml", max_expanded_bytes=max_docx_expanded)
    else:
        validate_archive(payload, "ppt/presentation.xml", max_expanded_bytes=max_pptx_expanded)

    converter = _get_converter(suffix, artifacts)
    try:
        from docling.datamodel.base_models import DocumentStream
    except ImportError:
        raise ApiProblem(503, "document_parser_unavailable", "Docling 解析组件尚未安装。") from None
    try:
        if suffix != ".pdf" or max_pages <= PDF_BATCH_PAGES:
            result = converter.convert(
                DocumentStream(name=f"upload{suffix}", stream=BytesIO(payload)),
                max_num_pages=max_pages,
                max_file_size=max_file_size,
            )
            if result.status.value != "success":
                raise ValueError("Document conversion was incomplete")
            return _export_document_pages(result.document)

        # Batch-convert large PDFs in fixed page windows to bound 2C8G peak memory.
        sections: list[tuple[int | None, str, str]] = []
        for start_page in range(1, max_pages + 1, PDF_BATCH_PAGES):
            end_page = min(start_page + PDF_BATCH_PAGES - 1, max_pages)
            result = converter.convert(
                DocumentStream(name=f"upload{suffix}", stream=BytesIO(payload)),
                max_num_pages=max_pages,
                max_file_size=max_file_size,
                page_range=(start_page, end_page),
            )
            if result.status.value != "success":
                if start_page == 1:
                    raise ValueError("Document conversion was incomplete")
                break
            batch_pages = sorted(result.document.pages)
            if not batch_pages:
                break
            sections.extend(_export_document_pages(result.document))
            if max(batch_pages) < end_page:
                break
        return sections
    except Exception:
        raise ApiProblem(422, "document_parse_failed", "资料解析失败或超出本地转换限制。") from None
