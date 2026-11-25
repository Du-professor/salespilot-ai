import base64
import os
from pathlib import Path

import httpx

from app.api.errors import ApiProblem
from app.contracts.rag_workspace import RagDocumentError, RagUnavailable
from app.infrastructure.uploads import parse_library_sections as parse_sections

from .industrial import industrial_docx
from .models import ParsedSection


def call_external_parser(
    filename: str,
    payload: bytes,
    *,
    base_url: str,
    api_key: str | None = None,
    timeout: float = 180.0,
) -> tuple[ParsedSection, ...]:
    """Call an external OCR / high-precision document parser service (e.g. MinerU / Doc2X gateway)."""
    clean_base = base_url.strip().rstrip("/")
    url = clean_base if clean_base.endswith("/parse") else f"{clean_base}/parse"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        response = httpx.post(
            url,
            json={
                "filename": filename,
                "content_base64": base64.b64encode(payload).decode("ascii"),
            },
            headers=headers,
            timeout=timeout,
        )
    except (httpx.TimeoutException, httpx.TransportError) as exc:
        raise RagUnavailable(f"外部文档解析服务不可用：{exc.__class__.__name__}") from None
    if response.status_code == 429 or response.status_code >= 500:
        raise RagUnavailable(f"外部文档解析服务返回 HTTP {response.status_code}。")
    if response.status_code >= 400:
        raise RagDocumentError(f"外部文档解析拒绝该文件（HTTP {response.status_code}）。")
    try:
        items = response.json()["sections"]
        parsed = tuple(
            ParsedSection(
                page=item.get("page"),
                section=(item.get("section") or (f"第 {item['page']} 页" if item.get("page") else "正文"))[:180],
                text=str(item["text"]).strip(),
                model=item.get("model"),
            )
            for item in items
            if str(item.get("text", "")).strip()
        )
    except (KeyError, TypeError, ValueError):
        raise RagUnavailable("外部文档解析服务返回了无效结构。") from None
    if not parsed:
        raise RagDocumentError("资料没有可索引文字。")
    return parsed


def _external_parse(
    filename: str,
    payload: bytes,
    external_parser: tuple[str, str | None] | None = None,
) -> tuple[ParsedSection, ...] | None:
    suffix = Path(filename).suffix.lower()
    if suffix not in {".pdf", ".docx", ".pptx"}:
        return None
    if external_parser is not None:
        base_url, api_key = external_parser
        timeout = float(os.getenv("SALESPILOT_PARSER_TIMEOUT_SECONDS", "180"))
        return call_external_parser(filename, payload, base_url=base_url, api_key=api_key, timeout=timeout)
    base_url = (os.getenv("SALESPILOT_PARSER_BASE_URL") or "").strip().rstrip("/")
    if not base_url:
        return None
    api_key = (os.getenv("SALESPILOT_PARSER_API_KEY") or "").strip() or None
    timeout = float(os.getenv("SALESPILOT_PARSER_TIMEOUT_SECONDS", "180"))
    return call_external_parser(filename, payload, base_url=base_url, api_key=api_key, timeout=timeout)


def parse_document(
    filename: str,
    payload: bytes,
    artifacts: Path | None,
    *,
    external_parser: tuple[str, str | None] | None = None,
) -> tuple[ParsedSection, ...]:
    if Path(filename).suffix.lower() == ".docx":
        try:
            if industrial := industrial_docx(filename, payload):
                return industrial
        except ApiProblem as exc:
            if exc.status >= 500:
                raise RagUnavailable(str(exc)) from None
            raise RagDocumentError(str(exc)) from None
    if external := _external_parse(filename, payload, external_parser):
        return external
    try:
        sections = parse_sections(filename, payload, artifacts)
    except ApiProblem as exc:
        if exc.status >= 500:
            raise RagUnavailable(str(exc)) from None
        raise RagDocumentError(str(exc)) from None
    parsed = tuple(ParsedSection(page=page, section=section, text=text) for page, section, text in sections)
    if not parsed:
        raise RagDocumentError("资料没有可索引文字。")
    return parsed
