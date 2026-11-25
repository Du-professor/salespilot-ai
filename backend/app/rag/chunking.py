import hashlib
import re
from collections.abc import Callable

from app.contracts.rag_workspace import RagDocumentError

from .models import Chunk, ParsedSection
from .specs import MODEL_PATTERN

MAX_CHARS = 900
OVERLAP = 120
_TABLE_SEP_RE = re.compile(r"^\s*\|?(?:\s*:?-+:?\s*\|)+\s*:?-+:?\s*\|?\s*$")


def _is_table_line(line: str) -> bool:
    stripped = line.strip()
    return "|" in stripped and (stripped.startswith("|") or stripped.count("|") >= 2)


def _split_plain_text(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    effective_overlap = min(overlap, max(0, max_chars // 4))
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        step = max(1, max_chars - effective_overlap)
        return [text[index:index + max_chars] for index in range(0, len(text), step)]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=effective_overlap,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )
    return splitter.split_text(text)


def _split_table_block(lines: list[str], max_chars: int) -> list[str]:
    """Split a Markdown table by complete rows while repeating header + separator on every chunk."""
    header = "\n".join(lines[:2])
    rows = lines[2:]
    if not rows:
        return [header]
    full = "\n".join(lines)
    if len(full) <= max_chars:
        return [full]
    chunks: list[str] = []
    current_rows: list[str] = []
    current_len = len(header)
    for row in rows:
        projected = current_len + 1 + len(row)
        if current_rows and projected > max_chars:
            chunks.append("\n".join([header, *current_rows]))
            current_rows = [row]
            current_len = len(header) + 1 + len(row)
        else:
            current_rows.append(row)
            current_len = projected
    if current_rows:
        chunks.append("\n".join([header, *current_rows]))
    return chunks


def _split_text(text: str, *, max_chars: int = MAX_CHARS, overlap: int = OVERLAP) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    lines = text.splitlines()
    pieces: list[str] = []
    prose_buf: list[str] = []
    idx = 0
    total = len(lines)

    def flush_prose() -> None:
        if not prose_buf:
            return
        block = "\n".join(prose_buf).strip()
        prose_buf.clear()
        if block:
            pieces.extend(_split_plain_text(block, max_chars, overlap))

    while idx < total:
        if idx + 1 < total and _is_table_line(lines[idx]) and _TABLE_SEP_RE.match(lines[idx + 1]):
            flush_prose()
            tbl = [lines[idx], lines[idx + 1]]
            idx += 2
            while idx < total and _is_table_line(lines[idx]) and lines[idx].strip():
                tbl.append(lines[idx])
                idx += 1
            pieces.extend(_split_table_block(tbl, max_chars))
        else:
            prose_buf.append(lines[idx])
            idx += 1
    flush_prose()
    return pieces or _split_plain_text(text, max_chars, overlap)


def _enforce_token_budget(
    pieces: list[str],
    token_budget: int,
    token_counter: Callable[[list[str]], list[int]],
) -> list[str]:
    """Subdivide any chunk whose real tokenizer count exceeds token_budget."""
    if not pieces or token_budget <= 0:
        return pieces
    try:
        counts = token_counter(pieces)
    except Exception:
        return pieces
    if len(counts) != len(pieces):
        return pieces

    result: list[str] = []
    for piece, count in zip(pieces, counts, strict=True):
        if count <= token_budget or len(piece) <= 64:
            result.append(piece)
            continue
        # Estimate tighter character cap proportional to observed token ratio.
        ratio = max(1.15, count / max(1, token_budget))
        tighter_chars = max(64, int(len(piece) / ratio) - 16)
        sub_pieces = _split_text(piece, max_chars=tighter_chars, overlap=min(OVERLAP, tighter_chars // 6))
        if len(sub_pieces) <= 1:
            # Fallback hard bisection if no separator split occurred.
            mid = len(piece) // 2
            sub_pieces = [piece[:mid].strip(), piece[mid:].strip()]
        sub_pieces = [item for item in sub_pieces if item]
        if len(sub_pieces) <= 1:
            result.append(piece)
        else:
            result.extend(_enforce_token_budget(sub_pieces, token_budget, token_counter))
    return result


def _chunk_id(
    org_id: str, document_id: str, version: str, page: int | None, section: str | None, ordinal: int, text: str,
) -> str:
    identity = "\0".join((
        org_id, document_id, version, str(page), section or "", str(ordinal), text,
    ))
    return "chk_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def _parent_id(org_id: str, document_id: str, version: str, section: str | None) -> str:
    identity = "\0".join((org_id, document_id, version, section or ""))
    return "par_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]


def build_chunks(
    org_id: str,
    document_id: str,
    version: str,
    sections: tuple[ParsedSection, ...],
    *,
    max_tokens: int = 0,
    token_counter: Callable[[list[str]], list[int]] | None = None,
) -> tuple[Chunk, ...]:
    token_budget = max(64, int(max_tokens * 0.85)) if max_tokens > 0 else 0
    max_chars = min(MAX_CHARS, max(240, token_budget)) if token_budget > 0 else MAX_CHARS
    overlap = min(OVERLAP, max_chars // 5)

    heading_models = set(MODEL_PATTERN.findall(
        "\n".join(item.section or "" for item in sections),
    ))
    default_model = next(iter(heading_models)) if len(heading_models) == 1 else None
    chunks: list[Chunk] = []
    ordinal = 0
    for section in sections:
        section_models = set(MODEL_PATTERN.findall(section.section or ""))
        text_models = set(MODEL_PATTERN.findall(section.text))
        if section.model:
            model = section.model
        elif len(section_models) == 1:
            model = next(iter(section_models))
        elif default_model is not None:
            model = default_model
        elif len(text_models) == 1:
            model = next(iter(text_models))
        else:
            model = None
        pieces = _split_text(section.text, max_chars=max_chars, overlap=overlap)
        if token_budget > 0 and callable(token_counter):
            pieces = _enforce_token_budget(pieces, token_budget, token_counter)
        parent = _parent_id(org_id, document_id, version, section.section)
        for piece in pieces:
            text = piece.strip()
            if not text:
                continue
            chunks.append(Chunk(
                org_id=org_id,
                document_id=document_id,
                version=version,
                chunk_id=_chunk_id(org_id, document_id, version, section.page, section.section, ordinal, text),
                model=model,
                page=section.page,
                section=section.section,
                ordinal=ordinal,
                text=text,
                parent_id=parent,
            ))
            ordinal += 1
    if not chunks:
        raise RagDocumentError("资料没有可索引文字。")
    return tuple(chunks)
