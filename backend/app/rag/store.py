import json
import re
import sqlite3
from array import array
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from .models import Chunk

_CODE_TOKEN = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)+", re.I)
_WORD_TOKEN = re.compile(r"[a-z0-9]+", re.I)
_CJK_BLOCK = re.compile(r"[\u4e00-\u9fff]+")


def lexical_tokens(text: str) -> list[str]:
    """Tokenize bilingual technical specs: exact model codes + alphanumeric words + CJK unigrams/bigrams."""
    lowered = text.casefold()
    tokens: list[str] = []
    # Preserve compound model/spec codes (e.g. sp-vc-120 -> sp_vc_120, rs-485 -> rs_485)
    for code in _CODE_TOKEN.findall(lowered):
        tokens.append(re.sub(r"[._-]+", "_", code))
    tokens.extend(_WORD_TOKEN.findall(lowered))
    for block in _CJK_BLOCK.findall(lowered):
        if len(block) >= 2:
            tokens.extend(block[i:i + 2] for i in range(len(block) - 1))
        tokens.extend(block)
    return tokens


def lexical_document(text: str, section: str | None = None, model: str | None = None) -> str:
    combined = " ".join(part for part in (model or "", section or "", text) if part)
    return " ".join(lexical_tokens(combined))


def fts_match_query(query: str, *, max_terms: int = 24) -> str:
    """Build a safe SQLite FTS5 OR query prioritizing model codes, English terms, and CJK bigrams."""
    raw = lexical_tokens(query)
    if not raw:
        return ""
    # Prioritize multi-char tokens (codes, English words, CJK bigrams); fall back to single chars only if needed.
    multi = list(dict.fromkeys(tok for tok in raw if len(tok) >= 2))
    single = list(dict.fromkeys(tok for tok in raw if len(tok) == 1))
    selected = (multi if multi else single)[:max_terms]
    safe = [re.sub(r"[^a-z0-9_\u4e00-\u9fff]", "", tok) for tok in selected]
    safe = [tok for tok in safe if tok]
    if not safe:
        return ""
    return " OR ".join(f'"{tok}"' for tok in safe)

LEGACY_KNOWLEDGE_BASE = "default"
LEGACY_INDEX_VERSION = "legacy"
LEGACY_EMBEDDING_MODEL = "workspace-default"
LEGACY_MODEL_REVISION = "workspace"

SCHEMA = """
CREATE TABLE IF NOT EXISTS rag_documents (
    org_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    version TEXT NOT NULL,
    digest TEXT NOT NULL,
    chunk_count INTEGER NOT NULL,
    indexed_at TEXT NOT NULL,
    PRIMARY KEY (org_id, document_id, version)
);
CREATE TABLE IF NOT EXISTS rag_chunks (
    org_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    version TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    model TEXT,
    page INTEGER,
    section TEXT,
    ordinal INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL,
    dimensions INTEGER NOT NULL,
    PRIMARY KEY (org_id, document_id, version, chunk_id)
);
CREATE INDEX IF NOT EXISTS idx_rag_chunks_scope ON rag_chunks (org_id, document_id, version);
"""

SCHEMA += """
CREATE TABLE IF NOT EXISTS rag_index_versions (
    org_id TEXT NOT NULL,
    knowledge_base_id TEXT NOT NULL,
    index_version TEXT NOT NULL,
    embedding_model_id TEXT NOT NULL,
    embedding_model_revision TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    vector_config_json TEXT NOT NULL,
    state TEXT NOT NULL,
    source_snapshot_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (org_id, knowledge_base_id, index_version)
);
CREATE TABLE IF NOT EXISTS rag_index_documents (
    org_id TEXT NOT NULL,
    knowledge_base_id TEXT NOT NULL,
    index_version TEXT NOT NULL,
    document_id TEXT NOT NULL,
    version TEXT NOT NULL,
    digest TEXT NOT NULL,
    chunk_count INTEGER NOT NULL,
    embedding_model_id TEXT NOT NULL,
    embedding_model_revision TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    vector_config_json TEXT NOT NULL,
    indexed_at TEXT NOT NULL,
    PRIMARY KEY (org_id, knowledge_base_id, index_version, document_id, version)
);
CREATE TABLE IF NOT EXISTS rag_index_chunks (
    org_id TEXT NOT NULL,
    knowledge_base_id TEXT NOT NULL,
    index_version TEXT NOT NULL,
    document_id TEXT NOT NULL,
    version TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    model TEXT,
    page INTEGER,
    section TEXT,
    ordinal INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL,
    dimensions INTEGER NOT NULL,
    truncated INTEGER NOT NULL DEFAULT 0,
    parent_id TEXT,
    PRIMARY KEY (org_id, knowledge_base_id, index_version, document_id, version, chunk_id)
);
CREATE INDEX IF NOT EXISTS idx_rag_index_chunks_scope
    ON rag_index_chunks (org_id, knowledge_base_id, index_version, document_id, version);
CREATE VIRTUAL TABLE IF NOT EXISTS rag_chunks_fts USING fts5(
    org_id UNINDEXED,
    document_id UNINDEXED,
    version UNINDEXED,
    chunk_id UNINDEXED,
    lexical_text
);
CREATE VIRTUAL TABLE IF NOT EXISTS rag_index_chunks_fts USING fts5(
    org_id UNINDEXED,
    knowledge_base_id UNINDEXED,
    index_version UNINDEXED,
    document_id UNINDEXED,
    version UNINDEXED,
    chunk_id UNINDEXED,
    lexical_text
);
CREATE TABLE IF NOT EXISTS rag_rebuild_tasks (
    task_id TEXT PRIMARY KEY,
    org_id TEXT NOT NULL,
    knowledge_base_id TEXT NOT NULL,
    target_index_version TEXT NOT NULL,
    embedding_model_id TEXT NOT NULL,
    embedding_model_revision TEXT NOT NULL,
    state TEXT NOT NULL,
    total INTEGER NOT NULL DEFAULT 0,
    completed INTEGER NOT NULL DEFAULT 0,
    failed INTEGER NOT NULL DEFAULT 0,
    snapshot_json TEXT,
    error_code TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rag_rebuild_items (
    task_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    version TEXT NOT NULL,
    digest TEXT,
    state TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (task_id, document_id, version)
);
CREATE INDEX IF NOT EXISTS idx_rag_rebuild_tasks_state ON rag_rebuild_tasks (state, updated_at);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _pack(vector: Sequence[float]) -> bytes:
    return array("f", vector).tobytes()


def _unpack(blob: bytes, dimensions: int) -> list[float]:
    vector = array("f")
    vector.frombytes(blob)
    if len(vector) != dimensions:
        raise ValueError("stored vector dimensions do not match")
    return list(vector)


class RagChunkStore:
    def __init__(self, path: Path):
        self.path = path
        self.initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(SCHEMA)
            connection.execute("BEGIN IMMEDIATE")
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(rag_index_chunks)")}
            if "parent_id" not in columns:
                connection.execute("ALTER TABLE rag_index_chunks ADD COLUMN parent_id TEXT")
            # Backfill FTS5 tables seamlessly for pre-existing SQLite index files.
            if (
                connection.execute("SELECT 1 FROM rag_chunks LIMIT 1").fetchone() is not None
                and connection.execute("SELECT 1 FROM rag_chunks_fts LIMIT 1").fetchone() is None
            ):
                rows = connection.execute(
                    "SELECT org_id, document_id, version, chunk_id, model, section, text FROM rag_chunks",
                ).fetchall()
                connection.executemany(
                    """INSERT INTO rag_chunks_fts (org_id, document_id, version, chunk_id, lexical_text)
                    VALUES (?, ?, ?, ?, ?)""",
                    [
                        (
                            r["org_id"], r["document_id"], r["version"], r["chunk_id"],
                            lexical_document(r["text"], r["section"], r["model"]),
                        )
                        for r in rows
                    ],
                )
            if (
                connection.execute("SELECT 1 FROM rag_index_chunks LIMIT 1").fetchone() is not None
                and connection.execute("SELECT 1 FROM rag_index_chunks_fts LIMIT 1").fetchone() is None
            ):
                rows = connection.execute(
                    """SELECT org_id, knowledge_base_id, index_version, document_id, version, chunk_id,
                              model, section, text FROM rag_index_chunks""",
                ).fetchall()
                connection.executemany(
                    """INSERT INTO rag_index_chunks_fts
                    (org_id, knowledge_base_id, index_version, document_id, version, chunk_id, lexical_text)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    [
                        (
                            r["org_id"], r["knowledge_base_id"], r["index_version"],
                            r["document_id"], r["version"], r["chunk_id"],
                            lexical_document(r["text"], r["section"], r["model"]),
                        )
                        for r in rows
                    ],
                )

    def replace_document(
        self,
        org_id: str,
        document_id: str,
        version: str,
        digest: str,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        *,
        knowledge_base_id: str = LEGACY_KNOWLEDGE_BASE,
        index_version: str = LEGACY_INDEX_VERSION,
        embedding_model_id: str = LEGACY_EMBEDDING_MODEL,
        embedding_model_revision: str = LEGACY_MODEL_REVISION,
        vector_config: dict | None = None,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if index_version == LEGACY_INDEX_VERSION:
            self._replace_legacy(org_id, document_id, version, digest, chunks, vectors)
            return
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1 or 0 in dimensions:
            raise ValueError("indexed vectors must share one non-zero dimension")
        dimension = dimensions.pop()
        config_json = json.dumps(vector_config or {}, sort_keys=True, ensure_ascii=True)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """DELETE FROM rag_index_chunks
                WHERE org_id=? AND knowledge_base_id=? AND index_version=?
                  AND document_id=? AND version=?""",
                (org_id, knowledge_base_id, index_version, document_id, version),
            )
            connection.execute(
                """DELETE FROM rag_index_chunks_fts
                WHERE org_id=? AND knowledge_base_id=? AND index_version=?
                  AND document_id=? AND version=?""",
                (org_id, knowledge_base_id, index_version, document_id, version),
            )
            connection.executemany(
                """INSERT INTO rag_index_chunks
                (org_id, knowledge_base_id, index_version, document_id, version, chunk_id, model,
                 page, section, ordinal, text, embedding, dimensions, truncated, parent_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        chunk.org_id, knowledge_base_id, index_version, chunk.document_id, chunk.version,
                        chunk.chunk_id, chunk.model, chunk.page, chunk.section, chunk.ordinal, chunk.text,
                        _pack(vector), len(vector), 1 if chunk.truncated else 0, chunk.parent_id,
                    )
                    for chunk, vector in zip(chunks, vectors, strict=True)
                ],
            )
            connection.executemany(
                """INSERT INTO rag_index_chunks_fts
                (org_id, knowledge_base_id, index_version, document_id, version, chunk_id, lexical_text)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        chunk.org_id, knowledge_base_id, index_version, chunk.document_id, chunk.version,
                        chunk.chunk_id, lexical_document(chunk.text, chunk.section, chunk.model),
                    )
                    for chunk in chunks
                ],
            )
            connection.execute(
                """INSERT INTO rag_index_documents
                (org_id, knowledge_base_id, index_version, document_id, version, digest, chunk_count,
                 embedding_model_id, embedding_model_revision, dimensions, vector_config_json, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (org_id, knowledge_base_id, index_version, document_id, version) DO UPDATE SET
                    digest=excluded.digest,
                    chunk_count=excluded.chunk_count,
                    embedding_model_id=excluded.embedding_model_id,
                    embedding_model_revision=excluded.embedding_model_revision,
                    dimensions=excluded.dimensions,
                    vector_config_json=excluded.vector_config_json,
                    indexed_at=excluded.indexed_at""",
                (
                    org_id, knowledge_base_id, index_version, document_id, version, digest, len(chunks),
                    embedding_model_id, embedding_model_revision, dimension, config_json, _now(),
                ),
            )

    def _replace_legacy(
        self,
        org_id: str,
        document_id: str,
        version: str,
        digest: str,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "DELETE FROM rag_chunks WHERE org_id=? AND document_id=? AND version=?",
                (org_id, document_id, version),
            )
            connection.execute(
                "DELETE FROM rag_chunks_fts WHERE org_id=? AND document_id=? AND version=?",
                (org_id, document_id, version),
            )
            connection.executemany(
                """INSERT INTO rag_chunks
                (org_id, document_id, version, chunk_id, model, page, section, ordinal, text, embedding, dimensions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        chunk.org_id, chunk.document_id, chunk.version, chunk.chunk_id, chunk.model,
                        chunk.page, chunk.section, chunk.ordinal, chunk.text, _pack(vector), len(vector),
                    )
                    for chunk, vector in zip(chunks, vectors, strict=True)
                ],
            )
            connection.executemany(
                """INSERT INTO rag_chunks_fts (org_id, document_id, version, chunk_id, lexical_text)
                VALUES (?, ?, ?, ?, ?)""",
                [
                    (
                        chunk.org_id, chunk.document_id, chunk.version, chunk.chunk_id,
                        lexical_document(chunk.text, chunk.section, chunk.model),
                    )
                    for chunk in chunks
                ],
            )
            connection.execute(
                """INSERT INTO rag_documents
                (org_id, document_id, version, digest, chunk_count, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (org_id, document_id, version) DO UPDATE SET
                    digest=excluded.digest,
                    chunk_count=excluded.chunk_count,
                    indexed_at=excluded.indexed_at""",
                (org_id, document_id, version, digest, len(chunks), _now()),
            )

    def remove_document(self, org_id: str, document_id: str, version: str) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "DELETE FROM rag_chunks WHERE org_id=? AND document_id=? AND version=?",
                (org_id, document_id, version),
            )
            connection.execute(
                "DELETE FROM rag_chunks_fts WHERE org_id=? AND document_id=? AND version=?",
                (org_id, document_id, version),
            )
            connection.execute(
                "DELETE FROM rag_documents WHERE org_id=? AND document_id=? AND version=?",
                (org_id, document_id, version),
            )
            connection.execute(
                """DELETE FROM rag_index_chunks
                WHERE org_id=? AND document_id=? AND version=?""",
                (org_id, document_id, version),
            )
            connection.execute(
                """DELETE FROM rag_index_chunks_fts
                WHERE org_id=? AND document_id=? AND version=?""",
                (org_id, document_id, version),
            )
            connection.execute(
                """DELETE FROM rag_index_documents
                WHERE org_id=? AND document_id=? AND version=?""",
                (org_id, document_id, version),
            )

    def remove_document_from_version(
        self,
        org_id: str,
        knowledge_base_id: str,
        index_version: str,
        document_id: str,
        version: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if index_version == LEGACY_INDEX_VERSION:
                connection.execute(
                    "DELETE FROM rag_chunks WHERE org_id=? AND document_id=? AND version=?",
                    (org_id, document_id, version),
                )
                connection.execute(
                    "DELETE FROM rag_chunks_fts WHERE org_id=? AND document_id=? AND version=?",
                    (org_id, document_id, version),
                )
                connection.execute(
                    "DELETE FROM rag_documents WHERE org_id=? AND document_id=? AND version=?",
                    (org_id, document_id, version),
                )
            else:
                connection.execute(
                    """DELETE FROM rag_index_chunks
                    WHERE org_id=? AND knowledge_base_id=? AND index_version=?
                      AND document_id=? AND version=?""",
                    (org_id, knowledge_base_id, index_version, document_id, version),
                )
                connection.execute(
                    """DELETE FROM rag_index_chunks_fts
                    WHERE org_id=? AND knowledge_base_id=? AND index_version=?
                      AND document_id=? AND version=?""",
                    (org_id, knowledge_base_id, index_version, document_id, version),
                )
                connection.execute(
                    """DELETE FROM rag_index_documents
                    WHERE org_id=? AND knowledge_base_id=? AND index_version=?
                      AND document_id=? AND version=?""",
                    (org_id, knowledge_base_id, index_version, document_id, version),
                )

    def search_bm25(
        self,
        org_id: str,
        keys: set[tuple[str, str]],
        query_text: str,
        *,
        knowledge_base_id: str = LEGACY_KNOWLEDGE_BASE,
        index_version: str = LEGACY_INDEX_VERSION,
        limit: int = 200,
    ) -> dict[str, float]:
        """Run FTS5 BM25 lexical recall scoped strictly to (org, kb, index_version, authorized keys)."""
        match_expr = fts_match_query(query_text)
        if not keys or not match_expr:
            return {}
        legacy = index_version == LEGACY_INDEX_VERSION
        scores: dict[str, float] = {}
        try:
            with self._connect() as connection:
                if legacy:
                    rows = connection.execute(
                        """SELECT document_id, version, chunk_id, bm25(rag_chunks_fts) AS rank_score
                        FROM rag_chunks_fts
                        WHERE rag_chunks_fts MATCH ? AND org_id=?
                        ORDER BY rank_score LIMIT ?""",
                        (match_expr, org_id, max(limit * 2, 200)),
                    ).fetchall()
                else:
                    rows = connection.execute(
                        """SELECT document_id, version, chunk_id, bm25(rag_index_chunks_fts) AS rank_score
                        FROM rag_index_chunks_fts
                        WHERE rag_index_chunks_fts MATCH ?
                          AND org_id=? AND knowledge_base_id=? AND index_version=?
                        ORDER BY rank_score LIMIT ?""",
                        (match_expr, org_id, knowledge_base_id, index_version, max(limit * 2, 200)),
                    ).fetchall()
        except sqlite3.OperationalError:
            return {}
        for row in rows:
            if (row["document_id"], row["version"]) not in keys:
                continue
            raw = max(0.0, -float(row["rank_score"]))
            normalized = raw / (1.0 + raw) if raw > 0 else 0.0
            cid = row["chunk_id"]
            if cid not in scores or normalized > scores[cid]:
                scores[cid] = normalized
            if len(scores) >= limit:
                break
        return scores

    def _query_scoped_rows(
        self,
        connection: sqlite3.Connection,
        *,
        org_id: str,
        keys: set[tuple[str, str]],
        knowledge_base_id: str,
        index_version: str,
        include_embedding: bool,
    ) -> list[sqlite3.Row]:
        if not keys:
            return []
        ordered_keys = sorted(keys)
        legacy = index_version == LEGACY_INDEX_VERSION
        if legacy:
            columns = (
                "c.document_id, c.version, c.chunk_id, c.model, c.page, c.section, c.ordinal, c.text"
                + (", c.embedding, c.dimensions" if include_embedding else "")
            )
            table = "rag_chunks"
            scope_sql = "c.org_id=?"
            scope_params: tuple = (org_id,)
        else:
            columns = (
                "c.document_id, c.version, c.chunk_id, c.model, c.page, c.section, c.ordinal, c.text, c.truncated, c.parent_id"
                + (", c.embedding, c.dimensions" if include_embedding else "")
            )
            table = "rag_index_chunks"
            scope_sql = "c.org_id=? AND c.knowledge_base_id=? AND c.index_version=?"
            scope_params = (org_id, knowledge_base_id, index_version)
        rows: list[sqlite3.Row] = []
        batch_size = 200
        for start in range(0, len(ordered_keys), batch_size):
            batch = ordered_keys[start:start + batch_size]
            placeholders = ", ".join("(?, ?)" for _ in batch)
            params: list = []
            for doc_id, ver in batch:
                params.extend((doc_id, ver))
            params.extend(scope_params)
            rows.extend(connection.execute(
                f"""WITH wanted(document_id, version) AS (VALUES {placeholders})
                SELECT {columns}
                FROM {table} AS c
                JOIN wanted AS w ON c.document_id=w.document_id AND c.version=w.version
                WHERE {scope_sql}
                ORDER BY c.document_id, c.version, c.ordinal""",
                params,
            ).fetchall())
        if len(ordered_keys) > batch_size:
            rows.sort(key=lambda r: (r["document_id"], r["version"], r["ordinal"]))
        return rows

    def load_chunks(
        self,
        org_id: str,
        keys: set[tuple[str, str]],
        *,
        knowledge_base_id: str = LEGACY_KNOWLEDGE_BASE,
        index_version: str = LEGACY_INDEX_VERSION,
    ) -> list[tuple[Chunk, list[float]]]:
        if not keys:
            return []
        legacy = index_version == LEGACY_INDEX_VERSION
        with self._connect() as connection:
            rows = self._query_scoped_rows(
                connection,
                org_id=org_id,
                keys=keys,
                knowledge_base_id=knowledge_base_id,
                index_version=index_version,
                include_embedding=True,
            )
        return [
            (
                Chunk(
                    org_id=org_id,
                    document_id=row["document_id"],
                    version=row["version"],
                    chunk_id=row["chunk_id"],
                    model=row["model"],
                    page=row["page"],
                    section=row["section"],
                    ordinal=row["ordinal"],
                    text=row["text"],
                    truncated=False if legacy else bool(row["truncated"]),
                    parent_id=None if legacy else row["parent_id"],
                ),
                _unpack(row["embedding"], row["dimensions"]),
            )
            for row in rows
        ]

    def load_text_chunks(
        self,
        org_id: str,
        keys: set[tuple[str, str]],
        *,
        knowledge_base_id: str = LEGACY_KNOWLEDGE_BASE,
        index_version: str = LEGACY_INDEX_VERSION,
    ) -> list[Chunk]:
        if not keys:
            return []
        legacy = index_version == LEGACY_INDEX_VERSION
        with self._connect() as connection:
            rows = self._query_scoped_rows(
                connection,
                org_id=org_id,
                keys=keys,
                knowledge_base_id=knowledge_base_id,
                index_version=index_version,
                include_embedding=False,
            )
        return [
            Chunk(
                org_id=org_id,
                document_id=row["document_id"],
                version=row["version"],
                chunk_id=row["chunk_id"],
                model=row["model"],
                page=row["page"],
                section=row["section"],
                ordinal=row["ordinal"],
                text=row["text"],
                truncated=False if legacy else bool(row["truncated"]),
            )
            for row in rows
        ]

    def read_document_chunks(
        self,
        org_id: str,
        document_id: str,
        version: str,
        *,
        knowledge_base_id: str = LEGACY_KNOWLEDGE_BASE,
        index_version: str = LEGACY_INDEX_VERSION,
        offset: int = 0,
        limit: int = 20,
    ) -> tuple[int, list[Chunk]]:
        legacy = index_version == LEGACY_INDEX_VERSION
        with self._connect() as connection:
            if legacy:
                total = connection.execute(
                    "SELECT count(*) FROM rag_chunks WHERE org_id=? AND document_id=? AND version=?",
                    (org_id, document_id, version),
                ).fetchone()[0]
                rows = connection.execute(
                    """SELECT document_id, version, chunk_id, model, page, section, ordinal, text
                    FROM rag_chunks
                    WHERE org_id=? AND document_id=? AND version=?
                    ORDER BY ordinal LIMIT ? OFFSET ?""",
                    (org_id, document_id, version, limit, offset),
                ).fetchall()
            else:
                total = connection.execute(
                    """SELECT count(*) FROM rag_index_chunks
                    WHERE org_id=? AND knowledge_base_id=? AND index_version=?
                      AND document_id=? AND version=?""",
                    (org_id, knowledge_base_id, index_version, document_id, version),
                ).fetchone()[0]
                rows = connection.execute(
                    """SELECT document_id, version, chunk_id, model, page, section, ordinal, text, truncated
                    FROM rag_index_chunks
                    WHERE org_id=? AND knowledge_base_id=? AND index_version=?
                      AND document_id=? AND version=?
                    ORDER BY ordinal LIMIT ? OFFSET ?""",
                    (org_id, knowledge_base_id, index_version, document_id, version, limit, offset),
                ).fetchall()
        chunks = [
            Chunk(
                org_id=org_id,
                document_id=row["document_id"],
                version=row["version"],
                chunk_id=row["chunk_id"],
                model=row["model"],
                page=row["page"],
                section=row["section"],
                ordinal=row["ordinal"],
                text=row["text"],
                truncated=False if legacy else bool(row["truncated"]),
            )
            for row in rows
        ]
        return total, chunks

    def count_chunks(
        self,
        org_id: str,
        document_id: str,
        version: str,
        *,
        knowledge_base_id: str | None = None,
        index_version: str | None = None,
    ) -> int:
        with self._connect() as connection:
            if index_version is not None:
                if index_version == LEGACY_INDEX_VERSION:
                    return connection.execute(
                        """SELECT count(*) FROM rag_chunks
                        WHERE org_id=? AND document_id=? AND version=?""",
                        (org_id, document_id, version),
                    ).fetchone()[0]
                return connection.execute(
                    """SELECT count(*) FROM rag_index_chunks
                    WHERE org_id=? AND knowledge_base_id=? AND index_version=?
                      AND document_id=? AND version=?""",
                    (org_id, knowledge_base_id or LEGACY_KNOWLEDGE_BASE, index_version, document_id, version),
                ).fetchone()[0]
            legacy = connection.execute(
                """SELECT count(*) FROM rag_chunks
                WHERE org_id=? AND document_id=? AND version=?""",
                (org_id, document_id, version),
            ).fetchone()[0]
            current = connection.execute(
                """SELECT count(*) FROM rag_index_chunks
                WHERE org_id=? AND document_id=? AND version=?""",
                (org_id, document_id, version),
            ).fetchone()[0]
            return legacy + current

    def chunk_ids(
        self,
        org_id: str,
        document_id: str,
        version: str,
        *,
        knowledge_base_id: str = LEGACY_KNOWLEDGE_BASE,
        index_version: str = LEGACY_INDEX_VERSION,
    ) -> tuple[str, ...]:
        if index_version == LEGACY_INDEX_VERSION:
            table, extra = "rag_chunks", ""
            params = (org_id, document_id, version)
        else:
            table = "rag_index_chunks"
            extra = " AND knowledge_base_id=? AND index_version=?"
            params = (org_id, document_id, version, knowledge_base_id, index_version)
        with self._connect() as connection:
            rows = connection.execute(
                f"""SELECT chunk_id FROM {table}
                WHERE org_id=? AND document_id=? AND version=?{extra} ORDER BY ordinal""",
                params,
            ).fetchall()
        return tuple(row["chunk_id"] for row in rows)

    def document_count(
        self,
        org_id: str,
        document_id: str,
        version: str,
        *,
        knowledge_base_id: str | None = None,
        index_version: str | None = None,
    ) -> int:
        with self._connect() as connection:
            if index_version is not None:
                if index_version == LEGACY_INDEX_VERSION:
                    return connection.execute(
                        """SELECT count(*) FROM rag_documents
                        WHERE org_id=? AND document_id=? AND version=?""",
                        (org_id, document_id, version),
                    ).fetchone()[0]
                return connection.execute(
                    """SELECT count(*) FROM rag_index_documents
                    WHERE org_id=? AND knowledge_base_id=? AND index_version=?
                      AND document_id=? AND version=?""",
                    (org_id, knowledge_base_id or LEGACY_KNOWLEDGE_BASE, index_version, document_id, version),
                ).fetchone()[0]
            legacy = connection.execute(
                """SELECT count(*) FROM rag_documents
                WHERE org_id=? AND document_id=? AND version=?""",
                (org_id, document_id, version),
            ).fetchone()[0]
            current = connection.execute(
                """SELECT count(*) FROM rag_index_documents
                WHERE org_id=? AND document_id=? AND version=?""",
                (org_id, document_id, version),
            ).fetchone()[0]
            return legacy + current

    def upsert_index_version(
        self,
        org_id: str,
        knowledge_base_id: str,
        index_version: str,
        embedding_model_id: str,
        embedding_model_revision: str,
        dimensions: int,
        vector_config: dict,
        *,
        state: str = "ready",
        source_snapshot_id: str | None = None,
    ) -> None:
        timestamp = _now()
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO rag_index_versions
                (org_id, knowledge_base_id, index_version, embedding_model_id, embedding_model_revision,
                 dimensions, vector_config_json, state, source_snapshot_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (org_id, knowledge_base_id, index_version) DO UPDATE SET
                    embedding_model_id=excluded.embedding_model_id,
                    embedding_model_revision=excluded.embedding_model_revision,
                    dimensions=excluded.dimensions,
                    vector_config_json=excluded.vector_config_json,
                    state=excluded.state,
                    source_snapshot_id=COALESCE(excluded.source_snapshot_id, rag_index_versions.source_snapshot_id),
                    updated_at=excluded.updated_at""",
                (
                    org_id, knowledge_base_id, index_version, embedding_model_id, embedding_model_revision,
                    dimensions, json.dumps(vector_config, sort_keys=True, ensure_ascii=True), state,
                    source_snapshot_id, timestamp, timestamp,
                ),
            )

    def get_index_version(self, org_id: str, knowledge_base_id: str, index_version: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT * FROM rag_index_versions
                WHERE org_id=? AND knowledge_base_id=? AND index_version=?""",
                (org_id, knowledge_base_id, index_version),
            ).fetchone()
        return dict(row) if row else None

    def list_index_versions(self, org_id: str, knowledge_base_id: str) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM rag_index_versions
                WHERE org_id=? AND knowledge_base_id=? ORDER BY created_at, index_version""",
                (org_id, knowledge_base_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def active_index_version(self, org_id: str, knowledge_base_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT * FROM rag_index_versions
                WHERE org_id=? AND knowledge_base_id=? AND state='active'
                ORDER BY updated_at DESC LIMIT 1""",
                (org_id, knowledge_base_id),
            ).fetchone()
        return dict(row) if row else None

    def set_index_version_state(
        self, org_id: str, knowledge_base_id: str, index_version: str, state: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """UPDATE rag_index_versions SET state=?, updated_at=?
                WHERE org_id=? AND knowledge_base_id=? AND index_version=?""",
                (state, _now(), org_id, knowledge_base_id, index_version),
            )

    def activate_index_version(self, org_id: str, knowledge_base_id: str, index_version: str) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """UPDATE rag_index_versions SET state='superseded', updated_at=?
                WHERE org_id=? AND knowledge_base_id=? AND state='active'""",
                (_now(), org_id, knowledge_base_id),
            )
            connection.execute(
                """UPDATE rag_index_versions SET state='active', updated_at=?
                WHERE org_id=? AND knowledge_base_id=? AND index_version=?""",
                (_now(), org_id, knowledge_base_id, index_version),
            )

    def index_version_counts(self, org_id: str, knowledge_base_id: str, index_version: str) -> tuple[int, int]:
        with self._connect() as connection:
            documents = connection.execute(
                """SELECT count(*) FROM rag_index_documents
                WHERE org_id=? AND knowledge_base_id=? AND index_version=?""",
                (org_id, knowledge_base_id, index_version),
            ).fetchone()[0]
            chunks = connection.execute(
                """SELECT count(*) FROM rag_index_chunks
                WHERE org_id=? AND knowledge_base_id=? AND index_version=?""",
                (org_id, knowledge_base_id, index_version),
            ).fetchone()[0]
        return documents, chunks

    def index_version_documents(self, org_id: str, knowledge_base_id: str, index_version: str) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT document_id, version, digest, chunk_count, indexed_at
                FROM rag_index_documents
                WHERE org_id=? AND knowledge_base_id=? AND index_version=?
                ORDER BY document_id, version""",
                (org_id, knowledge_base_id, index_version),
            ).fetchall()
        return [dict(row) for row in rows]

    def index_version_document(
        self, org_id: str, knowledge_base_id: str, index_version: str, document_id: str, version: str,
    ) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT document_id, version, digest, chunk_count, indexed_at
                FROM rag_index_documents
                WHERE org_id=? AND knowledge_base_id=? AND index_version=?
                  AND document_id=? AND version=?""",
                (org_id, knowledge_base_id, index_version, document_id, version),
            ).fetchone()
        return dict(row) if row else None

    def legacy_chunk_count(self, org_id: str) -> int:
        with self._connect() as connection:
            return connection.execute(
                "SELECT count(*) FROM rag_chunks WHERE org_id=?", (org_id,),
            ).fetchone()[0]

    def legacy_counts(self, org_id: str) -> tuple[int, int]:
        with self._connect() as connection:
            documents = connection.execute(
                "SELECT count(*) FROM rag_documents WHERE org_id=?", (org_id,),
            ).fetchone()[0]
            chunks = connection.execute(
                "SELECT count(*) FROM rag_chunks WHERE org_id=?", (org_id,),
            ).fetchone()[0]
        return documents, chunks

    def legacy_documents(self, org_id: str) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT document_id, version, digest FROM rag_documents
                WHERE org_id=? ORDER BY document_id, version""",
                (org_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def activate_legacy(
        self,
        org_id: str,
        knowledge_base_id: str,
        embedding_model_id: str,
        embedding_model_revision: str,
        dimensions: int,
        vector_config: dict,
    ) -> None:
        timestamp = _now()
        config_json = json.dumps(vector_config, sort_keys=True, ensure_ascii=True)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """UPDATE rag_index_versions SET state='superseded', updated_at=?
                WHERE org_id=? AND knowledge_base_id=? AND state='active'
                  AND index_version<>'legacy'""",
                (timestamp, org_id, knowledge_base_id),
            )
            connection.execute(
                """INSERT INTO rag_index_versions
                (org_id, knowledge_base_id, index_version, embedding_model_id, embedding_model_revision,
                 dimensions, vector_config_json, state, source_snapshot_id, created_at, updated_at)
                VALUES (?, ?, 'legacy', ?, ?, ?, ?, 'active', NULL, ?, ?)
                ON CONFLICT (org_id, knowledge_base_id, index_version) DO UPDATE SET
                    embedding_model_id=excluded.embedding_model_id,
                    embedding_model_revision=excluded.embedding_model_revision,
                    dimensions=excluded.dimensions,
                    vector_config_json=excluded.vector_config_json,
                    state='active',
                    updated_at=excluded.updated_at""",
                (
                    org_id, knowledge_base_id, embedding_model_id, embedding_model_revision,
                    dimensions, config_json, timestamp, timestamp,
                ),
            )

    def create_rebuild_task(
        self,
        task_id: str,
        org_id: str,
        knowledge_base_id: str,
        target_index_version: str,
        embedding_model_id: str,
        embedding_model_revision: str,
        items: Sequence[tuple[str, str, str | None]],
        snapshot_json: str | None = None,
    ) -> None:
        timestamp = _now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """INSERT INTO rag_rebuild_tasks
                (task_id, org_id, knowledge_base_id, target_index_version, embedding_model_id,
                 embedding_model_revision, state, total, completed, failed, snapshot_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, 0, 0, ?, ?, ?)""",
                (
                    task_id, org_id, knowledge_base_id, target_index_version, embedding_model_id,
                    embedding_model_revision, len(items), snapshot_json, timestamp, timestamp,
                ),
            )
            connection.executemany(
                """INSERT INTO rag_rebuild_items
                (task_id, document_id, version, digest, state, attempts, updated_at)
                VALUES (?, ?, ?, ?, 'pending', 0, ?)
                ON CONFLICT (task_id, document_id, version) DO NOTHING""",
                [(task_id, document_id, version, digest, timestamp) for document_id, version, digest in items],
            )

    def get_rebuild_task(self, task_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM rag_rebuild_tasks WHERE task_id=?", (task_id,),
            ).fetchone()
        return dict(row) if row else None

    def active_rebuild_task(self, org_id: str, knowledge_base_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT * FROM rag_rebuild_tasks
                WHERE org_id=? AND knowledge_base_id=? AND state IN ('pending', 'running')
                ORDER BY created_at DESC LIMIT 1""",
                (org_id, knowledge_base_id),
            ).fetchone()
        return dict(row) if row else None

    def list_rebuild_tasks(self, org_id: str, knowledge_base_id: str, *, limit: int = 10) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM rag_rebuild_tasks
                WHERE org_id=? AND knowledge_base_id=?
                ORDER BY created_at DESC LIMIT ?""",
                (org_id, knowledge_base_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def rebuild_items(self, task_id: str) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT document_id, version, digest, state, attempts, last_error
                FROM rag_rebuild_items WHERE task_id=? ORDER BY document_id, version""",
                (task_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def set_rebuild_task_state(
        self,
        task_id: str,
        state: str,
        *,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """UPDATE rag_rebuild_tasks
                SET state=?, error_code=?, error_message=?, updated_at=?
                WHERE task_id=?""",
                (state, error_code, error_message, _now(), task_id),
            )

    def refresh_rebuild_counts(self, task_id: str) -> None:
        with self._connect() as connection:
            completed = connection.execute(
                """SELECT count(*) FROM rag_rebuild_items
                WHERE task_id=? AND state IN ('succeeded', 'skipped')""",
                (task_id,),
            ).fetchone()[0]
            failed = connection.execute(
                """SELECT count(*) FROM rag_rebuild_items
                WHERE task_id=? AND state='failed'""",
                (task_id,),
            ).fetchone()[0]
            connection.execute(
                """UPDATE rag_rebuild_tasks
                SET completed=?, failed=?, updated_at=? WHERE task_id=?""",
                (completed, failed, _now(), task_id),
            )

    def set_rebuild_item_state(
        self,
        task_id: str,
        document_id: str,
        version: str,
        state: str,
        *,
        last_error: str | None = None,
        increment_attempts: bool = False,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """UPDATE rag_rebuild_items
                SET state=?, attempts=attempts + ?, last_error=?, updated_at=?
                WHERE task_id=? AND document_id=? AND version=?""",
                (state, 1 if increment_attempts else 0, last_error, _now(), task_id, document_id, version),
            )

    def reset_rebuild_items(self, task_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """UPDATE rag_rebuild_items
                SET state='pending', last_error=NULL, updated_at=?
                WHERE task_id=? AND state IN ('failed', 'running', 'pending')""",
                (_now(), task_id),
            )

    def interrupt_running_rebuilds(self) -> list[str]:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            task_ids = [
                row["task_id"] for row in connection.execute(
                    "SELECT task_id FROM rag_rebuild_tasks WHERE state='running'",
                )
            ]
            connection.execute(
                """UPDATE rag_rebuild_tasks SET state='interrupted', updated_at=?
                WHERE state='running'""",
                (_now(),),
            )
            connection.execute(
                """UPDATE rag_rebuild_items SET state='pending', updated_at=?
                WHERE state='running'""",
                (_now(),),
            )
        return task_ids

    def deletable_index_versions(self, org_id: str, knowledge_base_id: str) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM rag_index_versions
                WHERE org_id=? AND knowledge_base_id=? AND state IN ('superseded', 'retired')
                ORDER BY created_at""",
                (org_id, knowledge_base_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def purge_retired_versions(
        self,
        org_id: str,
        knowledge_base_id: str,
        *,
        keep_superseded: int = 1,
    ) -> tuple[tuple[str, ...], int, int]:
        """Physically delete chunks, FTS entries, documents, and metadata for retired/old superseded indexes."""
        keep = max(0, int(keep_superseded))
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            busy_targets = {
                row["target_index_version"]
                for row in connection.execute(
                    """SELECT target_index_version FROM rag_rebuild_tasks
                    WHERE org_id=? AND knowledge_base_id=? AND state IN ('pending', 'running')""",
                    (org_id, knowledge_base_id),
                ).fetchall()
            }
            superseded = connection.execute(
                """SELECT index_version FROM rag_index_versions
                WHERE org_id=? AND knowledge_base_id=? AND state='superseded' AND index_version<>'legacy'
                ORDER BY updated_at DESC, index_version DESC""",
                (org_id, knowledge_base_id),
            ).fetchall()
            protected_superseded = {row["index_version"] for row in superseded[:keep]}
            candidates = connection.execute(
                """SELECT index_version, state FROM rag_index_versions
                WHERE org_id=? AND knowledge_base_id=?
                  AND state IN ('retired', 'superseded', 'failed')
                  AND index_version<>'legacy'
                ORDER BY created_at, index_version""",
                (org_id, knowledge_base_id),
            ).fetchall()
            to_purge = [
                row["index_version"]
                for row in candidates
                if row["index_version"] not in busy_targets
                and not (row["state"] == "superseded" and row["index_version"] in protected_superseded)
            ]
            if not to_purge:
                return (), 0, 0
            purged_docs = 0
            purged_chunks = 0
            for ver in to_purge:
                doc_cnt = connection.execute(
                    """SELECT count(*) FROM rag_index_documents
                    WHERE org_id=? AND knowledge_base_id=? AND index_version=?""",
                    (org_id, knowledge_base_id, ver),
                ).fetchone()[0]
                chunk_cnt = connection.execute(
                    """SELECT count(*) FROM rag_index_chunks
                    WHERE org_id=? AND knowledge_base_id=? AND index_version=?""",
                    (org_id, knowledge_base_id, ver),
                ).fetchone()[0]
                purged_docs += int(doc_cnt)
                purged_chunks += int(chunk_cnt)
                connection.execute(
                    """DELETE FROM rag_index_chunks
                    WHERE org_id=? AND knowledge_base_id=? AND index_version=?""",
                    (org_id, knowledge_base_id, ver),
                )
                connection.execute(
                    """DELETE FROM rag_index_chunks_fts
                    WHERE org_id=? AND knowledge_base_id=? AND index_version=?""",
                    (org_id, knowledge_base_id, ver),
                )
                connection.execute(
                    """DELETE FROM rag_index_documents
                    WHERE org_id=? AND knowledge_base_id=? AND index_version=?""",
                    (org_id, knowledge_base_id, ver),
                )
                connection.execute(
                    """DELETE FROM rag_index_versions
                    WHERE org_id=? AND knowledge_base_id=? AND index_version=?""",
                    (org_id, knowledge_base_id, ver),
                )
            return tuple(to_purge), purged_docs, purged_chunks