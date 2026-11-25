"""PostgreSQL + pgvector (HNSW) + tsvector/pg_trgm hybrid RAG index store."""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from app.contracts.rag_workspace import RagUnavailable

from .models import Chunk
from .store import (
    LEGACY_EMBEDDING_MODEL,
    LEGACY_INDEX_VERSION,
    LEGACY_KNOWLEDGE_BASE,
    LEGACY_MODEL_REVISION,
    lexical_document,
    lexical_tokens,
)

PG_SCHEMA_STATEMENTS: tuple[str, ...] = (
    "CREATE EXTENSION IF NOT EXISTS vector",
    "CREATE EXTENSION IF NOT EXISTS pg_trgm",
    """
    CREATE TABLE IF NOT EXISTS rag_documents (
        org_id TEXT NOT NULL,
        document_id TEXT NOT NULL,
        version TEXT NOT NULL,
        digest TEXT NOT NULL,
        chunk_count INTEGER NOT NULL,
        indexed_at TEXT NOT NULL,
        PRIMARY KEY (org_id, document_id, version)
    )
    """,
    """
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
        lexical_text TEXT NOT NULL DEFAULT '',
        lexical_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', lexical_text)) STORED,
        embedding vector NOT NULL,
        dimensions INTEGER NOT NULL,
        PRIMARY KEY (org_id, document_id, version, chunk_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_pg_rag_chunks_scope ON rag_chunks (org_id, document_id, version)",
    "CREATE INDEX IF NOT EXISTS idx_pg_rag_chunks_tsv ON rag_chunks USING gin (lexical_tsv)",
    "CREATE INDEX IF NOT EXISTS idx_pg_rag_chunks_trgm ON rag_chunks USING gin (lexical_text gin_trgm_ops)",
    """
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
    )
    """,
    """
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
    )
    """,
    """
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
        lexical_text TEXT NOT NULL DEFAULT '',
        lexical_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', lexical_text)) STORED,
        embedding vector NOT NULL,
        dimensions INTEGER NOT NULL,
        truncated BOOLEAN NOT NULL DEFAULT FALSE,
        parent_id TEXT,
        PRIMARY KEY (org_id, knowledge_base_id, index_version, document_id, version, chunk_id)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_pg_rag_index_chunks_scope
        ON rag_index_chunks (org_id, knowledge_base_id, index_version, document_id, version)
    """,
    "CREATE INDEX IF NOT EXISTS idx_pg_rag_index_chunks_tsv ON rag_index_chunks USING gin (lexical_tsv)",
    "CREATE INDEX IF NOT EXISTS idx_pg_rag_index_chunks_trgm ON rag_index_chunks USING gin (lexical_text gin_trgm_ops)",
    """
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
    )
    """,
    """
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
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_pg_rag_rebuild_tasks_state ON rag_rebuild_tasks (state, updated_at)",
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _safe_dim(dimensions: int) -> int:
    dim = int(dimensions)
    if not 1 <= dim <= 4096:
        raise ValueError(f"Unsupported embedding dimension: {dim}")
    return dim


def format_pgvector(vector: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(v):.8g}" for v in vector) + "]"


def parse_pgvector(raw: Any, dimensions: int) -> list[float]:
    if isinstance(raw, (list, tuple)):
        vec = [float(v) for v in raw]
    else:
        text = str(raw).strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        vec = [float(item) for item in text.split(",") if item.strip()]
    if len(vec) != dimensions:
        raise ValueError("stored pgvector dimensions do not match")
    return vec


def build_tsquery_or(query_text: str, *, max_terms: int = 24) -> tuple[str, str]:
    """Build a safe PostgreSQL tsquery ('a | b | c') and normalized trigram string."""
    raw = lexical_tokens(query_text)
    if not raw:
        return "", ""
    multi = list(dict.fromkeys(tok for tok in raw if len(tok) >= 2))
    single = list(dict.fromkeys(tok for tok in raw if len(tok) == 1))
    selected = (multi if multi else single)[:max_terms]
    safe = [re.sub(r"[^a-z0-9_\u4e00-\u9fff]", "", tok) for tok in selected]
    safe = [tok for tok in safe if tok]
    if not safe:
        return "", ""
    return " | ".join(safe), " ".join(safe)


def _default_psycopg_factory(dsn: str) -> Callable[[], Any]:
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise RagUnavailable(
            "启用 PostgreSQL + pgvector 存储需要安装 psycopg[binary]（pip install 'salespilot-ai-backend[enterprise]'）。"
        ) from exc

    def open_conn():
        return psycopg.connect(dsn, row_factory=dict_row)

    return open_conn


class PgVectorRagStore:
    """Enterprise RAG store backed by PostgreSQL + pgvector (HNSW) + tsvector/pg_trgm."""

    def __init__(
        self,
        dsn: str | None = None,
        *,
        conn_factory: Callable[[], Any] | None = None,
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 64,
    ):
        if conn_factory is None:
            if not dsn:
                raise ValueError("PgVectorRagStore requires a postgres dsn or conn_factory")
            conn_factory = _default_psycopg_factory(dsn)
        self._conn_factory = conn_factory
        self.hnsw_m = max(4, min(64, int(hnsw_m)))
        self.hnsw_ef_construction = max(16, min(512, int(hnsw_ef_construction)))
        self._ensured_dims: set[int] = set()
        self.initialize()

    @contextmanager
    def _connect(self) -> Iterator[Any]:
        conn = self._conn_factory()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize(self) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            for stmt in PG_SCHEMA_STATEMENTS:
                cur.execute(stmt)

    def _ensure_hnsw_index(self, cur: Any, dimensions: int, *, legacy: bool = False) -> None:
        dim = _safe_dim(dimensions)
        if dim > 2000 or (not legacy and dim in self._ensured_dims):
            return
        table = "rag_chunks" if legacy else "rag_index_chunks"
        idx_name = f"idx_pg_{table}_hnsw_d{dim}"
        cur.execute(
            f"""CREATE INDEX IF NOT EXISTS {idx_name}
            ON {table} USING hnsw ((embedding::vector({dim})) vector_cosine_ops)
            WITH (m = {self.hnsw_m}, ef_construction = {self.hnsw_ef_construction})
            WHERE dimensions = {dim}"""
        )
        if not legacy:
            self._ensured_dims.add(dim)

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
        dims = {len(vector) for vector in vectors}
        if len(dims) != 1 or 0 in dims:
            raise ValueError("indexed vectors must share one non-zero dimension")
        dimension = _safe_dim(dims.pop())
        if index_version == LEGACY_INDEX_VERSION:
            self._replace_legacy(org_id, document_id, version, digest, chunks, vectors, dimension)
            return
        config_json = json.dumps(vector_config or {}, sort_keys=True, ensure_ascii=True)
        with self._connect() as conn, conn.cursor() as cur:
            self._ensure_hnsw_index(cur, dimension, legacy=False)
            cur.execute(
                """DELETE FROM rag_index_chunks
                WHERE org_id=%s AND knowledge_base_id=%s AND index_version=%s
                  AND document_id=%s AND version=%s""",
                (org_id, knowledge_base_id, index_version, document_id, version),
            )
            cur.executemany(
                """INSERT INTO rag_index_chunks
                (org_id, knowledge_base_id, index_version, document_id, version, chunk_id, model,
                 page, section, ordinal, text, lexical_text, embedding, dimensions, truncated, parent_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector, %s, %s, %s)""",
                [
                    (
                        chunk.org_id, knowledge_base_id, index_version, chunk.document_id, chunk.version,
                        chunk.chunk_id, chunk.model, chunk.page, chunk.section, chunk.ordinal, chunk.text,
                        lexical_document(chunk.text, chunk.section, chunk.model),
                        format_pgvector(vector), dimension, bool(chunk.truncated), chunk.parent_id,
                    )
                    for chunk, vector in zip(chunks, vectors, strict=True)
                ],
            )
            cur.execute(
                """INSERT INTO rag_index_documents
                (org_id, knowledge_base_id, index_version, document_id, version, digest, chunk_count,
                 embedding_model_id, embedding_model_revision, dimensions, vector_config_json, indexed_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (org_id, knowledge_base_id, index_version, document_id, version) DO UPDATE SET
                    digest=EXCLUDED.digest,
                    chunk_count=EXCLUDED.chunk_count,
                    embedding_model_id=EXCLUDED.embedding_model_id,
                    embedding_model_revision=EXCLUDED.embedding_model_revision,
                    dimensions=EXCLUDED.dimensions,
                    vector_config_json=EXCLUDED.vector_config_json,
                    indexed_at=EXCLUDED.indexed_at""",
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
        dimension: int,
    ) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            self._ensure_hnsw_index(cur, dimension, legacy=True)
            cur.execute(
                "DELETE FROM rag_chunks WHERE org_id=%s AND document_id=%s AND version=%s",
                (org_id, document_id, version),
            )
            cur.executemany(
                """INSERT INTO rag_chunks
                (org_id, document_id, version, chunk_id, model, page, section, ordinal, text, lexical_text, embedding, dimensions)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector, %s)""",
                [
                    (
                        chunk.org_id, chunk.document_id, chunk.version, chunk.chunk_id, chunk.model,
                        chunk.page, chunk.section, chunk.ordinal, chunk.text,
                        lexical_document(chunk.text, chunk.section, chunk.model),
                        format_pgvector(vector), dimension,
                    )
                    for chunk, vector in zip(chunks, vectors, strict=True)
                ],
            )
            cur.execute(
                """INSERT INTO rag_documents
                (org_id, document_id, version, digest, chunk_count, indexed_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (org_id, document_id, version) DO UPDATE SET
                    digest=EXCLUDED.digest,
                    chunk_count=EXCLUDED.chunk_count,
                    indexed_at=EXCLUDED.indexed_at""",
                (org_id, document_id, version, digest, len(chunks), _now()),
            )

    def remove_document(self, org_id: str, document_id: str, version: str) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            for table in ("rag_chunks", "rag_documents", "rag_index_chunks", "rag_index_documents"):
                cur.execute(
                    f"DELETE FROM {table} WHERE org_id=%s AND document_id=%s AND version=%s",
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
        with self._connect() as conn, conn.cursor() as cur:
            if index_version == LEGACY_INDEX_VERSION:
                for table in ("rag_chunks", "rag_documents"):
                    cur.execute(
                        f"DELETE FROM {table} WHERE org_id=%s AND document_id=%s AND version=%s",
                        (org_id, document_id, version),
                    )
            else:
                for table in ("rag_index_chunks", "rag_index_documents"):
                    cur.execute(
                        f"""DELETE FROM {table}
                        WHERE org_id=%s AND knowledge_base_id=%s AND index_version=%s
                          AND document_id=%s AND version=%s""",
                        (org_id, knowledge_base_id, index_version, document_id, version),
                    )

    def search_vector(
        self,
        org_id: str,
        keys: set[tuple[str, str]],
        query_vector: Sequence[float],
        *,
        knowledge_base_id: str = LEGACY_KNOWLEDGE_BASE,
        index_version: str = LEGACY_INDEX_VERSION,
        limit: int = 200,
    ) -> list[tuple[Chunk, float]]:
        """Server-side pgvector ANN cosine recall scoped strictly to authorized (doc_id, version) keys."""
        if not keys or not query_vector:
            return []
        dim = _safe_dim(len(query_vector))
        vec_literal = format_pgvector(query_vector)
        ordered_keys = sorted(keys)
        values_sql = ", ".join("(%s, %s)" for _ in ordered_keys)
        key_params: list[Any] = []
        for doc_id, ver in ordered_keys:
            key_params.extend((doc_id, ver))
        legacy = index_version == LEGACY_INDEX_VERSION
        with self._connect() as conn, conn.cursor() as cur:
            if legacy:
                sql = f"""
                    WITH wanted(document_id, version) AS (VALUES {values_sql})
                    SELECT c.document_id, c.version, c.chunk_id, c.model, c.page, c.section, c.ordinal, c.text,
                           1.0 - (c.embedding::vector({dim}) <=> %s::vector({dim})) AS cosine_score
                    FROM rag_chunks AS c
                    JOIN wanted AS w ON c.document_id = w.document_id AND c.version = w.version
                    WHERE c.org_id = %s AND c.dimensions = %s
                    ORDER BY c.embedding::vector({dim}) <=> %s::vector({dim}), c.chunk_id
                    LIMIT %s
                """
                params = [*key_params, vec_literal, org_id, dim, vec_literal, max(1, int(limit))]
            else:
                sql = f"""
                    WITH wanted(document_id, version) AS (VALUES {values_sql})
                    SELECT c.document_id, c.version, c.chunk_id, c.model, c.page, c.section, c.ordinal, c.text,
                           c.truncated, c.parent_id,
                           1.0 - (c.embedding::vector({dim}) <=> %s::vector({dim})) AS cosine_score
                    FROM rag_index_chunks AS c
                    JOIN wanted AS w ON c.document_id = w.document_id AND c.version = w.version
                    WHERE c.org_id = %s AND c.knowledge_base_id = %s AND c.index_version = %s AND c.dimensions = %s
                    ORDER BY c.embedding::vector({dim}) <=> %s::vector({dim}), c.chunk_id
                    LIMIT %s
                """
                params = [
                    *key_params, vec_literal, org_id, knowledge_base_id, index_version, dim,
                    vec_literal, max(1, int(limit)),
                ]
            cur.execute(sql, params)
            rows = cur.fetchall()
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
                max(-1.0, min(1.0, float(row["cosine_score"]))),
            )
            for row in rows
        ]

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
        """PostgreSQL full-text (ts_rank_cd) + pg_trgm (word_similarity) lexical recall."""
        tsquery_str, trgm_str = build_tsquery_or(query_text)
        if not keys or not tsquery_str:
            return {}
        ordered_keys = sorted(keys)
        values_sql = ", ".join("(%s, %s)" for _ in ordered_keys)
        key_params: list[Any] = []
        for doc_id, ver in ordered_keys:
            key_params.extend((doc_id, ver))
        legacy = index_version == LEGACY_INDEX_VERSION
        with self._connect() as conn, conn.cursor() as cur:
            if legacy:
                sql = f"""
                    WITH wanted(document_id, version) AS (VALUES {values_sql})
                    SELECT c.chunk_id,
                           ts_rank_cd(c.lexical_tsv, to_tsquery('simple', %s))
                           + 0.35 * word_similarity(%s, c.lexical_text) AS rank_score
                    FROM rag_chunks AS c
                    JOIN wanted AS w ON c.document_id = w.document_id AND c.version = w.version
                    WHERE c.org_id = %s
                      AND (c.lexical_tsv @@ to_tsquery('simple', %s) OR %s <%% c.lexical_text)
                    ORDER BY rank_score DESC
                    LIMIT %s
                """
                params = [*key_params, tsquery_str, trgm_str, org_id, tsquery_str, trgm_str, max(1, int(limit))]
            else:
                sql = f"""
                    WITH wanted(document_id, version) AS (VALUES {values_sql})
                    SELECT c.chunk_id,
                           ts_rank_cd(c.lexical_tsv, to_tsquery('simple', %s))
                           + 0.35 * word_similarity(%s, c.lexical_text) AS rank_score
                    FROM rag_index_chunks AS c
                    JOIN wanted AS w ON c.document_id = w.document_id AND c.version = w.version
                    WHERE c.org_id = %s AND c.knowledge_base_id = %s AND c.index_version = %s
                      AND (c.lexical_tsv @@ to_tsquery('simple', %s) OR %s <%% c.lexical_text)
                    ORDER BY rank_score DESC
                    LIMIT %s
                """
                params = [
                    *key_params, tsquery_str, trgm_str, org_id, knowledge_base_id, index_version,
                    tsquery_str, trgm_str, max(1, int(limit)),
                ]
            cur.execute(sql, params)
            rows = cur.fetchall()
        scores: dict[str, float] = {}
        for row in rows:
            raw = max(0.0, float(row["rank_score"]))
            if raw <= 0.0:
                continue
            scores[row["chunk_id"]] = raw / (1.0 + raw)
        return scores

    def load_chunks_by_ids(
        self,
        org_id: str,
        chunk_ids: set[str],
        *,
        knowledge_base_id: str = LEGACY_KNOWLEDGE_BASE,
        index_version: str = LEGACY_INDEX_VERSION,
    ) -> list[Chunk]:
        if not chunk_ids:
            return []
        ordered = sorted(chunk_ids)
        placeholders = ", ".join("%s" for _ in ordered)
        legacy = index_version == LEGACY_INDEX_VERSION
        with self._connect() as conn, conn.cursor() as cur:
            if legacy:
                cur.execute(
                    f"""SELECT document_id, version, chunk_id, model, page, section, ordinal, text
                    FROM rag_chunks WHERE org_id=%s AND chunk_id IN ({placeholders})""",
                    [org_id, *ordered],
                )
            else:
                cur.execute(
                    f"""SELECT document_id, version, chunk_id, model, page, section, ordinal, text, truncated, parent_id
                    FROM rag_index_chunks
                    WHERE org_id=%s AND knowledge_base_id=%s AND index_version=%s
                      AND chunk_id IN ({placeholders})""",
                    [org_id, knowledge_base_id, index_version, *ordered],
                )
            rows = cur.fetchall()
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
                parent_id=None if legacy else row["parent_id"],
            )
            for row in rows
        ]

    def load_parent_siblings(
        self,
        org_id: str,
        parents: set[tuple[str, str, str]],
        *,
        knowledge_base_id: str = LEGACY_KNOWLEDGE_BASE,
        index_version: str = LEGACY_INDEX_VERSION,
    ) -> list[Chunk]:
        if not parents or index_version == LEGACY_INDEX_VERSION:
            return []
        ordered = sorted(parents)
        values_sql = ", ".join("(%s, %s, %s)" for _ in ordered)
        params: list[Any] = []
        for doc_id, ver, parent_id in ordered:
            params.extend((doc_id, ver, parent_id))
        params.extend((org_id, knowledge_base_id, index_version))
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""WITH wanted(document_id, version, parent_id) AS (VALUES {values_sql})
                SELECT c.document_id, c.version, c.chunk_id, c.model, c.page, c.section, c.ordinal,
                       c.text, c.truncated, c.parent_id
                FROM rag_index_chunks AS c
                JOIN wanted AS w
                  ON c.document_id = w.document_id AND c.version = w.version AND c.parent_id = w.parent_id
                WHERE c.org_id = %s AND c.knowledge_base_id = %s AND c.index_version = %s
                ORDER BY c.document_id, c.version, c.ordinal""",
                params,
            )
            rows = cur.fetchall()
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
                truncated=bool(row["truncated"]),
                parent_id=row["parent_id"],
            )
            for row in rows
        ]

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
        ordered = sorted(keys)
        values_sql = ", ".join("(%s, %s)" for _ in ordered)
        params: list[Any] = []
        for doc_id, ver in ordered:
            params.extend((doc_id, ver))
        legacy = index_version == LEGACY_INDEX_VERSION
        with self._connect() as conn, conn.cursor() as cur:
            if legacy:
                cur.execute(
                    f"""WITH wanted(document_id, version) AS (VALUES {values_sql})
                    SELECT c.document_id, c.version, c.chunk_id, c.model, c.page, c.section, c.ordinal,
                           c.text, c.embedding::text AS embedding_text, c.dimensions
                    FROM rag_chunks AS c
                    JOIN wanted AS w ON c.document_id=w.document_id AND c.version=w.version
                    WHERE c.org_id=%s ORDER BY c.document_id, c.version, c.ordinal""",
                    [*params, org_id],
                )
            else:
                cur.execute(
                    f"""WITH wanted(document_id, version) AS (VALUES {values_sql})
                    SELECT c.document_id, c.version, c.chunk_id, c.model, c.page, c.section, c.ordinal,
                           c.text, c.truncated, c.parent_id, c.embedding::text AS embedding_text, c.dimensions
                    FROM rag_index_chunks AS c
                    JOIN wanted AS w ON c.document_id=w.document_id AND c.version=w.version
                    WHERE c.org_id=%s AND c.knowledge_base_id=%s AND c.index_version=%s
                    ORDER BY c.document_id, c.version, c.ordinal""",
                    [*params, org_id, knowledge_base_id, index_version],
                )
            rows = cur.fetchall()
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
                parse_pgvector(row["embedding_text"], int(row["dimensions"])),
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
        ordered = sorted(keys)
        values_sql = ", ".join("(%s, %s)" for _ in ordered)
        params: list[Any] = []
        for doc_id, ver in ordered:
            params.extend((doc_id, ver))
        legacy = index_version == LEGACY_INDEX_VERSION
        with self._connect() as conn, conn.cursor() as cur:
            if legacy:
                cur.execute(
                    f"""WITH wanted(document_id, version) AS (VALUES {values_sql})
                    SELECT c.document_id, c.version, c.chunk_id, c.model, c.page, c.section, c.ordinal, c.text
                    FROM rag_chunks AS c
                    JOIN wanted AS w ON c.document_id=w.document_id AND c.version=w.version
                    WHERE c.org_id=%s ORDER BY c.document_id, c.version, c.ordinal""",
                    [*params, org_id],
                )
            else:
                cur.execute(
                    f"""WITH wanted(document_id, version) AS (VALUES {values_sql})
                    SELECT c.document_id, c.version, c.chunk_id, c.model, c.page, c.section, c.ordinal,
                           c.text, c.truncated, c.parent_id
                    FROM rag_index_chunks AS c
                    JOIN wanted AS w ON c.document_id=w.document_id AND c.version=w.version
                    WHERE c.org_id=%s AND c.knowledge_base_id=%s AND c.index_version=%s
                    ORDER BY c.document_id, c.version, c.ordinal""",
                    [*params, org_id, knowledge_base_id, index_version],
                )
            rows = cur.fetchall()
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
                parent_id=None if legacy else row["parent_id"],
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
        with self._connect() as conn, conn.cursor() as cur:
            if legacy:
                cur.execute(
                    "SELECT count(*) AS cnt FROM rag_chunks WHERE org_id=%s AND document_id=%s AND version=%s",
                    (org_id, document_id, version),
                )
                total = int(cur.fetchone()["cnt"])
                cur.execute(
                    """SELECT document_id, version, chunk_id, model, page, section, ordinal, text
                    FROM rag_chunks
                    WHERE org_id=%s AND document_id=%s AND version=%s
                    ORDER BY ordinal LIMIT %s OFFSET %s""",
                    (org_id, document_id, version, limit, offset),
                )
            else:
                cur.execute(
                    """SELECT count(*) AS cnt FROM rag_index_chunks
                    WHERE org_id=%s AND knowledge_base_id=%s AND index_version=%s
                      AND document_id=%s AND version=%s""",
                    (org_id, knowledge_base_id, index_version, document_id, version),
                )
                total = int(cur.fetchone()["cnt"])
                cur.execute(
                    """SELECT document_id, version, chunk_id, model, page, section, ordinal, text, truncated
                    FROM rag_index_chunks
                    WHERE org_id=%s AND knowledge_base_id=%s AND index_version=%s
                      AND document_id=%s AND version=%s
                    ORDER BY ordinal LIMIT %s OFFSET %s""",
                    (org_id, knowledge_base_id, index_version, document_id, version, limit, offset),
                )
            rows = cur.fetchall()
        return total, [
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

    def count_chunks(
        self,
        org_id: str,
        document_id: str,
        version: str,
        *,
        knowledge_base_id: str | None = None,
        index_version: str | None = None,
    ) -> int:
        with self._connect() as conn, conn.cursor() as cur:
            if index_version is not None:
                if index_version == LEGACY_INDEX_VERSION:
                    cur.execute(
                        "SELECT count(*) AS cnt FROM rag_chunks WHERE org_id=%s AND document_id=%s AND version=%s",
                        (org_id, document_id, version),
                    )
                    return int(cur.fetchone()["cnt"])
                cur.execute(
                    """SELECT count(*) AS cnt FROM rag_index_chunks
                    WHERE org_id=%s AND knowledge_base_id=%s AND index_version=%s
                      AND document_id=%s AND version=%s""",
                    (org_id, knowledge_base_id or LEGACY_KNOWLEDGE_BASE, index_version, document_id, version),
                )
                return int(cur.fetchone()["cnt"])
            cur.execute(
                "SELECT count(*) AS cnt FROM rag_chunks WHERE org_id=%s AND document_id=%s AND version=%s",
                (org_id, document_id, version),
            )
            legacy_cnt = int(cur.fetchone()["cnt"])
            cur.execute(
                "SELECT count(*) AS cnt FROM rag_index_chunks WHERE org_id=%s AND document_id=%s AND version=%s",
                (org_id, document_id, version),
            )
            return legacy_cnt + int(cur.fetchone()["cnt"])

    def chunk_ids(
        self,
        org_id: str,
        document_id: str,
        version: str,
        *,
        knowledge_base_id: str = LEGACY_KNOWLEDGE_BASE,
        index_version: str = LEGACY_INDEX_VERSION,
    ) -> tuple[str, ...]:
        with self._connect() as conn, conn.cursor() as cur:
            if index_version == LEGACY_INDEX_VERSION:
                cur.execute(
                    """SELECT chunk_id FROM rag_chunks
                    WHERE org_id=%s AND document_id=%s AND version=%s ORDER BY ordinal""",
                    (org_id, document_id, version),
                )
            else:
                cur.execute(
                    """SELECT chunk_id FROM rag_index_chunks
                    WHERE org_id=%s AND document_id=%s AND version=%s
                      AND knowledge_base_id=%s AND index_version=%s ORDER BY ordinal""",
                    (org_id, document_id, version, knowledge_base_id, index_version),
                )
            return tuple(row["chunk_id"] for row in cur.fetchall())

    def document_count(
        self,
        org_id: str,
        document_id: str,
        version: str,
        *,
        knowledge_base_id: str | None = None,
        index_version: str | None = None,
    ) -> int:
        with self._connect() as conn, conn.cursor() as cur:
            if index_version is not None:
                if index_version == LEGACY_INDEX_VERSION:
                    cur.execute(
                        "SELECT count(*) AS cnt FROM rag_documents WHERE org_id=%s AND document_id=%s AND version=%s",
                        (org_id, document_id, version),
                    )
                    return int(cur.fetchone()["cnt"])
                cur.execute(
                    """SELECT count(*) AS cnt FROM rag_index_documents
                    WHERE org_id=%s AND knowledge_base_id=%s AND index_version=%s
                      AND document_id=%s AND version=%s""",
                    (org_id, knowledge_base_id or LEGACY_KNOWLEDGE_BASE, index_version, document_id, version),
                )
                return int(cur.fetchone()["cnt"])
            cur.execute(
                "SELECT count(*) AS cnt FROM rag_documents WHERE org_id=%s AND document_id=%s AND version=%s",
                (org_id, document_id, version),
            )
            legacy_cnt = int(cur.fetchone()["cnt"])
            cur.execute(
                "SELECT count(*) AS cnt FROM rag_index_documents WHERE org_id=%s AND document_id=%s AND version=%s",
                (org_id, document_id, version),
            )
            return legacy_cnt + int(cur.fetchone()["cnt"])

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
        dim = _safe_dim(dimensions)
        with self._connect() as conn, conn.cursor() as cur:
            self._ensure_hnsw_index(cur, dim, legacy=(index_version == LEGACY_INDEX_VERSION))
            cur.execute(
                """INSERT INTO rag_index_versions
                (org_id, knowledge_base_id, index_version, embedding_model_id, embedding_model_revision,
                 dimensions, vector_config_json, state, source_snapshot_id, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (org_id, knowledge_base_id, index_version) DO UPDATE SET
                    embedding_model_id=EXCLUDED.embedding_model_id,
                    embedding_model_revision=EXCLUDED.embedding_model_revision,
                    dimensions=EXCLUDED.dimensions,
                    vector_config_json=EXCLUDED.vector_config_json,
                    state=EXCLUDED.state,
                    source_snapshot_id=COALESCE(EXCLUDED.source_snapshot_id, rag_index_versions.source_snapshot_id),
                    updated_at=EXCLUDED.updated_at""",
                (
                    org_id, knowledge_base_id, index_version, embedding_model_id, embedding_model_revision,
                    dim, json.dumps(vector_config, sort_keys=True, ensure_ascii=True), state,
                    source_snapshot_id, timestamp, timestamp,
                ),
            )

    def get_index_version(self, org_id: str, knowledge_base_id: str, index_version: str) -> dict | None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT * FROM rag_index_versions
                WHERE org_id=%s AND knowledge_base_id=%s AND index_version=%s""",
                (org_id, knowledge_base_id, index_version),
            )
            row = cur.fetchone()
        return dict(row) if row else None

    def list_index_versions(self, org_id: str, knowledge_base_id: str) -> list[dict]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT * FROM rag_index_versions
                WHERE org_id=%s AND knowledge_base_id=%s ORDER BY created_at, index_version""",
                (org_id, knowledge_base_id),
            )
            return [dict(row) for row in cur.fetchall()]

    def active_index_version(self, org_id: str, knowledge_base_id: str) -> dict | None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT * FROM rag_index_versions
                WHERE org_id=%s AND knowledge_base_id=%s AND state='active'
                ORDER BY updated_at DESC LIMIT 1""",
                (org_id, knowledge_base_id),
            )
            row = cur.fetchone()
        return dict(row) if row else None

    def set_index_version_state(
        self, org_id: str, knowledge_base_id: str, index_version: str, state: str,
    ) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """UPDATE rag_index_versions SET state=%s, updated_at=%s
                WHERE org_id=%s AND knowledge_base_id=%s AND index_version=%s""",
                (state, _now(), org_id, knowledge_base_id, index_version),
            )

    def activate_index_version(self, org_id: str, knowledge_base_id: str, index_version: str) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """UPDATE rag_index_versions SET state='superseded', updated_at=%s
                WHERE org_id=%s AND knowledge_base_id=%s AND state='active'""",
                (_now(), org_id, knowledge_base_id),
            )
            cur.execute(
                """UPDATE rag_index_versions SET state='active', updated_at=%s
                WHERE org_id=%s AND knowledge_base_id=%s AND index_version=%s""",
                (_now(), org_id, knowledge_base_id, index_version),
            )

    def index_version_counts(self, org_id: str, knowledge_base_id: str, index_version: str) -> tuple[int, int]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT count(*) AS cnt FROM rag_index_documents
                WHERE org_id=%s AND knowledge_base_id=%s AND index_version=%s""",
                (org_id, knowledge_base_id, index_version),
            )
            docs = int(cur.fetchone()["cnt"])
            cur.execute(
                """SELECT count(*) AS cnt FROM rag_index_chunks
                WHERE org_id=%s AND knowledge_base_id=%s AND index_version=%s""",
                (org_id, knowledge_base_id, index_version),
            )
            chunks = int(cur.fetchone()["cnt"])
        return docs, chunks

    def index_version_documents(self, org_id: str, knowledge_base_id: str, index_version: str) -> list[dict]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT document_id, version, digest, chunk_count, indexed_at
                FROM rag_index_documents
                WHERE org_id=%s AND knowledge_base_id=%s AND index_version=%s
                ORDER BY document_id, version""",
                (org_id, knowledge_base_id, index_version),
            )
            return [dict(row) for row in cur.fetchall()]

    def index_version_document(
        self, org_id: str, knowledge_base_id: str, index_version: str, document_id: str, version: str,
    ) -> dict | None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT document_id, version, digest, chunk_count, indexed_at
                FROM rag_index_documents
                WHERE org_id=%s AND knowledge_base_id=%s AND index_version=%s
                  AND document_id=%s AND version=%s""",
                (org_id, knowledge_base_id, index_version, document_id, version),
            )
            row = cur.fetchone()
        return dict(row) if row else None

    def legacy_chunk_count(self, org_id: str) -> int:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) AS cnt FROM rag_chunks WHERE org_id=%s", (org_id,))
            return int(cur.fetchone()["cnt"])

    def legacy_counts(self, org_id: str) -> tuple[int, int]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) AS cnt FROM rag_documents WHERE org_id=%s", (org_id,))
            docs = int(cur.fetchone()["cnt"])
            cur.execute("SELECT count(*) AS cnt FROM rag_chunks WHERE org_id=%s", (org_id,))
            chunks = int(cur.fetchone()["cnt"])
        return docs, chunks

    def legacy_documents(self, org_id: str) -> list[dict]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT document_id, version, digest FROM rag_documents WHERE org_id=%s ORDER BY document_id, version",
                (org_id,),
            )
            return [dict(row) for row in cur.fetchall()]

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
        dim = _safe_dim(dimensions)
        config_json = json.dumps(vector_config, sort_keys=True, ensure_ascii=True)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """UPDATE rag_index_versions SET state='superseded', updated_at=%s
                WHERE org_id=%s AND knowledge_base_id=%s AND state='active' AND index_version<>'legacy'""",
                (timestamp, org_id, knowledge_base_id),
            )
            cur.execute(
                """INSERT INTO rag_index_versions
                (org_id, knowledge_base_id, index_version, embedding_model_id, embedding_model_revision,
                 dimensions, vector_config_json, state, source_snapshot_id, created_at, updated_at)
                VALUES (%s, %s, 'legacy', %s, %s, %s, %s, 'active', NULL, %s, %s)
                ON CONFLICT (org_id, knowledge_base_id, index_version) DO UPDATE SET
                    embedding_model_id=EXCLUDED.embedding_model_id,
                    embedding_model_revision=EXCLUDED.embedding_model_revision,
                    dimensions=EXCLUDED.dimensions,
                    vector_config_json=EXCLUDED.vector_config_json,
                    state='active',
                    updated_at=EXCLUDED.updated_at""",
                (org_id, knowledge_base_id, embedding_model_id, embedding_model_revision, dim, config_json, timestamp, timestamp),
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
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """INSERT INTO rag_rebuild_tasks
                (task_id, org_id, knowledge_base_id, target_index_version, embedding_model_id,
                 embedding_model_revision, state, total, completed, failed, snapshot_json, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, 'pending', %s, 0, 0, %s, %s, %s)""",
                (
                    task_id, org_id, knowledge_base_id, target_index_version, embedding_model_id,
                    embedding_model_revision, len(items), snapshot_json, timestamp, timestamp,
                ),
            )
            cur.executemany(
                """INSERT INTO rag_rebuild_items
                (task_id, document_id, version, digest, state, attempts, updated_at)
                VALUES (%s, %s, %s, %s, 'pending', 0, %s)
                ON CONFLICT (task_id, document_id, version) DO NOTHING""",
                [(task_id, document_id, version, digest, timestamp) for document_id, version, digest in items],
            )

    def get_rebuild_task(self, task_id: str) -> dict | None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM rag_rebuild_tasks WHERE task_id=%s", (task_id,))
            row = cur.fetchone()
        return dict(row) if row else None

    def active_rebuild_task(self, org_id: str, knowledge_base_id: str) -> dict | None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT * FROM rag_rebuild_tasks
                WHERE org_id=%s AND knowledge_base_id=%s AND state IN ('pending', 'running')
                ORDER BY created_at DESC LIMIT 1""",
                (org_id, knowledge_base_id),
            )
            row = cur.fetchone()
        return dict(row) if row else None

    def list_rebuild_tasks(self, org_id: str, knowledge_base_id: str, *, limit: int = 10) -> list[dict]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT * FROM rag_rebuild_tasks
                WHERE org_id=%s AND knowledge_base_id=%s
                ORDER BY created_at DESC LIMIT %s""",
                (org_id, knowledge_base_id, limit),
            )
            return [dict(row) for row in cur.fetchall()]

    def rebuild_items(self, task_id: str) -> list[dict]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT document_id, version, digest, state, attempts, last_error
                FROM rag_rebuild_items WHERE task_id=%s ORDER BY document_id, version""",
                (task_id,),
            )
            return [dict(row) for row in cur.fetchall()]

    def set_rebuild_task_state(
        self,
        task_id: str,
        state: str,
        *,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """UPDATE rag_rebuild_tasks
                SET state=%s, error_code=%s, error_message=%s, updated_at=%s WHERE task_id=%s""",
                (state, error_code, error_message, _now(), task_id),
            )

    def refresh_rebuild_counts(self, task_id: str) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) AS cnt FROM rag_rebuild_items WHERE task_id=%s AND state IN ('succeeded', 'skipped')",
                (task_id,),
            )
            completed = int(cur.fetchone()["cnt"])
            cur.execute(
                "SELECT count(*) AS cnt FROM rag_rebuild_items WHERE task_id=%s AND state='failed'",
                (task_id,),
            )
            failed = int(cur.fetchone()["cnt"])
            cur.execute(
                "UPDATE rag_rebuild_tasks SET completed=%s, failed=%s, updated_at=%s WHERE task_id=%s",
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
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """UPDATE rag_rebuild_items
                SET state=%s, attempts=attempts + %s, last_error=%s, updated_at=%s
                WHERE task_id=%s AND document_id=%s AND version=%s""",
                (state, 1 if increment_attempts else 0, last_error, _now(), task_id, document_id, version),
            )

    def reset_rebuild_items(self, task_id: str) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """UPDATE rag_rebuild_items
                SET state='pending', last_error=NULL, updated_at=%s
                WHERE task_id=%s AND state IN ('failed', 'running', 'pending')""",
                (_now(), task_id),
            )

    def interrupt_running_rebuilds(self) -> list[str]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT task_id FROM rag_rebuild_tasks WHERE state='running'")
            task_ids = [row["task_id"] for row in cur.fetchall()]
            cur.execute("UPDATE rag_rebuild_tasks SET state='interrupted', updated_at=%s WHERE state='running'", (_now(),))
            cur.execute("UPDATE rag_rebuild_items SET state='pending', updated_at=%s WHERE state='running'", (_now(),))
        return task_ids

    def deletable_index_versions(self, org_id: str, knowledge_base_id: str) -> list[dict]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT * FROM rag_index_versions
                WHERE org_id=%s AND knowledge_base_id=%s AND state IN ('superseded', 'retired')
                ORDER BY created_at""",
                (org_id, knowledge_base_id),
            )
            return [dict(row) for row in cur.fetchall()]

    def purge_retired_versions(
        self,
        org_id: str,
        knowledge_base_id: str,
        *,
        keep_superseded: int = 1,
    ) -> tuple[tuple[str, ...], int, int]:
        keep = max(0, int(keep_superseded))
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT target_index_version FROM rag_rebuild_tasks
                WHERE org_id=%s AND knowledge_base_id=%s AND state IN ('pending', 'running')""",
                (org_id, knowledge_base_id),
            )
            busy_targets = {row["target_index_version"] for row in cur.fetchall()}
            cur.execute(
                """SELECT index_version FROM rag_index_versions
                WHERE org_id=%s AND knowledge_base_id=%s AND state='superseded' AND index_version<>'legacy'
                ORDER BY updated_at DESC, index_version DESC""",
                (org_id, knowledge_base_id),
            )
            protected = {row["index_version"] for row in cur.fetchall()[:keep]}
            cur.execute(
                """SELECT index_version, state FROM rag_index_versions
                WHERE org_id=%s AND knowledge_base_id=%s
                  AND state IN ('retired', 'superseded', 'failed')
                  AND index_version<>'legacy'
                ORDER BY created_at, index_version""",
                (org_id, knowledge_base_id),
            )
            to_purge = [
                row["index_version"]
                for row in cur.fetchall()
                if row["index_version"] not in busy_targets
                and not (row["state"] == "superseded" and row["index_version"] in protected)
            ]
            if not to_purge:
                return (), 0, 0
            purged_docs = 0
            purged_chunks = 0
            for ver in to_purge:
                cur.execute(
                    """SELECT count(*) AS cnt FROM rag_index_documents
                    WHERE org_id=%s AND knowledge_base_id=%s AND index_version=%s""",
                    (org_id, knowledge_base_id, ver),
                )
                purged_docs += int(cur.fetchone()["cnt"])
                cur.execute(
                    """SELECT count(*) AS cnt FROM rag_index_chunks
                    WHERE org_id=%s AND knowledge_base_id=%s AND index_version=%s""",
                    (org_id, knowledge_base_id, ver),
                )
                purged_chunks += int(cur.fetchone()["cnt"])
                for table in ("rag_index_chunks", "rag_index_documents", "rag_index_versions"):
                    cur.execute(
                        f"""DELETE FROM {table}
                        WHERE org_id=%s AND knowledge_base_id=%s AND index_version=%s""",
                        (org_id, knowledge_base_id, ver),
                    )
            return tuple(to_purge), purged_docs, purged_chunks
