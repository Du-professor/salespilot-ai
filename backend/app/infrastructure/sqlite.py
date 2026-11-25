import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

DEMO_ORG = "demo"

SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    org_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    username TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('admin', 'sales', 'catalog_viewer')),
    password_hash TEXT NOT NULL,
    PRIMARY KEY (org_id, user_id),
    UNIQUE (org_id, username)
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    org_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    expires_at INTEGER NOT NULL,
    FOREIGN KEY (org_id, user_id) REFERENCES accounts (org_id, user_id)
);
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY,
    recorded_at INTEGER NOT NULL,
    org_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    action TEXT NOT NULL,
    object_id TEXT,
    request_id TEXT NOT NULL,
    versions_json TEXT
);
CREATE TABLE IF NOT EXISTS library_document_versions (
    org_id TEXT NOT NULL,
    knowledge_base_id TEXT NOT NULL DEFAULT 'default',
    document_id TEXT NOT NULL,
    version TEXT NOT NULL,
    filename TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    digest TEXT NOT NULL,
    content BLOB NOT NULL DEFAULT X'',
    storage_uri TEXT,
    status TEXT NOT NULL CHECK (status IN ('draft', 'published', 'withdrawn')),
    uploaded_at TEXT NOT NULL,
    PRIMARY KEY (org_id, document_id, version)
);
CREATE TABLE IF NOT EXISTS library_publish_tasks (
    task_id TEXT PRIMARY KEY,
    org_id TEXT NOT NULL,
    knowledge_base_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    version TEXT NOT NULL,
    stage TEXT NOT NULL CHECK (stage IN ('queued', 'parsing', 'indexing', 'succeeded', 'failed')),
    error_code TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_library_publish_tasks_scope
    ON library_publish_tasks (org_id, knowledge_base_id, created_at);
CREATE INDEX IF NOT EXISTS idx_library_publish_tasks_doc
    ON library_publish_tasks (org_id, document_id, version, created_at);
CREATE INDEX IF NOT EXISTS idx_library_visibility ON library_document_versions (org_id, status);
CREATE TABLE IF NOT EXISTS library_folders (
    org_id TEXT NOT NULL,
    knowledge_base_id TEXT NOT NULL DEFAULT 'default',
    folder_id TEXT NOT NULL,
    parent_id TEXT,
    name TEXT NOT NULL,
    name_key TEXT NOT NULL,
    PRIMARY KEY (org_id, folder_id),
    FOREIGN KEY (org_id, parent_id) REFERENCES library_folders (org_id, folder_id)
);
CREATE TABLE IF NOT EXISTS library_knowledge_bases (
    org_id TEXT NOT NULL,
    knowledge_base_id TEXT NOT NULL,
    name TEXT NOT NULL,
    name_key TEXT NOT NULL,
    description TEXT NOT NULL,
    embedding_model_id TEXT NOT NULL,
    embedding_model_revision TEXT,
    active_index_version TEXT,
    index_state TEXT NOT NULL DEFAULT 'awaiting_adapter',
    access_policy TEXT NOT NULL DEFAULT 'org_published' CHECK (access_policy IN ('org_published', 'restricted')),
    created_at TEXT NOT NULL,
    PRIMARY KEY (org_id, knowledge_base_id),
    UNIQUE (org_id, name_key)
);
CREATE TABLE IF NOT EXISTS library_kb_grants (
    org_id TEXT NOT NULL,
    knowledge_base_id TEXT NOT NULL,
    subject_type TEXT NOT NULL CHECK (subject_type IN ('user', 'role', 'group')),
    subject_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (org_id, knowledge_base_id, subject_type, subject_id)
);
CREATE INDEX IF NOT EXISTS idx_library_kb_grants_lookup
    ON library_kb_grants (org_id, knowledge_base_id, subject_type, subject_id);
CREATE TABLE IF NOT EXISTS library_retrieval_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    org_id TEXT NOT NULL,
    knowledge_base_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    source TEXT NOT NULL CHECK (source IN ('selection', 'retrieval_test')),
    strategy TEXT NOT NULL,
    status TEXT NOT NULL,
    hit_count INTEGER NOT NULL DEFAULT 0,
    top_score REAL,
    latency_ms REAL NOT NULL DEFAULT 0,
    query_preview TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_library_retrieval_logs_scope
    ON library_retrieval_logs (org_id, knowledge_base_id, created_at DESC);
CREATE TABLE IF NOT EXISTS library_index_activations (
    sequence INTEGER PRIMARY KEY,
    org_id TEXT NOT NULL,
    knowledge_base_id TEXT NOT NULL,
    index_version TEXT NOT NULL,
    embedding_model_id TEXT NOT NULL,
    embedding_model_revision TEXT NOT NULL,
    reason TEXT NOT NULL,
    activated_at TEXT NOT NULL,
    user_id TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_library_index_activations
    ON library_index_activations (org_id, knowledge_base_id, sequence);
CREATE TABLE IF NOT EXISTS library_document_locations (
    org_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    folder_id TEXT NOT NULL,
    PRIMARY KEY (org_id, document_id),
    FOREIGN KEY (org_id, folder_id) REFERENCES library_folders (org_id, folder_id)
);
CREATE TABLE IF NOT EXISTS library_deleted_versions (
    org_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    version TEXT NOT NULL,
    PRIMARY KEY (org_id, document_id, version)
);
CREATE TABLE IF NOT EXISTS rag_recommendation_turns (
    sequence INTEGER PRIMARY KEY,
    turn_id TEXT NOT NULL UNIQUE,
    org_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    knowledge_base_id TEXT NOT NULL DEFAULT 'default',
    client_request_id TEXT NOT NULL,
    user_text TEXT NOT NULL,
    response_json TEXT NOT NULL,
    archived INTEGER NOT NULL CHECK (archived IN (0, 1)),
    UNIQUE (org_id, user_id, client_request_id),
    FOREIGN KEY (org_id, user_id) REFERENCES accounts (org_id, user_id)
);
CREATE INDEX IF NOT EXISTS idx_rag_conversation ON rag_recommendation_turns (org_id, user_id, conversation_id, sequence);
CREATE INDEX IF NOT EXISTS idx_rag_history ON rag_recommendation_turns (org_id, user_id, archived, sequence);
CREATE TABLE IF NOT EXISTS rag_conversation_settings (
    org_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    title TEXT,
    pinned INTEGER NOT NULL DEFAULT 0 CHECK (pinned IN (0, 1)),
    deleted INTEGER NOT NULL DEFAULT 0 CHECK (deleted IN (0, 1)),
    PRIMARY KEY (org_id, user_id, conversation_id),
    FOREIGN KEY (org_id, user_id) REFERENCES accounts (org_id, user_id)
);
CREATE TABLE IF NOT EXISTS model_configs (
    config_id TEXT PRIMARY KEY,
    org_id TEXT NOT NULL,
    capability TEXT NOT NULL DEFAULT 'chat' CHECK (capability IN ('chat', 'embedding', 'rerank', 'parser')),
    name TEXT NOT NULL,
    provider TEXT NOT NULL,
    model_id TEXT NOT NULL,
    base_url TEXT NOT NULL,
    encrypted_key BLOB,
    dimensions INTEGER,
    revision INTEGER NOT NULL CHECK (revision >= 1),
    test_status TEXT NOT NULL DEFAULT 'untested' CHECK (test_status IN ('untested', 'passed', 'failed')),
    test_message TEXT,
    test_at INTEGER,
    active INTEGER NOT NULL DEFAULT 0 CHECK (active IN (0, 1))
);
CREATE INDEX IF NOT EXISTS idx_model_configs_org ON model_configs (org_id, capability, name);
CREATE UNIQUE INDEX IF NOT EXISTS idx_model_configs_active_capability
    ON model_configs (org_id, capability) WHERE active=1;
"""


class Database:
    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(SCHEMA)
            connection.execute("BEGIN IMMEDIATE")
            for table in ("library_document_versions", "library_folders", "rag_recommendation_turns"):
                columns = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}
                if "knowledge_base_id" not in columns:
                    connection.execute(
                        f"ALTER TABLE {table} ADD COLUMN knowledge_base_id TEXT NOT NULL DEFAULT 'default'"
                    )
            doc_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(library_document_versions)")
            }
            if "storage_uri" not in doc_columns:
                connection.execute("ALTER TABLE library_document_versions ADD COLUMN storage_uri TEXT")
            base_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(library_knowledge_bases)")
            }
            for name, ddl in (
                ("embedding_model_revision", "TEXT"),
                ("active_index_version", "TEXT"),
                ("index_state", "TEXT NOT NULL DEFAULT 'awaiting_adapter'"),
                ("access_policy", "TEXT NOT NULL DEFAULT 'org_published'"),
            ):
                if name not in base_columns:
                    connection.execute(f"ALTER TABLE library_knowledge_bases ADD COLUMN {name} {ddl}")
            model_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(model_configs)")
            }
            if "capability" not in model_columns:
                connection.execute(
                    "ALTER TABLE model_configs ADD COLUMN capability TEXT NOT NULL DEFAULT 'chat'"
                )
            if "dimensions" not in model_columns:
                connection.execute("ALTER TABLE model_configs ADD COLUMN dimensions INTEGER")
            connection.execute("DROP INDEX IF EXISTS idx_model_configs_active")
            connection.execute(
                """CREATE UNIQUE INDEX IF NOT EXISTS idx_model_configs_active_capability
                ON model_configs (org_id, capability) WHERE active=1"""
            )
            connection.execute("DROP INDEX IF EXISTS idx_library_folder_name")
            connection.execute(
                """CREATE UNIQUE INDEX IF NOT EXISTS idx_library_folder_base_name
                ON library_folders (org_id, knowledge_base_id, coalesce(parent_id, ''), name_key)"""
            )
            connection.execute(
                """CREATE INDEX IF NOT EXISTS idx_library_base_visibility
                ON library_document_versions (org_id, knowledge_base_id, status)"""
            )


def record_audit(
    connection: sqlite3.Connection,
    *,
    org_id: str,
    user_id: str,
    action: str,
    request_id: str,
    object_id: str | None = None,
    versions_json: str | None = None,
) -> None:
    connection.execute(
        """
        INSERT INTO audit
            (recorded_at, org_id, user_id, action, object_id, request_id, versions_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (int(time.time()), org_id, user_id, action, object_id, request_id, versions_json),
    )
