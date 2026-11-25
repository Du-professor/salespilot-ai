import asyncio
import base64
import hashlib
import json
import math
import threading
import time
from typing import Any

import httpx
import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from app.auth.oidc import OidcVerifier
from app.contracts.rag_workspace import DocumentVersionKey, LibraryDocument, RagContext
from app.infrastructure.distributed_lock import RedisIndexWriteLock
from app.infrastructure.library import LibraryIndexBusy
from app.infrastructure.publish_queue import LibraryPublishQueue
from app.infrastructure.sqlite import Database
from app.main import create_app
from app.rag.config import RagSettings
from app.rag.engine import SqliteRagEngine
from app.rag.pg_store import PgVectorRagStore, parse_pgvector
from app.settings import Settings

from workspace_helpers import ORIGIN, PASSWORD_A, PASSWORD_B, PREFIX, browser, login

BASES = f"{PREFIX}/library/knowledge-bases"
LIBRARY = f"{PREFIX}/library/documents"
RETRIEVAL_TEST = f"{PREFIX}/library/retrieval-test"
TELEMETRY = f"{PREFIX}/library/telemetry"
TURNS = f"{PREFIX}/recommendations/turns"
SESSION = f"{PREFIX}/session"


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _sign_rs256_jwt(private_key: rsa.RSAPrivateKey, kid: str, claims: dict[str, Any]) -> str:
    header = {"alg": "RS256", "typ": "JWT", "kid": kid}
    header_b64 = _b64url(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_b64 = _b64url(json.dumps(claims, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    sig = private_key.sign(signing_input, padding.PKCS1v15(), hashes.SHA256())
    return f"{header_b64}.{payload_b64}.{_b64url(sig)}"


def _jwk_from_public_key(public_key: rsa.RSAPublicKey, kid: str) -> dict[str, Any]:
    numbers = public_key.public_numbers()
    n_bytes = numbers.n.to_bytes((numbers.n.bit_length() + 7) // 8, "big")
    e_bytes = numbers.e.to_bytes((numbers.e.bit_length() + 7) // 8, "big")
    return {"kty": "RSA", "use": "sig", "alg": "RS256", "kid": kid, "n": _b64url(n_bytes), "e": _b64url(e_bytes)}


def _upload_md(filename: str, text: str, *, knowledge_base_id: str = "default") -> dict:
    return {
        "file": {
            "filename": filename,
            "content_base64": base64.b64encode(text.encode("utf-8")).decode("ascii"),
        },
        "knowledge_base_id": knowledge_base_id,
    }


class FakeRedis:
    """Thread-safe in-memory Redis test double supporting Lua locks, SET NX PX, and RPUSH/BLPOP."""

    def __init__(self):
        self._lock = threading.Condition()
        self._kv: dict[str, tuple[str, float | None]] = {}
        self._hashes: dict[str, tuple[dict[str, str], float | None]] = {}
        self._lists: dict[str, list[str]] = {}

    def _purge_expired(self) -> None:
        now = time.monotonic()
        for k in [key for key, (_, exp) in self._kv.items() if exp is not None and exp <= now]:
            del self._kv[k]
        for k in [key for key, (_, exp) in self._hashes.items() if exp is not None and exp <= now]:
            del self._hashes[k]

    def set(self, key: str, value: str, *, nx: bool = False, px: int | None = None) -> bool:
        with self._lock:
            self._purge_expired()
            if nx and key in self._kv:
                return False
            exp = (time.monotonic() + px / 1000.0) if px is not None else None
            self._kv[key] = (str(value), exp)
            return True

    def eval(self, script: str, numkeys: int, *keys_and_args: Any) -> int:
        keys = keys_and_args[:numkeys]
        argv = keys_and_args[numkeys:]
        with self._lock:
            self._purge_expired()
            # Global lock acquire
            if "HLEN" in script and numkeys == 2:
                g_key, r_key = keys
                token, ttl_ms = str(argv[0]), int(argv[1])
                if g_key in self._kv or len(self._hashes.get(r_key, ({}, None))[0]) > 0:
                    return 0
                self._kv[g_key] = (token, time.monotonic() + ttl_ms / 1000.0)
                return 1
            # Scoped lock acquire
            if "HSET" in script and numkeys == 3:
                g_key, r_key, s_key = keys
                token, ttl_ms, digest = str(argv[0]), int(argv[1]), str(argv[2])
                if g_key in self._kv or s_key in self._kv:
                    return 0
                exp = time.monotonic() + ttl_ms / 1000.0
                self._kv[s_key] = (token, exp)
                readers = dict(self._hashes.get(r_key, ({}, None))[0])
                readers[token] = digest
                self._hashes[r_key] = (readers, exp)
                return 1
            # Scoped lock release
            if "HDEL" in script and numkeys == 3:
                _, r_key, s_key = keys
                token = str(argv[0])
                if r_key in self._hashes:
                    readers, exp = self._hashes[r_key]
                    readers.pop(token, None)
                    if readers:
                        self._hashes[r_key] = (readers, exp)
                    else:
                        del self._hashes[r_key]
                if self._kv.get(s_key, (None, None))[0] == token:
                    del self._kv[s_key]
                    return 1
                return 0
            # Single-key compare-and-delete (global lock release / worker slot release)
            key = keys[0]
            token = str(argv[0])
            if self._kv.get(key, (None, None))[0] == token:
                del self._kv[key]
                return 1
            return 0

    def rpush(self, key: str, value: str) -> int:
        with self._lock:
            lst = self._lists.setdefault(key, [])
            lst.append(str(value))
            self._lock.notify_all()
            return len(lst)

    def lpush(self, key: str, value: str) -> int:
        with self._lock:
            lst = self._lists.setdefault(key, [])
            lst.insert(0, str(value))
            self._lock.notify_all()
            return len(lst)

    def blpop(self, key: str, timeout: int = 1) -> tuple[str, str] | None:
        deadline = time.monotonic() + timeout
        with self._lock:
            while not self._lists.get(key):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._lock.wait(timeout=remaining)
            val = self._lists[key].pop(0)
            return key, val


class FakePgBackend:
    """In-memory PostgreSQL + pgvector + tsvector/pg_trgm SQL engine verifying PgVectorRagStore SQL pushdown."""

    def __init__(self):
        self.executed_sql: list[str] = []
        self.hnsw_indexes: set[str] = set()
        self.versions: dict[tuple[str, str, str], dict] = {}
        self.index_docs: dict[tuple[str, str, str, str, str], dict] = {}
        self.index_chunks: dict[tuple[str, str, str, str, str, str], dict] = {}
        self.legacy_docs: dict[tuple[str, str, str], dict] = {}
        self.legacy_chunks: dict[tuple[str, str, str, str], dict] = {}
        self.rebuild_tasks: dict[str, dict] = {}

    def connect(self):
        return _FakePgConn(self)


class _FakePgConn:
    def __init__(self, backend: FakePgBackend):
        self.backend = backend

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def cursor(self):
        return _FakePgCursor(self.backend)

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        pass


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class _FakePgCursor:
    def __init__(self, b: FakePgBackend):
        self.b = b
        self._rows: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def fetchone(self) -> dict | None:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[dict]:
        return list(self._rows)

    def executemany(self, sql: str, seq_of_params):
        for params in seq_of_params:
            self.execute(sql, params)

    def execute(self, sql: str, params=()):
        norm = " ".join(sql.split())
        self.b.executed_sql.append(norm)
        self._rows = []
        p = list(params)

        if norm.startswith(("CREATE EXTENSION", "CREATE TABLE", "CREATE INDEX IF NOT EXISTS idx_pg_rag")):
            if "USING hnsw" in norm:
                name = norm.split("CREATE INDEX IF NOT EXISTS ")[1].split()[0]
                self.b.hnsw_indexes.add(name)
            return
        if "USING hnsw" in norm:
            name = norm.split("CREATE INDEX IF NOT EXISTS ")[1].split()[0]
            self.b.hnsw_indexes.add(name)
            return

        if norm.startswith("INSERT INTO rag_index_versions"):
            org, kb, ver, mid, mrev, dim, cfg, state, snap, cat, uat = p
            key = (org, kb, ver)
            prev = self.b.versions.get(key)
            self.b.versions[key] = {
                "org_id": org, "knowledge_base_id": kb, "index_version": ver,
                "embedding_model_id": mid, "embedding_model_revision": mrev,
                "dimensions": dim, "vector_config_json": cfg, "state": state,
                "source_snapshot_id": snap or (prev["source_snapshot_id"] if prev else None),
                "created_at": prev["created_at"] if prev else cat, "updated_at": uat,
            }
            return

        if norm.startswith("SELECT * FROM rag_index_versions WHERE org_id=%s AND knowledge_base_id=%s AND index_version=%s"):
            row = self.b.versions.get((p[0], p[1], p[2]))
            self._rows = [dict(row)] if row else []
            return

        if norm.startswith("SELECT * FROM rag_index_versions WHERE org_id=%s AND knowledge_base_id=%s AND state='active'"):
            matches = [v for v in self.b.versions.values() if v["org_id"] == p[0] and v["knowledge_base_id"] == p[1] and v["state"] == "active"]
            matches.sort(key=lambda r: r["updated_at"], reverse=True)
            self._rows = matches[:1]
            return

        if norm.startswith("SELECT * FROM rag_index_versions WHERE org_id=%s AND knowledge_base_id=%s ORDER BY created_at"):
            matches = [v for v in self.b.versions.values() if v["org_id"] == p[0] and v["knowledge_base_id"] == p[1]]
            matches.sort(key=lambda r: (r["created_at"], r["index_version"]))
            self._rows = matches
            return

        if norm.startswith("UPDATE rag_index_versions SET state='superseded'"):
            uat, org, kb = p[0], p[1], p[2]
            for v in self.b.versions.values():
                if v["org_id"] == org and v["knowledge_base_id"] == kb and v["state"] == "active":
                    v["state"] = "superseded"
                    v["updated_at"] = uat
            return

        if norm.startswith("UPDATE rag_index_versions SET state='active'"):
            uat, org, kb, ver = p
            if (org, kb, ver) in self.b.versions:
                self.b.versions[(org, kb, ver)]["state"] = "active"
                self.b.versions[(org, kb, ver)]["updated_at"] = uat
            return

        if norm.startswith("UPDATE rag_index_versions SET state=%s"):
            state, uat, org, kb, ver = p
            if (org, kb, ver) in self.b.versions:
                self.b.versions[(org, kb, ver)]["state"] = state
                self.b.versions[(org, kb, ver)]["updated_at"] = uat
            return

        if norm.startswith("DELETE FROM rag_index_chunks WHERE org_id=%s AND knowledge_base_id=%s AND index_version=%s AND document_id=%s"):
            org, kb, ver, doc, dver = p
            for k in [k for k in self.b.index_chunks if k[:5] == (org, kb, ver, doc, dver)]:
                del self.b.index_chunks[k]
            return

        if norm.startswith("INSERT INTO rag_index_chunks"):
            org, kb, ver, doc, dver, cid, model, page, section, ordinal, text, lex, emb, dim, trunc, pid = p
            self.b.index_chunks[(org, kb, ver, doc, dver, cid)] = {
                "org_id": org, "knowledge_base_id": kb, "index_version": ver,
                "document_id": doc, "version": dver, "chunk_id": cid, "model": model,
                "page": page, "section": section, "ordinal": ordinal, "text": text,
                "lexical_text": lex, "embedding": parse_pgvector(emb, dim),
                "dimensions": dim, "truncated": trunc, "parent_id": pid,
            }
            return

        if norm.startswith("INSERT INTO rag_index_documents"):
            org, kb, ver, doc, dver, digest, cnt, mid, mrev, dim, cfg, iat = p
            self.b.index_docs[(org, kb, ver, doc, dver)] = {
                "org_id": org, "knowledge_base_id": kb, "index_version": ver,
                "document_id": doc, "version": dver, "digest": digest, "chunk_count": cnt,
                "embedding_model_id": mid, "embedding_model_revision": mrev,
                "dimensions": dim, "vector_config_json": cfg, "indexed_at": iat,
            }
            return

        if "AS cosine_score FROM rag_index_chunks" in norm:
            limit = p[-1]
            vec_lit = p[-2]
            dim = p[-3]
            ver = p[-4]
            kb = p[-5]
            org = p[-6]
            key_pairs = {(p[i], p[i + 1]) for i in range(0, len(p) - 7, 2)}
            qvec = parse_pgvector(vec_lit, dim)
            hits = []
            for row in self.b.index_chunks.values():
                if (row["org_id"], row["knowledge_base_id"], row["index_version"], row["dimensions"]) == (org, kb, ver, dim):
                    if (row["document_id"], row["version"]) in key_pairs:
                        hits.append({**row, "cosine_score": _cosine(qvec, row["embedding"])})
            hits.sort(key=lambda r: (-r["cosine_score"], r["chunk_id"]))
            self._rows = hits[:limit]
            return

        if "AS rank_score FROM rag_index_chunks" in norm:
            limit = p[-1]
            tsq = p[-3]
            ver = p[-4]
            kb = p[-5]
            org = p[-6]
            key_pairs = {(p[i], p[i + 1]) for i in range(0, len(p) - 8, 2)}
            terms = [t.strip() for t in tsq.split("|") if t.strip()]
            hits = []
            for row in self.b.index_chunks.values():
                if (row["org_id"], row["knowledge_base_id"], row["index_version"]) == (org, kb, ver):
                    if (row["document_id"], row["version"]) in key_pairs:
                        lex_tokens = set(row["lexical_text"].split())
                        matched = sum(1 for t in terms if t in lex_tokens)
                        if matched > 0:
                            hits.append({"chunk_id": row["chunk_id"], "rank_score": matched * 0.6})
            hits.sort(key=lambda r: -r["rank_score"])
            self._rows = hits[:limit]
            return

        if norm.startswith("SELECT document_id, version, chunk_id, model, page, section, ordinal, text, truncated, parent_id FROM rag_index_chunks WHERE org_id=%s AND knowledge_base_id=%s AND index_version=%s AND chunk_id IN"):
            org, kb, ver = p[:3]
            wanted = set(p[3:])
            self._rows = [
                row for row in self.b.index_chunks.values()
                if (row["org_id"], row["knowledge_base_id"], row["index_version"]) == (org, kb, ver)
                and row["chunk_id"] in wanted
            ]
            return

        if norm.startswith("SELECT count(*) AS cnt FROM rag_index_documents WHERE org_id=%s AND knowledge_base_id=%s AND index_version=%s"):
            org, kb, ver = p[:3]
            cnt = sum(1 for k in self.b.index_docs if k[:3] == (org, kb, ver))
            self._rows = [{"cnt": cnt}]
            return

        if norm.startswith("SELECT count(*) AS cnt FROM rag_index_chunks WHERE org_id=%s AND knowledge_base_id=%s AND index_version=%s"):
            org, kb, ver = p[:3]
            cnt = sum(1 for k in self.b.index_chunks if k[:3] == (org, kb, ver))
            self._rows = [{"cnt": cnt}]
            return

        if norm.startswith("SELECT target_index_version FROM rag_rebuild_tasks"):
            self._rows = []
            return

        if norm.startswith("SELECT index_version FROM rag_index_versions WHERE org_id=%s AND knowledge_base_id=%s AND state='superseded'"):
            org, kb = p
            rows = [v for v in self.b.versions.values() if v["org_id"] == org and v["knowledge_base_id"] == kb and v["state"] == "superseded" and v["index_version"] != "legacy"]
            rows.sort(key=lambda r: (r["updated_at"], r["index_version"]), reverse=True)
            self._rows = rows
            return

        if norm.startswith("SELECT index_version, state FROM rag_index_versions WHERE org_id=%s AND knowledge_base_id=%s AND state IN ('retired', 'superseded', 'failed')"):
            org, kb = p
            rows = [v for v in self.b.versions.values() if v["org_id"] == org and v["knowledge_base_id"] == kb and v["state"] in {"retired", "superseded", "failed"} and v["index_version"] != "legacy"]
            rows.sort(key=lambda r: (r["created_at"], r["index_version"]))
            self._rows = rows
            return

        if norm.startswith("DELETE FROM rag_index_chunks WHERE org_id=%s AND knowledge_base_id=%s AND index_version=%s"):
            org, kb, ver = p
            for k in [k for k in self.b.index_chunks if k[:3] == (org, kb, ver)]:
                del self.b.index_chunks[k]
            return

        if norm.startswith("DELETE FROM rag_index_documents WHERE org_id=%s AND knowledge_base_id=%s AND index_version=%s"):
            org, kb, ver = p
            for k in [k for k in self.b.index_docs if k[:3] == (org, kb, ver)]:
                del self.b.index_docs[k]
            return

        if norm.startswith("DELETE FROM rag_index_versions WHERE org_id=%s AND knowledge_base_id=%s AND index_version=%s"):
            org, kb, ver = p
            self.b.versions.pop((org, kb, ver), None)
            return

        if norm.startswith("SELECT document_id, version, digest, chunk_count, indexed_at FROM rag_index_documents"):
            org, kb, ver = p[:3]
            rows = [v for k, v in self.b.index_docs.items() if k[:3] == (org, kb, ver)]
            rows.sort(key=lambda r: (r["document_id"], r["version"]))
            self._rows = rows
            return


def test_pgvector_store_hybrid_server_ann_and_physical_gc(tmp_path):
    fake_pg = FakePgBackend()
    pg_store = PgVectorRagStore(conn_factory=fake_pg.connect)

    # Extensions must be initialized on startup
    assert any("CREATE EXTENSION IF NOT EXISTS vector" in s for s in fake_pg.executed_sql)
    assert any("CREATE EXTENSION IF NOT EXISTS pg_trgm" in s for s in fake_pg.executed_sql)

    # Forbid full-table load_chunks to prove RecommendationGraph uses server-side pgvector ANN pushdown!
    pg_store.load_chunks = lambda *args, **kwargs: (_ for _ in ()).throw(  # type: ignore[method-assign]
        AssertionError("load_chunks must NOT be called when server-side pgvector ANN is available!")
    )

    engine = SqliteRagEngine(RagSettings(index_path=tmp_path / "unused.sqlite3"), store=pg_store)
    v1 = engine.ensure_index_version("acme", "kb-pg", "workspace-default", index_version="idx-v1", activate=True)
    assert "idx_pg_rag_index_chunks_hnsw_d256" in fake_pg.hnsw_indexes

    payload = "# SP-PG-500 工业相机\n型号 SP-PG-500 支持 GigE 接口、分辨率 16 MP、帧率 180 fps。\n".encode("utf-8")
    doc = LibraryDocument(
        document_id="doc-pg-1",
        version="v1",
        filename="sp-pgvector.md",
        size_bytes=len(payload),
        uploaded_at="2026-11-24T00:00:00Z",
        status="published",
        index_status="ready",
        knowledge_base_id="kb-pg",
        index_version=v1.index_version,
        embedding_model_id="workspace-default",
        embedding_model_revision=v1.embedding_model_revision,
        dimensions=v1.dimensions,
        digest=hashlib.sha256(payload).hexdigest(),
    )
    engine.index_document("acme", doc, payload)

    diag, _ = engine.diagnose_detailed(
        RagContext(
            org_id="acme",
            user_id="u1",
            text="SP-PG-500 GigE 180 fps",
            previous_turns=(),
            documents=(doc,),
            knowledge_base_id="kb-pg",
            index_version="idx-v1",
            embedding_model_id="workspace-default",
        ),
        mode="raw_retrieval",
        strategy="hybrid",
    )
    assert diag.hits and "SP-PG-500" in diag.hits[0].text
    assert diag.hits[0].vector_score is not None
    assert diag.hits[0].bm25_score is not None
    assert any("<=>" in sql for sql in fake_pg.executed_sql)

    # Create idx-v2 (activate -> idx-v1 becomes superseded) and idx-retired (retire -> eligible for GC)
    engine.ensure_index_version("acme", "kb-pg", "workspace-default", index_version="idx-retired", activate=False)
    engine.index_document("acme", doc.model_copy(update={"index_version": "idx-retired"}), payload)
    engine.retire("acme", "kb-pg", "idx-retired")

    engine.ensure_index_version("acme", "kb-pg", "workspace-default", index_version="idx-v2", activate=False)
    engine.index_document("acme", doc.model_copy(update={"index_version": "idx-v2"}), payload)
    engine.mark_active("acme", "kb-pg", "idx-v2", documents=(doc,))

    gc_res = engine.purge_retired("acme", "kb-pg", keep_superseded=1)
    assert gc_res.purged_versions == ("idx-retired",)
    assert gc_res.purged_chunks >= 1
    # idx-v1 (most recent superseded rollback target) and idx-v2 (active) must remain untouched!
    remaining = {v.index_version for v in engine.list_index_versions("acme", "kb-pg")}
    assert remaining == {"idx-v1", "idx-v2"}


def test_redis_distributed_lock_and_publish_queue(tmp_path):
    fake_redis = FakeRedis()
    lock = RedisIndexWriteLock(fake_redis, wait_ms=40, poll_interval_ms=10)

    # Distinct (org, kb) scopes acquire concurrently
    with lock.acquire(scope=("org-1", "kb-a")):
        with lock.acquire(scope=("org-1", "kb-b")):
            pass
        # Same scope contends and raises LibraryIndexBusy
        with pytest.raises(LibraryIndexBusy):
            with lock.acquire(scope=("org-1", "kb-a")):
                pass
        # Global maintenance lock (scope=None) is blocked while any scoped reader holds a lock
        with pytest.raises(LibraryIndexBusy):
            with lock.acquire(scope=None):
                pass

    # Global maintenance lock blocks new scoped locks until released
    with lock.acquire(scope=None):
        with pytest.raises(LibraryIndexBusy):
            with lock.acquire(scope=("org-1", "kb-a")):
                pass

    # Redis-backed LibraryPublishQueue processes tasks sequentially via RPUSH / BLPOP + worker slot
    db = Database(tmp_path / "redis-queue.sqlite3")
    db.initialize()
    queue = LibraryPublishQueue(db, redis_client=fake_redis)
    try:
        done = threading.Event()
        task, created = queue.create_or_reuse(
            org_id="org-1", knowledge_base_id="default", document_id="doc-1", version="v1",
        )
        assert created is True

        def runner(tid: str) -> None:
            queue.update_stage(tid, "indexing")
            queue.update_stage(tid, "succeeded")
            done.set()

        queue.submit(task.task_id, runner)
        assert done.wait(timeout=3.0)
        finished = queue.get("org-1", task.task_id)
        assert finished is not None and finished.stage == "succeeded"
    finally:
        queue.shutdown()


def test_keycloak_oidc_jwt_multi_org_and_kb_acl(tmp_path, monkeypatch):
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    kid = "kc-key-2026"
    jwks = {"keys": [_jwk_from_public_key(private_key.public_key(), kid)]}
    issuer = "https://sso.example.internal/realms/salespilot"
    audience = "salespilot-web"

    def fake_jwks_get(url, *, timeout):
        assert url == f"{issuer}/protocol/openid-connect/certs"
        return httpx.Response(200, json=jwks)

    monkeypatch.setattr("app.auth.oidc.httpx.get", fake_jwks_get)
    verifier = OidcVerifier(issuer=issuer, audience=audience)

    settings = Settings(
        database_path=tmp_path / "oidc-acl.sqlite3",
        rag_mode="demo",
        web_origins=(ORIGIN,),
    )
    app = create_app(settings, oidc_verifier=verifier)

    now = int(time.time())
    alice_token = _sign_rs256_jwt(private_key, kid, {
        "iss": issuer, "aud": audience, "sub": "alice-admin", "preferred_username": "alice",
        "org_id": "acme-corp", "realm_access": {"roles": ["workspace_admin"]},
        "groups": ["/leadership"], "exp": now + 3600,
    })
    bob_token = _sign_rs256_jwt(private_key, kid, {
        "iss": issuer, "aud": audience, "sub": "bob-vip", "preferred_username": "bob",
        "org_id": "acme-corp", "realm_access": {"roles": ["sales"]},
        "groups": ["/vip-sales"], "exp": now + 3600,
    })
    charlie_token = _sign_rs256_jwt(private_key, kid, {
        "iss": issuer, "aud": audience, "sub": "charlie-reg", "preferred_username": "charlie",
        "org_id": "acme-corp", "realm_access": {"roles": ["sales"]},
        "groups": ["/general-sales"], "exp": now + 3600,
    })
    mallory_token = _sign_rs256_jwt(private_key, kid, {
        "iss": issuer, "aud": audience, "sub": "mallory", "preferred_username": "mallory",
        "org_id": "rival-org", "realm_access": {"roles": ["admin"]},
        "exp": now + 3600,
    })
    expired_token = _sign_rs256_jwt(private_key, kid, {
        "iss": issuer, "aud": audience, "sub": "alice-admin", "org_id": "acme-corp", "exp": now - 120,
    })

    def auth(token: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {token}"}

    async def check():
        async with app.router.lifespan_context(app), browser(app) as client:
            # Auth config reflects OIDC enabled
            cfg = (await client.get(f"{SESSION}/auth-config")).json()
            assert cfg["oidc_enabled"] is True and cfg["oidc_issuer"] == issuer

            # Expired token rejected with 401
            bad = await client.get(SESSION, headers=auth(expired_token))
            assert (bad.status_code, bad.json()["code"]) == (401, "invalid_oidc_token")

            # Alice (non-DEMO_ORG OIDC admin!) has full workspace management rights
            me = (await client.get(SESSION, headers=auth(alice_token))).json()
            assert me["org_id"] == "acme-corp" and me["role"] == "admin" and me["can_manage_workspace"] is True

            # Alice creates a restricted KB and publishes a confidential product spec
            kb_res = await client.post(BASES, headers=auth(alice_token), json={
                "name": "VIP 专享机密库",
                "description": "仅对 vip-sales 组开放",
                "embedding_model_id": "workspace-default",
                "access_policy": "restricted",
            })
            assert kb_res.status_code == 201, kb_res.text
            kb_id = kb_res.json()["knowledge_base_id"]

            doc = (await client.post(LIBRARY, headers=auth(alice_token), json=_upload_md(
                "vip-camera.md",
                "# SP-VIP-900 定制工业相机\n型号 SP-VIP-900 支持 GigE 接口，分辨率 24 MP，帧率 120 fps。\n",
                knowledge_base_id=kb_id,
            ))).json()
            doc_id = doc["document_id"]
            assert (await client.post(f"{LIBRARY}/{doc_id}/v1/publish", headers=auth(alice_token))).status_code == 200

            # Before granting vip-sales, neither Bob nor Charlie can see the restricted KB
            bob_kbs_before = [k["knowledge_base_id"] for k in (await client.get(BASES, headers=auth(bob_token))).json()]
            assert kb_id not in bob_kbs_before

            # Alice grants group 'vip-sales' access via ACL API
            acl_res = await client.put(f"{BASES}/{kb_id}/acl", headers=auth(alice_token), json={
                "access_policy": "restricted",
                "grants": [{"subject_type": "group", "subject_id": "vip-sales"}],
            })
            assert acl_res.status_code == 200 and len(acl_res.json()["grants"]) == 1

            # Bob (in /vip-sales) CAN list KB, download file, and run product selection!
            bob_kbs = [k["knowledge_base_id"] for k in (await client.get(BASES, headers=auth(bob_token))).json()]
            assert kb_id in bob_kbs
            assert (await client.get(f"{LIBRARY}/{doc_id}/v1/file", headers=auth(bob_token))).status_code == 200
            bob_turn = await client.post(TURNS, headers=auth(bob_token), json={
                "conversation_id": "vip-conv-1",
                "client_request_id": "vip-req-1",
                "knowledge_base_id": kb_id,
                "text": "需要 GigE 接口 120 fps 工业相机",
            })
            assert bob_turn.status_code == 200, bob_turn.text

            # Charlie (same org acme-corp, NOT in vip-sales) is blocked everywhere (list, turns, direct file/chunks)!
            charlie_kbs = [k["knowledge_base_id"] for k in (await client.get(BASES, headers=auth(charlie_token))).json()]
            assert kb_id not in charlie_kbs
            assert (await client.get(f"{LIBRARY}?knowledge_base_id={kb_id}", headers=auth(charlie_token))).status_code == 404
            assert (await client.get(f"{LIBRARY}/{doc_id}/v1/file", headers=auth(charlie_token))).status_code == 404
            assert (await client.get(f"{LIBRARY}/{doc_id}/v1/chunks", headers=auth(charlie_token))).status_code == 404
            assert (await client.post(TURNS, headers=auth(charlie_token), json={
                "conversation_id": "c-conv-1",
                "client_request_id": "c-req-1",
                "knowledge_base_id": kb_id,
                "text": "SP-VIP-900",
            })).status_code == 404

            # Mallory (different org) cannot see acme-corp's KB even with admin role
            mallory_kbs = [k["knowledge_base_id"] for k in (await client.get(BASES, headers=auth(mallory_token))).json()]
            assert kb_id not in mallory_kbs

    asyncio.run(check())


def test_retired_index_gc_and_retrieval_telemetry_endpoints(tmp_path):
    app = create_app(Settings(
        database_path=tmp_path / "gc-telemetry.sqlite3",
        rag_mode="demo",
        demo_password_a=PASSWORD_A,
        demo_password_b=PASSWORD_B,
        web_origins=(ORIGIN,),
    ))

    async def check():
        async with app.router.lifespan_context(app), browser(app) as admin:
            await login(admin)
            kb = (await admin.post(BASES, json={
                "name": "运维治理测试库",
                "description": "测试退役索引 GC 与检索遥测",
                "embedding_model_id": "workspace-default",
            })).json()
            kb_id = kb["knowledge_base_id"]

            doc = (await admin.post(LIBRARY, json=_upload_md(
                "sp-gov.md",
                "# SP-GOV-300 伺服驱动器\n型号 SP-GOV-300 额定功率 750W，支持 EtherCAT 总线。\n",
                knowledge_base_id=kb_id,
            ))).json()
            assert (await admin.post(f"{LIBRARY}/{doc['document_id']}/v1/publish")).status_code == 200

            # Trigger synchronous rebuild to create a second index version, activate it, then retire the old one
            rebuilt = (await admin.post(f"{BASES}/{kb_id}/rebuild", json={
                "embedding_model_id": "workspace-default",
                "background": False,
            })).json()
            assert rebuilt["state"] == "succeeded"
            new_ver = rebuilt["target_index_version"]
            old_ver = (await admin.get(f"{BASES}/{kb_id}/index")).json()["active_index_version"]

            assert (await admin.post(f"{BASES}/{kb_id}/indexes/{new_ver}/activate", json={"allow_stale": False})).status_code == 200
            assert (await admin.post(f"{BASES}/{kb_id}/indexes/{old_ver}/retire")).status_code == 204

            gc = (await admin.post(f"{BASES}/{kb_id}/indexes/gc", json={"keep_superseded": 1})).json()
            assert old_ver in gc["purged_versions"]
            assert gc["purged_chunks"] >= 1

            # Run 1 hit query and 1 zero-hit query, then verify telemetry aggregation
            hit_q = await admin.post(RETRIEVAL_TEST, json={
                "text": "SP-GOV-300 EtherCAT 750W",
                "knowledge_base_id": kb_id,
                "mode": "raw_retrieval",
                "strategy": "hybrid",
            })
            assert hit_q.status_code == 200 and len(hit_q.json()["hits"]) >= 1

            miss_q = await admin.post(RETRIEVAL_TEST, json={
                "text": "不存在的量子反重力推进器 XYZ-9999",
                "knowledge_base_id": kb_id,
                "mode": "selection",
                "strategy": "bm25",
            })
            assert miss_q.status_code == 200 and len(miss_q.json()["hits"]) == 0

            stats = (await admin.get(TELEMETRY, params={"knowledge_base_id": kb_id, "days": 7})).json()
            assert stats["total_queries"] == 2
            assert stats["zero_hit_count"] == 1
            assert stats["zero_hit_rate"] == 0.5
            assert stats["avg_latency_ms"] >= 0.0
            assert len(stats["zero_hit_samples"]) == 1
            assert "XYZ-9999" in stats["zero_hit_samples"][0]["query_preview"]

    asyncio.run(check())
