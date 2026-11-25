"""Distributed index write lock adapters (SQLite local file lock & Redis cluster lock)."""
from __future__ import annotations

import hashlib
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from app.contracts.rag_workspace import RagUnavailable
from app.infrastructure.library import LibraryIndexBusy, library_index_write

_ACQUIRE_GLOBAL_LUA = """
if redis.call('EXISTS', KEYS[1]) == 1 then
    return 0
end
if redis.call('HLEN', KEYS[2]) > 0 then
    return 0
end
redis.call('SET', KEYS[1], ARGV[1], 'PX', tonumber(ARGV[2]))
return 1
"""

_RELEASE_GLOBAL_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
end
return 0
"""

_ACQUIRE_SCOPED_LUA = """
if redis.call('EXISTS', KEYS[1]) == 1 then
    return 0
end
if redis.call('EXISTS', KEYS[3]) == 1 then
    return 0
end
redis.call('SET', KEYS[3], ARGV[1], 'PX', tonumber(ARGV[2]))
redis.call('HSET', KEYS[2], ARGV[1], ARGV[3])
redis.call('PEXPIRE', KEYS[2], tonumber(ARGV[2]))
return 1
"""

_RELEASE_SCOPED_LUA = """
redis.call('HDEL', KEYS[2], ARGV[1])
if redis.call('GET', KEYS[3]) == ARGV[1] then
    return redis.call('DEL', KEYS[3])
end
return 0
"""


class IndexWriteLock(Protocol):
    @contextmanager
    def acquire(self, *, scope: tuple[str, str] | None = None) -> Iterator[None]: ...


def build_redis_client(redis_url: str) -> Any:
    try:
        import redis
    except ImportError as exc:
        raise RagUnavailable(
            "启用 Redis 分布式锁与队列需要安装 redis 客户端（pip install 'salespilot-ai-backend[enterprise]'）。"
        ) from exc
    return redis.Redis.from_url(redis_url, decode_responses=True)


class SqliteIndexWriteLock:
    """Single-host index write lock backed by auxiliary SQLite lock files."""

    def __init__(self, database_path: Path):
        self.database_path = database_path

    @contextmanager
    def acquire(self, *, scope: tuple[str, str] | None = None) -> Iterator[None]:
        with library_index_write(self.database_path, scope=scope):
            yield


class RedisIndexWriteLock:
    """Multi-replica distributed lock supporting global exclusive + per-(org, kb) concurrency."""

    def __init__(
        self,
        redis_client: Any,
        *,
        prefix: str = "salespilot:index_lock",
        ttl_ms: int = 300_000,
        wait_ms: int = 250,
        poll_interval_ms: int = 25,
    ):
        self.redis = redis_client
        self.prefix = prefix.rstrip(":")
        self.ttl_ms = max(1_000, int(ttl_ms))
        self.wait_ms = max(0, int(wait_ms))
        self.poll_interval_ms = max(5, int(poll_interval_ms))

    def _keys(self, scope: tuple[str, str] | None) -> tuple[str, str, str | None, str]:
        global_key = f"{self.prefix}:global"
        readers_key = f"{self.prefix}:readers"
        if scope is None:
            return global_key, readers_key, None, "global"
        scope_label = f"{scope[0]}\0{scope[1]}"
        digest = hashlib.sha256(scope_label.encode("utf-8")).hexdigest()[:16]
        return global_key, readers_key, f"{self.prefix}:scope:{digest}", digest

    def _try_acquire(self, token: str, scope: tuple[str, str] | None) -> bool:
        global_key, readers_key, scope_key, digest = self._keys(scope)
        if scope_key is None:
            acquired = self.redis.eval(_ACQUIRE_GLOBAL_LUA, 2, global_key, readers_key, token, self.ttl_ms)
        else:
            acquired = self.redis.eval(
                _ACQUIRE_SCOPED_LUA, 3, global_key, readers_key, scope_key, token, self.ttl_ms, digest,
            )
        return bool(int(acquired))

    def _release(self, token: str, scope: tuple[str, str] | None) -> None:
        global_key, readers_key, scope_key, _ = self._keys(scope)
        try:
            if scope_key is None:
                self.redis.eval(_RELEASE_GLOBAL_LUA, 1, global_key, token)
            else:
                self.redis.eval(_RELEASE_SCOPED_LUA, 3, global_key, readers_key, scope_key, token)
        except Exception:
            # Lock expires automatically via PX TTL if Redis disconnects during release.
            pass

    @contextmanager
    def acquire(self, *, scope: tuple[str, str] | None = None) -> Iterator[None]:
        token = uuid4().hex
        deadline = time.monotonic() + (self.wait_ms / 1000.0)
        acquired = self._try_acquire(token, scope)
        while not acquired and time.monotonic() < deadline:
            time.sleep(self.poll_interval_ms / 1000.0)
            acquired = self._try_acquire(token, scope)
        if not acquired:
            raise LibraryIndexBusy()
        try:
            yield
        finally:
            self._release(token, scope)
