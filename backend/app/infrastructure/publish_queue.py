import queue
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from app.contracts.common import ContractModel, Identifier

_RELEASE_SLOT_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
end
return 0
"""

PublishStage = Literal["queued", "parsing", "indexing", "succeeded", "failed"]


class PublishTask(ContractModel):
    task_id: Identifier
    org_id: str
    knowledge_base_id: Identifier
    document_id: Identifier
    version: Identifier
    stage: PublishStage
    error_code: str | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _task_from_row(row) -> PublishTask:
    return PublishTask(
        task_id=row["task_id"],
        org_id=row["org_id"],
        knowledge_base_id=row["knowledge_base_id"],
        document_id=row["document_id"],
        version=row["version"],
        stage=row["stage"],
        error_code=row["error_code"],
        error_message=row["error_message"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


class LibraryPublishQueue:
    """Single-concurrency background publish queue (in-process or Redis brokered for multi-replica)."""

    def __init__(
        self,
        database,
        *,
        redis_client: Any | None = None,
        queue_key: str = "salespilot:publish_queue",
        slot_key: str = "salespilot:publish_worker_slot",
        slot_ttl_ms: int = 600_000,
        default_runner: Callable[[str], None] | None = None,
    ):
        self.database = database
        self.redis = redis_client
        self.queue_key = queue_key
        self.slot_key = slot_key
        self.slot_ttl_ms = max(5_000, int(slot_ttl_ms))
        self.default_runner = default_runner
        self._queue: queue.Queue[tuple[str, Callable[[str], None]] | None] = queue.Queue()
        self._runners: dict[str, Callable[[str], None]] = {}
        self._lock = threading.Lock()
        self._started = False
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
            self._stop_event.clear()
            target = self._redis_loop if self.redis is not None else self._loop
            self._worker = threading.Thread(
                target=target, name="salespilot-publish-worker", daemon=True,
            )
            self._worker.start()

    def _execute_task(self, task_id: str, runner: Callable[[str], None] | None) -> None:
        effective_runner = runner or self.default_runner
        if effective_runner is None:
            self.update_stage(
                task_id,
                "failed",
                error_code="publish_worker_unconfigured",
                error_message="当前工作节点未注册异步发布执行器。",
            )
            return
        try:
            effective_runner(task_id)
        except Exception as exc:
            self.update_stage(
                task_id,
                "failed",
                error_code="publish_failed",
                error_message=str(exc) or "资料索引发布失败，请稍后重试。",
            )

    def _loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            task_id, runner = item
            try:
                self._execute_task(task_id, runner)
            finally:
                self._queue.task_done()

    def _acquire_global_slot(self, token: str) -> bool:
        assert self.redis is not None
        return bool(self.redis.set(self.slot_key, token, nx=True, px=self.slot_ttl_ms))

    def _release_global_slot(self, token: str) -> None:
        if self.redis is None:
            return
        try:
            self.redis.eval(_RELEASE_SLOT_LUA, 1, self.slot_key, token)
        except Exception:
            pass

    def _redis_loop(self) -> None:
        assert self.redis is not None
        while not self._stop_event.is_set():
            try:
                popped = self.redis.blpop(self.queue_key, timeout=1)
            except Exception:
                time.sleep(0.1)
                continue
            if not popped:
                continue
            _, task_id = popped
            if task_id == "__STOP__":
                return
            token = uuid4().hex
            while not self._stop_event.is_set() and not self._acquire_global_slot(token):
                time.sleep(0.05)
            if self._stop_event.is_set():
                # Re-queue so another replica can process it cleanly.
                try:
                    self.redis.lpush(self.queue_key, task_id)
                except Exception:
                    pass
                return
            with self._lock:
                runner = self._runners.pop(task_id, None)
            try:
                self._execute_task(task_id, runner)
            finally:
                self._release_global_slot(token)

    def shutdown(self) -> None:
        with self._lock:
            if not self._started:
                return
            self._started = False
            self._stop_event.set()
            worker = self._worker
        if self.redis is not None:
            try:
                self.redis.rpush(self.queue_key, "__STOP__")
            except Exception:
                pass
        else:
            self._queue.put(None)
        if worker is not None:
            worker.join(timeout=2.0)

    def recover_interrupted(self) -> int:
        now = _now()
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """UPDATE library_publish_tasks
                SET stage='failed', error_code='publish_interrupted',
                    error_message='服务重启导致发布任务中断，请重新发起发布。', updated_at=?
                WHERE stage IN ('queued', 'parsing', 'indexing')""",
                (now,),
            )
            return cursor.rowcount

    def find_active(self, org_id: str, document_id: str, version: str) -> PublishTask | None:
        with self.database.connect() as connection:
            row = connection.execute(
                """SELECT * FROM library_publish_tasks
                WHERE org_id=? AND document_id=? AND version=?
                  AND stage IN ('queued', 'parsing', 'indexing')
                ORDER BY created_at DESC LIMIT 1""",
                (org_id, document_id, version),
            ).fetchone()
        return _task_from_row(row) if row else None

    def create_or_reuse(
        self,
        *,
        org_id: str,
        knowledge_base_id: str,
        document_id: str,
        version: str,
    ) -> tuple[PublishTask, bool]:
        now = _now()
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """SELECT * FROM library_publish_tasks
                WHERE org_id=? AND document_id=? AND version=?
                  AND stage IN ('queued', 'parsing', 'indexing')
                ORDER BY created_at DESC LIMIT 1""",
                (org_id, document_id, version),
            ).fetchone()
            if existing is not None:
                return _task_from_row(existing), False
            task_id = f"pub-{uuid4().hex[:12]}"
            connection.execute(
                """INSERT INTO library_publish_tasks
                (task_id, org_id, knowledge_base_id, document_id, version, stage, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 'queued', ?, ?)""",
                (task_id, org_id, knowledge_base_id, document_id, version, now, now),
            )
            row = connection.execute(
                "SELECT * FROM library_publish_tasks WHERE task_id=?", (task_id,),
            ).fetchone()
        return _task_from_row(row), True

    def submit(self, task_id: str, runner: Callable[[str], None]) -> None:
        self._ensure_worker()
        if self.redis is not None:
            with self._lock:
                self._runners[task_id] = runner
            self.redis.rpush(self.queue_key, task_id)
        else:
            self._queue.put((task_id, runner))

    def update_stage(
        self,
        task_id: str,
        stage: PublishStage,
        *,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """UPDATE library_publish_tasks
                SET stage=?, error_code=?, error_message=?, updated_at=?
                WHERE task_id=?""",
                (stage, error_code, error_message, _now(), task_id),
            )

    def get(self, org_id: str, task_id: str) -> PublishTask | None:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM library_publish_tasks WHERE org_id=? AND task_id=?",
                (org_id, task_id),
            ).fetchone()
        return _task_from_row(row) if row else None

    def list_tasks(
        self, org_id: str, knowledge_base_id: str | None = None, *, limit: int = 30,
    ) -> list[PublishTask]:
        with self.database.connect() as connection:
            rows = connection.execute(
                """SELECT * FROM library_publish_tasks
                WHERE org_id=? AND (? IS NULL OR knowledge_base_id=?)
                ORDER BY created_at DESC LIMIT ?""",
                (org_id, knowledge_base_id, knowledge_base_id, limit),
            ).fetchall()
        return [_task_from_row(row) for row in rows]
