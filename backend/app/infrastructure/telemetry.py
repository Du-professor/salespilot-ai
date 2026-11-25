"""Retrieval observability & hit-rate / latency telemetry repository."""
from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from typing import Literal

from app.contracts.common import ContractModel, Identifier


class RetrievalEventSample(ContractModel):
    created_at: datetime
    source: Literal["selection", "retrieval_test"]
    strategy: str
    status: str
    hit_count: int
    top_score: float | None = None
    latency_ms: float
    query_preview: str


class RetrievalTelemetrySummary(ContractModel):
    knowledge_base_id: Identifier
    window_days: int
    total_queries: int = 0
    zero_hit_count: int = 0
    zero_hit_rate: float = 0.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    avg_top_score: float | None = None
    zero_hit_samples: tuple[RetrievalEventSample, ...] = ()
    recent_events: tuple[RetrievalEventSample, ...] = ()


def record_retrieval_event(
    database,
    *,
    org_id: str,
    knowledge_base_id: str,
    user_id: str,
    source: Literal["selection", "retrieval_test"],
    strategy: str,
    status: str,
    hit_count: int,
    top_score: float | None,
    latency_ms: float,
    query_text: str,
) -> None:
    preview = " ".join(query_text.strip().split())[:120]
    clean_score = float(top_score) if top_score is not None and math.isfinite(top_score) else None
    clean_latency = max(0.0, round(float(latency_ms), 2))
    try:
        with database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """INSERT INTO library_retrieval_logs
                (org_id, knowledge_base_id, user_id, source, strategy, status,
                 hit_count, top_score, latency_ms, query_preview, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    org_id,
                    knowledge_base_id,
                    user_id,
                    source,
                    strategy or "hybrid",
                    status,
                    max(0, int(hit_count)),
                    clean_score,
                    clean_latency,
                    preview,
                    datetime.now(UTC).isoformat(),
                ),
            )
    except Exception:
        # Telemetry must never fail user-facing selection or diagnostic requests.
        pass


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return round(sorted_values[0], 2)
    idx = min(len(sorted_values) - 1, max(0, math.ceil(pct * len(sorted_values)) - 1))
    return round(sorted_values[idx], 2)


def _row_to_sample(row) -> RetrievalEventSample:
    return RetrievalEventSample(
        created_at=datetime.fromisoformat(row["created_at"]),
        source=row["source"],
        strategy=row["strategy"],
        status=row["status"],
        hit_count=int(row["hit_count"]),
        top_score=float(row["top_score"]) if row["top_score"] is not None else None,
        latency_ms=round(float(row["latency_ms"]), 2),
        query_preview=row["query_preview"],
    )


def summarize_retrieval_telemetry(
    database,
    *,
    org_id: str,
    knowledge_base_id: str,
    days: int = 7,
    sample_limit: int = 10,
) -> RetrievalTelemetrySummary:
    window = max(1, min(90, int(days)))
    cutoff = (datetime.now(UTC) - timedelta(days=window)).isoformat()
    with database.connect() as connection:
        rows = connection.execute(
            """SELECT source, strategy, status, hit_count, top_score, latency_ms, query_preview, created_at
            FROM library_retrieval_logs
            WHERE org_id=? AND knowledge_base_id=? AND created_at>=?
            ORDER BY created_at DESC, id DESC
            LIMIT 2000""",
            (org_id, knowledge_base_id, cutoff),
        ).fetchall()

    if not rows:
        return RetrievalTelemetrySummary(knowledge_base_id=knowledge_base_id, window_days=window)

    total = len(rows)
    zero_rows = [
        r for r in rows
        if int(r["hit_count"]) == 0 or r["status"] == "insufficient_evidence"
    ]
    zero_cnt = len(zero_rows)
    latencies = sorted(float(r["latency_ms"]) for r in rows)
    scores = [float(r["top_score"]) for r in rows if r["top_score"] is not None]

    return RetrievalTelemetrySummary(
        knowledge_base_id=knowledge_base_id,
        window_days=window,
        total_queries=total,
        zero_hit_count=zero_cnt,
        zero_hit_rate=round(zero_cnt / total, 4),
        avg_latency_ms=round(sum(latencies) / total, 2),
        p95_latency_ms=_percentile(latencies, 0.95),
        avg_top_score=round(sum(scores) / len(scores), 4) if scores else None,
        zero_hit_samples=tuple(_row_to_sample(r) for r in zero_rows[:sample_limit]),
        recent_events=tuple(_row_to_sample(r) for r in rows[:sample_limit]),
    )
