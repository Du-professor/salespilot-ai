import time
from datetime import UTC, datetime
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, Query, Request
from pydantic import ValidationError

from app.api.errors import ApiProblem
from app.api.knowledge_bases import DEFAULT_BASE_ID, accessible_base_ids, index_scope, require_base
from app.api.session import check_origin, require_permission, require_principal
from app.auth.sessions import Principal
from app.contracts import ApiError
from app.contracts.common import Identifier
from app.contracts.rag_workspace import (
    ConversationHistory, ConversationList, ConversationSummary, ConversationTurn, ConversationUpdate,
    HistoryEntry, LocatedCitation, ManualAttachment, PreviousTurn, RagContext, RagDocumentError, RagIndexConflict,
    RagModelInvalid, RagModelUnavailable, RagRecommendation, RagUnavailable, RecommendationInput, RecommendationTurn,
)
from app.infrastructure.library import download_response, published_documents
from app.infrastructure.sqlite import record_audit
from app.infrastructure.telemetry import record_retrieval_event

router = APIRouter(prefix="/recommendations", tags=["recommendations"])
Actor = Annotated[Principal, Depends(require_principal)]
HISTORY_PAGE_SIZE = 50
CONVERSATION_PAGE_SIZE = 50
CONVERSATION_SEARCH = """
WHERE instr(lower(summary.title), lower(:query)) > 0 OR EXISTS (
    SELECT 1 FROM rag_recommendation_turns AS message
    WHERE message.org_id=:org_id AND message.user_id=:user_id AND message.conversation_id=summary.conversation_id
    AND (
        instr(lower(message.user_text), lower(:query)) > 0 OR (
            instr(lower(json_extract(message.response_json, '$.answer')), lower(:query)) > 0
            AND NOT EXISTS (
                SELECT 1 FROM (
                    SELECT value FROM json_each(message.response_json, '$.citations')
                    UNION ALL
                    SELECT value FROM json_each(message.response_json, '$.manuals')
                ) AS source
                WHERE NOT EXISTS (
                    SELECT 1 FROM library_document_versions AS doc
                    WHERE doc.org_id=:org_id AND doc.status='published'
                    AND doc.document_id=json_extract(source.value, '$.document_id')
                    AND doc.version=json_extract(source.value, '$.version')
                )
            )
        )
    )
)
"""


def _key(reference) -> tuple[str, str]:
    return reference.document_id, reference.version


def _sources(reply) -> set[tuple[str, str]]:
    return {_key(item) for item in (*reply.citations, *reply.manuals)}


def _prior_request(connection, owner, body):
    row = connection.execute(
        """SELECT conversation_id, knowledge_base_id, user_text, response_json FROM rag_recommendation_turns
        WHERE org_id=? AND user_id=? AND client_request_id=?""", (*owner, body.client_request_id),
    ).fetchone()
    if row is not None and (
        row["conversation_id"] != body.conversation_id or row["user_text"] != body.text
        or row["knowledge_base_id"] != body.knowledge_base_id
    ):
        raise ApiProblem(409, "recommendation_request_conflict", "这条请求已用于其他内容或知识库。")
    return row


def _sources_changed():
    return ApiProblem(409, "recommendation_sources_changed", "依据资料的可见状态已变更，请重新发送需求。")


def _conversation_not_found():
    return ApiProblem(404, "conversation_not_found", "找不到这段对话。")


def _is_deleted(connection, scope) -> bool:
    return connection.execute(
        """SELECT 1 FROM rag_conversation_settings
        WHERE org_id=? AND user_id=? AND conversation_id=? AND deleted=1""", scope,
    ).fetchone() is not None


def _require_active_conversation(connection, scope):
    if _is_deleted(connection, scope):
        raise _conversation_not_found()


def _require_saved_turn(connection, scope):
    if connection.execute(
        """SELECT 1 FROM rag_recommendation_turns
        WHERE org_id=? AND user_id=? AND conversation_id=? LIMIT 1""", scope,
    ).fetchone() is None:
        raise _conversation_not_found()


def _replay(row, visible) -> RecommendationTurn:
    reply = RecommendationTurn.model_validate_json(row["response_json"])
    if not _sources(reply) <= visible:
        raise _sources_changed()
    return reply


def _recent_turns(connection, owner, conversation_id):
    return connection.execute(
        """SELECT sequence, knowledge_base_id, user_text, response_json FROM rag_recommendation_turns
        WHERE org_id=? AND user_id=? AND conversation_id=? ORDER BY sequence DESC LIMIT 8""",
        (*owner, conversation_id),
    ).fetchall()


def _generate(engine, context) -> RagRecommendation:
    if engine is None:
        raise ApiProblem(503, "rag_unavailable", "选型 Agent 尚未接入。请在服务接通后重试。")
    try:
        return RagRecommendation.model_validate(engine.recommend(context))
    except RagModelUnavailable as exc:
        raise ApiProblem(503, exc.code, str(exc)) from None
    except (RagModelInvalid, RagIndexConflict) as exc:
        status = 503 if getattr(exc, "retryable", False) else 409
        raise ApiProblem(status, getattr(exc, "code", "rag_unavailable"), str(exc)) from None
    except RagDocumentError:
        raise ApiProblem(503, "rag_unavailable", "索引配置与资料不匹配，请重新发布或重建索引。") from None
    except RagUnavailable:
        raise ApiProblem(503, "rag_unavailable", "选型 Agent 暂不可用，请稍后重试。") from None
    except ValidationError:
        raise ApiProblem(502, "rag_invalid_response", "选型 Agent 未返回可使用的推荐。") from None


@router.post(
    "/turns", response_model=RecommendationTurn, dependencies=[Depends(check_origin)],
    responses={502: {"model": ApiError}, 503: {"model": ApiError}},
)
def recommend(body: RecommendationInput, request: Request, actor: Actor):
    require_permission(actor, "selection:run")
    database = request.app.state.database
    owner = (actor.org_id, actor.user_id)
    scope = (*owner, body.conversation_id)
    with database.connect() as connection:
        connection.execute("BEGIN")
        _require_active_conversation(connection, scope)
        prior = _prior_request(connection, owner, body)
        recent = _recent_turns(connection, owner, body.conversation_id)
        if recent and recent[0]["knowledge_base_id"] != body.knowledge_base_id:
            raise ApiProblem(409, "conversation_knowledge_base_conflict", "请新建对话以切换知识库。")
        require_base(connection, actor, body.knowledge_base_id)
        base_scope = index_scope(connection, actor.org_id, body.knowledge_base_id)
        if body.knowledge_base_id != DEFAULT_BASE_ID and not base_scope["active_index_version"]:
            raise ApiProblem(503, "knowledge_base_index_pending", "此知识库的索引尚未接入。")
        documents = published_documents(connection, actor.org_id, base_id=body.knowledge_base_id)
        visible = {_key(doc): doc for doc in documents}
        if prior is not None:
            return _replay(prior, visible.keys())

    previous_turns = []
    context_sources = set()
    for row in reversed(recent):
        reply = RecommendationTurn.model_validate_json(row["response_json"])
        sources = _sources(reply)
        if sources <= visible.keys():
            previous_turns.append(PreviousTurn(text=row["user_text"], answer=reply.answer))
            context_sources.update(sources)
    t0 = time.perf_counter()
    generated = _generate(request.app.state.rag_engine, RagContext(
        org_id=actor.org_id, user_id=actor.user_id, text=body.text,
        previous_turns=tuple(previous_turns), documents=documents,
        knowledge_base_id=base_scope["knowledge_base_id"],
        index_version=base_scope["active_index_version"],
        embedding_model_id=base_scope["embedding_model_id"],
    ))
    latency_ms = (time.perf_counter() - t0) * 1000.0
    if not _sources(generated) <= visible.keys():
        raise ApiProblem(502, "rag_invalid_response", "选型 Agent 引用了本次无法访问的资料。")
    record_retrieval_event(
        database,
        org_id=actor.org_id,
        knowledge_base_id=body.knowledge_base_id,
        user_id=actor.user_id,
        source="selection",
        strategy="hybrid",
        status=generated.status,
        hit_count=len(generated.citations),
        top_score=None,
        latency_ms=latency_ms,
        query_text=body.text,
    )
    manuals = tuple(
        ManualAttachment(document_id=key[0], version=key[1], filename=visible[key].filename)
        for key in dict.fromkeys(map(_key, generated.manuals))
    )
    turn = RecommendationTurn(
        turn_id=uuid4().hex, conversation_id=body.conversation_id, created_at=datetime.now(UTC),
        status=generated.status, answer=generated.answer, models=generated.models,
        citations=tuple(LocatedCitation(**item.model_dump(), filename=visible[_key(item)].filename)
                        for item in generated.citations),
        manuals=manuals, has_report=generated.include_report,
    )
    archived = bool(turn.manuals or turn.has_report)
    with database.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        _require_active_conversation(connection, scope)
        current = {_key(doc) for doc in published_documents(connection, actor.org_id, base_id=body.knowledge_base_id)}
        current_scope = index_scope(connection, actor.org_id, body.knowledge_base_id)
        prior = _prior_request(connection, owner, body)
        if prior is not None:
            return _replay(prior, current)
        newest = connection.execute(
            """SELECT max(sequence) FROM rag_recommendation_turns
            WHERE org_id=? AND user_id=? AND conversation_id=?""", (*owner, body.conversation_id),
        ).fetchone()[0]
        if (recent[0]["sequence"] if recent else None) != newest:
            raise ApiProblem(409, "conversation_changed", "此会话已有新回复，请重试当前需求。")
        if current_scope["active_index_version"] != base_scope["active_index_version"]:
            raise _sources_changed()
        if not (_sources(turn) | context_sources) <= current:
            raise _sources_changed()
        connection.execute(
            """INSERT INTO rag_recommendation_turns
            (turn_id, org_id, user_id, conversation_id, knowledge_base_id, client_request_id, user_text, response_json, archived)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (turn.turn_id, *owner, body.conversation_id, body.knowledge_base_id, body.client_request_id, body.text,
             turn.model_dump_json(), archived),
        )
        record_audit(connection, org_id=actor.org_id, user_id=actor.user_id, action="recommendation_generated",
                     request_id=request.state.request_id, object_id=turn.turn_id)
    return turn


@router.get("/conversations", response_model=ConversationList)
def conversations(
    request: Request, actor: Actor, offset: Annotated[int, Query(ge=0, le=1_000_000)] = 0,
    q: Annotated[str, Query(max_length=160)] = "",
):
    require_permission(actor, "selection:run")
    query = q.strip()
    search = CONVERSATION_SEARCH if query else ""
    with request.app.state.database.connect() as connection:
        rows = connection.execute(
            f"""WITH sessions AS (
                SELECT conversation_id, min(sequence) AS first_turn, max(sequence) AS last_turn
                FROM rag_recommendation_turns
                WHERE org_id=:org_id AND user_id=:user_id
                GROUP BY conversation_id
            ), summary AS (
                SELECT sessions.conversation_id, sessions.last_turn,
                       coalesce(settings.title, substr(first.user_text, 1, 160)) AS title,
                       coalesce(settings.pinned, 0) AS pinned
                FROM sessions
                JOIN rag_recommendation_turns AS first ON first.sequence=sessions.first_turn
                LEFT JOIN rag_conversation_settings AS settings
                    ON settings.conversation_id=sessions.conversation_id
                    AND settings.org_id=:org_id AND settings.user_id=:user_id
                WHERE coalesce(settings.deleted, 0)=0
            )
            SELECT summary.conversation_id, summary.title, summary.pinned,
                   json_extract(last.response_json, '$.created_at') AS updated_at
            FROM summary JOIN rag_recommendation_turns AS last ON last.sequence=summary.last_turn
            {search}
            ORDER BY summary.pinned DESC, summary.last_turn DESC LIMIT :limit OFFSET :offset""",
            {"org_id": actor.org_id, "user_id": actor.user_id, "query": query,
             "limit": CONVERSATION_PAGE_SIZE + 1, "offset": offset},
        ).fetchall()
    return ConversationList(
        items=[ConversationSummary(conversation_id=row["conversation_id"], title=row["title"],
                                   updated_at=row["updated_at"], pinned=bool(row["pinned"]))
               for row in rows[:CONVERSATION_PAGE_SIZE]],
        next_offset=offset + CONVERSATION_PAGE_SIZE if len(rows) > CONVERSATION_PAGE_SIZE else None,
    )


@router.get("/conversations/{conversation_id}", response_model=ConversationHistory)
def conversation_history(conversation_id: Identifier, request: Request, actor: Actor, before: Identifier | None = None):
    require_permission(actor, "selection:run")
    scope = (actor.org_id, actor.user_id, conversation_id)
    with request.app.state.database.connect() as connection:
        connection.execute("BEGIN")
        _require_active_conversation(connection, scope)
        if before is None:
            rows = connection.execute(
                """SELECT turn_id, knowledge_base_id, user_text, response_json FROM rag_recommendation_turns
                WHERE org_id=? AND user_id=? AND conversation_id=? ORDER BY sequence DESC LIMIT ?""",
                (*scope, CONVERSATION_PAGE_SIZE + 1),
            ).fetchall()
            if not rows:
                raise _conversation_not_found()
            base_id = rows[0]["knowledge_base_id"]
        else:
            cursor = connection.execute(
                """SELECT sequence, knowledge_base_id FROM rag_recommendation_turns
                WHERE org_id=? AND user_id=? AND conversation_id=? AND turn_id=?""", (*scope, before),
            ).fetchone()
            if cursor is None:
                raise _conversation_not_found()
            base_id = cursor["knowledge_base_id"]
            rows = connection.execute(
                """SELECT turn_id, user_text, response_json FROM rag_recommendation_turns
                WHERE org_id=? AND user_id=? AND conversation_id=? AND sequence<?
                ORDER BY sequence DESC LIMIT ?""", (*scope, cursor["sequence"], CONVERSATION_PAGE_SIZE + 1),
            ).fetchall()
        allowed_bases = accessible_base_ids(connection, actor)
        visible = {_key(doc) for doc in published_documents(connection, actor.org_id, allowed_bases=allowed_bases)}
    turns = []
    for row in reversed(rows[:CONVERSATION_PAGE_SIZE]):
        reply = RecommendationTurn.model_validate_json(row["response_json"])
        turns.append(ConversationTurn(turn_id=row["turn_id"], text=row["user_text"],
                                      reply=reply if _sources(reply) <= visible else None))
    return ConversationHistory(
        conversation_id=conversation_id, knowledge_base_id=base_id, turns=turns,
        before=turns[0].turn_id if len(rows) > CONVERSATION_PAGE_SIZE else None,
    )


@router.patch("/conversations/{conversation_id}", status_code=204, dependencies=[Depends(check_origin)])
def update_conversation(conversation_id: Identifier, body: ConversationUpdate, request: Request, actor: Actor):
    require_permission(actor, "selection:run")
    scope = (actor.org_id, actor.user_id, conversation_id)
    with request.app.state.database.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        _require_active_conversation(connection, scope)
        _require_saved_turn(connection, scope)
        connection.execute(
            """INSERT INTO rag_conversation_settings (org_id, user_id, conversation_id, title, pinned)
            VALUES (?, ?, ?, ?, coalesce(?, 0))
            ON CONFLICT (org_id, user_id, conversation_id) DO UPDATE SET
                title=coalesce(excluded.title, title), pinned=coalesce(?, pinned)""",
            (*scope, body.title, body.pinned, body.pinned),
        )


@router.delete("/conversations/{conversation_id}", status_code=204, dependencies=[Depends(check_origin)])
def delete_conversation(conversation_id: Identifier, request: Request, actor: Actor):
    require_permission(actor, "selection:run")
    scope = (actor.org_id, actor.user_id, conversation_id)
    with request.app.state.database.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        if _is_deleted(connection, scope):
            return
        _require_saved_turn(connection, scope)
        connection.execute(
            """INSERT INTO rag_conversation_settings (org_id, user_id, conversation_id, deleted)
            VALUES (?, ?, ?, 1)
            ON CONFLICT (org_id, user_id, conversation_id) DO UPDATE SET title=NULL, pinned=0, deleted=1""", scope,
        )
        connection.execute(
            """DELETE FROM rag_recommendation_turns
            WHERE org_id=? AND user_id=? AND conversation_id=? AND archived=0""", scope,
        )
        record_audit(connection, org_id=actor.org_id, user_id=actor.user_id, action="conversation_deleted",
                     request_id=request.state.request_id, object_id=conversation_id)


@router.get("/history", response_model=list[HistoryEntry])
def history(request: Request, actor: Actor, offset: Annotated[int, Query(ge=0, le=1_000_000)] = 0):
    require_permission(actor, "report:read")
    with request.app.state.database.connect() as connection:
        connection.execute("BEGIN")
        allowed_bases = accessible_base_ids(connection, actor)
        visible = {_key(doc) for doc in published_documents(connection, actor.org_id, allowed_bases=allowed_bases)}
        rows = connection.execute(
            """SELECT user_text, response_json FROM rag_recommendation_turns
            WHERE org_id=? AND user_id=? AND archived=1 ORDER BY sequence DESC LIMIT ? OFFSET ?""",
            (actor.org_id, actor.user_id, HISTORY_PAGE_SIZE, offset),
        )
        entries = []
        for row in rows:
            reply = RecommendationTurn.model_validate_json(row["response_json"])
            available = _sources(reply) <= visible
            entries.append(HistoryEntry(
                turn_id=reply.turn_id, created_at=reply.created_at, request_summary=row["user_text"][:160],
                available=available, reply=reply if available else None,
            ))
        return entries


@router.get("/history/{turn_id}/report.md")
def report(turn_id: Identifier, request: Request, actor: Actor):
    require_permission(actor, "report:read")
    with request.app.state.database.connect() as connection:
        connection.execute("BEGIN")
        row = connection.execute(
            """SELECT user_text, response_json FROM rag_recommendation_turns
            WHERE org_id=? AND user_id=? AND turn_id=? AND archived=1""",
            (actor.org_id, actor.user_id, turn_id),
        ).fetchone()
        if row is None:
            raise ApiProblem(404, "report_not_found", "找不到可下载的推荐报告。")
        reply = RecommendationTurn.model_validate_json(row["response_json"])
        allowed_bases = accessible_base_ids(connection, actor)
        visible = {_key(doc): doc for doc in published_documents(connection, actor.org_id, allowed_bases=allowed_bases)}
        if not reply.has_report or not _sources(reply) <= visible.keys():
            raise ApiProblem(404, "report_not_found", "找不到可下载的推荐报告。")
    lines = ["# 产品推荐报告", "", "## 客户需求", "", row["user_text"], "",
             "## 推荐型号", "", *[f"- {model}" for model in reply.models],
             "", "## 推荐说明", "", reply.answer, "", "## 手册依据", ""]
    for citation in reply.citations:
        doc = visible[_key(citation)]
        location = " · ".join(part for part in (
            f"第 {citation.page} 页" if citation.page else None, citation.section,
        ) if part)
        lines.extend([f"- {doc.filename} · {doc.version} · {location}",
                      *[f"  > {line}" for line in citation.excerpt.splitlines()]])
    return download_response("\n".join(lines) + "\n", f"salespilot-{turn_id}.md")
