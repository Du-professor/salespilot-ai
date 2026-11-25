import hashlib
import json
import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from langchain_core.embeddings import Embeddings

from app.contracts.rag_workspace import (
    ChunkReadPage,
    DiagnosticHit,
    DocumentVersionKey,
    IndexAvailability,
    IndexDescriptor,
    IndexGcResult,
    IndexScopeDifference,
    IndexSwitchPlan,
    LibraryDocument,
    RagContext,
    RagDocumentError,
    RagIndexConflict,
    RagIndexNotFound,
    RagModelInvalid,
    RagModelUnavailable,
    RagRecommendation,
    RagUnavailable,
    ReadChunk,
    RebuildItem,
    RebuildSource,
    RebuildTask,
    RetrievalDiagnostics,
)

from .chat import ChatModel, build_chat
from .chunking import build_chunks
from .config import RagSettings
from .embeddings import build_embeddings
from .graph import RecommendationGraph
from .model_catalog import WORKSPACE_MODEL_ID, WORKSPACE_REVISION
from .model_registry import RagModelRegistry
from .models import ScoredChunk, StructuredDraft
from .parsing import parse_document
from .pg_store import PgVectorRagStore
from .rerank import Reranker, build_reranker
from .store import LEGACY_INDEX_VERSION, LEGACY_KNOWLEDGE_BASE, RagChunkStore


def build_rag_store(settings: RagSettings) -> RagChunkStore | PgVectorRagStore:
    if settings.store_backend == "postgres":
        dsn = settings.postgres_dsn.get_secret_value() if settings.postgres_dsn is not None else None
        if not dsn:
            raise RagUnavailable("启用 PostgreSQL 索引存储时必须配置 SALESPILOT_POSTGRES_DSN。")
        return PgVectorRagStore(dsn=dsn)
    return RagChunkStore(settings.index_path)

_LEGACY = LEGACY_INDEX_VERSION
_DEFAULT_BASE = LEGACY_KNOWLEDGE_BASE


@dataclass(frozen=True)
class _Target:
    knowledge_base_id: str
    index_version: str
    embedding_model_id: str
    embedding_model_revision: str
    dimensions: int | None
    vector_config: dict
    legacy: bool


def _safe_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


class SqliteRagEngine:
    def __init__(
        self,
        settings: RagSettings,
        *,
        store: RagChunkStore | None = None,
        embedder: Embeddings | None = None,
        chat: ChatModel | None = None,
        chat_resolver: Callable[[str], ChatModel | None] | None = None,
        model_registry: RagModelRegistry | None = None,
        models_dir: Path | None = None,
        local_factory=None,
        reranker: Reranker | None = None,
        remote_embedding_resolver=None,
        reranker_resolver: Callable[[str], Reranker | None] | None = None,
        parser_resolver: Callable[[str], tuple[str, str | None] | None] | None = None,
    ):
        self.settings = settings
        self.store = store or build_rag_store(settings)
        self.embedder = embedder or build_embeddings(settings)
        self.models = model_registry or RagModelRegistry(
            settings,
            self.embedder,
            models_dir=models_dir,
            local_factory=local_factory,
            remote_resolver=remote_embedding_resolver,
        )
        self.chat = chat if chat is not None else build_chat(settings)
        self.reranker = reranker if reranker is not None else build_reranker(settings)
        self.reranker_resolver = reranker_resolver
        self.parser_resolver = parser_resolver
        self.graph = RecommendationGraph(
            store=self.store,
            embedder=self.embedder,
            chat=self.chat,
            top_k=settings.top_k,
            chat_resolver=chat_resolver,
            embedder_for=self._embedder_for_context,
            retrieval_strategy=settings.retrieval_strategy,
            reranker=self.reranker,
            reranker_resolver=reranker_resolver,
        )

    # ----- index metadata -----

    def _legacy_descriptor(self, org_id: str, knowledge_base_id: str) -> IndexDescriptor:
        row = self.store.get_index_version(org_id, knowledge_base_id, _LEGACY)
        documents, chunks = self.store.legacy_counts(org_id)
        if row is not None:
            try:
                vector_config = json.loads(row["vector_config_json"])
            except (TypeError, ValueError):
                vector_config = {}
            return IndexDescriptor(
                org_id=org_id,
                knowledge_base_id=knowledge_base_id,
                index_version=_LEGACY,
                embedding_model_id=row["embedding_model_id"],
                embedding_model_revision=row["embedding_model_revision"],
                dimensions=row["dimensions"],
                vector_config=vector_config,
                state=row["state"],
                document_count=documents,
                chunk_count=chunks,
                source_snapshot_id=row["source_snapshot_id"],
                created_at=_safe_datetime(row["created_at"]),
                updated_at=_safe_datetime(row["updated_at"]),
            )
        default_dims = self.models.descriptor(WORKSPACE_MODEL_ID).dimensions
        return IndexDescriptor(
            org_id=org_id,
            knowledge_base_id=knowledge_base_id,
            index_version=_LEGACY,
            embedding_model_id=WORKSPACE_MODEL_ID,
            embedding_model_revision=WORKSPACE_REVISION,
            dimensions=default_dims,
            vector_config={"provider": self.settings.embedding_provider, "legacy": True},
            state="legacy",
            document_count=documents,
            chunk_count=chunks,
        )

    def _descriptor(self, org_id: str, knowledge_base_id: str, index_version: str) -> IndexDescriptor:
        if index_version == _LEGACY:
            if knowledge_base_id != _DEFAULT_BASE:
                raise RagIndexConflict(
                    "legacy 索引只属于默认知识库。", code="index_version_not_found", retryable=False,
                )
            return self._legacy_descriptor(org_id, knowledge_base_id)
        row = self.store.get_index_version(org_id, knowledge_base_id, index_version)
        if row is None:
            raise RagIndexNotFound("找不到指定的索引版本。", code="index_version_not_found", retryable=False)
        documents, chunks = self.store.index_version_counts(org_id, knowledge_base_id, index_version)
        try:
            vector_config = json.loads(row["vector_config_json"])
        except (TypeError, ValueError):
            vector_config = {}
        return IndexDescriptor(
            org_id=org_id,
            knowledge_base_id=knowledge_base_id,
            index_version=index_version,
            embedding_model_id=row["embedding_model_id"],
            embedding_model_revision=row["embedding_model_revision"],
            dimensions=row["dimensions"],
            vector_config=vector_config,
            state=row["state"],
            document_count=documents,
            chunk_count=chunks,
            source_snapshot_id=row["source_snapshot_id"],
            created_at=_safe_datetime(row["created_at"]),
            updated_at=_safe_datetime(row["updated_at"]),
        )

    def _resolve_scope(self, context: RagContext) -> RagContext:
        knowledge_base_id = context.knowledge_base_id
        index_version = context.index_version
        if index_version is None:
            active = self.store.active_index_version(context.org_id, knowledge_base_id)
            if active is not None:
                index_version = active["index_version"]
            elif knowledge_base_id == _DEFAULT_BASE:
                index_version = _LEGACY
            else:
                raise RagIndexNotFound(
                    "知识库还没有可用的索引版本，请先完成索引接入。",
                    code="index_not_ready",
                    retryable=True,
                )
        descriptor = self._descriptor(context.org_id, knowledge_base_id, index_version)
        if descriptor.state not in {"legacy", "ready", "active", "superseded", "retired"}:
            raise RagIndexNotFound(
                "索引版本尚未就绪，请稍后重试。",
                code="index_not_ready",
                retryable=True,
            )
        if context.embedding_model_id is not None and context.embedding_model_id != descriptor.embedding_model_id:
            raise RagIndexConflict(
                "查询模型与索引版本的模型配置不一致。",
                code="embedding_config_mismatch",
                retryable=False,
            )
        documents = tuple(
            document for document in context.documents
            if document.knowledge_base_id == knowledge_base_id
            and (document.index_version is None or document.index_version == index_version)
        )
        return context.model_copy(update={
            "documents": documents,
            "index_version": descriptor.index_version,
            "embedding_model_id": descriptor.embedding_model_id,
            "embedding_model_revision": descriptor.embedding_model_revision,
        })

    def _embedder_for_context(self, context: RagContext) -> Embeddings:
        if context.index_version in (None, _LEGACY):
            return self.embedder
        row = self.store.get_index_version(context.org_id, context.knowledge_base_id, context.index_version)
        if row is None:
            return self.embedder
        return self.models.embedder(row["embedding_model_id"])

    def _embedder_for_target(self, target: _Target) -> Embeddings:
        if target.legacy:
            return self.embedder
        return self.models.embedder(target.embedding_model_id)

    def _target(self, org_id: str, document: LibraryDocument) -> _Target:
        knowledge_base_id = document.knowledge_base_id
        index_version = document.index_version
        if index_version is None:
            if knowledge_base_id != _DEFAULT_BASE:
                raise RagIndexNotFound(
                    "发布前请先为该知识库创建索引版本。",
                    code="index_not_ready",
                    retryable=True,
                )
            index_version = _LEGACY
        descriptor = self._descriptor(org_id, knowledge_base_id, index_version)
        if descriptor.state not in {"legacy", "ready", "active", "building"}:
            raise RagIndexConflict("索引版本当前不可写入。", code="index_not_ready", retryable=True)
        if document.embedding_model_id is not None and document.embedding_model_id != descriptor.embedding_model_id:
            raise RagIndexConflict(
                "资料模型与索引版本的模型配置不一致。",
                code="embedding_config_mismatch",
                retryable=False,
            )
        if document.embedding_model_revision is not None and document.embedding_model_revision != descriptor.embedding_model_revision:
            raise RagIndexConflict(
                "资料模型 revision 与索引版本不一致。",
                code="embedding_config_mismatch",
                retryable=False,
            )
        return _Target(
            knowledge_base_id=knowledge_base_id,
            index_version=index_version,
            embedding_model_id=descriptor.embedding_model_id,
            embedding_model_revision=descriptor.embedding_model_revision,
            dimensions=descriptor.dimensions,
            vector_config=descriptor.vector_config,
            legacy=index_version == _LEGACY,
        )

    # ----- RagEngine -----

    def index_document(self, org_id: str, document: LibraryDocument, payload: bytes) -> None:
        if not payload:
            raise RagDocumentError("文件内容为空，无法索引。")
        target = self._target(org_id, document)
        embedder = self._embedder_for_target(target)
        external_parser = self.parser_resolver(org_id) if self.parser_resolver is not None else None
        if external_parser is not None:
            sections = parse_document(
                document.filename, payload, self.settings.docling_artifacts, external_parser=external_parser,
            )
        else:
            sections = parse_document(document.filename, payload, self.settings.docling_artifacts)
        truncate_at = int(target.vector_config.get("max_tokens") or 0)
        counter = getattr(embedder, "token_counts", None)
        chunks = build_chunks(
            org_id,
            document.document_id,
            document.version,
            sections,
            max_tokens=truncate_at,
            token_counter=counter if callable(counter) else None,
        )
        if truncate_at and callable(counter):
            try:
                counts = counter([chunk.text for chunk in chunks])
            except RagUnavailable:
                raise
            except Exception as exc:
                raise RagUnavailable("Embedding 分词器不可用，请稍后重试。") from exc
            if len(counts) != len(chunks):
                raise RagUnavailable("Embedding 分词结果不完整。")
            chunks = tuple(
                chunk.model_copy(update={"truncated": True}) if count > truncate_at else chunk
                for chunk, count in zip(chunks, counts, strict=True)
            )
        try:
            vectors = embedder.embed_documents([chunk.text for chunk in chunks])
        except (RagUnavailable, RagDocumentError):
            raise
        except Exception as exc:
            raise RagUnavailable("Embedding 服务不可用。") from exc
        if len(vectors) != len(chunks):
            raise RagUnavailable("Embedding 服务返回的向量数量不完整。")
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1 or 0 in dimensions:
            raise RagUnavailable("Embedding 服务返回的向量维度不一致。")
        actual_dimensions = next(iter(dimensions))
        if target.legacy:
            if self.settings.embedding_provider == "hash" and actual_dimensions != self.settings.embedding_dimensions:
                raise RagUnavailable("Embedding 向量维度与配置不一致，请重新索引。")
        elif target.dimensions is not None and actual_dimensions != target.dimensions:
            raise RagModelInvalid(
                "Embedding 向量维度与索引版本不一致，请重新索引。",
                code="embedding_dimension_mismatch",
            )
        try:
            self.store.replace_document(
                org_id,
                document.document_id,
                document.version,
                hashlib.sha256(payload).hexdigest(),
                chunks,
                vectors,
                knowledge_base_id=target.knowledge_base_id,
                index_version=target.index_version,
                embedding_model_id=target.embedding_model_id,
                embedding_model_revision=target.embedding_model_revision,
                vector_config=target.vector_config,
            )
        except (sqlite3.Error, OSError, ValueError) as exc:
            raise RagUnavailable("索引写入失败，请稍后重试。") from exc

    def remove_document(self, org_id: str, key: DocumentVersionKey) -> None:
        try:
            self.store.remove_document(org_id, key.document_id, key.version)
        except RagUnavailable:
            raise
        except (sqlite3.Error, OSError) as exc:
            raise RagUnavailable("索引清理失败，请稍后重试。") from exc

    def recommend(self, context: RagContext) -> RagRecommendation:
        try:
            validated = RagContext.model_validate(context)
            resolved = self._resolve_scope(validated)
            return self.graph.run(resolved)
        except (RagUnavailable, RagDocumentError, RagIndexConflict):
            raise
        except Exception as exc:
            raise RagUnavailable("选型服务暂时不可用，请稍后重试。") from exc

    # ----- diagnostics and chunks -----

    def _hits_from_scored(
        self, resolved: RagContext, chunks: list[ScoredChunk],
    ) -> tuple[tuple[DiagnosticHit, ...], set[tuple[str, str]]]:
        used = {(item.chunk.document_id, item.chunk.version) for item in chunks}
        seen: set[str] = set()
        hits: list[DiagnosticHit] = []
        for item in chunks:
            if item.chunk.chunk_id in seen:
                continue
            seen.add(item.chunk.chunk_id)
            hits.append(DiagnosticHit(
                chunk_id=item.chunk.chunk_id,
                document_id=item.chunk.document_id,
                version=item.chunk.version,
                knowledge_base_id=resolved.knowledge_base_id,
                index_version=resolved.index_version or _LEGACY,
                embedding_model_id=resolved.embedding_model_id or WORKSPACE_MODEL_ID,
                page=item.chunk.page,
                section=item.chunk.section,
                text=item.chunk.text,
                score=item.score,
                truncated=item.chunk.truncated,
                vector_score=item.vector_score,
                bm25_score=item.bm25_score,
                rerank_score=item.rerank_score,
            ))
        return tuple(hits), used

    def diagnose_detailed(
        self,
        context: RagContext,
        *,
        mode: str = "selection",
        strategy: str | None = None,
        top_k: int | None = None,
    ) -> tuple[RetrievalDiagnostics, set[tuple[str, str]]]:
        validated = RagContext.model_validate(context)
        resolved = self._resolve_scope(validated)
        effective_strategy = strategy or self.settings.retrieval_strategy
        effective_top_k = top_k or self.settings.top_k
        graph = RecommendationGraph(
            store=self.store,
            embedder=self.embedder,
            chat=None,
            top_k=effective_top_k,
            embedder_for=self._embedder_for_context,
            retrieval_strategy=effective_strategy,
            reranker=self.reranker,
            reranker_resolver=self.reranker_resolver,
        )
        if mode == "raw_retrieval":
            if not resolved.documents:
                return RetrievalDiagnostics(
                    status="insufficient_evidence",
                    message="当前知识库没有已发布的资料可供检索。",
                    knowledge_base_id=resolved.knowledge_base_id,
                    index_version=resolved.index_version or _LEGACY,
                ), set()
            scored = graph.retrieve_scored(resolved, strategy=effective_strategy, top_k=effective_top_k)
            if not scored:
                return RetrievalDiagnostics(
                    status="insufficient_evidence",
                    message="当前策略下未召回匹配片段，可尝试切换混合检索或调整关键词。",
                    knowledge_base_id=resolved.knowledge_base_id,
                    index_version=resolved.index_version or _LEGACY,
                ), set()
            hits, used = self._hits_from_scored(resolved, scored)
            return RetrievalDiagnostics(
                status="retrieved",
                message=f"纯召回测试完成（策略：{effective_strategy}）。",
                knowledge_base_id=resolved.knowledge_base_id,
                index_version=resolved.index_version or _LEGACY,
                total=len(scored),
                hits=hits,
            ), used

        for event in graph.graph.stream(
            {"context": resolved.model_dump(mode="json")},
            stream_mode="updates",
            interrupt_before=["shortlist", "verify"],
        ):
            if event.get("route", {}).get("path") == "spec":
                return RetrievalDiagnostics(
                    status="spec_query",
                    message="这条消息走指定型号查证。输入完整选型需求或切换到「纯召回测试」可测试检索召回。",
                    knowledge_base_id=resolved.knowledge_base_id,
                    index_version=resolved.index_version or _LEGACY,
                ), set()
            for node in ("route", "understand", "retrieve"):
                update = event.get(node, {})
                if "draft" in update:
                    draft = StructuredDraft.model_validate(update["draft"])
                    status = "needs_input" if draft.status == "needs_input" else "insufficient_evidence"
                    return RetrievalDiagnostics(
                        status=status,
                        message=draft.answer,
                        knowledge_base_id=resolved.knowledge_base_id,
                        index_version=resolved.index_version or _LEGACY,
                    ), set()
            if "retrieve" in event and "chunks" in event["retrieve"]:
                chunks = [ScoredChunk.model_validate(item) for item in event["retrieve"]["chunks"]]
                hits, used = self._hits_from_scored(resolved, chunks)
                return RetrievalDiagnostics(
                    status="retrieved",
                    message="这是候选召回结果。推荐还需通过硬性条件和引用核验。",
                    knowledge_base_id=resolved.knowledge_base_id,
                    index_version=resolved.index_version or _LEGACY,
                    total=len(chunks),
                    hits=hits,
                ), used
        raise RagUnavailable("检索诊断未产生可用结果。")

    def diagnose(self, context: RagContext) -> RetrievalDiagnostics:
        result, _ = self.diagnose_detailed(context)
        return result

    def read_chunks(
        self,
        org_id: str,
        knowledge_base_id: str,
        index_version: str,
        key: DocumentVersionKey,
        *,
        offset: int = 0,
        limit: int = 20,
    ) -> ChunkReadPage:
        if index_version != _LEGACY:
            descriptor = self._descriptor(org_id, knowledge_base_id, index_version)
            if descriptor.state not in {"ready", "active", "superseded", "retired"}:
                raise RagIndexNotFound("索引版本尚未就绪。", code="index_not_ready", retryable=True)
        total, window = self.store.read_document_chunks(
            org_id,
            key.document_id,
            key.version,
            knowledge_base_id=knowledge_base_id,
            index_version=index_version,
            offset=offset,
            limit=limit,
        )
        return ChunkReadPage(
            total=total,
            chunks=tuple(
                ReadChunk(
                    chunk_id=chunk.chunk_id,
                    page=chunk.page,
                    section=chunk.section,
                    text=chunk.text,
                    truncated=chunk.truncated,
                )
                for chunk in window
            ),
        )

    # ----- index manager -----

    def model_availability(self, model_id: str) -> tuple[bool, str | None, str | None]:
        return self.models.availability(model_id)

    def availability(self, org_id: str, knowledge_base_id: str) -> IndexAvailability:
        active = self.store.active_index_version(org_id, knowledge_base_id)
        if active is not None and active["index_version"] == _LEGACY:
            descriptor = self._legacy_descriptor(org_id, knowledge_base_id)
            return IndexAvailability(
                knowledge_base_id=knowledge_base_id,
                state="legacy",
                active_index_version=_LEGACY,
                embedding_model_id=descriptor.embedding_model_id,
                embedding_model_revision=descriptor.embedding_model_revision,
                dimensions=descriptor.dimensions,
                ready=True,
            )
        if active is None and knowledge_base_id == _DEFAULT_BASE:
            default_dims = self.models.descriptor(WORKSPACE_MODEL_ID).dimensions
            return IndexAvailability(
                knowledge_base_id=knowledge_base_id,
                state="legacy",
                active_index_version=_LEGACY,
                embedding_model_id=WORKSPACE_MODEL_ID,
                embedding_model_revision=WORKSPACE_REVISION,
                dimensions=default_dims,
                ready=True,
            )
        if active is None:
            versions = self.store.list_index_versions(org_id, knowledge_base_id)
            building = [row for row in versions if row["state"] == "building"]
            ready = [row for row in versions if row["state"] in {"ready", "superseded"}]
            if building:
                return IndexAvailability(
                    knowledge_base_id=knowledge_base_id,
                    state="building",
                    embedding_model_id=WORKSPACE_MODEL_ID,
                    ready=False,
                    reason="索引版本正在构建。",
                    retryable=True,
                )
            if ready:
                row = ready[-1]
                return IndexAvailability(
                    knowledge_base_id=knowledge_base_id,
                    state="ready",
                    embedding_model_id=row["embedding_model_id"],
                    embedding_model_revision=row["embedding_model_revision"],
                    dimensions=row["dimensions"],
                    ready=False,
                    reason="索引版本已就绪，等待切换为活动版本。",
                    retryable=True,
                )
            return IndexAvailability(
                knowledge_base_id=knowledge_base_id,
                state="awaiting_adapter",
                embedding_model_id=WORKSPACE_MODEL_ID,
                ready=False,
                reason="知识库尚未建立可用索引版本。",
                retryable=True,
            )
        descriptor = self._descriptor(org_id, knowledge_base_id, active["index_version"])
        ready, reason, code = self.models.availability(descriptor.embedding_model_id)
        if not ready:
            return IndexAvailability(
                knowledge_base_id=knowledge_base_id,
                state="failed",
                active_index_version=descriptor.index_version,
                embedding_model_id=descriptor.embedding_model_id,
                embedding_model_revision=descriptor.embedding_model_revision,
                dimensions=descriptor.dimensions,
                ready=False,
                reason=reason,
                retryable=True,
            )
        return IndexAvailability(
            knowledge_base_id=knowledge_base_id,
            state="available",
            active_index_version=descriptor.index_version,
            embedding_model_id=descriptor.embedding_model_id,
            embedding_model_revision=descriptor.embedding_model_revision,
            dimensions=descriptor.dimensions,
            ready=True,
        )

    def ensure_index_version(
        self,
        org_id: str,
        knowledge_base_id: str,
        embedding_model_id: str,
        *,
        index_version: str | None = None,
        activate: bool = False,
    ) -> IndexDescriptor:
        if index_version == _LEGACY or (index_version is None and knowledge_base_id == _DEFAULT_BASE and embedding_model_id == WORKSPACE_MODEL_ID):
            if knowledge_base_id != _DEFAULT_BASE:
                raise RagIndexConflict("只有默认知识库可以使用 legacy 索引。", code="index_version_not_found")
            return self._legacy_descriptor(org_id, knowledge_base_id)
        ready, reason, code = self.models.availability(embedding_model_id)
        if not ready:
            raise RagModelUnavailable(reason or "Embedding 模型不可用。", code=code or "embedding_model_unavailable")
        descriptor = self.models.check(embedding_model_id)
        index_version = index_version or f"idx-{uuid4().hex[:12]}"
        existing = self.store.get_index_version(org_id, knowledge_base_id, index_version)
        if existing is not None:
            if existing["embedding_model_id"] != embedding_model_id:
                raise RagIndexConflict(
                    "索引版本已绑定其他模型。", code="embedding_config_mismatch", retryable=False,
                )
            if activate and existing["state"] != "active":
                self.store.activate_index_version(org_id, knowledge_base_id, index_version)
            return self._descriptor(org_id, knowledge_base_id, index_version)
        self.store.upsert_index_version(
            org_id,
            knowledge_base_id,
            index_version,
            embedding_model_id,
            descriptor.revision,
            descriptor.dimensions,
            descriptor.vector_config,
            state="active" if activate else "ready",
        )
        if activate:
            self.store.activate_index_version(org_id, knowledge_base_id, index_version)
        return self._descriptor(org_id, knowledge_base_id, index_version)

    def index_descriptor(self, org_id: str, knowledge_base_id: str, index_version: str) -> IndexDescriptor:
        return self._descriptor(org_id, knowledge_base_id, index_version)

    def list_index_versions(self, org_id: str, knowledge_base_id: str) -> tuple[IndexDescriptor, ...]:
        rows = self.store.list_index_versions(org_id, knowledge_base_id)
        descriptors = [self._descriptor(org_id, knowledge_base_id, row["index_version"]) for row in rows]
        if knowledge_base_id == _DEFAULT_BASE and not any(row["index_version"] == _LEGACY for row in rows):
            descriptors.insert(0, self._legacy_descriptor(org_id, knowledge_base_id))
        return tuple(descriptors)

    def mark_ready(self, org_id: str, knowledge_base_id: str, index_version: str) -> IndexDescriptor:
        if index_version == _LEGACY:
            return self._legacy_descriptor(org_id, knowledge_base_id)
        descriptor = self._descriptor(org_id, knowledge_base_id, index_version)
        if descriptor.state in {"building", "failed"}:
            self.store.set_index_version_state(org_id, knowledge_base_id, index_version, "ready")
        return self._descriptor(org_id, knowledge_base_id, index_version)

    def validate_scope(
        self,
        org_id: str,
        knowledge_base_id: str,
        index_version: str,
        documents: tuple[LibraryDocument, ...],
    ) -> IndexSwitchPlan:
        descriptor = self._descriptor(org_id, knowledge_base_id, index_version)
        if index_version == _LEGACY:
            target = {
                (row["document_id"], row["version"]): row["digest"]
                for row in self.store.legacy_documents(org_id)
            }
        else:
            target = {
                (row["document_id"], row["version"]): row["digest"]
                for row in self.store.index_version_documents(org_id, knowledge_base_id, index_version)
            }
        current = {
            (document.document_id, document.version): document.digest
            or f"{document.size_bytes}:{document.filename}"
            for document in documents
        }
        differences: list[IndexScopeDifference] = []
        for key, digest in current.items():
            if key not in target:
                differences.append(IndexScopeDifference(kind="missing", document_id=key[0], version=key[1]))
            elif target[key] != digest:
                differences.append(IndexScopeDifference(kind="changed", document_id=key[0], version=key[1]))
        for key in target:
            if key not in current:
                differences.append(IndexScopeDifference(kind="extra", document_id=key[0], version=key[1]))
        ready = descriptor.state in {"legacy", "ready", "active", "superseded"} and not differences
        return IndexSwitchPlan(
            org_id=org_id,
            knowledge_base_id=knowledge_base_id,
            target_index_version=index_version,
            embedding_model_id=descriptor.embedding_model_id,
            embedding_model_revision=descriptor.embedding_model_revision,
            dimensions=descriptor.dimensions,
            ready=ready,
            scope_changed=bool(differences),
            differences=tuple(differences),
        )

    def mark_active(
        self,
        org_id: str,
        knowledge_base_id: str,
        index_version: str,
        *,
        documents: tuple[LibraryDocument, ...] | None = None,
        allow_stale: bool = False,
    ) -> IndexSwitchPlan:
        descriptor = self._descriptor(org_id, knowledge_base_id, index_version)
        if index_version == _LEGACY:
            plan = self.validate_scope(org_id, knowledge_base_id, index_version, documents) if documents is not None else IndexSwitchPlan(
                org_id=org_id,
                knowledge_base_id=knowledge_base_id,
                target_index_version=index_version,
                embedding_model_id=descriptor.embedding_model_id,
                embedding_model_revision=descriptor.embedding_model_revision,
                dimensions=descriptor.dimensions,
                ready=True,
            )
            if plan.scope_changed and not allow_stale:
                raise RagIndexConflict(
                    "legacy 索引与当前资料范围不一致，请先重新索引。",
                    code="index_scope_changed",
                    retryable=True,
                )
            self.store.activate_legacy(
                org_id,
                knowledge_base_id,
                descriptor.embedding_model_id,
                descriptor.embedding_model_revision,
                descriptor.dimensions,
                descriptor.vector_config,
            )
            return plan
        if descriptor.state == "retired":
            raise RagIndexNotFound("索引版本已退役。", code="index_retired", retryable=False)
        if descriptor.state not in {"ready", "active", "superseded"}:
            raise RagIndexConflict("索引版本尚未就绪，无法切换。", code="index_not_ready", retryable=True)
        plan = self.validate_scope(org_id, knowledge_base_id, index_version, documents) if documents is not None else IndexSwitchPlan(
            org_id=org_id,
            knowledge_base_id=knowledge_base_id,
            target_index_version=index_version,
            embedding_model_id=descriptor.embedding_model_id,
            embedding_model_revision=descriptor.embedding_model_revision,
            dimensions=descriptor.dimensions,
            ready=True,
        )
        if plan.scope_changed and not allow_stale:
            raise RagIndexConflict(
                "索引版本与当前资料范围不一致，请先补齐或重建。",
                code="index_scope_changed",
                retryable=True,
            )
        self.store.activate_index_version(org_id, knowledge_base_id, index_version)
        return plan

    def retire(self, org_id: str, knowledge_base_id: str, index_version: str) -> None:
        descriptor = self._descriptor(org_id, knowledge_base_id, index_version)
        if descriptor.state == "active":
            raise RagIndexConflict("活动索引版本不能直接退役，请先切换。", code="index_task_conflict")
        if descriptor.index_version == _LEGACY:
            raise RagIndexConflict("legacy 索引由默认库兼容路径管理。", code="index_task_conflict")
        self.store.set_index_version_state(org_id, knowledge_base_id, index_version, "retired")

    def purge_retired(
        self,
        org_id: str,
        knowledge_base_id: str,
        *,
        keep_superseded: int = 1,
    ) -> IndexGcResult:
        purged_versions, purged_docs, purged_chunks = self.store.purge_retired_versions(
            org_id, knowledge_base_id, keep_superseded=keep_superseded,
        )
        return IndexGcResult(
            org_id=org_id,
            knowledge_base_id=knowledge_base_id,
            purged_versions=purged_versions,
            purged_documents=purged_docs,
            purged_chunks=purged_chunks,
        )

    # ----- rebuild -----

    def start_rebuild(
        self,
        org_id: str,
        knowledge_base_id: str,
        embedding_model_id: str,
        source: RebuildSource,
        *,
        task_id: str | None = None,
        background: bool = True,
    ) -> RebuildTask:
        if task_id is not None:
            existing = self.store.get_rebuild_task(task_id)
            if existing is not None:
                if (
                    existing["org_id"] != org_id
                    or existing["knowledge_base_id"] != knowledge_base_id
                    or existing["embedding_model_id"] != embedding_model_id
                ):
                    raise RagIndexConflict(
                        "重建任务 ID 已用于其他知识库或模型。",
                        code="index_task_conflict",
                        retryable=False,
                    )
                if existing["state"] in {"pending", "running", "succeeded"}:
                    return self.rebuild_status(task_id)
                return self.resume_rebuild(task_id, source, background=background)
        active = self.store.active_rebuild_task(org_id, knowledge_base_id)
        if active is not None:
            raise RagIndexConflict(
                "当前知识库已有正在运行的重建任务，请等待完成后重试。",
                code="index_task_conflict",
                retryable=True,
            )
        ready, reason, code = self.models.availability(embedding_model_id)
        if not ready:
            raise RagModelUnavailable(reason or "Embedding 模型不可用。", code=code or "embedding_model_unavailable")
        descriptor = self.models.check(embedding_model_id)
        snapshot = tuple(source.snapshot(org_id, knowledge_base_id))
        task_id = task_id or f"rebuild-{uuid4().hex[:12]}"
        index_version = f"idx-{uuid4().hex[:12]}"
        self.store.upsert_index_version(
            org_id,
            knowledge_base_id,
            index_version,
            embedding_model_id,
            descriptor.revision,
            descriptor.dimensions,
            descriptor.vector_config,
            state="building",
            source_snapshot_id=task_id,
        )
        items = tuple(
            (
                document.document_id,
                document.version,
                document.digest or f"{document.size_bytes}:{document.filename}",
            )
            for document in snapshot
        )
        self.store.create_rebuild_task(
            task_id,
            org_id,
            knowledge_base_id,
            index_version,
            embedding_model_id,
            descriptor.revision,
            items,
            snapshot_json=json.dumps(
                [document.model_dump(mode="json") for document in snapshot], ensure_ascii=False,
            ),
        )
        if background:
            thread = threading.Thread(
                target=self._run_rebuild, args=(task_id, source), name=f"rag-rebuild-{task_id}", daemon=True,
            )
            thread.start()
        else:
            self._run_rebuild(task_id, source)
        return self.rebuild_status(task_id)

    def _run_rebuild(self, task_id: str, source: RebuildSource) -> None:
        task = self.store.get_rebuild_task(task_id)
        if task is None:
            return
        self.store.set_rebuild_task_state(task_id, "running")
        self.store.set_index_version_state(
            task["org_id"], task["knowledge_base_id"], task["target_index_version"], "building",
        )
        self.store.refresh_rebuild_counts(task_id)
        try:
            snapshot = {
                (document.document_id, document.version): document
                for document in (
                    LibraryDocument.model_validate(item)
                    for item in json.loads(task["snapshot_json"] or "[]")
                )
            }
            check_published = getattr(source, "is_published", None)
            for item in self.store.rebuild_items(task_id):
                if item["state"] in {"succeeded", "skipped"}:
                    continue
                self.store.set_rebuild_item_state(
                    task_id, item["document_id"], item["version"], "running", increment_attempts=True,
                )
                doc_key = (item["document_id"], item["version"])
                document = snapshot.get(doc_key)
                if document is None:
                    self.store.set_rebuild_item_state(
                        task_id, item["document_id"], item["version"], "failed", last_error="快照中缺少该资料。",
                    )
                    continue
                version_key = DocumentVersionKey(document_id=item["document_id"], version=item["version"])
                try:
                    payload = source.read_payload(task["org_id"], version_key)
                    target_document = document.model_copy(update={
                        "knowledge_base_id": task["knowledge_base_id"],
                        "index_version": task["target_index_version"],
                        "embedding_model_id": task["embedding_model_id"],
                        "embedding_model_revision": task["embedding_model_revision"],
                    })
                    self.index_document(task["org_id"], target_document, payload)
                    if callable(check_published) and not check_published(task["org_id"], version_key):
                        self.store.remove_document_from_version(
                            task["org_id"],
                            task["knowledge_base_id"],
                            task["target_index_version"],
                            item["document_id"],
                            item["version"],
                        )
                        raise RagDocumentError("资料在重建期间已撤回或删除。")
                except (RagUnavailable, RagDocumentError, RagIndexConflict) as exc:
                    self.store.set_rebuild_item_state(
                        task_id, item["document_id"], item["version"], "failed", last_error=str(exc),
                    )
                else:
                    self.store.set_rebuild_item_state(task_id, item["document_id"], item["version"], "succeeded")
        except Exception as exc:
            self.store.set_rebuild_task_state(
                task_id, "failed", error_code="index_rebuild_failed", error_message=str(exc),
            )
            self.store.set_index_version_state(
                task["org_id"], task["knowledge_base_id"], task["target_index_version"], "failed",
            )
            return
        self.store.refresh_rebuild_counts(task_id)
        refreshed = self.store.get_rebuild_task(task_id) or task
        failed = refreshed["failed"]
        completed = refreshed["completed"]
        if failed == 0 and completed == refreshed["total"]:
            self.store.set_rebuild_task_state(task_id, "succeeded")
            self.store.set_index_version_state(
                task["org_id"], task["knowledge_base_id"], task["target_index_version"], "ready",
            )
        elif completed > 0:
            self.store.set_rebuild_task_state(
                task_id, "partial_failed", error_code="index_rebuild_failed",
                error_message="部分资料重建失败，请查看任务项后重试。",
            )
            self.store.set_index_version_state(
                task["org_id"], task["knowledge_base_id"], task["target_index_version"], "failed",
            )
        else:
            self.store.set_rebuild_task_state(
                task_id, "failed", error_code="index_rebuild_failed", error_message="重建任务未完成任何资料。",
            )
            self.store.set_index_version_state(
                task["org_id"], task["knowledge_base_id"], task["target_index_version"], "failed",
            )

    def rebuild_status(self, task_id: str) -> RebuildTask:
        row = self.store.get_rebuild_task(task_id)
        if row is None:
            raise RagIndexNotFound("找不到重建任务。", code="index_version_not_found", retryable=False)
        items = tuple(
            RebuildItem(
                document_id=item["document_id"],
                version=item["version"],
                state=item["state"],
                attempts=item["attempts"],
                last_error=item["last_error"],
            )
            for item in self.store.rebuild_items(task_id)
        )
        return RebuildTask(
            task_id=row["task_id"],
            org_id=row["org_id"],
            knowledge_base_id=row["knowledge_base_id"],
            target_index_version=row["target_index_version"],
            embedding_model_id=row["embedding_model_id"],
            embedding_model_revision=row["embedding_model_revision"],
            state=row["state"],
            total=row["total"],
            completed=row["completed"],
            failed=row["failed"],
            items=items,
            error_code=row["error_code"],
            error_message=row["error_message"],
            created_at=_safe_datetime(row["created_at"]) or datetime.now(UTC),
            updated_at=_safe_datetime(row["updated_at"]) or datetime.now(UTC),
        )

    def list_rebuilds(
        self, org_id: str, knowledge_base_id: str, *, limit: int = 10,
    ) -> tuple[RebuildTask, ...]:
        rows = self.store.list_rebuild_tasks(org_id, knowledge_base_id, limit=limit)
        return tuple(self.rebuild_status(row["task_id"]) for row in rows)

    def resume_rebuild(
        self,
        task_id: str,
        source: RebuildSource,
        *,
        background: bool = True,
    ) -> RebuildTask:
        row = self.store.get_rebuild_task(task_id)
        if row is None:
            raise RagIndexNotFound("找不到重建任务。", code="index_version_not_found", retryable=False)
        if row["state"] in {"pending", "running", "succeeded"}:
            return self.rebuild_status(task_id)
        active = self.store.active_rebuild_task(row["org_id"], row["knowledge_base_id"])
        if active is not None and active["task_id"] != task_id:
            raise RagIndexConflict(
                "当前知识库已有正在运行的重建任务，请等待完成后重试。",
                code="index_task_conflict",
                retryable=True,
            )
        self.store.reset_rebuild_items(task_id)
        self.store.set_rebuild_task_state(task_id, "pending")
        if background:
            thread = threading.Thread(
                target=self._run_rebuild, args=(task_id, source), name=f"rag-rebuild-{task_id}", daemon=True,
            )
            thread.start()
        else:
            self._run_rebuild(task_id, source)
        return self.rebuild_status(task_id)

    def recover_rebuilds(self) -> None:
        self.store.interrupt_running_rebuilds()


def build_rag_engine(
    settings: RagSettings | None = None,
    *,
    chat_resolver: Callable[[str], ChatModel | None] | None = None,
    remote_embedding_resolver=None,
    reranker_resolver: Callable[[str], Reranker | None] | None = None,
    parser_resolver: Callable[[str], tuple[str, str | None] | None] | None = None,
) -> SqliteRagEngine:
    return SqliteRagEngine(
        settings or RagSettings.from_environment(),
        chat_resolver=chat_resolver,
        remote_embedding_resolver=remote_embedding_resolver,
        reranker_resolver=reranker_resolver,
        parser_resolver=parser_resolver,
    )