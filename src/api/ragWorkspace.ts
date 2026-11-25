import { ApiError, jsonRequest, post, request, type UploadedFile } from "./salespilot";

export interface DocumentVersionKey {
  document_id: string;
  version: string;
}

export interface LibraryDocument extends DocumentVersionKey {
  filename: string;
  size_bytes: number;
  uploaded_at: string;
  status: "draft" | "published" | "withdrawn" | "deleting";
  index_status: "pending" | "ready";
}

export interface LibraryFolder {
  folder_id: string;
  parent_id: string | null;
  name: string;
}

export interface LibraryEntry extends LibraryDocument {
  folder_id: string | null;
}

export interface ChunkPreview {
  chunk_id: string;
  page: number | null;
  section: string | null;
  text: string;
  truncated?: boolean;
}

export interface KnowledgeBaseInput {
  name: string;
  description: string;
  embedding_model_id: string;
  access_policy?: "org_published" | "restricted";
}

export interface KnowledgeBase extends KnowledgeBaseInput {
  knowledge_base_id: string;
  index_state: "legacy" | "awaiting_adapter" | "building" | "available" | "failed";
  active_index_version?: string | null;
}

export interface KnowledgeBaseGrant {
  subject_type: "user" | "role" | "group";
  subject_id: string;
}

export interface KnowledgeBaseAcl {
  knowledge_base_id: string;
  access_policy: "org_published" | "restricted";
  grants: KnowledgeBaseGrant[];
}

export interface IndexGcResult {
  org_id: string;
  knowledge_base_id: string;
  purged_versions: string[];
  purged_documents: number;
  purged_chunks: number;
}

export interface RetrievalEventSample {
  created_at: string;
  source: "selection" | "retrieval_test";
  strategy: string;
  status: string;
  hit_count: number;
  top_score: number | null;
  latency_ms: number;
  query_preview: string;
}

export interface RetrievalTelemetrySummary {
  knowledge_base_id: string;
  window_days: number;
  total_queries: number;
  zero_hit_count: number;
  zero_hit_rate: number;
  avg_latency_ms: number;
  p95_latency_ms: number;
  avg_top_score: number | null;
  zero_hit_samples: RetrievalEventSample[];
  recent_events: RetrievalEventSample[];
}

export interface EmbeddingModelOption {
  model_id: string;
  name: string;
  selectable: boolean;
}

export interface IndexAvailability {
  active_index_version: string | null;
  embedding_model_id: string;
  dimensions: number | null;
  ready: boolean;
  reason: string | null;
}

export interface IndexVersion {
  index_version: string;
  embedding_model_id: string;
  state: "legacy" | "ready" | "active" | "superseded" | "building" | "retired" | "failed";
  document_count: number;
  chunk_count: number;
}

export interface RebuildTask {
  task_id: string;
  target_index_version: string;
  state: "pending" | "running" | "succeeded" | "partial_failed" | "failed" | "interrupted";
  total: number;
  completed: number;
  failed: number;
  error_message: string | null;
  items: Array<{ document_id: string; version: string; state: string; last_error: string | null }>;
}

export interface PublishTask {
  task_id: string;
  knowledge_base_id: string;
  document_id: string;
  version: string;
  stage: "queued" | "parsing" | "indexing" | "succeeded" | "failed";
  error_code: string | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export interface ChunkPage {
  total: number;
  chunks: ChunkPreview[];
}

export interface RetrievalHit extends ChunkPreview {
  document_id: string;
  version: string;
  filename: string;
  score: number;
  vector_score?: number | null;
  bm25_score?: number | null;
  rerank_score?: number | null;
}

export interface RetrievalTestOptions {
  mode?: "selection" | "raw_retrieval";
  strategy?: "hybrid" | "vector" | "bm25";
  index_version?: string;
  top_k?: number;
}

export interface RetrievalDiagnostic {
  status: "retrieved" | "needs_input" | "insufficient_evidence" | "spec_query";
  message: string;
  total: number;
  hits: RetrievalHit[];
}

interface ManualAttachment extends DocumentVersionKey {
  filename: string;
}

interface Citation extends ManualAttachment {
  page: number | null;
  section: string | null;
  excerpt: string;
}

export interface TurnInput {
  conversation_id: string;
  client_request_id: string;
  text: string;
  knowledge_base_id?: string;
}

export interface RecommendationTurn {
  turn_id: string;
  conversation_id: string;
  created_at: string;
  status: "recommended" | "needs_input" | "insufficient_evidence";
  answer: string;
  models: string[];
  citations: Citation[];
  manuals: ManualAttachment[];
  has_report: boolean;
}

export interface HistoryEntry {
  turn_id: string;
  created_at: string;
  request_summary: string;
  available: boolean;
  reply: RecommendationTurn | null;
}

export interface ConversationSummary {
  conversation_id: string;
  title: string;
  updated_at: string;
  pinned: boolean;
}

export type ConversationUpdate = Partial<Pick<ConversationSummary, "title" | "pinned">>;

export interface ConversationTurn {
  turn_id: string;
  text: string;
  reply: RecommendationTurn | null;
}

interface ConversationList {
  items: ConversationSummary[];
  next_offset: number | null;
}

interface ConversationHistory {
  conversation_id: string;
  knowledge_base_id?: string;
  turns: ConversationTurn[];
  before: string | null;
}

const documentPath = (key: DocumentVersionKey) =>
  "/library/documents/" + encodeURIComponent(key.document_id) + "/" + encodeURIComponent(key.version);
const conversationPath = (id: string) => "/recommendations/conversations/" + encodeURIComponent(id);
const knowledgeBasePath = (id: string) => "/library/knowledge-bases/" + encodeURIComponent(id);
const rebuildPath = (id: string) => "/library/rebuilds/" + encodeURIComponent(id);

async function download(path: string, filename: string) {
  const response = await request(path);
  const url = URL.createObjectURL(await response.blob());
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
}

export const ragWorkspace = {
  async knowledgeBases(): Promise<KnowledgeBase[] | null> {
    try {
      return (await request("/library/knowledge-bases")).json();
    } catch (failure) {
      if (failure instanceof ApiError && failure.status === 404) return null;
      throw failure;
    }
  },
  async embeddingModels(): Promise<EmbeddingModelOption[]> {
    return (await request("/library/embedding-models")).json();
  },
  async createKnowledgeBase(body: KnowledgeBaseInput): Promise<KnowledgeBase> {
    return (await request("/library/knowledge-bases", post(body))).json();
  },
  async updateKnowledgeBase(id: string, body: { name: string; description?: string }): Promise<KnowledgeBase> {
    return (await request(knowledgeBasePath(id), jsonRequest(body, "PATCH"))).json();
  },
  async indexAvailability(id: string): Promise<IndexAvailability> {
    return (await request(knowledgeBasePath(id) + "/index")).json();
  },
  async indexVersions(id: string): Promise<IndexVersion[]> {
    return (await request(knowledgeBasePath(id) + "/indexes")).json();
  },
  async startRebuild(id: string, modelId: string, taskId: string): Promise<RebuildTask> {
    return (await request(knowledgeBasePath(id) + "/rebuild", post({
      task_id: taskId, embedding_model_id: modelId, background: true,
    }))).json();
  },
  async rebuildTasks(id: string, limit = 10): Promise<RebuildTask[]> {
    return (await request(knowledgeBasePath(id) + "/rebuilds?limit=" + limit)).json();
  },
  async rebuildStatus(taskId: string): Promise<RebuildTask> {
    return (await request(rebuildPath(taskId))).json();
  },
  async resumeRebuild(taskId: string): Promise<RebuildTask> {
    return (await request(rebuildPath(taskId) + "/resume", post({ background: true }))).json();
  },
  async activateIndex(id: string, version: string, rollback: boolean): Promise<void> {
    await request(knowledgeBasePath(id) + "/indexes/" + encodeURIComponent(version) + "/activate", post({
      allow_stale: false, reason: rollback ? "rollback" : "activated",
    }));
  },
  async retireIndex(id: string, version: string): Promise<void> {
    await request(knowledgeBasePath(id) + "/indexes/" + encodeURIComponent(version) + "/retire", {
      method: "POST",
    });
  },
  async purgeRetiredIndexes(id: string, keepSuperseded = 1): Promise<IndexGcResult> {
    return (await request(knowledgeBasePath(id) + "/indexes/gc", post({
      keep_superseded: keepSuperseded,
    }))).json();
  },
  async getKnowledgeBaseAcl(id: string): Promise<KnowledgeBaseAcl> {
    return (await request(knowledgeBasePath(id) + "/acl")).json();
  },
  async updateKnowledgeBaseAcl(
    id: string,
    body: { access_policy: "org_published" | "restricted"; grants: KnowledgeBaseGrant[] },
  ): Promise<KnowledgeBaseAcl> {
    return (await request(knowledgeBasePath(id) + "/acl", jsonRequest(body, "PUT"))).json();
  },
  async telemetry(knowledgeBaseId = "default", days = 7): Promise<RetrievalTelemetrySummary> {
    const query = new URLSearchParams({ knowledge_base_id: knowledgeBaseId, days: String(days) });
    return (await request("/library/telemetry?" + query)).json();
  },
  async browseLibrary(knowledgeBaseId?: string): Promise<{
    folders: LibraryFolder[]; documents: LibraryEntry[]; foldersAvailable: boolean;
  }> {
    try {
      const query = knowledgeBaseId ? "?knowledge_base_id=" + encodeURIComponent(knowledgeBaseId) : "";
      const result = await (await request("/library/browser" + query)).json();
      return { ...result, foldersAvailable: true };
    } catch (failure) {
      if (knowledgeBaseId || !(failure instanceof ApiError) || failure.status !== 404) throw failure;
      const documents: LibraryDocument[] = await (await request("/library/documents")).json();
      return {
        folders: [], foldersAvailable: false,
        documents: documents.map(document => ({ ...document, folder_id: null })),
      };
    }
  },
  async createFolder(name: string, parentId: string | null, knowledgeBaseId?: string): Promise<LibraryFolder> {
    return (await request("/library/folders", post({
      name, parent_id: parentId, knowledge_base_id: knowledgeBaseId,
    }))).json();
  },
  async moveDocument(documentId: string, folderId: string | null): Promise<void> {
    await request("/library/documents/" + encodeURIComponent(documentId) + "/location",
      jsonRequest({ folder_id: folderId }, "PATCH"));
  },
  async uploadDocument(body: {
    file: UploadedFile; version: string; document_id?: string; folder_id?: string | null; knowledge_base_id?: string;
  }): Promise<LibraryDocument> {
    return (await request("/library/documents", post(body))).json();
  },
  async uploadDocumentStream(body: {
    file: File; version: string; document_id?: string; folder_id?: string | null; knowledge_base_id?: string;
  }): Promise<LibraryDocument> {
    const query = new URLSearchParams({ filename: body.file.name, version: body.version });
    if (body.document_id) query.set("document_id", body.document_id);
    if (body.folder_id !== undefined) query.set("folder_id", body.folder_id ?? "");
    if (body.knowledge_base_id) query.set("knowledge_base_id", body.knowledge_base_id);
    return (await request("/library/documents/upload?" + query, {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: body.file,
    })).json();
  },
  async publishDocument(doc: DocumentVersionKey): Promise<LibraryDocument> {
    return (await request(documentPath(doc) + "/publish", { method: "POST" })).json();
  },
  async publishDocumentAsync(doc: DocumentVersionKey): Promise<PublishTask> {
    return (await request(documentPath(doc) + "/publish-async", { method: "POST" })).json();
  },
  async publishTasks(knowledgeBaseId?: string): Promise<PublishTask[]> {
    const query = knowledgeBaseId ? "?knowledge_base_id=" + encodeURIComponent(knowledgeBaseId) : "";
    return (await request("/library/publish-tasks" + query)).json();
  },
  async publishTask(taskId: string): Promise<PublishTask> {
    return (await request("/library/publish-tasks/" + encodeURIComponent(taskId))).json();
  },
  async deleteDocument(doc: DocumentVersionKey): Promise<void> {
    await request(documentPath(doc), { method: "DELETE" });
  },
  async documentVersions(documentId: string): Promise<LibraryDocument[]> {
    return (await request("/library/documents/" + encodeURIComponent(documentId) + "/versions")).json();
  },
  async documentChunks(doc: DocumentVersionKey, offset: number): Promise<ChunkPage> {
    return (await request(documentPath(doc) + "/chunks?offset=" + offset)).json();
  },
  async documentFile(doc: DocumentVersionKey): Promise<Blob> {
    return (await request(documentPath(doc) + "/file")).blob();
  },
  async inspectRetrieval(
    text: string,
    knowledgeBaseId?: string,
    options?: RetrievalTestOptions,
  ): Promise<RetrievalDiagnostic> {
    return (await request("/library/retrieval-test", post({
      text,
      knowledge_base_id: knowledgeBaseId,
      ...options,
    }))).json();
  },
  downloadManual: (doc: ManualAttachment) => download(documentPath(doc) + "/file", doc.filename),
  async sendTurn(body: TurnInput, signal?: AbortSignal): Promise<RecommendationTurn> {
    return (await request("/recommendations/turns", { ...post(body), signal })).json();
  },
  async conversations(offset = 0, search = ""): Promise<ConversationList> {
    const query = new URLSearchParams({ offset: String(offset), q: search });
    return (await request("/recommendations/conversations?" + query)).json();
  },
  async conversation(id: string, before?: string): Promise<ConversationHistory> {
    const query = before ? "?before=" + encodeURIComponent(before) : "";
    return (await request(conversationPath(id) + query)).json();
  },
  async updateConversation(id: string, changes: ConversationUpdate): Promise<void> {
    await request(conversationPath(id), jsonRequest(changes, "PATCH"));
  },
  async deleteConversation(id: string): Promise<void> {
    await request(conversationPath(id), { method: "DELETE" });
  },
  async history(offset = 0): Promise<HistoryEntry[]> {
    return (await request("/recommendations/history?offset=" + offset)).json();
  },
  downloadReport: (turnId: string) =>
    download("/recommendations/history/" + encodeURIComponent(turnId) + "/report.md", "salespilot-" + turnId + ".md"),
};

export function workspaceError(failure: unknown, expire: () => void) {
  if (failure instanceof ApiError && failure.status === 401) expire();
  return failure instanceof Error ? failure.message : "操作未完成，请稍后重试。";
}
