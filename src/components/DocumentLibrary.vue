<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import { IconDownload, IconFile, IconFolder, IconFolderPlus, IconTrash, IconUpload } from "@tabler/icons-vue";
import { ApiError } from "../api/salespilot";
import {
  ragWorkspace, workspaceError, type IndexAvailability, type KnowledgeBase, type LibraryEntry, type LibraryFolder,
  type PublishTask,
} from "../api/ragWorkspace";
import LibraryDetail from "./LibraryDetail.vue";
import LibraryIndexManagement from "./LibraryIndexManagement.vue";
import LibraryRetrievalTest from "./LibraryRetrievalTest.vue";

const MAX_UPLOAD_BYTES = 200_000_000;
const props = defineProps<{ canManage: boolean; knowledgeBase?: KnowledgeBase; accountScope?: string }>();
const emit = defineEmits<{ expired: [] }>();
const folders = ref<LibraryFolder[]>([]);
const foldersAvailable = ref(false);
const documents = ref<LibraryEntry[]>([]);
const activeFolder = ref<string | null>(null);
const folderName = ref("");
const creatingFolder = ref(false);
const moving = ref<LibraryEntry | null>(null);
const destination = ref<string | null>(null);
const loading = ref(true);
const busy = ref(false);
const publishing = ref(false);
const publishTasks = ref<Record<string, PublishTask>>({});
const error = ref("");
const notice = ref("");
const fileInput = ref<HTMLInputElement | null>(null);
const files = ref<File[]>([]);
const uploadIssues = ref<{ name: string; message: string }[]>([]);
const newVersionOf = ref<LibraryEntry | null>(null);
const version = ref("v1");
const selectedDocument = ref<LibraryEntry | null>(null);
const view = ref<"documents" | "retrieval" | "indexes">("documents");
const search = ref("");
const fileType = ref("");
const statusFilter = ref("");
const index = ref<IndexAvailability | null>(null);
let alive = true;
let publishPollTimer: ReturnType<typeof setTimeout> | undefined;
const readyForRetrieval = computed(() => index.value?.ready ?? (
  !props.knowledgeBase || props.knowledgeBase.index_state === "legacy" || Boolean(props.knowledgeBase.active_index_version)
));
const readyForPublish = computed(() => index.value !== null || readyForRetrieval.value);
const scopeId = computed(() => {
  const id = props.knowledgeBase?.knowledge_base_id;
  return id === "default" ? undefined : id;
});

const statusLabel = { draft: "待发布", published: "已发布", withdrawn: "已撤回", deleting: "待完成删除" };
const statusTone = { draft: "unknown", published: "pass", withdrawn: "fail", deleting: "unknown" };
const indexLabel = { ready: "索引完成", pending: "待索引" };
const publishStageLabel: Record<PublishTask["stage"], string> = {
  queued: "排队等待索引",
  parsing: "正在解析资料",
  indexing: "正在向量化入库",
  succeeded: "索引完成",
  failed: "发布失败",
};
function docKey(doc: { document_id: string; version: string }) {
  return doc.document_id + "/" + doc.version;
}
function activeTask(doc: LibraryEntry) {
  return publishTasks.value[docKey(doc)];
}
function isTaskRunning(task: PublishTask | undefined) {
  return task !== undefined && (task.stage === "queued" || task.stage === "parsing" || task.stage === "indexing");
}
const folderById = computed(() => new Map(folders.value.map(folder => [folder.folder_id, folder])));
const subfolders = computed(() => folders.value.filter(folder => folder.parent_id === activeFolder.value));
const visibleDocuments = computed(() => {
  const query = search.value.trim().toLocaleLowerCase();
  return documents.value.filter(doc => (
    doc.folder_id === activeFolder.value
    && (!query || doc.filename.toLocaleLowerCase().includes(query))
    && (!fileType.value || doc.filename.toLocaleLowerCase().endsWith("." + fileType.value))
    && (!statusFilter.value || doc.status === statusFilter.value)
  ));
});
const crumbs = computed(() => folderTrail(activeFolder.value));

function folderTrail(id: string | null) {
  const trail: LibraryFolder[] = [];
  let folder = id === null ? undefined : folderById.value.get(id);
  while (folder) {
    trail.push(folder);
    folder = folder.parent_id === null ? undefined : folderById.value.get(folder.parent_id);
  }
  return trail.reverse();
}

const folderChoices = computed(() => folders.value.map(folder => ({
  id: folder.folder_id,
  label: folderTrail(folder.folder_id).map(item => item.name).join(" / "),
})).sort((left, right) => left.label.localeCompare(right.label, "zh-CN")));

async function run(action: () => Promise<void>) {
  busy.value = true;
  error.value = "";
  notice.value = "";
  try {
    await action();
  } catch (failure) {
    error.value = workspaceError(failure, () => emit("expired"));
  } finally {
    busy.value = false;
  }
}

async function refresh() {
  loading.value = true;
  await run(loadBrowser);
  loading.value = false;
}

async function loadBrowser() {
  const result = await ragWorkspace.browseLibrary(scopeId.value);
  folders.value = result.folders;
  foldersAvailable.value = result.foldersAvailable;
  documents.value = result.documents;
  if (activeFolder.value !== null && !folderById.value.has(activeFolder.value)) navigate(null);
}

function resetFiles() {
  files.value = [];
  uploadIssues.value = [];
  if (fileInput.value) fileInput.value.value = "";
}

function navigate(folderId: string | null) {
  activeFolder.value = folderId;
  creatingFolder.value = false;
  folderName.value = "";
  moving.value = null;
  targetDocument(null);
  error.value = "";
  notice.value = "";
}

function showView(next: "documents" | "retrieval" | "indexes") {
  selectedDocument.value = null;
  view.value = next;
}

async function readIndex() {
  if (!props.canManage || !props.knowledgeBase) return;
  try {
    index.value = await ragWorkspace.indexAvailability(props.knowledgeBase.knowledge_base_id);
  } catch (failure) {
    if (failure instanceof ApiError && failure.status === 401) emit("expired");
  }
}

function createFolder() {
  const name = folderName.value.trim();
  if (!name) return;
  void run(async () => {
    const folder = await ragWorkspace.createFolder(name, activeFolder.value, scopeId.value);
    folders.value = [...folders.value, folder];
    folders.value.sort((left, right) => left.name.localeCompare(right.name, "zh-CN"));
    folderName.value = "";
    creatingFolder.value = false;
    notice.value = "已创建「" + folder.name + "」。";
  });
}

function openMove(doc: LibraryEntry) {
  moving.value = doc;
  destination.value = doc.folder_id;
}

function moveDocument() {
  const doc = moving.value;
  const target = destination.value;
  if (!doc || doc.folder_id === target) return;
  void run(async () => {
    await ragWorkspace.moveDocument(doc.document_id, target);
    documents.value = documents.value.map(item => item.document_id === doc.document_id
      ? { ...item, folder_id: target } : item);
    moving.value = null;
    if (newVersionOf.value?.document_id === doc.document_id) targetDocument(null);
    notice.value = "已移动资料及其所有版本。";
  });
}

function selectFile(event: Event) {
  files.value = Array.from((event.target as HTMLInputElement).files ?? []);
  uploadIssues.value = [];
}

function targetDocument(doc: LibraryEntry | null) {
  resetFiles();
  newVersionOf.value = doc;
  version.value = doc ? "" : "v1";
  if (doc) fileInput.value?.focus();
}

function upload() {
  const selected = files.value;
  if (!selected.length || !version.value) return;
  const target = newVersionOf.value;
  const folderId = activeFolder.value;
  const tag = version.value;
  void run(async () => {
    const failed: File[] = [];
    uploadIssues.value = [];
    for (const item of selected) {
      try {
        if (item.size === 0 || item.size > MAX_UPLOAD_BYTES) {
          throw new Error(`请选择不超过 ${MAX_UPLOAD_BYTES / 1_000_000} MB 的非空文件。`);
        }
        const placement = target ? { document_id: target.document_id }
          : foldersAvailable.value ? { folder_id: folderId } : {};
        const uploaded = await ragWorkspace.uploadDocumentStream({
          file: item, version: tag, ...placement, knowledge_base_id: scopeId.value,
        });
        const assignedFolder = documents.value.find(doc => doc.document_id === uploaded.document_id);
        documents.value = [{ ...uploaded, folder_id: assignedFolder ? assignedFolder.folder_id : folderId },
          ...documents.value.filter(doc => doc.document_id !== uploaded.document_id || doc.version !== uploaded.version)];
      } catch (failure) {
        if (failure instanceof ApiError && failure.status === 401) throw failure;
        failed.push(item);
        uploadIssues.value.push({
          name: item.name,
          message: workspaceError(failure, () => emit("expired")),
        });
      }
    }
    files.value = failed;
    if (fileInput.value) fileInput.value.value = "";
    const saved = selected.length - failed.length;
    if (!failed.length) targetDocument(null);
    if (saved) notice.value = !readyForPublish.value ? "已保存 " + saved + " 份草稿。"
      : "已保存 " + saved + " 份资料。待发布资料可以逐份建立索引。";
  });
}

function schedulePublishPoll() {
  clearTimeout(publishPollTimer);
  const hasRunning = Object.values(publishTasks.value).some(isTaskRunning);
  if (hasRunning && alive) {
    publishPollTimer = setTimeout(() => void loadPublishTasks(true), 1200);
  }
}

async function loadPublishTasks(fromPoll = false) {
  if (!props.canManage || !alive) return;
  try {
    const previousRunning = new Set(
      Object.values(publishTasks.value).filter(isTaskRunning).map(task => docKey(task)),
    );
    const items = await ragWorkspace.publishTasks(scopeId.value);
    if (!alive) return;
    const latest: Record<string, PublishTask> = {};
    for (const item of items) {
      const key = docKey(item);
      if (!(key in latest)) latest[key] = item;
    }
    publishTasks.value = latest;
    const finishedNow = Array.from(previousRunning).some(key => !isTaskRunning(latest[key]));
    if (fromPoll && finishedNow) {
      await loadBrowser();
      await readIndex();
      const succeeded = Array.from(previousRunning).some(key => latest[key]?.stage === "succeeded");
      if (succeeded) notice.value = "索引完成，资料已发布。";
    }
    schedulePublishPoll();
  } catch (failure) {
    if (failure instanceof ApiError && failure.status === 401) emit("expired");
  }
}

function publish(doc: LibraryEntry) {
  publishing.value = true;
  void run(async () => {
    try {
      const task = await ragWorkspace.publishDocumentAsync(doc);
      publishTasks.value = { ...publishTasks.value, [docKey(doc)]: task };
      if (task.stage === "succeeded") {
        await loadBrowser();
        await readIndex();
        notice.value = "索引完成，资料已发布。";
      } else {
        notice.value = "已加入异步索引队列，正在后台解析与向量化…";
        schedulePublishPoll();
      }
    } catch (failure) {
      if (!(failure instanceof ApiError) || failure.status !== 404) throw failure;
      const updated = await ragWorkspace.publishDocument(doc);
      documents.value = documents.value.map(item =>
        item.document_id === updated.document_id && item.version === updated.version
          ? { ...updated, folder_id: item.folder_id } : item);
      notice.value = "索引完成，资料已发布。";
      await readIndex();
    }
  }).finally(() => { publishing.value = false; });
}

function removeDocument(doc: LibraryEntry) {
  if (!window.confirm(`删除「${doc.filename}」版本 ${doc.version}？原件和索引将被清理，相关引用和报告可能无法打开。此操作不可恢复。`)) return;
  const sameVersion = (item: LibraryEntry) => item.document_id === doc.document_id && item.version === doc.version;
  void run(async () => {
    try {
      await ragWorkspace.deleteDocument(doc);
    } catch (failure) {
      if (failure instanceof ApiError && failure.code === "rag_index_removal_failed") {
        documents.value = documents.value.map(item => sameVersion(item) ? { ...item, status: "deleting" } : item);
      }
      throw failure;
    }
    documents.value = documents.value.filter(item => !sameVersion(item));
    if (newVersionOf.value && sameVersion(newVersionOf.value)) targetDocument(null);
    notice.value = "已删除所选资料版本。";
  });
}

onMounted(() => {
  void refresh();
  void readIndex();
  void loadPublishTasks();
});
onBeforeUnmount(() => {
  alive = false;
  clearTimeout(publishPollTimer);
});
</script>

<template>
  <nav class="library-view-nav" aria-label="资料库视图">
    <button class="text-button" type="button" :disabled="busy"
      :aria-current="view === 'documents' && !selectedDocument ? 'page' : undefined"
      @click="showView('documents')">资料管理</button>
    <button v-if="canManage && foldersAvailable && readyForRetrieval" class="text-button" type="button" :disabled="busy"
      :aria-current="view === 'retrieval' ? 'page' : undefined" @click="showView('retrieval')">检索测试</button>
    <button v-if="canManage && index && knowledgeBase" class="text-button" type="button" :disabled="busy"
      :aria-current="view === 'indexes' ? 'page' : undefined" @click="showView('indexes')">索引管理</button>
  </nav>
  <p v-if="!readyForRetrieval && view === 'documents'" class="b2-panel library-index-notice" role="status">
    {{ index?.reason ?? "等待索引服务接入。可先上传和整理草稿。" }}
  </p>
  <LibraryDetail v-if="selectedDocument" :key="selectedDocument.document_id" :document="selectedDocument"
    @back="selectedDocument = null" @expired="emit('expired')" />
  <LibraryRetrievalTest v-else-if="view === 'retrieval'" :knowledge-base-id="scopeId" @expired="emit('expired')" />
  <LibraryIndexManagement v-else-if="view === 'indexes' && knowledgeBase && index"
    :knowledge-base="knowledgeBase" :availability="index" :account-scope="accountScope" @updated="index = $event"
    @expired="emit('expired')" />
  <template v-else>
  <section v-if="foldersAvailable" class="b2-panel library-folders" aria-label="资料文件夹">
    <div class="section-heading">
      <nav class="library-breadcrumb" aria-label="当前文件夹路径">
        <button v-if="crumbs.length" class="text-button" type="button" :disabled="busy"
          @click="navigate(null)">{{ knowledgeBase?.name ?? "资料库" }}</button>
        <strong v-else aria-current="location">{{ knowledgeBase?.name ?? "资料库" }}</strong>
        <template v-for="(folder, index) in crumbs" :key="folder.folder_id">
          <span aria-hidden="true">/</span>
          <strong v-if="index === crumbs.length - 1" aria-current="location">{{ folder.name }}</strong>
          <button v-else class="text-button" type="button" :disabled="busy"
            @click="navigate(folder.folder_id)">{{ folder.name }}</button>
        </template>
      </nav>
      <button v-if="canManage && !creatingFolder" class="text-button" type="button" :disabled="busy"
        @click="creatingFolder = true"><IconFolderPlus :size="16" aria-hidden="true" /> 新建文件夹</button>
    </div>
    <form v-if="creatingFolder" class="library-inline-form" aria-label="新建文件夹" @submit.prevent="createFolder">
      <label class="form-field">
        <span>文件夹名称</span>
        <input v-model.trim="folderName" maxlength="100" required :disabled="busy" />
      </label>
      <button class="primary-button" type="submit" :disabled="busy || !folderName">创建</button>
      <button class="text-button" type="button" :disabled="busy"
        @click="creatingFolder = false; folderName = ''">取消</button>
    </form>
    <ul v-if="subfolders.length" class="library-folder-list">
      <li v-for="folder in subfolders" :key="folder.folder_id">
        <button class="text-button library-folder" type="button" :disabled="busy" @click="navigate(folder.folder_id)">
          <IconFolder :size="18" aria-hidden="true" /> <span>{{ folder.name }}</span>
        </button>
      </li>
    </ul>
  </section>
  <form v-if="canManage" class="b2-panel library-form" aria-label="上传产品资料" @submit.prevent="upload">
    <div class="section-heading">
      <h2>{{ newVersionOf ? "上传新版本" : "上传产品资料" }}</h2>
      <button v-if="newVersionOf" class="text-button" type="button" :disabled="busy"
        @click="targetDocument(null)">取消新增版本</button>
    </div>
    <p v-if="newVersionOf" class="intro">{{ newVersionOf.filename }}</p>
    <div class="library-fields">
      <label class="form-field">
        <span>资料文件</span>
        <input ref="fileInput" type="file" accept=".pdf,.docx,.pptx,.txt,.md"
          :multiple="!newVersionOf" :disabled="busy" @change="selectFile" />
      </label>
      <label class="form-field">
        <span>版本号</span>
        <input v-model.trim="version" maxlength="128" pattern="[A-Za-z0-9][A-Za-z0-9._\-]*"
          required :disabled="busy" />
      </label>
    </div>
    <div class="b2-actions">
      <button class="primary-button" type="submit" :disabled="busy || !files.length || !version">
        <IconUpload :size="16" /> 上传资料
      </button>
      <span class="muted">PDF、DOCX、PPTX、TXT、Markdown，单文件不超过 200 MB</span>
    </div>
    <ul v-if="uploadIssues.length" class="library-upload-issues" role="alert">
      <li v-for="(issue, index) in uploadIssues" :key="index">{{ issue.name }} · {{ issue.message }}</li>
    </ul>
  </form>

  <p v-if="error" class="b1-error" role="alert">{{ error }}</p>
  <p v-if="publishing" class="intro" role="status">正在建立资料索引…</p>
  <p v-else-if="notice" class="feedback" role="status">{{ notice }}</p>

  <section class="b2-panel library-list" aria-label="资料列表">
    <div class="section-heading">
      <h2>资料列表</h2>
      <button class="text-button" type="button" :disabled="busy" @click="refresh">刷新列表</button>
    </div>
    <div class="library-filters">
      <label class="form-field">
        <span>搜索文件名</span>
        <input v-model="search" type="search" placeholder="输入文件名" />
      </label>
      <label class="form-field">
        <span>文件类型</span>
        <select v-model="fileType">
          <option value="">全部类型</option>
          <option v-for="type in ['pdf', 'docx', 'pptx', 'txt', 'md']" :key="type" :value="type">
            {{ type === 'md' ? 'Markdown' : type.toUpperCase() }}
          </option>
        </select>
      </label>
      <label class="form-field">
        <span>发布状态</span>
        <select v-model="statusFilter">
          <option value="">全部状态</option>
          <option v-for="(label, state) in statusLabel" :key="state" :value="state">{{ label }}</option>
        </select>
      </label>
    </div>
    <p v-if="loading" class="intro" role="status">正在读取资料…</p>
    <p v-else-if="!visibleDocuments.length && !error" class="intro">
      {{ search.trim() || fileType || statusFilter ? "没有符合筛选条件的资料。" : "此目录暂无可查看的资料。" }}
    </p>
    <ul v-else class="workspace-list">
      <li v-for="doc in visibleDocuments" :key="doc.document_id + '/' + doc.version" class="workspace-row">
        <div class="workspace-row-title">
          <IconFile :size="17" :stroke="1.7" aria-hidden="true" />
          <button v-if="foldersAvailable" class="library-document-link" type="button" :disabled="busy"
            @click="selectedDocument = doc">{{ doc.filename }}</button>
          <strong v-else>{{ doc.filename }}</strong>
          <span class="outcome" :class="statusTone[doc.status]">{{ statusLabel[doc.status] }}</span>
        </div>
        <p class="muted">版本 {{ doc.version }} ·
          {{ doc.status === "deleting"
            ? "待清理索引"
            : isTaskRunning(activeTask(doc))
              ? publishStageLabel[activeTask(doc)!.stage]
              : indexLabel[doc.index_status] }} ·
          {{ Math.ceil(doc.size_bytes / 1024) }} KB</p>
        <p v-if="doc.status === 'draft' && activeTask(doc)?.stage === 'failed' && activeTask(doc)?.error_message"
          class="b1-error" role="alert">{{ activeTask(doc)!.error_message }}</p>
        <div class="b2-actions">
          <button v-if="doc.status === 'draft' || doc.status === 'published'" class="text-button" type="button" :disabled="busy"
            @click="run(() => ragWorkspace.downloadManual(doc))"><IconDownload :size="15" /> 下载原件</button>
          <template v-if="canManage">
            <button v-if="doc.status === 'draft' && readyForPublish" class="primary-button" type="button"
              :disabled="busy || isTaskRunning(activeTask(doc))"
              @click="publish(doc)">{{ isTaskRunning(activeTask(doc)) ? publishStageLabel[activeTask(doc)!.stage] : "建立索引并发布" }}</button>
            <button class="text-button destructive-button" type="button" :disabled="busy" @click="removeDocument(doc)">
              <IconTrash :size="15" aria-hidden="true" /> {{ doc.status === "deleting" ? "重试删除" : "删除" }}
            </button>
            <button v-if="doc.status !== 'deleting'" class="text-button" type="button" :disabled="busy" @click="targetDocument(doc)">新增版本</button>
            <button v-if="doc.status !== 'deleting' && folders.length" class="text-button" type="button"
              :disabled="busy" @click="openMove(doc)">移动资料</button>
          </template>
        </div>
        <form v-if="moving?.document_id === doc.document_id && moving?.version === doc.version"
          class="library-inline-form" aria-label="移动资料及所有版本" @submit.prevent="moveDocument">
          <label class="form-field">
            <span>目标文件夹</span>
            <select v-model="destination" :disabled="busy">
              <option :value="null">知识库根目录</option>
              <option v-for="option in folderChoices" :key="option.id" :value="option.id">{{ option.label }}</option>
            </select>
          </label>
          <button class="primary-button" type="submit" :disabled="busy || destination === doc.folder_id">移动所有版本</button>
          <button class="text-button" type="button" :disabled="busy" @click="moving = null">取消</button>
        </form>
      </li>
    </ul>
  </section>
  </template>
</template>
