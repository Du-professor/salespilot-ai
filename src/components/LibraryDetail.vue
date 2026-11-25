<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import { IconArrowLeft, IconDownload } from "@tabler/icons-vue";
import { ragWorkspace, workspaceError, type ChunkPage, type LibraryDocument } from "../api/ragWorkspace";

const props = defineProps<{ document: LibraryDocument }>();
const emit = defineEmits<{ back: []; expired: [] }>();
const versions = ref<LibraryDocument[]>([]);
const current = ref(props.document);
const page = ref<ChunkPage>({ total: 0, chunks: [] });
const offset = ref(0);
const busy = ref(false);
const error = ref("");
const previewUrl = ref("");
const previewText = ref<string | null>(null);
const previewTruncated = ref(false);
const pageSize = 20;
const previews = computed(() => /\.(pdf|txt|md)$/i.test(current.value.filename));
const hasFile = computed(() => current.value.status === "draft" || current.value.status === "published");
const status = { draft: "待发布", published: "已发布", withdrawn: "已撤回", deleting: "待完成删除" };
let mounted = true;

async function run(action: () => Promise<void>) {
  busy.value = true;
  error.value = "";
  try { await action(); }
  catch (failure) { error.value = workspaceError(failure, () => emit("expired")); }
  finally { busy.value = false; }
}

function clearPreview() {
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value);
  previewUrl.value = "";
  previewText.value = null;
  previewTruncated.value = false;
}

async function readChunks(nextOffset: number) {
  page.value = await ragWorkspace.documentChunks(current.value, nextOffset);
  offset.value = nextOffset;
}

function switchVersion(event: Event) {
  const version = (event.target as HTMLSelectElement).value;
  const target = versions.value.find(item => item.version === version);
  if (!target) return;
  void run(async () => {
    clearPreview();
    current.value = target;
    page.value = { total: 0, chunks: [] };
    offset.value = 0;
    await readChunks(0);
  });
}

function preview() {
  void run(async () => {
    clearPreview();
    const blob = await ragWorkspace.documentFile(current.value);
    if (!mounted) return;
    if (/\.pdf$/i.test(current.value.filename)) {
      previewUrl.value = URL.createObjectURL(blob);
    } else {
      const limit = 96 * 1024;
      previewText.value = await blob.slice(0, limit).text();
      previewTruncated.value = blob.size > limit;
    }
  });
}

onMounted(() => run(async () => {
  versions.value = await ragWorkspace.documentVersions(props.document.document_id);
  current.value = versions.value.find(item => item.version === props.document.version) ?? versions.value[0] ?? props.document;
  await readChunks(0);
}));
onBeforeUnmount(() => { mounted = false; clearPreview(); });
</script>

<template>
  <section class="b2-panel library-detail" aria-label="文档详情">
    <button class="text-button" type="button" @click="$emit('back')">
      <IconArrowLeft :size="16" aria-hidden="true" /> 返回资料列表
    </button>
    <div class="library-detail-heading">
      <h2>{{ current.filename }}</h2>
      <p class="muted">{{ status[current.status] }} · {{ current.index_status === "ready" ? "已建立索引" : "待索引" }} · {{ Math.ceil(current.size_bytes / 1024) }} KB</p>
    </div>
    <div class="library-detail-actions">
      <label class="form-field">
        <span>文档版本</span>
        <select :value="current.version" :disabled="busy" @change="switchVersion">
          <option v-for="version in (versions.length ? versions : [current])" :key="version.version" :value="version.version">
            {{ version.version }} · {{ status[version.status] }}
          </option>
        </select>
      </label>
      <button v-if="hasFile && previews" class="text-button" type="button" :disabled="busy" @click="preview">预览原件</button>
      <button v-if="hasFile" class="text-button" type="button" :disabled="busy" @click="run(() => ragWorkspace.downloadManual(current))">
        <IconDownload :size="15" aria-hidden="true" /> 下载原件
      </button>
    </div>
    <p v-if="error" class="b1-error" role="alert">{{ error }}</p>
    <iframe v-if="previewUrl" class="library-pdf" :src="previewUrl" sandbox="allow-same-origin" title="PDF 原件预览"></iframe>
    <template v-if="previewText !== null">
      <p v-if="previewTruncated" class="muted">此处显示原件前 96 KB，可下载完整文件。</p>
      <pre class="library-chunk-text">{{ previewText }}</pre>
    </template>
    <div class="section-heading library-chunks-heading">
      <h2>索引片段</h2>
      <span v-if="busy" class="muted" role="status">正在读取…</span>
      <span v-else class="muted">{{ page.total }} 个片段</span>
    </div>
    <p v-if="!page.total && !busy && !error" class="intro">
      {{ current.status === "published" ? "当前索引没有此版本的片段。" : "此版本发布后可查看片段。" }}
    </p>
    <ul v-if="page.chunks.length" class="library-chunk-list">
      <li v-for="(chunk, index) in page.chunks" :key="chunk.chunk_id">
        <strong>片段 {{ offset + index + 1 }}</strong>
        <p class="muted">
          <span v-if="chunk.page">第 {{ chunk.page }} 页</span>
          <span v-if="chunk.page && chunk.section"> · </span>
          <span v-if="chunk.section">{{ chunk.section }}</span>
        </p>
        <p v-if="chunk.truncated" class="muted">向量化时已按模型输入上限截断</p>
        <p class="library-chunk-text">{{ chunk.text }}</p>
      </li>
    </ul>
    <div v-if="page.total > pageSize" class="b2-actions">
      <button class="text-button" type="button" :disabled="busy || offset === 0"
        @click="run(() => readChunks(offset - pageSize))">上一页</button>
      <button class="text-button" type="button" :disabled="busy || offset + pageSize >= page.total"
        @click="run(() => readChunks(offset + pageSize))">下一页</button>
    </div>
  </section>
</template>
