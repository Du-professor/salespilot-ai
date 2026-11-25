<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import {
  ragWorkspace, workspaceError, type IndexVersion, type RetrievalDiagnostic, type RetrievalHit,
} from "../api/ragWorkspace";

const emit = defineEmits<{ expired: [] }>();
const props = defineProps<{ knowledgeBaseId?: string }>();
const text = ref("");
const comparison = ref("");
const mode = ref<"selection" | "raw_retrieval">("selection");
const strategy = ref<"hybrid" | "vector" | "bm25">("hybrid");
const compareVersion = ref("");
const indexVersions = ref<IndexVersion[]>([]);
const busy = ref(false);
const error = ref("");
const results = ref<{ label: string; query: string; result: RetrievalDiagnostic }[]>([]);
const hitKeys = computed(() => results.value.map(item => new Set(item.result.hits.map(key))));

function key(hit: RetrievalHit) {
  return JSON.stringify([hit.document_id, hit.version, hit.chunk_id]);
}

async function loadVersions() {
  try {
    const all = await ragWorkspace.indexVersions(props.knowledgeBaseId ?? "default");
    indexVersions.value = all.filter(item => ["ready", "active", "superseded", "legacy"].includes(item.state));
  } catch {
    indexVersions.value = [];
  }
}

async function run() {
  const query = text.value.trim();
  if (!query || busy.value) return;
  busy.value = true;
  error.value = "";
  results.value = [];
  try {
    const baseOptions = { mode: mode.value, strategy: strategy.value };
    if (compareVersion.value) {
      results.value = await Promise.all([
        (async () => ({
          label: "当前活动索引",
          query,
          result: await ragWorkspace.inspectRetrieval(query, props.knowledgeBaseId, baseOptions),
        }))(),
        (async () => ({
          label: "对比版本 · " + compareVersion.value,
          query,
          result: await ragWorkspace.inspectRetrieval(query, props.knowledgeBaseId, {
            ...baseOptions, index_version: compareVersion.value,
          }),
        }))(),
      ]);
    } else {
      const queries = comparison.value.trim() ? [query, comparison.value.trim()] : [query];
      results.value = await Promise.all(queries.map(async (input, index) => ({
        label: queries.length > 1 ? "询盘 " + (index === 0 ? "A" : "B") : "召回结果",
        query: input,
        result: await ragWorkspace.inspectRetrieval(input, props.knowledgeBaseId, baseOptions),
      })));
    }
  } catch (failure) {
    error.value = workspaceError(failure, () => emit("expired"));
  } finally {
    busy.value = false;
  }
}

onMounted(() => void loadVersions());
</script>

<template>
  <section class="b2-panel library-retrieval" aria-label="检索测试">
    <form class="library-retrieval-form" @submit.prevent="run">
      <div class="library-retrieval-controls">
        <label class="form-field">
          <span>诊断模式</span>
          <select v-model="mode" name="retrieval-mode" :disabled="busy">
            <option value="selection">选型链路诊断</option>
            <option value="raw_retrieval">纯召回测试（不限品类）</option>
          </select>
        </label>
        <label class="form-field">
          <span>检索策略</span>
          <select v-model="strategy" name="retrieval-strategy" :disabled="busy">
            <option value="hybrid">混合检索（向量 + BM25 RRF）</option>
            <option value="vector">仅向量检索</option>
            <option value="bm25">仅关键词（BM25）</option>
          </select>
        </label>
        <label v-if="indexVersions.length > 1" class="form-field">
          <span>跨索引版本 A/B 对比（选填）</span>
          <select v-model="compareVersion" name="retrieval-compare-version" :disabled="busy">
            <option value="">不对比索引版本</option>
            <option v-for="ver in indexVersions" :key="ver.index_version" :value="ver.index_version">
              {{ ver.embedding_model_id }} · {{ ver.index_version }}（{{ ver.document_count }} 份）
            </option>
          </select>
        </label>
      </div>
      <label class="form-field">
        <span>测试询盘</span>
        <textarea v-model="text" name="retrieval-query" rows="3" maxlength="4000" required :disabled="busy"
          placeholder="需要一款 GigE 工业相机，至少 8 MP、100 fps。" />
      </label>
      <label v-if="!compareVersion" class="form-field">
        <span>对比询盘（选填）</span>
        <textarea v-model="comparison" name="retrieval-comparison" rows="3" maxlength="4000" :disabled="busy"
          placeholder="可将原询盘加上「请附手册和推荐报告」后再对比。" />
      </label>
      <button class="primary-button" type="submit" :disabled="busy || !text.trim()">
        {{ busy ? "正在检索…" : "测试召回" }}
      </button>
    </form>
    <p v-if="error" class="b1-error" role="alert">{{ error }}</p>
    <div v-if="results.length" class="library-retrieval-results" :class="{ 'library-retrieval-compare': results.length > 1 }">
      <section v-for="(item, index) in results" :key="index" class="library-retrieval-result">
        <h2>{{ item.label }}</h2>
        <p class="library-retrieval-query">{{ item.query }}</p>
        <p class="muted" role="status">{{ item.result.message }}</p>
        <p v-if="item.result.total" class="muted">
          共 {{ item.result.total }} 个候选片段<template v-if="item.result.hits.length < item.result.total">，显示前 {{ item.result.hits.length }} 个</template>
        </p>
        <ol v-if="item.result.hits.length" class="library-chunk-list">
          <li v-for="hit in item.result.hits" :key="key(hit)">
            <strong>{{ hit.filename }} · {{ hit.version }}</strong>
            <p class="muted">
              <span v-if="hit.page">第 {{ hit.page }} 页 · </span>
              <span v-if="hit.section">{{ hit.section }} · </span>
              综合分 {{ hit.score.toFixed(3) }}
              <template v-if="hit.vector_score != null"> · 向量 {{ hit.vector_score.toFixed(3) }}</template>
              <template v-if="hit.bm25_score != null"> · BM25 {{ hit.bm25_score.toFixed(3) }}</template>
              <template v-if="hit.rerank_score != null"> · 重排 {{ hit.rerank_score.toFixed(3) }}</template>
            </p>
            <span v-if="hit.truncated" class="outcome unknown">向量化已截断</span>
            <span v-if="results.length > 1 && !hitKeys[1 - index]?.has(key(hit))" class="outcome unknown">仅本次展示</span>
            <p class="library-chunk-text">{{ hit.text }}</p>
          </li>
        </ol>
      </section>
    </div>
  </section>
</template>
