<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { IconActivityHeartbeat, IconPlus, IconSettings } from "@tabler/icons-vue";
import {
  modelConfigs, type ModelCapability, type ModelConfig, type ModelConfigInput, type ModelProvider, type ModelProviderId,
} from "../api/modelConfigs";
import { workspaceError } from "../api/ragWorkspace";

const emit = defineEmits<{ expired: [] }>();
const capabilityTabs: Array<{
  id: ModelCapability; label: string; intro: string; fallback: string; modelPlaceholder: string;
}> = [
  {
    id: "chat",
    label: "对话模型（选型）",
    intro: "为智能产品选型连接对话模型。测试通过后，选择一套配置启用。",
    fallback: "本地演示",
    modelPlaceholder: "例如 deepseek-chat / gpt-4o-mini",
  },
  {
    id: "embedding",
    label: "向量模型（Embedding）",
    intro: "2核 8G 节点推荐配置远程/私有 Embedding API。测试通过并探测维度后，可在新建或重建知识库时直接选用。",
    fallback: "本地轻量模型 / 工作区默认",
    modelPlaceholder: "例如 text-embedding-3-small / bge-m3",
  },
  {
    id: "rerank",
    label: "重排模型（Rerank）",
    intro: "启用后自动对混合检索（向量 + BM25）候选执行远程重排，提升精准排序且不占用本机显存。",
    fallback: "未启用远程重排（使用 RRF 融合）",
    modelPlaceholder: "例如 BAAI/bge-reranker-v2-m3",
  },
  {
    id: "parser",
    label: "文档解析（OCR/PDF）",
    intro: "面向扫描件 PDF 与复杂跨页多栏表格。启用后优先交由外部 OCR/解析网关（如 MinerU/Doc2X）解析，保护本机内存。",
    fallback: "本地 Docling 解析",
    modelPlaceholder: "例如 mineru-ocr / doc2x-pdf",
  },
];

const activeTab = ref<ModelCapability>("chat");
const currentMeta = computed(() => capabilityTabs.find(tab => tab.id === activeTab.value) ?? capabilityTabs[0]);
const providers = ref<ModelProvider[]>([]);
const configs = ref<ModelConfig[]>([]);
const keyStorageReady = ref(false);
const scopedConfigs = computed(() => configs.value.filter(item => (item.capability ?? "chat") === activeTab.value));
const active = computed(() => scopedConfigs.value.find(item => item.active));
const loading = ref(true);
const busy = ref("");
const feedback = ref("");
const error = ref("");
const editing = ref<ModelConfig | null>(null);
const formOpen = ref(false);
const form = ref({
  name: "", capability: "chat" as ModelCapability, provider: "deepseek" as ModelProviderId,
  modelId: "", baseUrl: "", apiKey: "", clearKey: false,
});
const provider = computed(() => providers.value.find(item => item.provider === form.value.provider));
const canRetainKey = computed(() => editing.value?.has_key && editing.value.provider === form.value.provider
  && editing.value.base_url === form.value.baseUrl.trim().replace(/\/+$/, "") && !form.value.clearKey);
const statusLabels = { untested: "待测试", passed: "测试通过", failed: "测试失败" };
const formatDate = new Intl.DateTimeFormat("zh-CN", { dateStyle: "medium", timeStyle: "short" });

async function reload() {
  const result = await modelConfigs.list();
  providers.value = result.providers;
  configs.value = result.configs;
  keyStorageReady.value = result.key_storage_ready;
}

async function run(label: string, operation: () => Promise<void>) {
  busy.value = label;
  feedback.value = "";
  error.value = "";
  try {
    await operation();
  } catch (failure) {
    error.value = workspaceError(failure, () => emit("expired"));
  } finally {
    busy.value = "";
  }
}

function switchTab(next: ModelCapability) {
  if (busy.value) return;
  activeTab.value = next;
  cancel();
  feedback.value = "";
  error.value = "";
}

function begin(item?: ModelConfig) {
  editing.value = item ?? null;
  const preferred = activeTab.value === "chat" ? "deepseek" : "custom";
  const firstProvider = providers.value.find(entry => entry.provider === preferred)
    ?? providers.value.find(entry => entry.provider === "openai")
    ?? providers.value[0];
  form.value = {
    name: item?.name ?? "",
    capability: item?.capability ?? activeTab.value,
    provider: item?.provider ?? firstProvider?.provider ?? "custom",
    modelId: item?.model_id ?? "",
    baseUrl: item?.base_url ?? firstProvider?.base_urls[0] ?? "",
    apiKey: "",
    clearKey: false,
  };
  formOpen.value = true;
}

function cancel() {
  form.value.apiKey = "";
  editing.value = null;
  formOpen.value = false;
}

function selectProvider() {
  form.value.baseUrl = provider.value?.base_urls[0] ?? "";
  form.value.apiKey = "";
  form.value.clearKey = false;
}

function save() {
  const values = form.value;
  const input: ModelConfigInput = {
    name: values.name.trim(),
    capability: values.capability,
    provider: values.provider,
    model_id: values.modelId.trim(),
    base_url: values.baseUrl.trim() || null,
  };
  if (values.apiKey.trim()) input.api_key = values.apiKey.trim();
  if (values.clearKey) input.clear_key = true;
  void run("正在保存配置…", async () => {
    await modelConfigs.save(input, editing.value);
    cancel();
    await reload();
    feedback.value = "配置已保存。请点击「测试连接」完成探针验证。";
  });
}

function applyAction(item: ModelConfig, action: "test" | "activate" | "deactivate") {
  const labels = { test: "正在执行连通性与能力探针测试…", activate: "正在启用配置…", deactivate: "正在停用配置…" };
  void run(labels[action], async () => {
    const result = await modelConfigs.action(item, action);
    await reload();
    if (action === "test" && result.test_status === "failed") error.value = result.test_message ?? "测试失败。";
    else feedback.value = action === "test" ? (result.test_message ?? "测试通过。")
      : action === "activate" ? "配置已启用并立即生效。" : "配置已停用。";
  });
}

function remove(item: ModelConfig) {
  if (!window.confirm(`删除「${item.name}」及其已保存的密钥？`)) return;
  void run("正在删除配置…", async () => {
    await modelConfigs.remove(item);
    if (editing.value?.config_id === item.config_id) cancel();
    await reload();
    feedback.value = "配置已删除。";
  });
}

onMounted(async () => {
  await run("正在读取模型配置…", reload);
  loading.value = false;
});
</script>

<template>
  <nav class="library-view-nav" aria-label="模型能力分类">
    <button v-for="tab in capabilityTabs" :key="tab.id" class="text-button" type="button" :disabled="!!busy"
      :aria-current="activeTab === tab.id ? 'page' : undefined" @click="switchTab(tab.id)">{{ tab.label }}</button>
  </nav>
  <p class="intro">{{ currentMeta.intro }}</p>
  <div class="b2-panel model-current">
    <div>
      <span class="muted">当前启用（{{ currentMeta.label }}）</span>
      <h2>{{ active?.name ?? currentMeta.fallback }}</h2>
      <p v-if="active" class="intro">{{ active.model_id }}<template v-if="active.dimensions"> · {{ active.dimensions }} 维</template></p>
    </div>
    <button class="primary-button" type="button" :disabled="!!busy || !providers.length" @click="begin()">
      <IconPlus :size="16" /> 新增配置
    </button>
  </div>

  <p v-if="busy" class="intro model-feedback" role="status">{{ busy }}</p>
  <p v-if="error" class="b1-error" role="alert">{{ error }}</p>
  <p v-if="feedback" class="feedback" role="status">{{ feedback }}</p>

  <form v-if="formOpen" class="b2-panel model-editor" aria-label="模型配置表单" @submit.prevent="save">
    <h2>{{ editing ? "编辑" : "新增" }} · {{ currentMeta.label }}</h2>
    <div class="model-fields">
      <label class="form-field">
        <span>配置名称</span>
        <input v-model="form.name" maxlength="80" required placeholder="例如 云端 BGE-M3 / MinerU 解析网关" :disabled="!!busy" />
      </label>
      <label class="form-field">
        <span>供应商</span>
        <select v-model="form.provider" :disabled="!!busy" @change="selectProvider">
          <option v-for="item in providers" :key="item.provider" :value="item.provider">{{ item.label }}</option>
        </select>
      </label>
      <label class="form-field">
        <span>模型 / 服务标识</span>
        <input v-model="form.modelId" maxlength="200" required :placeholder="currentMeta.modelPlaceholder" :disabled="!!busy" />
      </label>
      <label class="form-field">
        <span>API 基础地址</span>
        <input v-model="form.baseUrl" maxlength="300" list="model-base-urls" type="url"
          placeholder="例如 http://127.0.0.1:8000/v1" :disabled="!!busy" />
        <datalist id="model-base-urls">
          <option v-for="url in provider?.base_urls" :key="url" :value="url" />
        </datalist>
        <small v-if="form.provider === 'custom'" class="muted">填写部署环境允许的兼容接口地址（由 SALESPILOT_MODEL_ALLOWED_BASE_URLS 控制白名单）。</small>
      </label>
      <label class="form-field">
        <span>API 密钥 <span v-if="canRetainKey" class="field-unit">已配置</span></span>
        <input v-model="form.apiKey" type="password" autocomplete="new-password" maxlength="4000"
          :required="provider?.requires_key && !canRetainKey" :disabled="!!busy || !keyStorageReady || form.clearKey"
          :placeholder="canRetainKey ? '留空保留原密钥' : provider?.requires_key ? '填写供应商 API 密钥' : '按服务认证要求选填'" />
        <small v-if="!keyStorageReady" class="muted">服务器需配置模型密钥加密主密钥后才能保存 API 密钥。</small>
      </label>
      <label v-if="canRetainKey && !provider?.requires_key || form.clearKey" class="b2-checkbox">
        <input v-model="form.clearKey" type="checkbox" :disabled="!!busy || !!form.apiKey" />清除保存的密钥
      </label>
    </div>
    <p v-if="provider?.requires_key" class="intro">启用云端服务后，对应阶段的文本或文档内容会发送至该受信任端点。</p>
    <div class="b2-actions">
      <button class="primary-button" type="submit" :disabled="!!busy || (provider?.requires_key && !keyStorageReady && !canRetainKey)">保存配置</button>
      <button class="text-button" type="button" :disabled="!!busy" @click="cancel">取消</button>
    </div>
  </form>

  <section class="b2-panel" aria-label="已保存模型">
    <div class="section-heading">
      <h2>已保存配置（{{ currentMeta.label }}）</h2>
      <button v-if="!loading" type="button" class="text-button" :disabled="!!busy" @click="run('正在刷新配置…', reload)">刷新列表</button>
    </div>
    <p v-if="!loading && !scopedConfigs.length" class="intro">当前分类还没有保存配置。</p>
    <ul v-if="scopedConfigs.length" class="workspace-list">
      <li v-for="item in scopedConfigs" :key="item.config_id" class="workspace-row model-row">
        <div class="workspace-row-title">
          <IconSettings :size="17" :stroke="1.7" />
          <strong>{{ item.name }}</strong>
          <span v-if="item.active" class="outcome pass">使用中</span>
          <span class="outcome" :class="{ pass: item.test_status === 'passed', fail: item.test_status === 'failed', unknown: item.test_status === 'untested' }">
            {{ statusLabels[item.test_status] }}
          </span>
        </div>
        <p class="muted">
          {{ providers.find(p => p.provider === item.provider)?.label }} · {{ item.model_id }}
          <template v-if="item.dimensions"> · {{ item.dimensions }} 维</template>
          · {{ item.has_key ? "密钥已配置" : "无密钥" }}
        </p>
        <p class="muted model-url">{{ item.base_url }}</p>
        <p v-if="item.test_message" class="intro">{{ item.test_message }}<template v-if="item.test_at"> · {{ formatDate.format(new Date(item.test_at * 1000)) }}</template></p>
        <div class="b2-actions">
          <button class="text-button" type="button" :disabled="!!busy" @click="applyAction(item, 'test')">
            <IconActivityHeartbeat :size="15" /> 测试连接
          </button>
          <button v-if="item.active" class="text-button" type="button" :disabled="!!busy" @click="applyAction(item, 'deactivate')">停用</button>
          <template v-else>
            <button class="primary-button" type="button" :disabled="!!busy || item.test_status !== 'passed'" @click="applyAction(item, 'activate')">启用</button>
            <button class="text-button" type="button" :disabled="!!busy" @click="begin(item)">编辑</button>
            <button class="text-button" type="button" :disabled="!!busy" @click="remove(item)">删除</button>
          </template>
        </div>
      </li>
    </ul>
  </section>
</template>
