<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import {
  ragWorkspace, workspaceError, type EmbeddingModelOption, type IndexAvailability,
  type IndexVersion, type KnowledgeBase, type KnowledgeBaseGrant,
  type RebuildTask, type RetrievalTelemetrySummary,
} from "../api/ragWorkspace";
import { ApiError } from "../api/salespilot";

const props = defineProps<{ knowledgeBase: KnowledgeBase; availability: IndexAvailability; accountScope?: string }>();
const emit = defineEmits<{ updated: [index: IndexAvailability]; expired: [] }>();
const id = props.knowledgeBase.knowledge_base_id;
const isDefaultBase = id === "default";
const versions = ref<IndexVersion[]>([]);
const models = ref<EmbeddingModelOption[]>([]);
const selectedModel = ref("");
const task = ref<RebuildTask | null>(null);
const busy = ref(false);
const error = ref("");
const notice = ref("");
const aclPolicy = ref<"org_published" | "restricted">("org_published");
const aclGrants = ref<KnowledgeBaseGrant[]>([]);
const newGrantType = ref<"user" | "role" | "group">("user");
const newGrantId = ref("");
const telemetry = ref<RetrievalTelemetrySummary | null>(null);
const taskKey = "salespilot:rebuild:" + (props.accountScope ?? "") + ":" + id;
const running = computed(() => task.value?.state === "running" || task.value?.state === "pending");
const canResume = computed(() => ["failed", "partial_failed", "interrupted"].includes(task.value?.state ?? ""));
const hasCollectableVersions = computed(() =>
  versions.value.some(v => v.state === "retired" || v.state === "failed")
  || versions.value.filter(v => v.state === "superseded").length > 1,
);
const labels = {
  legacy: "历史索引", ready: "待启用", active: "使用中", superseded: "可回滚",
  building: "构建中", retired: "已退役", failed: "失败",
  pending: "排队中", running: "重建中", succeeded: "重建完成", partial_failed: "部分失败", interrupted: "已中断",
};
const subjectTypeLabels: Record<KnowledgeBaseGrant["subject_type"], string> = {
  user: "账号",
  role: "角色",
  group: "用户组",
};
let alive = true;
let timer: ReturnType<typeof setTimeout> | undefined;

function label(modelId: string) {
  return models.value.find(model => model.model_id === modelId)?.name ?? modelId;
}

async function run(action: () => Promise<void>) {
  busy.value = true;
  error.value = "";
  try {
    await action();
  } catch (failure) {
    if (alive) error.value = workspaceError(failure, () => emit("expired"));
  } finally {
    busy.value = false;
  }
}

async function refreshGovernance() {
  const [aclRes, telemetryRes] = await Promise.allSettled([
    ragWorkspace.getKnowledgeBaseAcl(id),
    ragWorkspace.telemetry(id, 7),
  ]);
  if (!alive) return;
  if (aclRes.status === "fulfilled") {
    aclPolicy.value = aclRes.value.access_policy;
    aclGrants.value = [...aclRes.value.grants];
  }
  if (telemetryRes.status === "fulfilled") {
    telemetry.value = telemetryRes.value;
  }
}

async function refresh() {
  const [availability, availableVersions] = await Promise.all([
    ragWorkspace.indexAvailability(id), ragWorkspace.indexVersions(id),
  ]);
  if (!alive) return;
  versions.value = availableVersions;
  emit("updated", availability);
  await refreshGovernance();
}

function remember(taskId: string | null) {
  try {
    if (taskId) sessionStorage.setItem(taskKey, taskId);
    else sessionStorage.removeItem(taskKey);
  } catch { /* The task is still available until this view closes. */ }
}

async function track(next: RebuildTask) {
  if (!alive) return;
  task.value = next;
  remember(next.task_id);
  clearTimeout(timer);
  if (running.value) {
    timer = setTimeout(readProgress, 1500);
  } else {
    await refresh();
  }
}

function readProgress() {
  const taskId = task.value?.task_id;
  if (taskId && alive) void run(async () => track(await ragWorkspace.rebuildStatus(taskId)));
}

async function initialize() {
  await refresh();
  models.value = (await ragWorkspace.embeddingModels()).filter(model => model.selectable);
  selectedModel.value = models.value.find(model => model.model_id === props.availability.embedding_model_id)?.model_id
    ?? models.value[0]?.model_id ?? "";
  try {
    const recent = await ragWorkspace.rebuildTasks(id, 5);
    const active = recent.find(item => ["pending", "running", "partial_failed", "failed", "interrupted"].includes(item.state))
      ?? recent[0];
    if (active) {
      await track(active);
      return;
    }
  } catch (failure) {
    if (failure instanceof ApiError && failure.status === 401) throw failure;
  }
  let taskId: string | null = null;
  try { taskId = sessionStorage.getItem(taskKey); } catch { /* Storage is optional. */ }
  if (!taskId) return;
  try {
    await track(await ragWorkspace.rebuildStatus(taskId));
  } catch (failure) {
    if (!(failure instanceof ApiError) || failure.status !== 404) throw failure;
    remember(null);
  }
}

function rebuild() {
  if (running.value || busy.value || !selectedModel.value) return;
  notice.value = "";
  const taskId = crypto.randomUUID();
  remember(taskId);
  void run(async () => track(await ragWorkspace.startRebuild(id, selectedModel.value, taskId)));
}

function resume() {
  const current = task.value;
  if (!current || !canResume.value || busy.value) return;
  notice.value = "";
  void run(async () => track(await ragWorkspace.resumeRebuild(current.task_id)));
}

function activate(version: IndexVersion) {
  notice.value = "";
  void run(async () => {
    await ragWorkspace.activateIndex(id, version.index_version,
      version.state === "superseded" || version.index_version === "legacy");
    notice.value = `已切换至索引版本 ${version.index_version}。`;
    await refresh();
  });
}

function retire(version: IndexVersion) {
  notice.value = "";
  void run(async () => {
    await ragWorkspace.retireIndex(id, version.index_version);
    notice.value = `索引版本 ${version.index_version} 已标记退役，可执行物理回收释放空间。`;
    await refresh();
  });
}

function garbageCollect() {
  notice.value = "";
  void run(async () => {
    const res = await ragWorkspace.purgeRetiredIndexes(id, 1);
    notice.value = res.purged_versions.length
      ? `已物理回收 ${res.purged_versions.length} 个退役/旧索引版本（释放 ${res.purged_chunks} 个向量分块）。`
      : "当前没有可物理回收的退役索引（默认保留最近 1 个可回滚版本）。";
    await refresh();
  });
}

function addGrant() {
  const cleanId = newGrantId.value.trim();
  if (!cleanId) return;
  if (!aclGrants.value.some(g => g.subject_type === newGrantType.value && g.subject_id === cleanId)) {
    aclGrants.value.push({ subject_type: newGrantType.value, subject_id: cleanId });
  }
  newGrantId.value = "";
}

function removeGrant(index: number) {
  aclGrants.value.splice(index, 1);
}

function saveAcl() {
  if (isDefaultBase || busy.value) return;
  notice.value = "";
  void run(async () => {
    const saved = await ragWorkspace.updateKnowledgeBaseAcl(id, {
      access_policy: aclPolicy.value,
      grants: aclGrants.value,
    });
    aclPolicy.value = saved.access_policy;
    aclGrants.value = [...saved.grants];
    notice.value = saved.access_policy === "restricted"
      ? `访问权限已保存：受限访问（共 ${saved.grants.length} 条授权规则）。`
      : "访问权限已保存：全组织已发布资料可见。";
  });
}

onMounted(() => void run(initialize));
onBeforeUnmount(() => { alive = false; clearTimeout(timer); });
</script>

<template>
  <section class="b2-panel library-index-management" aria-label="索引管理">
    <div class="section-heading">
      <h2>活动索引</h2>
      <p v-if="availability.active_index_version" class="muted">
        {{ label(availability.embedding_model_id) }} · {{ availability.dimensions ?? "—" }} 维
      </p>
    </div>
    <p v-if="availability.reason" class="muted">{{ availability.reason }}</p>
    <form class="b2-actions library-index-form" @submit.prevent="rebuild">
      <label class="form-field">
        <span>重建使用的模型</span>
        <select v-model="selectedModel" name="rebuild-model" :disabled="running || busy">
          <option v-if="!models.length" value="">暂无可用模型</option>
          <option v-for="model in models" :key="model.model_id" :value="model.model_id">{{ model.name }}</option>
        </select>
      </label>
      <button class="primary-button" type="submit" :disabled="running || busy || !selectedModel">启动重建</button>
    </form>
    <p v-if="error" class="b1-error" role="alert">{{ error }}</p>
    <p v-if="notice" class="library-index-notice" role="status">{{ notice }}</p>
    <button v-if="error && task && running" class="text-button" type="button" :disabled="busy"
      @click="readProgress">刷新进度</button>
    <div v-if="task" class="library-index-task">
      <p role="status"><strong>{{ labels[task.state] }}</strong> · {{ task.completed }}/{{ task.total }} 份
        <template v-if="task.failed">（{{ task.failed }} 份失败）</template>
      </p>
      <progress v-if="task.total" :max="task.total" :value="task.completed" aria-label="已完成的资料数" />
      <p v-if="task.error_message" class="b1-error">{{ task.error_message }}</p>
      <ul v-if="task.failed">
        <li v-for="item in task.items.filter(item => item.last_error)" :key="item.document_id + '/' + item.version">
          {{ item.document_id }} / {{ item.version }} · {{ item.last_error }}
        </li>
      </ul>
      <button v-if="canResume" class="text-button" type="button" :disabled="busy" @click="resume">恢复重建</button>
    </div>

    <div class="section-heading library-governance-heading">
      <h2>索引版本</h2>
      <button class="text-button" type="button" :disabled="busy || running || !versions.length"
        @click="garbageCollect">
        {{ hasCollectableVersions ? "清理退役索引（GC）" : "检查退役索引回收" }}
      </button>
    </div>
    <p v-if="!versions.length && !busy" class="muted">上传并发布资料后可查看索引版本。</p>
    <ul class="workspace-list">
      <li v-for="version in versions" :key="version.index_version" class="workspace-row">
        <strong>{{ label(version.embedding_model_id) }} · {{ version.index_version }}</strong>
        <p class="muted">
          {{ version.index_version === availability.active_index_version ? "使用中" : labels[version.state] }}
          <template v-if="version.document_count || version.chunk_count"> · {{ version.document_count }} 份资料 · {{ version.chunk_count }} 个片段</template>
        </p>
        <div class="b2-actions">
          <button v-if="version.index_version !== availability.active_index_version
            && ['ready', 'superseded', 'active', 'legacy'].includes(version.state)" class="text-button" type="button" :disabled="busy"
            @click="activate(version)">{{ version.state === "superseded" || version.index_version === "legacy"
              ? "回滚至此版本" : "启用此版本" }}</button>
          <button v-if="version.index_version !== availability.active_index_version
            && version.index_version !== 'legacy'
            && ['ready', 'superseded'].includes(version.state)" class="text-button" type="button" :disabled="busy"
            @click="retire(version)">退役此版本</button>
        </div>
      </li>
    </ul>

    <div class="section-heading library-governance-heading">
      <h2>访问权限（ACL）</h2>
      <span class="muted">{{ isDefaultBase ? "全组织默认共享" : (aclPolicy === "restricted" ? "受限访问" : "全组织可见") }}</span>
    </div>
    <p v-if="isDefaultBase" class="muted">
      默认知识库对同组织内所有角色开放已发布资料；如需部门/项目级隔离，请新建独立知识库并配置受限访问规则。
    </p>
    <div v-else class="library-acl-box">
      <label class="form-field">
        <span>可见范围策略</span>
        <select v-model="aclPolicy" name="kb-access-policy" :disabled="busy">
          <option value="org_published">全组织可见（资料发布后对本组织成员开放）</option>
          <option value="restricted">受限访问（仅管理员与授权的账号 / 角色 / OIDC 用户组可见）</option>
        </select>
      </label>
      <div v-if="aclPolicy === 'restricted'" class="library-acl-grants">
        <div class="library-acl-add">
          <label class="form-field">
            <span>授权主体类型</span>
            <select v-model="newGrantType" name="grant-subject-type" :disabled="busy">
              <option value="user">账号（user_id）</option>
              <option value="role">角色（sales / catalog_viewer）</option>
              <option value="group">OIDC 用户组（Keycloak group）</option>
            </select>
          </label>
          <label class="form-field">
            <span>主体标识</span>
            <input v-model="newGrantId" name="grant-subject-id" placeholder="例如 sales-b 或 vip-sales" :disabled="busy"
              @keydown.enter.prevent="addGrant" />
          </label>
          <button class="text-button" type="button" :disabled="busy || !newGrantId.trim()" @click="addGrant">+ 添加授权</button>
        </div>
        <p v-if="!aclGrants.length" class="muted">尚未添加普通成员授权（仅管理员可访问）。</p>
        <ul v-else class="library-grant-tags">
          <li v-for="(grant, idx) in aclGrants" :key="grant.subject_type + ':' + grant.subject_id" class="library-grant-tag">
            <span>{{ subjectTypeLabels[grant.subject_type] }}：{{ grant.subject_id }}</span>
            <button class="text-button" type="button" :disabled="busy" @click="removeGrant(idx)">移除</button>
          </li>
        </ul>
      </div>
      <div class="b2-actions">
        <button class="primary-button" type="button" :disabled="busy" @click="saveAcl">保存权限策略</button>
      </div>
    </div>

    <div class="section-heading library-governance-heading">
      <h2>检索质量可观测性（近 7 天）</h2>
      <button class="text-button" type="button" :disabled="busy" @click="run(refreshGovernance)">刷新指标</button>
    </div>
    <div v-if="telemetry" class="library-telemetry-grid">
      <div class="workspace-row library-metric-card">
        <span class="muted">总检索/选型次数</span>
        <strong>{{ telemetry.total_queries }}</strong>
      </div>
      <div class="workspace-row library-metric-card">
        <span class="muted">零命中率（无依据）</span>
        <strong>{{ (telemetry.zero_hit_rate * 100).toFixed(1) }}%（{{ telemetry.zero_hit_count }} 次）</strong>
      </div>
      <div class="workspace-row library-metric-card">
        <span class="muted">平均 / P95 耗时</span>
        <strong>{{ telemetry.avg_latency_ms }} ms / {{ telemetry.p95_latency_ms }} ms</strong>
      </div>
    </div>
    <div v-if="telemetry && telemetry.zero_hit_samples.length" class="library-zero-hits">
      <p class="muted">近期未命中查询（可用于补齐缺失产品资料）：</p>
      <ul class="library-chunk-list">
        <li v-for="(sample, idx) in telemetry.zero_hit_samples" :key="idx">
          <strong>{{ sample.query_preview || "（空查询）" }}</strong>
          <p class="muted">{{ sample.source === "selection" ? "选型对话" : "检索诊断" }} · {{ sample.strategy }} · {{ sample.latency_ms }} ms</p>
        </li>
      </ul>
    </div>
  </section>
</template>
