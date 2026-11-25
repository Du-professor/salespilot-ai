<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { IconArrowLeft, IconArrowRight, IconBooks, IconPlus } from "@tabler/icons-vue";
import { ragWorkspace, workspaceError, type EmbeddingModelOption, type KnowledgeBase } from "../api/ragWorkspace";
import DocumentLibrary from "./DocumentLibrary.vue";

const props = defineProps<{ canManage: boolean; baseId: string | null; accountScope?: string }>();
const emit = defineEmits<{ expired: [] }>();
const bases = ref<KnowledgeBase[]>([]);
const legacy = ref(false);
const loading = ref(true);
const error = ref("");
const search = ref("");
const dialog = ref<HTMLDialogElement | null>(null);
const name = ref("");
const description = ref("");
const modelId = ref("");
const models = ref<EmbeddingModelOption[]>([]);
const selectableModels = computed(() => models.value.filter(model => model.selectable));
const loadingModels = ref(false);
const creating = ref(false);
const formError = ref("");
const renameDialog = ref<HTMLDialogElement | null>(null);
const renameTarget = ref<KnowledgeBase | null>(null);
const renameName = ref("");
const renameDescription = ref("");
const renaming = ref(false);
const renameError = ref("");
const selected = computed(() => bases.value.find(base => base.knowledge_base_id === props.baseId));
const filtered = computed(() => {
  const term = search.value.trim().toLocaleLowerCase();
  return bases.value.filter(base => base.name.toLocaleLowerCase().includes(term));
});

function href(id: string) {
  return "#library/" + encodeURIComponent(id);
}

function modelLabel(id: string) {
  return models.value.find(model => model.model_id === id)?.name
    ?? (id === "workspace-default" ? "工作区默认配置" : id);
}

async function load() {
  error.value = "";
  loading.value = true;
  try {
    const result = await ragWorkspace.knowledgeBases();
    bases.value = result ?? [];
    legacy.value = result === null;
  } catch (failure) {
    error.value = workspaceError(failure, () => emit("expired"));
  } finally {
    loading.value = false;
  }
}

async function open() {
  name.value = "";
  description.value = "";
  modelId.value = "";
  formError.value = "";
  loadingModels.value = true;
  dialog.value?.showModal();
  try {
    models.value = await ragWorkspace.embeddingModels();
    modelId.value = selectableModels.value[0]?.model_id ?? "";
  } catch (failure) {
    formError.value = workspaceError(failure, () => emit("expired"));
  } finally {
    loadingModels.value = false;
  }
}

async function create() {
  if (!name.value.trim() || !modelId.value || creating.value || loadingModels.value) return;
  creating.value = true;
  formError.value = "";
  try {
    const base = await ragWorkspace.createKnowledgeBase({
      name: name.value.trim(),
      description: description.value.trim(),
      embedding_model_id: modelId.value,
    });
    bases.value = [bases.value[0], base, ...bases.value.slice(1)];
    dialog.value?.close();
    window.location.hash = href(base.knowledge_base_id);
  } catch (failure) {
    formError.value = workspaceError(failure, () => emit("expired"));
  } finally {
    creating.value = false;
  }
}

function openRename(base: KnowledgeBase) {
  renameTarget.value = base;
  renameName.value = base.name;
  renameDescription.value = base.description ?? "";
  renameError.value = "";
  renameDialog.value?.showModal();
}

async function saveRename() {
  if (!renameTarget.value || !renameName.value.trim() || renaming.value) return;
  renaming.value = true;
  renameError.value = "";
  try {
    const updated = await ragWorkspace.updateKnowledgeBase(renameTarget.value.knowledge_base_id, {
      name: renameName.value.trim(),
      description: renameDescription.value.trim(),
    });
    bases.value = bases.value.map(item =>
      item.knowledge_base_id === updated.knowledge_base_id ? updated : item
    );
    renameDialog.value?.close();
  } catch (failure) {
    renameError.value = workspaceError(failure, () => emit("expired"));
  } finally {
    renaming.value = false;
  }
}

onMounted(load);
watch(() => props.baseId, id => { if (id === null) void load(); });
</script>

<template>
  <p v-if="loading" class="intro" role="status">正在读取知识库…</p>
  <div v-else-if="error" class="b2-panel">
    <p class="b1-error" role="alert">{{ error }}</p>
    <button class="text-button" type="button" @click="load">重试</button>
  </div>
  <DocumentLibrary v-else-if="legacy" :can-manage="canManage" @expired="emit('expired')" />
  <template v-else-if="selected">
    <div class="knowledge-base-header">
      <a class="text-button" href="#library"><IconArrowLeft :size="16" aria-hidden="true" /> 全部知识库</a>
      <div>
        <h2>{{ selected.name }}</h2>
        <p v-if="selected.description" class="muted">{{ selected.description }}</p>
      </div>
      <button v-if="canManage" class="text-button" type="button" @click="openRename(selected)">重命名</button>
    </div>
    <DocumentLibrary :key="selected.knowledge_base_id" :knowledge-base="selected"
      :can-manage="canManage" :account-scope="accountScope" @expired="emit('expired')" />
  </template>
  <div v-else-if="baseId" class="b2-panel">
    <p class="b1-error" role="alert">找不到可访问的知识库。</p>
    <a class="text-button knowledge-base-return" href="#library">返回全部知识库</a>
  </div>
  <section v-else aria-label="全部知识库">
    <div class="knowledge-base-toolbar">
      <label class="form-field">
        <span>知识库名称</span>
        <input v-model="search" type="search" placeholder="搜索知识库" />
      </label>
      <button v-if="canManage" class="primary-button" type="button" @click="open">
        <IconPlus :size="18" aria-hidden="true" /> 新建知识库
      </button>
    </div>
    <p v-if="!filtered.length" class="intro">没有符合搜索条件的知识库。</p>
    <ul v-else class="workspace-list knowledge-base-list">
      <li v-for="base in filtered" :key="base.knowledge_base_id">
        <a class="knowledge-base-card workspace-row" :href="href(base.knowledge_base_id)">
          <span class="knowledge-base-card-title">
            <IconBooks :size="23" :stroke="1.5" aria-hidden="true" />
            <strong>{{ base.name }}</strong>
            <IconArrowRight class="knowledge-base-arrow" :size="17" aria-hidden="true" />
          </span>
          <span v-if="base.description" class="knowledge-base-description">{{ base.description }}</span>
          <span class="knowledge-base-card-meta">
            <span class="muted">{{ modelLabel(base.embedding_model_id) }}</span>
            <span v-if="base.index_state === 'awaiting_adapter'" class="outcome unknown">索引待接入</span>
          </span>
        </a>
      </li>
    </ul>
  </section>
  <dialog v-if="canManage" ref="dialog" class="knowledge-base-dialog" aria-labelledby="knowledge-create-title"
    @cancel="creating && $event.preventDefault()">
    <h2 id="knowledge-create-title">新建知识库</h2>
    <form class="knowledge-base-form" @submit.prevent="create">
      <label class="form-field">
        <span>名称</span>
        <input v-model="name" name="knowledge-base-name" maxlength="100" required autofocus :disabled="creating" />
      </label>
      <label class="form-field">
        <span>说明（选填）</span>
        <textarea v-model="description" name="knowledge-base-description" rows="3" maxlength="500" :disabled="creating" />
      </label>
      <label class="form-field">
        <span>索引模型</span>
        <select v-model="modelId" name="knowledge-base-model" required :disabled="loadingModels || creating">
          <option value="" disabled>{{ loadingModels ? "正在读取模型…" : "选择索引模型" }}</option>
          <option v-for="model in selectableModels" :key="model.model_id" :value="model.model_id">
            {{ model.name }}
          </option>
        </select>
      </label>
      <p v-if="!loadingModels && !formError && selectableModels.length < models.length" class="muted">
        本地索引模型通过可用性检查后会出现在此处。
      </p>
      <p v-if="formError" class="b1-error" role="alert">{{ formError }}</p>
      <div class="b2-actions">
        <button class="text-button" type="button" :disabled="creating" @click="dialog?.close()">取消</button>
        <button class="primary-button" type="submit"
          :disabled="creating || loadingModels || !name.trim() || !modelId">{{ creating ? "正在创建…" : "创建" }}</button>
      </div>
    </form>
  </dialog>
  <dialog v-if="canManage" ref="renameDialog" class="knowledge-base-dialog" aria-labelledby="knowledge-rename-title"
    @cancel="renaming && $event.preventDefault()">
    <h2 id="knowledge-rename-title">重命名知识库</h2>
    <form class="knowledge-base-form" @submit.prevent="saveRename">
      <label class="form-field">
        <span>名称</span>
        <input v-model="renameName" name="knowledge-base-rename-name" maxlength="100" required autofocus :disabled="renaming" />
      </label>
      <label class="form-field">
        <span>说明（选填）</span>
        <textarea v-model="renameDescription" name="knowledge-base-rename-description" rows="3" maxlength="500" :disabled="renaming" />
      </label>
      <p v-if="renameError" class="b1-error" role="alert">{{ renameError }}</p>
      <div class="b2-actions">
        <button class="text-button" type="button" :disabled="renaming" @click="renameDialog?.close()">取消</button>
        <button class="primary-button" type="submit"
          :disabled="renaming || !renameName.trim()">{{ renaming ? "正在保存…" : "保存" }}</button>
      </div>
    </form>
  </dialog>
</template>
