<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, ref, watch } from "vue";
import { IconHistory, IconMessageCircle, IconPlus, IconSend, IconX } from "@tabler/icons-vue";
import { ApiError } from "../api/salespilot";
import { modelConfigs, type ActiveModel } from "../api/modelConfigs";
import {
  ragWorkspace, workspaceError, type ConversationSummary, type ConversationTurn, type ConversationUpdate,
  type KnowledgeBase, type TurnInput,
} from "../api/ragWorkspace";
import ConversationHistoryList from "./ConversationHistoryList.vue";
import RecommendationReply from "./RecommendationReply.vue";

const props = defineProps<{ visible: boolean; conversationId: string | null; accountScope: string }>();
const emit = defineEmits<{ expired: []; selectConversation: [id: string | null] }>();
const activeModel = ref<ActiveModel | null>(null);
const bases = ref<KnowledgeBase[]>([]);
const baseId = ref("default");
const canChooseBase = computed(() => !openedId.value && !loadingThread.value && !busy.value);
const selectableBases = computed(() => bases.value.filter(base =>
  base.knowledge_base_id === "default" || base.active_index_version));
const selectedBaseAvailable = computed(() => baseId.value === "default"
  || selectableBases.value.some(base => base.knowledge_base_id === baseId.value));
const openedId = ref<string | null>(null);
const turns = ref<ConversationTurn[]>([]);
const before = ref<string | null>(null);
const loadingThread = ref(false);
const loadingOlder = ref(false);
const threadError = ref("");
const olderError = ref("");
const draft = ref("");
const busy = ref(false);
const error = ref("");
const transcript = ref<HTMLElement | null>(null);
const input = ref<HTMLTextAreaElement | null>(null);
const historyOpen = ref(false);
const historyItems = ref<ConversationSummary[]>([]);
const historyOffset = ref<number | null>(null);
const historyQuery = ref("");
const historyBusy = ref(false);
const historyMutating = ref(false);
const historyError = ref("");
const historyNotice = ref("");
const historyToggle = ref<HTMLButtonElement | null>(null);
const historyDrawer = ref<HTMLElement | null>(null);
const narrowScreen = window.matchMedia("(max-width: 1050px)");
const narrow = ref(narrowScreen.matches);
const updateWidth = () => { narrow.value = narrowScreen.matches; };
narrowScreen.addEventListener("change", updateWidth);

let mounted = true;
let threadVersion = 0;
let historyVersion = 0;
let searchTimer: ReturnType<typeof setTimeout> | undefined;
let attempt: TurnInput | null = null;
let sending: { id: string; controller: AbortController } | null = null;
const deletionChannel = typeof BroadcastChannel !== "undefined"
  ? new BroadcastChannel("salespilot:conversation-deleted:" + props.accountScope) : null;
deletionChannel?.addEventListener("message", ({ data }) => {
  if (typeof data === "string") clearDeletedConversation(data);
});

onBeforeUnmount(() => {
  mounted = false;
  narrowScreen.removeEventListener("change", updateWidth);
  clearTimeout(searchTimer);
  deletionChannel?.close();
  sending?.controller.abort();
});

function showError(failure: unknown) {
  return workspaceError(failure, () => { if (mounted) emit("expired"); });
}

async function refreshModel() {
  activeModel.value = null;
  try {
    activeModel.value = await modelConfigs.active();
  } catch (failure) {
    if (mounted && failure instanceof ApiError && failure.status === 401) emit("expired");
  }
}

async function refreshBases() {
  try {
    const available = await ragWorkspace.knowledgeBases();
    if (mounted) bases.value = available ?? [];
  } catch (failure) {
    if (mounted && failure instanceof ApiError && failure.status === 401) emit("expired");
  }
}

async function scrollToLatest() {
  await nextTick();
  transcript.value?.scrollTo({ top: transcript.value.scrollHeight });
}

async function loadThread(id: string | null, keepDraft = false) {
  const version = ++threadVersion;
  openedId.value = id;
  turns.value = [];
  before.value = null;
  threadError.value = "";
  olderError.value = "";
  loadingOlder.value = false;
  loadingThread.value = id !== null;
  if (!keepDraft) {
    draft.value = "";
    error.value = "";
    attempt = null;
    baseId.value = "default";
  }
  if (!id) return;
  try {
    const thread = await ragWorkspace.conversation(id);
    if (version !== threadVersion) return;
    baseId.value = thread.knowledge_base_id ?? "default";
    turns.value = thread.turns;
    before.value = thread.before;
    loadingThread.value = false;
    await scrollToLatest();
  } catch (failure) {
    if (version === threadVersion) threadError.value = showError(failure);
  } finally {
    if (version === threadVersion) loadingThread.value = false;
  }
}

async function loadOlder() {
  const id = openedId.value;
  if (!id || !before.value || loadingOlder.value) return;
  const version = threadVersion;
  loadingOlder.value = true;
  olderError.value = "";
  try {
    const thread = await ragWorkspace.conversation(id, before.value);
    if (version !== threadVersion) return;
    const area = transcript.value;
    const height = area?.scrollHeight ?? 0;
    const top = area?.scrollTop ?? 0;
    turns.value = [...thread.turns, ...turns.value];
    before.value = thread.before;
    await nextTick();
    if (area && version === threadVersion) area.scrollTop = top + area.scrollHeight - height;
  } catch (failure) {
    if (version === threadVersion) olderError.value = showError(failure);
  } finally {
    if (version === threadVersion) loadingOlder.value = false;
  }
}

async function loadHistory(append = false) {
  clearTimeout(searchTimer);
  const offset = append ? historyOffset.value : 0;
  if (offset === null) return;
  const version = ++historyVersion;
  historyBusy.value = true;
  historyError.value = "";
  try {
    const page = await ragWorkspace.conversations(offset, historyQuery.value.trim());
    if (version !== historyVersion) return;
    const items = append ? [...historyItems.value, ...page.items] : page.items;
    const seen = new Set<string>();
    historyItems.value = items.filter(item => {
      if (seen.has(item.conversation_id)) return false;
      seen.add(item.conversation_id);
      return true;
    });
    historyOffset.value = page.next_offset;
  } catch (failure) {
    if (version === historyVersion) historyError.value = showError(failure);
  } finally {
    if (version === historyVersion) historyBusy.value = false;
  }
}

watch(historyQuery, () => {
  ++historyVersion;
  historyItems.value = [];
  historyOffset.value = null;
  historyNotice.value = "";
  historyError.value = "";
  historyBusy.value = true;
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => { if (historyOpen.value) void loadHistory(); }, 200);
});

function clearDeletedConversation(id: string) {
  ++historyVersion;
  historyItems.value = historyItems.value.filter(item => item.conversation_id !== id);
  if (sending?.id === id) sending.controller.abort();
  if (openedId.value === id || props.conversationId === id || attempt?.conversation_id === id) reset(false);
  if (historyOpen.value) void loadHistory();
}

async function updateHistoryItem(id: string, changes: ConversationUpdate | null) {
  if (historyMutating.value) return;
  historyMutating.value = true;
  historyError.value = "";
  historyNotice.value = "";
  try {
    if (changes === null) {
      await ragWorkspace.deleteConversation(id);
      if (!mounted) return;
      clearDeletedConversation(id);
      historyNotice.value = "会话已删除。";
      deletionChannel?.postMessage(id);
    } else {
      await ragWorkspace.updateConversation(id, changes);
      if (mounted) await loadHistory();
    }
  } catch (failure) {
    if (mounted) historyError.value = showError(failure);
  } finally {
    historyMutating.value = false;
  }
}

function closeHistory(restoreFocus = true) {
  if (historyDrawer.value instanceof HTMLDialogElement) historyDrawer.value.close();
  historyOpen.value = false;
  if (restoreFocus) void nextTick(() => historyToggle.value?.focus());
}

function toggleHistory() {
  if (historyOpen.value) closeHistory();
  else {
    historyOpen.value = true;
    void loadHistory();
  }
}

watch([historyOpen, narrow], async () => {
  await nextTick();
  const drawer = historyDrawer.value;
  if (historyOpen.value && drawer instanceof HTMLDialogElement && !drawer.open) drawer.showModal();
});

watch(() => props.conversationId, id => {
  if (id !== openedId.value) void loadThread(id);
}, { immediate: true });

watch(() => props.visible, visible => {
  if (!visible) {
    closeHistory(false);
    return;
  }
  void refreshModel();
  void refreshBases();
  if (openedId.value && !busy.value && !loadingThread.value) void loadThread(openedId.value, true);
});
if (props.visible) {
  void refreshModel();
  void refreshBases();
}

function restoreConversation(id: string) {
  if (narrow.value) closeHistory();
  if (id === openedId.value) {
    if (!loadingThread.value) void loadThread(id, true);
    return;
  }
  emit("selectConversation", id);
}

function reset(focus = true) {
  void loadThread(null);
  emit("selectConversation", null);
  if (focus) void nextTick(() => input.value?.focus());
}

async function send() {
  const text = draft.value.trim();
  if (!text || busy.value || loadingThread.value || threadError.value || !selectedBaseAvailable.value) return;
  const id = props.conversationId;
  if (attempt?.text !== text || (attempt.knowledge_base_id ?? "default") !== baseId.value) {
    attempt = {
      conversation_id: id ?? crypto.randomUUID(), client_request_id: crypto.randomUUID(), text,
      knowledge_base_id: baseId.value === "default" ? undefined : baseId.value,
    };
  }
  const body = attempt;
  const controller = new AbortController();
  sending = { id: body.conversation_id, controller };
  error.value = "";
  busy.value = true;
  try {
    const reply = await ragWorkspace.sendTurn(body, controller.signal);
    if (!mounted || controller.signal.aborted || props.conversationId !== id) return;
    openedId.value = reply.conversation_id;
    turns.value.push({ turn_id: reply.turn_id, text, reply });
    draft.value = "";
    attempt = null;
    if (id !== reply.conversation_id) emit("selectConversation", reply.conversation_id);
    if (historyOpen.value) void loadHistory();
    await scrollToLatest();
  } catch (failure) {
    if (!mounted || controller.signal.aborted || props.conversationId !== id) return;
    if (failure instanceof ApiError && failure.code === "recommendation_sources_changed") attempt = null;
    if (failure instanceof ApiError && failure.code === "conversation_knowledge_base_conflict" && id) {
      void loadThread(id, true);
    }
    if (id && failure instanceof ApiError && failure.code === "conversation_not_found") {
      clearDeletedConversation(id);
      error.value = "会话已删除。请开始新对话。";
      return;
    }
    error.value = showError(failure);
    if (id && failure instanceof ApiError && failure.code === "conversation_changed") void loadThread(id, true);
  } finally {
    sending = null;
    busy.value = false;
    if (props.visible && !historyOpen.value) void nextTick(() => input.value?.focus());
  }
}

function sendOnEnter(event: KeyboardEvent) {
  if (event.isComposing || event.shiftKey || event.keyCode === 229) return;
  event.preventDefault();
  void send();
}
</script>

<template>
  <div class="chat-workspace" :class="{ 'chat-history-expanded': historyOpen && !narrow }">
    <component :is="narrow ? 'dialog' : 'aside'" v-if="historyOpen" id="chat-history"
      ref="historyDrawer" class="chat-history" aria-labelledby="chat-history-title" @cancel.prevent="closeHistory()">
      <div class="chat-history-layout">
        <div class="chat-history-header">
          <h2 id="chat-history-title">聊天历史</h2>
          <button class="icon-button" type="button" aria-label="关闭聊天历史" autofocus @click="closeHistory()">
            <IconX :size="16" aria-hidden="true" />
          </button>
        </div>
        <div class="chat-history-scroll">
          <p v-if="historyError" class="b1-error" role="alert">{{ historyError }}</p>
          <p v-if="historyNotice" class="muted chat-history-empty" role="status">{{ historyNotice }}</p>
          <ConversationHistoryList v-model:search="historyQuery" :items="historyItems" :selected-id="openedId"
            :selecting-disabled="busy" :mutating="historyMutating" @open="restoreConversation"
            @update="updateHistoryItem" @remove="updateHistoryItem($event, null)" />
          <button v-if="historyError" class="text-button" type="button" @click="loadHistory()">刷新会话</button>
          <p v-else-if="!historyItems.length" class="muted chat-history-empty" role="status">
            {{ historyBusy ? "正在加载聊天历史…" : historyQuery.trim() ? "没有找到匹配的会话。" :
              "发送消息后，会话会显示在这里。" }}
          </p>
          <button v-if="historyOffset !== null && !historyError" class="text-button" type="button"
            :disabled="historyBusy" @click="loadHistory(true)">加载更多会话</button>
        </div>
      </div>
    </component>
    <section class="chat-panel" aria-label="选型对话">
      <div class="chat-header">
        <div class="chat-agent-heading">
          <strong><IconMessageCircle :size="17" :stroke="1.8" aria-hidden="true" /> 选型 Agent</strong>
          <span v-if="activeModel" class="chat-model-label" :title="activeModel.model_id ?? activeModel.name">
            {{ activeModel.name }}<template v-if="activeModel.model_id"> · {{ activeModel.model_id }}</template>
          </span>
        </div>
        <div class="chat-actions">
          <button ref="historyToggle" class="text-button" type="button" :aria-expanded="historyOpen"
            :aria-controls="historyOpen ? 'chat-history' : undefined" :aria-haspopup="narrow ? 'dialog' : undefined"
            @click="toggleHistory">
            <IconHistory :size="16" aria-hidden="true" /> 聊天历史
          </button>
          <button class="text-button" type="button" :disabled="busy" @click="reset()">
            <IconPlus :size="15" aria-hidden="true" /> 新对话
          </button>
        </div>
      </div>
      <div ref="transcript" class="chat-transcript" role="log" aria-label="对话消息">
        <div class="chat-messages">
          <div v-if="loadingThread" class="chat-empty muted" role="status">正在加载对话…</div>
          <div v-else-if="threadError" class="chat-empty">
            <p role="alert">{{ threadError }}</p>
            <button class="text-button" type="button" @click="loadThread(openedId, true)">重试</button>
          </div>
          <div v-else-if="!turns.length" class="chat-empty">
            <IconMessageCircle :size="26" :stroke="1.6" aria-hidden="true" />
            <h2>客户需要什么产品？</h2>
            <p>输入客户需求，Agent 会从已发布的资料中查找匹配型号和手册依据。</p>
          </div>
          <div v-if="before" class="chat-older">
            <p v-if="olderError" class="b1-error" role="alert">{{ olderError }}</p>
            <button class="text-button" type="button" :disabled="loadingOlder" @click="loadOlder">
              {{ loadingOlder ? "正在加载…" : "加载更早的消息" }}
            </button>
          </div>
          <div v-for="turn in turns" :key="turn.turn_id" class="chat-turn">
            <div class="chat-user"><span class="chat-speaker">你</span><p>{{ turn.text }}</p></div>
            <div class="chat-agent">
              <span class="chat-speaker">选型 Agent</span>
              <RecommendationReply v-if="turn.reply" :reply="turn.reply" @expired="emit('expired')" />
              <p v-else class="chat-unavailable muted">依据资料已无法访问，原回复及附件不可用。</p>
            </div>
          </div>
        </div>
      </div>
      <form class="chat-composer" @submit.prevent="send">
        <div class="chat-composer-inner">
          <label v-if="bases.length || baseId !== 'default'" class="form-field chat-knowledge-base">
            <span>知识库</span>
            <select v-model="baseId" name="chat-knowledge-base" :disabled="!canChooseBase">
              <option v-if="!selectedBaseAvailable" :value="baseId">原知识库当前不可用</option>
              <option v-for="base in selectableBases" :key="base.knowledge_base_id" :value="base.knowledge_base_id">
                {{ base.name }}
              </option>
            </select>
          </label>
          <label class="form-field">
            <span>客户需求</span>
            <textarea ref="input" v-model="draft" maxlength="4000" rows="3"
              :readonly="busy || loadingThread || !!threadError"
              placeholder="例如：需要一款 GigE 工业相机，至少 8 MP、100 fps。请附选型报告。"
              @keydown.enter="sendOnEnter"></textarea>
          </label>
          <div class="chat-composer-footer">
            <span v-if="busy" class="muted" role="status">正在检索资料并生成回复…</span>
            <button class="primary-button" type="submit"
              :disabled="busy || loadingThread || !!threadError || !draft.trim() || !selectedBaseAvailable">
              <IconSend :size="16" aria-hidden="true" /> 发送
            </button>
          </div>
          <p v-if="error" class="b1-error" role="alert">{{ error }}</p>
        </div>
      </form>
    </section>
  </div>
</template>
