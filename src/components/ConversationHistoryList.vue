<script setup lang="ts">
import { computed, nextTick, ref } from "vue";
import { IconDots, IconPencil, IconPin, IconPinnedOff, IconSearch, IconTrash } from "@tabler/icons-vue";
import type { ConversationSummary, ConversationUpdate } from "../api/ragWorkspace";

const props = defineProps<{
  items: ConversationSummary[];
  selectedId: string | null;
  search: string;
  selectingDisabled: boolean;
  mutating: boolean;
}>();
const emit = defineEmits<{
  "update:search": [value: string];
  open: [id: string];
  update: [id: string, changes: ConversationUpdate];
  remove: [id: string];
}>();

const menuId = ref<string | null>(null);
const renameValue = ref<string | null>(null);
const list = ref<HTMLElement | null>(null);
let menuButton: HTMLButtonElement | null = null;

const dateFormat = new Intl.DateTimeFormat("zh-CN", { month: "short", day: "numeric" });
const sections = computed(() => {
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);
  const week = new Date(today);
  week.setDate(week.getDate() - 6);
  const groups = new Map<string, ConversationSummary[]>([
    ["置顶", []], ["今天", []], ["昨天", []], ["过去 7 天", []], ["更早", []],
  ]);
  for (const item of props.items) {
    const updated = new Date(item.updated_at);
    const label = item.pinned ? "置顶" : updated >= today ? "今天" : updated >= yesterday ? "昨天" :
      updated >= week ? "过去 7 天" : "更早";
    groups.get(label)?.push(item);
  }
  return [...groups].filter(([, items]) => items.length);
});

function closeMenu(restoreFocus = false) {
  menuId.value = null;
  renameValue.value = null;
  if (restoreFocus) menuButton?.focus();
}

function toggleMenu(item: ConversationSummary, event: MouseEvent) {
  if (menuId.value === item.conversation_id) closeMenu(true);
  else {
    menuButton = event.currentTarget as HTMLButtonElement;
    menuId.value = item.conversation_id;
    renameValue.value = null;
  }
}

function startRename(item: ConversationSummary) {
  renameValue.value = item.title;
  void nextTick(() => {
    const input = list.value?.querySelector<HTMLInputElement>(".chat-rename-input");
    input?.focus();
    input?.select();
  });
}

function saveRename(item: ConversationSummary) {
  const title = renameValue.value?.trim();
  if (!title) return;
  closeMenu(true);
  emit("update", item.conversation_id, { title });
}

function togglePin(item: ConversationSummary) {
  closeMenu(true);
  emit("update", item.conversation_id, { pinned: !item.pinned });
}

function remove(item: ConversationSummary) {
  if (!window.confirm(`删除会话「${item.title}」？聊天消息和续聊上下文将被清理。已归档的推荐和报告仍可在产品推荐历史记录中查看。`)) return;
  closeMenu();
  emit("remove", item.conversation_id);
}
</script>

<template>
  <label class="chat-history-search">
    <IconSearch :size="16" aria-hidden="true" />
    <input type="search" aria-label="搜索聊天历史" placeholder="搜索会话" maxlength="160"
      :value="search" @input="closeMenu(); emit('update:search', ($event.target as HTMLInputElement).value)" />
  </label>
  <div ref="list">
    <section v-for="[label, conversations] in sections" :key="label" class="chat-history-group">
      <h3>{{ label }}</h3>
      <ul class="chat-history-list">
        <li v-for="item in conversations" :key="item.conversation_id">
          <div class="chat-history-row" :class="{ current: selectedId === item.conversation_id }">
            <button class="chat-history-item" type="button" :disabled="selectingDisabled || mutating"
              :aria-current="selectedId === item.conversation_id ? 'true' : undefined"
              :title="item.title" @click="closeMenu(); emit('open', item.conversation_id)">
              <span>{{ item.title }}</span>
              <time :datetime="item.updated_at">{{ dateFormat.format(new Date(item.updated_at)) }}</time>
            </button>
            <button type="button" class="chat-history-menu-button icon-button" :disabled="mutating"
              :aria-label="'会话操作：' + item.title" :aria-expanded="menuId === item.conversation_id"
              @click="toggleMenu(item, $event)">
              <IconDots :size="17" aria-hidden="true" />
            </button>
          </div>
          <div v-if="menuId === item.conversation_id" class="chat-history-menu" @keydown.esc.stop.prevent="closeMenu(true)">
            <form v-if="renameValue !== null" @submit.prevent="saveRename(item)">
              <input v-model="renameValue" class="chat-rename-input" aria-label="会话名称" maxlength="160" required />
              <div class="b2-actions">
                <button type="submit" class="text-button" :disabled="!renameValue.trim()">保存</button>
                <button type="button" class="text-button" @click="closeMenu(true)">取消</button>
              </div>
            </form>
            <template v-else>
              <button type="button" class="text-button" @click="startRename(item)">
                <IconPencil :size="15" aria-hidden="true" /> 重命名
              </button>
              <button type="button" class="text-button" @click="togglePin(item)">
                <component :is="item.pinned ? IconPinnedOff : IconPin" :size="15" aria-hidden="true" />
                {{ item.pinned ? "取消置顶" : "置顶" }}
              </button>
              <button type="button" class="text-button destructive-button" @click="remove(item)">
                <IconTrash :size="15" aria-hidden="true" /> 删除会话
              </button>
            </template>
          </div>
        </li>
      </ul>
    </section>
  </div>
</template>
