<script setup lang="ts">
import { onMounted, ref } from "vue";
import { ragWorkspace, workspaceError, type HistoryEntry } from "../api/ragWorkspace";
import RecommendationReply from "./RecommendationReply.vue";

const emit = defineEmits<{ expired: [] }>();
const entries = ref<HistoryEntry[]>([]);
const busy = ref(false);
const hasMore = ref(false);
const error = ref("");
const date = new Intl.DateTimeFormat("zh-CN", { dateStyle: "medium", timeStyle: "short" });

async function load(append = false) {
  busy.value = true;
  error.value = "";
  try {
    const page = await ragWorkspace.history(append ? entries.value.length : 0);
    hasMore.value = page.length === 50;
    entries.value = append ? [...entries.value, ...page] : page;
  } catch (failure) {
    error.value = workspaceError(failure, () => emit("expired"));
  } finally {
    busy.value = false;
  }
}

onMounted(() => load());
</script>

<template>
  <p class="intro">Agent 发出产品手册或推荐报告后会自动记录在这里。</p>
  <section class="b2-panel history-panel" aria-label="推荐记录列表">
    <div class="section-heading">
      <h2>推荐记录</h2>
      <button class="text-button" type="button" :disabled="busy" @click="load()">刷新列表</button>
    </div>
    <p v-if="error" class="b1-error" role="alert">{{ error }}</p>
    <p v-if="busy" class="intro" role="status">正在读取历史…</p>
    <p v-else-if="!entries.length && !error" class="intro">暂无推荐记录。</p>
    <ul v-if="entries.length" class="workspace-list">
      <li v-for="entry in entries" :key="entry.turn_id" class="workspace-row">
        <details class="history-entry">
          <summary>
            <strong>{{ entry.request_summary }}</strong>
            <span class="history-meta">
              <time :datetime="entry.created_at">{{ date.format(new Date(entry.created_at)) }}</time>
              <span v-if="entry.reply?.models.length">{{ entry.reply.models.join(" · ") }}</span>
              <span v-else-if="!entry.available" class="outcome unknown">资料已失效</span>
            </span>
          </summary>
          <RecommendationReply v-if="entry.reply" :reply="entry.reply" @expired="emit('expired')" />
          <p v-else class="intro">依据资料已不可访问。</p>
        </details>
      </li>
    </ul>
    <button v-if="hasMore" class="text-button" type="button" :disabled="busy" @click="load(true)">查看更多</button>
  </section>
</template>
