<script setup lang="ts">
import { ref } from "vue";
import { IconDownload } from "@tabler/icons-vue";
import { ragWorkspace, workspaceError, type RecommendationTurn } from "../api/ragWorkspace";

defineProps<{ reply: RecommendationTurn }>();
const emit = defineEmits<{ expired: [] }>();
const error = ref("");
const busy = ref(false);

async function download(action: () => Promise<void>) {
  busy.value = true;
  error.value = "";
  try {
    await action();
  } catch (failure) {
    error.value = workspaceError(failure, () => emit("expired"));
  } finally {
    busy.value = false;
  }
}
</script>

<template>
  <div class="reply">
    <p v-if="reply.models.length" class="reply-models">{{ reply.models.join(" · ") }}</p>
    <p class="reply-text">{{ reply.answer }}</p>
    <details v-if="reply.citations.length" class="reply-sources">
      <summary>资料依据（{{ reply.citations.length }}）</summary>
      <ul>
        <li v-for="(source, index) in reply.citations" :key="index">
          <strong>{{ source.filename }} · {{ source.version }}</strong>
          <span v-if="source.page"> · 第 {{ source.page }} 页</span>
          <span v-if="source.section"> · {{ source.section }}</span>
          <blockquote>{{ source.excerpt }}</blockquote>
        </li>
      </ul>
    </details>
    <div v-if="reply.manuals.length || reply.has_report" class="b2-actions reply-downloads">
      <button v-for="manual in reply.manuals" :key="manual.document_id + '/' + manual.version"
        class="text-button" type="button" :disabled="busy" @click="download(() => ragWorkspace.downloadManual(manual))">
        <IconDownload :size="15" /> {{ manual.filename }} · {{ manual.version }}
      </button>
      <button v-if="reply.has_report" class="text-button" type="button" :disabled="busy"
        @click="download(() => ragWorkspace.downloadReport(reply.turn_id))">
        <IconDownload :size="15" /> 下载推荐报告
      </button>
    </div>
    <p v-if="error" class="b1-error" role="alert">{{ error }}</p>
  </div>
</template>
