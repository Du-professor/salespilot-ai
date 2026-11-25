<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import { IconAdjustments, IconBooks, IconHistory, IconLayoutSidebar, IconMessageCircle } from "@tabler/icons-vue";
import { ApiError, salespilot, type SessionUser } from "./api/salespilot";
import { workspaceError } from "./api/ragWorkspace";
import LoginForm from "./components/LoginForm.vue";
import RecommendationChat from "./components/RecommendationChat.vue";
import KnowledgeLibrary from "./components/KnowledgeLibrary.vue";
import RecommendationHistory from "./components/RecommendationHistory.vue";
import ModelConfigWorkspace from "./components/ModelConfigWorkspace.vue";

const pages = [
  { id: "selection", title: "智能产品选型", icon: IconMessageCircle },
  { id: "library", title: "资料库", icon: IconBooks },
  { id: "history", title: "产品推荐历史记录", icon: IconHistory },
  { id: "models", title: "模型 API 配置", icon: IconAdjustments },
] as const;

type Page = typeof pages[number]["id"];
const phase = ref<"checking" | "signed_out" | "ready" | "unavailable">("checking");
const account = ref<SessionUser | null>(null);
const busy = ref(false);
const error = ref("");
const sidebarVisible = ref(true);
const page = ref<Page>("selection");
const conversationId = ref<string | null>(null);
const knowledgeBaseId = ref<string | null>(null);
const visiblePages = computed(() => pages.filter(item => item.id !== "models" || account.value?.can_manage_workspace));
const title = computed(() => pages.find(item => item.id === page.value)?.title);

function syncPage() {
  const hash = window.location.hash.slice(1);
  const library = /^library(?:\/([A-Za-z0-9][A-Za-z0-9._-]{0,127}))?$/.exec(hash);
  if (library || hash === "knowledge") {
    page.value = "library";
    knowledgeBaseId.value = library?.[1] ?? null;
  }
  else if (hash === "history") page.value = "history";
  else if (hash === "models" && (!account.value || account.value.can_manage_workspace)) page.value = "models";
  else {
    page.value = "selection";
    conversationId.value = /^selection\/([A-Za-z0-9][A-Za-z0-9._-]{0,127})$/.exec(hash)?.[1] ?? null;
  }
}

function selectionHash() {
  return conversationId.value ? "#selection/" + conversationId.value : "#selection";
}

function selectConversation(id: string | null) {
  conversationId.value = id;
  if (page.value === "selection") window.history.replaceState(window.history.state, "", selectionHash());
}

function navigate(target: Page) {
  page.value = target;
  window.location.hash = target === "selection" ? selectionHash() : target;
}

function signedOut() {
  selectConversation(null);
  account.value = null;
  phase.value = "signed_out";
}

async function enterWorkspace() {
  phase.value = "checking";
  error.value = "";
  try {
    account.value = await salespilot.session();
    syncPage();
    phase.value = "ready";
  } catch (failure) {
    if (failure instanceof ApiError && failure.status === 401) signedOut();
    else {
      error.value = workspaceError(failure, signedOut);
      phase.value = "unavailable";
    }
  }
}

async function login(username: string, password: string) {
  busy.value = true;
  error.value = "";
  try {
    account.value = await salespilot.login(username, password);
    syncPage();
    phase.value = "ready";
  } catch (failure) {
    error.value = workspaceError(failure, signedOut);
  } finally {
    busy.value = false;
  }
}

async function logout() {
  busy.value = true;
  error.value = "";
  try {
    await salespilot.logout();
    signedOut();
  } catch (failure) {
    error.value = workspaceError(failure, signedOut);
  } finally {
    busy.value = false;
  }
}

onMounted(() => {
  syncPage();
  window.addEventListener("hashchange", syncPage);
  void enterWorkspace();
});
onBeforeUnmount(() => window.removeEventListener("hashchange", syncPage));
</script>

<template>
  <div class="app-frame" :class="{ 'selection-frame': account && page === 'selection' }">
    <header class="topbar">
      <div class="brand">
        <button v-if="account" class="icon-button" type="button"
          :aria-expanded="sidebarVisible" aria-controls="workspace-nav"
          aria-label="切换工作区侧栏" @click="sidebarVisible = !sidebarVisible">
          <IconLayoutSidebar :size="18" :stroke="2" aria-hidden="true" />
        </button>
        <span>SalesPilot AI</span>
      </div>
      <div v-if="account" class="header-actions">
        <span class="account-label">{{ account.username }}</span>
        <button class="text-button logout-button" type="button" :disabled="busy" @click="logout">退出登录</button>
      </div>
    </header>

    <div class="workspace" :class="{ 'sidebar-hidden': !account || !sidebarVisible, 'auth-workspace': !account }">
      <nav v-if="account" v-show="sidebarVisible" id="workspace-nav" class="sidebar" aria-label="工作空间">
        <p class="nav-label">工作空间</p>
        <button v-for="item in visiblePages" :key="item.id"
          class="nav-item" :class="{ selected: page === item.id }" type="button"
          :aria-current="page === item.id ? 'page' : undefined" @click="navigate(item.id)">
          <component :is="item.icon" :size="17" :stroke="2" aria-hidden="true" />
          {{ item.title }}
        </button>
      </nav>

      <main class="main" aria-labelledby="workspace-title">
        <h1 id="workspace-title" :class="{ 'visually-hidden': account }">{{ account ? title : "SalesPilot AI 工作台" }}</h1>
        <p v-if="error" class="b1-error" role="alert">{{ error }}</p>
        <p v-if="phase === 'checking'" class="intro" role="status">正在检查会话…</p>
        <button v-else-if="phase === 'unavailable'" type="button"
          class="text-button retry-button" @click="enterWorkspace">重试连接</button>
        <LoginForm v-else-if="phase === 'signed_out'" :busy="busy" @login="login" />
        <template v-else-if="account">
          <RecommendationChat v-show="page === 'selection'" :visible="page === 'selection'"
            :account-scope="JSON.stringify([account.org_id, account.user_id])"
            :conversation-id="conversationId" @select-conversation="selectConversation" @expired="signedOut" />
          <KnowledgeLibrary v-if="page === 'library'" :base-id="knowledgeBaseId"
            :account-scope="JSON.stringify([account.org_id, account.user_id])"
            :can-manage="account.can_manage_workspace" @expired="signedOut" />
          <RecommendationHistory v-if="page === 'history'" @expired="signedOut" />
          <ModelConfigWorkspace v-if="page === 'models' && account.can_manage_workspace" @expired="signedOut" />
        </template>
      </main>
    </div>
  </div>
</template>

