<script setup lang="ts">
import { ref } from "vue";

defineProps<{ busy: boolean }>();
const emit = defineEmits<{ login: [username: string, password: string] }>();
const username = ref("");
const password = ref("");

function submit() {
  emit("login", username.value, password.value);
  password.value = "";
}
</script>

<template>
  <form class="login-form" aria-label="工作区登录" @submit.prevent="submit">
    <label class="form-field">
      <span>账号</span>
      <input v-model.trim="username" name="username" autocomplete="username" maxlength="64" required :disabled="busy" />
    </label>
    <label class="form-field">
      <span>密码</span>
      <input v-model="password" name="password" type="password" autocomplete="current-password" maxlength="256" required :disabled="busy" />
    </label>
    <button class="primary-button" type="submit" :disabled="busy">{{ busy ? "登录中…" : "登录" }}</button>
  </form>
</template>
