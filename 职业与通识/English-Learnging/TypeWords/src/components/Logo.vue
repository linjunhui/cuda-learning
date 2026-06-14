<script setup lang="ts">
import {useSettingStore} from "@/stores/setting.ts";
import { useRouter } from 'vue-router'

const settingStore = useSettingStore()
const router = useRouter()

function goHome() {
  router.push('/')
}
</script>

<template>
  <div class="center mb-2" @click="goHome">
    <img v-show="settingStore.theme === 'dark'" src="/logo-text-white.png" alt="">
    <img v-show="settingStore.theme !== 'dark'" src="/logo-text-black.png" alt="">
  </div>
</template>

<style scoped lang="scss">
img {
  cursor: pointer;
  height: 2rem;
}
</style>
