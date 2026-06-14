<script setup lang="ts">

import { defineAsyncComponent, onMounted, watch } from "vue";
import { useSettingStore } from "@/stores/setting.ts";
import { jump2Feedback } from "@/utils";
import { useDisableEventListener } from "@/hooks/event.ts";

const Dialog = defineAsyncComponent(() => import('@/components/dialog/Dialog.vue'))

let settingStore = useSettingStore()
let show = $ref(false)

watch(() => settingStore.load, (n) => {
  if (n && settingStore.conflictNotice) {
    setTimeout(() => {
      show = true
    }, 300)
  }
}, { immediate: true })

useDisableEventListener(() => show)

</script>

<template>
  <Dialog
      v-model="show"
      title="重要提示"
      footer
      :closeOnClickBg="false"
      cancel-button-text="不再提醒"
      confirm-button-text="关闭"
      @cancel="settingStore.conflictNotice = false"
  >
    <div class="card w-150 center flex-col color-main py-0 mb-0">
      <div class="text">
        如果您安装了 <span class="font-bold text-red">“调速” “Vim”</span> 等插件/脚本，它们会拦截键盘按下事件，<span
          class="font-bold text-red">导致在本网站练习时按 'A'、 'S' 、'D' 等键无反应</span>，您可以根据以下步骤解决冲突：
      </div>
      <ul class="m-0">
        <li>用浏览器无痕模式打开本网站，确认能否正常输入？</li>
        <li>无痕模式下无法输入，请给<span class="color-link mx-1 cp" @click="jump2Feedback">点此</span>反馈</li>
        <li>无痕模式下可以输入，则是插件/脚本导致的冲突</li>
        <li>临时禁用对应插件/脚本，或在对应插件/脚本的设置里面排除本网站</li>
        <li>可安装此
          <a href="https://chromewebstore.google.com/detail/one-click-extensions-mana/pbgjpgbpljobkekbhnnmlikbbfhbhmem"
             target="_blank">插件(Chrome版本，需翻墙)</a>,
          <a href="https://microsoftedge.microsoft.com/addons/detail/%E5%BF%AB%E6%8D%B7%E6%89%A9%E5%B1%95%E7%AE%A1%E7%90%86/jdodenbllldnoogfmbmmgpieafbnaogm"
             target="_blank">插件(Edge版本，无需翻墙)</a>,
          来快速激活、禁用其他插件
        </li>
      </ul>
    </div>
  </Dialog>
</template>
