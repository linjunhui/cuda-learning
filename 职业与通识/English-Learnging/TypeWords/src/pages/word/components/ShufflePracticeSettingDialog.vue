<script setup lang="ts">

import Slider from "@/components/base/Slider.vue";
import {defineAsyncComponent, watch} from "vue";
import {useBaseStore} from "@/stores/base.ts";

const Dialog = defineAsyncComponent(() => import('@/components/dialog/Dialog.vue'))

const store = useBaseStore()

const model = defineModel()

const emit = defineEmits<{
  ok: [val: number];
}>()

let num = $ref(0)
let min = $ref(0)

watch(() => model.value, (n) => {
  if (n) {
    num = Math.floor(store.sdict.lastLearnIndex / 3)
    num = num > 50 ? 50 : num
    min = num < 10 ? num : 10
  }
})
</script>

<template>
  <Dialog v-model="model" title="随机复习设置"
          :footer="true"
          @ok="emit('ok',num)">
    <div class="target-modal color-main">
      <div class="flex gap-4 items-end  mb-2">
        <span>随机复习：<span class="font-bold">{{ store.sdict.name }}</span></span>
        <span class="text-3xl mx-2 lh">{{ num }}</span>个单词
      </div>
      <div class="flex gap-space">
        <span class="shrink-0">随机数量</span>
        <Slider :min="min"
                :step="10"
                show-text
                class="mt-1"
                :max="store.sdict.lastLearnIndex"
                v-model="num"/>
      </div>
    </div>
  </Dialog>
</template>

<style scoped lang="scss">

.target-modal {
  width: 30rem;
  padding: 0 var(--space);

  .lh {
    color: rgb(176, 116, 211)
  }

  .mode-item {
    @apply w-50% border border-blue border-solid p-2 rounded-lg cursor-pointer;
  }

  .active {
    @apply bg-blue color-white;
  }
}
</style>
