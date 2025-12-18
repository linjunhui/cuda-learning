<script setup lang="ts">

import { _getAccomplishDays } from "@/utils";
import BaseButton from "@/components/BaseButton.vue";
import Checkbox from "@/components/base/checkbox/Checkbox.vue";
import Slider from "@/components/base/Slider.vue";
import { defineAsyncComponent, watch } from "vue";
import { useSettingStore } from "@/stores/setting.ts";
import Toast from "@/components/base/toast/Toast.ts";
import ChangeLastPracticeIndexDialog from "@/pages/word/components/ChangeLastPracticeIndexDialog.vue";
import Tooltip from "@/components/base/Tooltip.vue";
import { useRuntimeStore } from "@/stores/runtime.ts";
import BaseInput from "@/components/base/BaseInput.vue";
import InputNumber from "@/components/base/InputNumber.vue";

const Dialog = defineAsyncComponent(() => import('@/components/dialog/Dialog.vue'))

const settings = useSettingStore()
const runtimeStore = useRuntimeStore()

const model = defineModel()

defineProps<{
  showLeftOption: boolean,
}>()

const emit = defineEmits<{
  ok: [];
}>()

let show = $ref(false)
let tempPerDayStudyNumber = $ref(0)
let tempWordReviewRatio = $ref(0)
let tempLastLearnIndex = $ref(0)
let temPracticeMode = $ref(0)
let tempDisableShowPracticeSettingDialog = $ref(false)


function changePerDayStudyNumber() {
  runtimeStore.editDict.perDayStudyNumber = tempPerDayStudyNumber
  runtimeStore.editDict.lastLearnIndex = tempLastLearnIndex
  settings.wordPracticeMode = temPracticeMode
  settings.wordReviewRatio = tempWordReviewRatio
  settings.disableShowPracticeSettingDialog = tempDisableShowPracticeSettingDialog
  emit('ok')
}

watch(() => model.value, (n) => {
  if (n) {
    if (runtimeStore.editDict.id) {
      tempPerDayStudyNumber = runtimeStore.editDict.perDayStudyNumber
      tempLastLearnIndex = runtimeStore.editDict.lastLearnIndex
      temPracticeMode = settings.wordPracticeMode
      tempWordReviewRatio = settings.wordReviewRatio
      tempDisableShowPracticeSettingDialog = settings.disableShowPracticeSettingDialog
    } else {
      Toast.warning('请先选择一本词典')
    }
  }
})
</script>

<template>
  <Dialog
      v-model="model"
      title="学习设置" :footer="true"
      @ok="changePerDayStudyNumber">
    <div class="target-modal color-main" id="mode">
      <div class="center">
        <div class="flex gap-4 text-center h-30 w-85">
          <div class="mode-item" :class="temPracticeMode == 0 && 'active'" @click=" temPracticeMode = 0">
            <div class="title text-align-center">智能模式</div>
            <div class="desc mt-2">自动规划学习、复习、听写、默写</div>
          </div>
          <div class="mode-item" :class="temPracticeMode == 1 && 'active'" @click=" temPracticeMode = 1">
            <div class="title">自由模式</div>
            <div class="desc mt-2">自由练习，系统不强制复习与默写</div>
          </div>
        </div>
      </div>

      <div class="text-center mt-4">
        <span>共<span class="text-3xl mx-2 inner">{{ runtimeStore.editDict.length }}</span>个单词，</span>
        <span>预计<span
            class="text-3xl mx-2 inner">{{
            _getAccomplishDays(runtimeStore.editDict.length - tempLastLearnIndex, tempPerDayStudyNumber)
          }}</span>天完成</span>
      </div>

      <div class="text-center mt-4 mb-8 flex gap-1 items-end justify-center">
        <span>从第</span>
        <div class="w-20">
          <BaseInput v-model="tempLastLearnIndex"/>
        </div>
        <span>个开始，每日</span>
        <div class="w-16">
          <BaseInput v-model="tempPerDayStudyNumber"/>
        </div>
        <span>个新词</span>
        <template v-if="temPracticeMode === 0">
          <span>，复习</span>
          <div class="inner -translate-y-1 mx-1">{{ tempPerDayStudyNumber * tempWordReviewRatio }}</div>
          <span>个</span>
        </template>
      </div>

      <div class="flex mb-4 gap-space" v-if="temPracticeMode === 0">
        <Tooltip title="复习词与新词的比例">
          <div class="flex items-center gap-1 w-20">
            <span>复习比</span>
            <IconFluentQuestionCircle20Regular/>
          </div>
        </Tooltip>
        <InputNumber :min="0" :max="10" v-model="tempWordReviewRatio"/>
      </div>

      <div class="flex mb-4 gap-space">
        <span class="shrink-0 w-20">每日学习</span>
        <Slider :min="10"
                :step="10"
                show-text
                class="mt-1"
                :max="200" v-model="tempPerDayStudyNumber"/>
      </div>
      <div class="mb-6 flex gap-space">
        <span class="shrink-0 w-20">学习进度</span>
        <div class="flex-1">
          <Slider :min="0"
                  :step="10"
                  show-text
                  class="my-1"
                  :max="runtimeStore.editDict.words.length" v-model="tempLastLearnIndex"/>
          <BaseButton @click="show = true">从词典选起始位置</BaseButton>
        </div>
      </div>
    </div>
    <template v-slot:footer-left v-if="showLeftOption">
      <div class="flex items-center">
        <Checkbox v-model="tempDisableShowPracticeSettingDialog"/>
        <Tooltip title="可在设置页面更改">
          <span class="text-sm">保持默认，不再显示</span>
        </Tooltip>
      </div>
    </template>
  </Dialog>
  <ChangeLastPracticeIndexDialog
      v-model="show"
      @ok="e => {
        tempLastLearnIndex = e
        show = false
      }"
  />
</template>

<style scoped lang="scss">

.target-modal {
  width: 35rem;
  padding: 0 var(--space);

  :deep(.inner) {
    font-size: 1.8rem;
    color: rgb(176, 116, 211)
  }

  .mode-item {
    @apply w-50% border border-blue border-solid p-2 rounded-lg cursor-pointer;
  }

  .active {
    @apply bg-blue color-white;
  }
}

// 移动端适配
@media (max-width: 768px) {
  .target-modal {
    width: 90vw !important;
    max-width: 400px;
    padding: 0 1rem;

    // 模式选择
    .center .flex.gap-4 {
      width: 100%;
      flex-direction: column;
      height: auto;
      gap: 0.8rem;

      .mode-item {
        width: 100%;
        padding: 1rem;

        .title {
          font-size: 1rem;
        }

        .desc {
          font-size: 0.85rem;
          margin-top: 0.5rem;
        }
      }
    }

    // 统计显示
    .text-center {
      font-size: 0.9rem;

      .text-3xl {
        font-size: 1.5rem;
      }
    }

    // 滑块控件
    .flex.mb-4, .flex.mb-6 {
      flex-direction: column;
      align-items: flex-start;
      gap: 0.5rem;

      span {
        width: 100%;
      }

      .flex-1 {
        width: 100%;
      }
    }

    // 按钮
    .base-button {
      width: 100%;
      min-height: 44px;
    }
  }
}

@media (max-width: 480px) {
  .target-modal {
    width: 95vw !important;
    padding: 0 0.5rem;

    .text-center {
      font-size: 0.8rem;

      .text-3xl {
        font-size: 1.2rem;
      }
    }
  }
}
</style>
