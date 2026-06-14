<script setup lang="ts">
import Dialog from "@/components/dialog/Dialog.vue";
import { watch } from "vue";
import { getWordTranslation, IcibaWordInfo } from "@/apis/iciba.ts";
import Toast from "@/components/base/toast/Toast.ts";

interface Props {
  modelValue: boolean
  word: string
}

const props = withDefaults(defineProps<Props>(), {
  modelValue: false,
  word: ''
})

const emit = defineEmits<{
  'update:modelValue': [value: boolean]
}>()

let loading = $ref(false)
let wordInfo = $ref<IcibaWordInfo | null>(null)

// 监听单词变化，重新获取翻译
watch(() => props.word, async (newWord) => {
  if (newWord && props.modelValue) {
    await fetchWordTranslation(newWord)
  }
}, { immediate: true })

// 监听弹窗显示状态
watch(() => props.modelValue, async (visible) => {
  if (visible && props.word) {
    await fetchWordTranslation(props.word)
  } else {
    wordInfo = null
  }
})

async function fetchWordTranslation(word: string) {
  if (!word || !word.trim()) {
    wordInfo = null
    return
  }

  loading = true
  try {
    const info = await getWordTranslation(word.trim().toLowerCase())
    if (info) {
      wordInfo = info
    } else {
      Toast.error('未找到该单词的翻译信息')
      wordInfo = null
    }
  } catch (error) {
    console.error('获取单词翻译失败:', error)
    Toast.error('获取单词翻译失败，请稍后重试')
    wordInfo = null
  } finally {
    loading = false
  }
}

function close() {
  emit('update:modelValue', false)
}
</script>

<template>
  <Dialog
    :model-value="modelValue"
    @update:model-value="emit('update:modelValue', $event)"
    :title="`单词翻译：${word}`"
    :show-close="true"
    :footer="false"
    :padding="true"
    :close-on-click-bg="false"
    :keyboard="false"
    :show-mask="false"
  >
    <div class="word-translation-dialog" v-loading="loading">
      <div v-if="wordInfo" class="word-info">
        <!-- 音标 -->
        <div class="phonetic-section" v-if="wordInfo.symbols && wordInfo.symbols.length > 0">
          <div class="phonetic-item" v-for="(symbol, index) in wordInfo.symbols" :key="index">
            <div class="phonetic-row" v-if="symbol.ph_en || symbol.ph_am">
              <span class="phonetic-label">音标：</span>
              <span class="phonetic-value" v-if="symbol.ph_en">
                <span class="phonetic-type">英</span>{{ symbol.ph_en }}
              </span>
              <span class="phonetic-value" v-if="symbol.ph_am">
                <span class="phonetic-type">美</span>{{ symbol.ph_am }}
              </span>
            </div>
          </div>
        </div>

        <!-- 中文意思 -->
        <div class="meaning-section" v-if="wordInfo.symbols && wordInfo.symbols.length > 0">
          <div class="meaning-item" v-for="(symbol, symbolIndex) in wordInfo.symbols" :key="symbolIndex">
            <div v-for="(part, partIndex) in symbol.parts" :key="partIndex" class="part-item">
              <div class="part-name">{{ part.part }}</div>
              <div class="part-means">
                <span v-for="(mean, meanIndex) in part.means" :key="meanIndex" class="mean-item">
                  {{ mean }}
                  <span v-if="meanIndex < part.means.length - 1">；</span>
                </span>
              </div>
            </div>
          </div>
        </div>

        <!-- 如果没有找到信息 -->
        <div v-if="!wordInfo.symbols || wordInfo.symbols.length === 0" class="no-data">
          未找到该单词的翻译信息
        </div>
      </div>

      <!-- 加载中或错误状态 -->
      <div v-else-if="!loading" class="no-data">
        未找到该单词的翻译信息
      </div>
    </div>
  </Dialog>
</template>

<style scoped lang="scss">
.word-translation-dialog {
  min-width: 20rem;
  max-width: 30rem;
  color: var(--color-font-1);

  .word-info {
    .phonetic-section {
      margin-bottom: 1.5rem;

      .phonetic-item {
        .phonetic-row {
          display: flex;
          align-items: center;
          gap: 0.8rem;
          margin-bottom: 0.5rem;

          .phonetic-label {
            font-weight: 600;
            color: var(--color-font-2);
            min-width: 3rem;
          }

          .phonetic-value {
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            font-size: 1.2rem;
            color: var(--color-font-1);

            .phonetic-type {
              font-size: 0.9rem;
              color: var(--color-font-3);
              padding: 0.1rem 0.3rem;
              background: var(--color-select-bg);
              border-radius: 0.2rem;
            }
          }
        }
      }
    }

    .meaning-section {
      .meaning-item {
        margin-bottom: 1rem;

        .part-item {
          margin-bottom: 0.8rem;

          .part-name {
            font-weight: 600;
            color: var(--color-font-active-1);
            margin-bottom: 0.3rem;
            font-size: 1rem;
          }

          .part-means {
            color: var(--color-font-1);
            line-height: 1.8;
            padding-left: 1rem;

            .mean-item {
              display: inline;
            }
          }
        }
      }
    }
  }

  .no-data {
    text-align: center;
    color: var(--color-font-3);
    padding: 2rem 0;
  }
}
</style>
