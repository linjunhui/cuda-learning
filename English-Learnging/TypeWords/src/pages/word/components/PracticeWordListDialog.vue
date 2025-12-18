<script setup lang="ts">

import BaseTable from "@/components/BaseTable.vue";
import WordItem from "@/components/WordItem.vue";
import {defineAsyncComponent} from "vue";
import {TaskWords} from "@/types/types.ts";
import Checkbox from "@/components/base/checkbox/Checkbox.vue";

const Dialog = defineAsyncComponent(() => import('@/components/dialog/Dialog.vue'))

const model = defineModel()
defineProps<{
  data: TaskWords
}>()

let showTranslate = $ref(false)

</script>

<template>
  <Dialog v-model="model" title="任务">
    <div class="px-4 pb-4 h-80vh flex gap-4">
      <div class="h-full flex flex-col gap-2">
        <div class="flex justify-between items-center">
          <span class="title">新词 {{data.new.length}}</span>
        </div>
        <BaseTable
          class="overflow-auto flex-1 w-85"
          :list='data.new'
          :loading='false'
          :show-toolbar="false"
          :showPagination="false"
        >
          <template v-slot="item">
            <WordItem
              :item="item.item"
              :show-translate="showTranslate">
              <template v-slot:prefix>
                {{ item.index }}
              </template>
            </WordItem>
          </template>
        </BaseTable>
      </div>
      <div class="h-full flex flex-col gap-2" v-if="data.review.length">
        <div class="flex justify-between items-center">
          <span class="title">复习上次 {{data.review.length}}</span>
        </div>
        <BaseTable
            class="overflow-auto flex-1 w-85"
            :list='data.review'
            :loading='false'
            :show-toolbar="false"
            :showPagination="false"
        >
          <template v-slot="item">
            <WordItem
                :item="item.item"
                :show-translate="showTranslate">
              <template v-slot:prefix>
                {{ item.index }}
              </template>
            </WordItem>
          </template>
        </BaseTable>
      </div>
      <div class="h-full flex flex-col gap-2" v-if="data.write.length">
        <div class="flex justify-between items-center">
          <span class="title">复习之前 {{data.write.length}}</span>
          <Checkbox v-model="showTranslate">翻译</Checkbox>
        </div>
        <BaseTable
          class="overflow-auto flex-1 w-85"
          :list='data.write'
          :loading='false'
          :show-toolbar="false"
          :showPagination="false"
        >
          <template v-slot="item">
            <WordItem
              :item="item.item"
              :show-translate="showTranslate">
              <template v-slot:prefix>
                {{ item.index }}
              </template>
            </WordItem>
          </template>
        </BaseTable>
      </div>
    </div>
  </Dialog>
</template>

<style scoped lang="scss">

</style>
