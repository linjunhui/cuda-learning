<script setup lang="ts">
import { Dict } from "@/types/types.ts";
import Progress from '@/components/base/Progress.vue'
import Checkbox from "@/components/base/checkbox/Checkbox.vue";

interface IProps {
  item?: Partial<Dict>;
  quantifier?: string
  isAdd: boolean
  showCheckbox?: boolean
  checked?: boolean
  showProgress?: boolean
  isUser?: boolean //是否是用户的词典
}

const props = withDefaults(defineProps<IProps>(), {
  showProgress: true,
  isUser: false
})

defineEmits<{
  check: []
}>()

const progress = $computed(() => {
  return Number(((props.item?.lastLearnIndex / props.item?.length) * 100).toFixed())
})

const studyProgress = $computed(() => {
  if (!props.showProgress) return
  return props.item?.lastLearnIndex ? props.item?.lastLearnIndex + '/' : ''
})
</script>

<template>
  <div :id="item?.id" v-if="!isAdd">
    <div class="book overflow-hidden relative">
      <img class="absolute top-0 left-0 w-full object-cover" v-if="item?.cover" :src='item.cover' alt=""/>
      <div class="text-base mt-1" v-else>{{ item?.name }}</div>
      <div class="absolute bottom-4 right-3 z-1" :class="item?.cover && 'color-white'">
        <div>{{ studyProgress }}{{ item?.length }}{{ quantifier }}</div>
      </div>
      <div class="absolute bottom-2 left-3 right-3">
        <Progress v-if="(item?.lastLearnIndex) && showProgress" class="mt-1"
                  :percentage="progress"
                  :show-text="false"></Progress>
      </div>
      <Checkbox v-if="showCheckbox"
                :model-value="checked"
                @change="$emit('check')"
                class="absolute left-3 bottom-3 z-2"/>
      <div class="custom z-1" v-if="item.custom">自定义</div>
      <div class="custom bg-red! color-white z-1" v-else-if="item.update">更新中</div>
<!--      <div class="sync bg-red! color-white z-1" v-if="!item.sync && isUser && !showCheckbox">未同步</div>-->
    </div>
    <div class="text-base mt-1" v-if="item?.cover">{{ item?.name }}</div>
  </div>
  <div v-else class="book" id="no-book">
    <div class="h-full center text-2xl">
      <IconFluentAdd16Regular/>
    </div>
  </div>
</template>

<style scoped lang="scss">
.custom {
  position: absolute;
  top: 4px;
  right: -22px;
  padding: 1px 20px;
  background: var(--color-label-bg);
  font-size: 11px;
  transform: rotate(45deg);
}

.sync {
  @extend .custom;
  bottom: 4px;
  left: -22px;
  top: unset;
  right: unset;
}
</style>
