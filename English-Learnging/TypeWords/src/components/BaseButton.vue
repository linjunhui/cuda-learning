<script setup lang="ts">
import Tooltip from "@/components/base/Tooltip.vue";

interface IProps {
  keyboard?: string,
  active?: boolean
  disabled?: boolean
  loading?: boolean
  size?: 'small' | 'normal' | 'large',
  type?: 'primary' | 'link' | 'info' | 'orange'
}

withDefaults(defineProps<IProps>(), {
  type: 'primary',
  size: 'normal',
})

defineEmits(['click'])

</script>

<template>
  <Tooltip :disabled="!keyboard" :title="`${keyboard}`">
    <div class="base-button"
         v-bind="$attrs"
         @click="e => (!disabled && !loading) && $emit('click',e)"
         :class="[
             active && 'active',
             size,
             type,
             (disabled||loading) && 'disabled',
         ]">
      <span :style="{opacity:loading?0:1}"><slot></slot></span>
      <IconEosIconsLoading
          v-if="loading"
          class="loading"
          width="18"
          :color="type === 'info'?'#000000':'#ffffff'"
      />
    </div>
  </Tooltip>
</template>

<style scoped lang="scss">

.base-button {
  cursor: pointer;
  box-sizing: border-box;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  outline: none;
  text-align: center;
  transition: .1s;
  user-select: none;
  vertical-align: middle;
  white-space: nowrap;
  border-radius: .3rem;
  padding: 0 0.9rem;
  font-size: .9rem;
  height: 2rem;
  color: white;

  & + .base-button {
    margin-left: 1rem;
  }

  .loading {
    position: absolute;
  }

  &.disabled {
    opacity: .6;
    cursor: not-allowed;
    user-select: none;
  }

  &.small {
    border-radius: 0.3rem;
    padding: 0 0.6rem;
    height: 1.6rem;
    font-size: .8rem;
  }

  &.large {
    padding: 0 1.3rem;
    height: 2.4rem;
    font-size: 0.9rem;
    border-radius: .5rem;
  }

  & > span {
    line-height: 1;
    transform: translateY(-5%);

    :deep(a) {
      color: white;
    }
  }

  &.primary {
    background: var(--btn-primary);

    &:hover:not(.disabled) {
      opacity: 0.6;
    }
  }

  &.link {
    border-radius: 0;
    border-bottom: 2px solid transparent;

    &:hover:not(.disabled) {
      border-bottom: 2px solid var(--color-font-2);
    }
  }

  &.info {
    background: var(--btn-info);
    border: 1px solid var(--color-main-text);
    color: var(--color-main-text);

    &:hover:not(.disabled) {
      opacity: 0.6;
    }
  }

  &.orange {
    background: #FACC15;
    color: black;

    &:hover:not(.disabled) {
      background: #fbe27e;
      color: rgba(0, 0, 0, 0.6);
    }
  }

  &.active {
    opacity: .4;
  }
}
</style>
