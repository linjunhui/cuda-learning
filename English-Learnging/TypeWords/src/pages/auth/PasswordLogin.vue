<script setup lang="ts">
import { ref } from 'vue'
import BaseInput from '@/components/base/BaseInput.vue'
import BaseButton from '@/components/BaseButton.vue'
import { useAuthStore } from '@/stores/auth.ts'
import Toast from '@/components/base/toast/Toast.ts'
import { useRouter } from 'vue-router'

const authStore = useAuthStore()
const router = useRouter()

let password = $ref('')
let loading = $ref(false)

function handleLogin() {
  if (!password.trim()) {
    Toast.warning('请输入密码')
    return
  }

  loading = true
  try {
    const success = authStore.login(password)
    if (success) {
      Toast.success('登录成功')
      // 跳转到之前想访问的页面，或默认首页（articles）
      const redirect = router.currentRoute.value.query.redirect as string || '/articles'
      router.push(redirect)
    } else {
      Toast.error('密码错误')
      password = ''
    }
  } catch (error) {
    Toast.error('登录失败，请重试')
  } finally {
    loading = false
  }
}

// 支持回车键登录
function handleKeydown(e: KeyboardEvent) {
  if (e.key === 'Enter' && !loading) {
    handleLogin()
  }
}
</script>

<template>
  <div class="password-login-page">
    <div class="login-container">
      <div class="login-card">
        <h1 class="title">请输入访问密码</h1>
        <div class="form-group">
          <BaseInput
            v-model="password"
            type="password"
            size="large"
            placeholder="请输入密码"
            autocomplete="current-password"
            @keydown="handleKeydown"
            class="password-input"
          />
        </div>
        <BaseButton
          class="login-button"
          size="large"
          :loading="loading"
          @click="handleLogin"
        >
          登录
        </BaseButton>
      </div>
    </div>
  </div>
</template>

<style scoped lang="scss">
.password-login-page {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--bg-page, #f5f5f5);
  padding: 1rem;
}

.login-container {
  width: 100%;
  max-width: 400px;
}

.login-card {
  background: var(--bg-card, #ffffff);
  border-radius: 0.5rem;
  padding: 2.5rem;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);

  .title {
    font-size: 1.5rem;
    font-weight: 600;
    text-align: center;
    margin-bottom: 2rem;
    color: var(--color-font-1, #333);
  }

  .form-group {
    margin-bottom: 1.5rem;

    .password-input {
      width: 100%;
    }
  }

  .login-button {
    width: 100%;
  }
}
</style>
