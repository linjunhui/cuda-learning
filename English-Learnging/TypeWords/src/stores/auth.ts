import { defineStore } from 'pinia'
import { ref } from 'vue'

// 默认密码
const DEFAULT_PASSWORD = 'Zxvmax_911'
const AUTH_KEY = 'typewords_auth'

export const useAuthStore = defineStore('auth', () => {
  const isAuthenticated = ref<boolean>(false)
  let initialized = false

  // 初始化：从 localStorage 读取认证状态（只初始化一次）
  function init() {
    if (initialized) return
    const savedAuth = localStorage.getItem(AUTH_KEY)
    if (savedAuth === 'true') {
      isAuthenticated.value = true
    }
    initialized = true
  }

  // 登录：验证密码
  function login(password: string): boolean {
    if (password === DEFAULT_PASSWORD) {
      isAuthenticated.value = true
      localStorage.setItem(AUTH_KEY, 'true')
      return true
    }
    return false
  }

  // 登出
  function logout() {
    isAuthenticated.value = false
    localStorage.removeItem(AUTH_KEY)
  }

  // 检查是否已认证
  function checkAuth(): boolean {
    return isAuthenticated.value
  }

  return {
    isAuthenticated,
    init,
    login,
    logout,
    checkAuth
  }
})
