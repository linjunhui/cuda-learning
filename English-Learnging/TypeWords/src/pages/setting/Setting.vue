<script setup lang="ts">
import { nextTick, onMounted, ref, watch } from "vue";
import { useSettingStore } from "@/stores/setting.ts";
import { getAudioFileUrl, usePlayAudio } from "@/hooks/sound.ts";
import { getShortcutKey, useEventListener } from "@/hooks/event.ts";
import {
  checkAndUpgradeSaveDict,
  checkAndUpgradeSaveSetting,
  cloneDeep,
  loadJsLib,
  shakeCommonDict,
  sleep
} from "@/utils";
import { DefaultShortcutKeyMap, ShortcutKey, WordPracticeMode } from "@/types/types.ts";
import BaseButton from "@/components/BaseButton.vue";
import VolumeIcon from "@/components/icon/VolumeIcon.vue";
import { useBaseStore } from "@/stores/base.ts";
import { saveAs } from "file-saver";
import {
  APP_NAME, APP_VERSION, EMAIL,
  EXPORT_DATA_KEY, GITHUB, Host, LIB_JS_URL,
  LOCAL_FILE_KEY,
  Origin,
  PracticeSaveArticleKey,
  PracticeSaveWordKey, SAVE_DICT_KEY, SAVE_SETTING_KEY, SoundFileOptions
} from "@/config/env.ts";
import dayjs from "dayjs";
import BasePage from "@/components/BasePage.vue";
import Toast from '@/components/base/toast/Toast.ts'
import { Option, Select } from "@/components/base/select";
import Switch from "@/components/base/Switch.vue";
import Slider from "@/components/base/Slider.vue";
import RadioGroup from "@/components/base/radio/RadioGroup.vue";
import Radio from "@/components/base/radio/Radio.vue";
import InputNumber from "@/components/base/InputNumber.vue";
import PopConfirm from "@/components/PopConfirm.vue";
import Textarea from "@/components/base/Textarea.vue";
import SettingItem from "@/pages/setting/SettingItem.vue";
import { get, set, del, keys as getIDBKeys } from "idb-keyval";
import { useRuntimeStore } from "@/stores/runtime.ts";
import { useUserStore } from "@/stores/user.ts";
import { useAuthStore } from "@/stores/auth.ts";
import { useExport } from "@/hooks/export.ts";
import MigrateDialog from "@/components/MigrateDialog.vue";
import Log from "@/pages/setting/Log.vue";
import About from "@/components/About.vue";
import { useRouter } from "vue-router";
import { MessageBox } from "@/utils/MessageBox.tsx";

const emit = defineEmits<{
  toggleDisabledDialogEscKey: [val: boolean]
}>()

const tabIndex = $ref(4)
const settingStore = useSettingStore()
const runtimeStore = useRuntimeStore()
const store = useBaseStore()
const authStore = useAuthStore()
const router = useRouter()

//@ts-ignore
const gitLastCommitHash = ref(LATEST_COMMIT_HASH);
const simpleWords = $computed({
  get: () => store.simpleWords.join(','),
  set: v => {
    try {
      store.simpleWords = v.split(',');
    } catch (e) {

    }
  }
})

let editShortcutKey = $ref('')

const disabledDefaultKeyboardEvent = $computed(() => {
  return editShortcutKey && tabIndex === 3
})

watch(() => disabledDefaultKeyboardEvent, v => {
  emit('toggleDisabledDialogEscKey', !!v)
})

// 监听编辑快捷键状态变化，自动聚焦输入框
watch(() => editShortcutKey, (newVal) => {
  if (newVal) {
    // 使用nextTick确保DOM已更新
    nextTick(() => {
      focusShortcutInput()
    })
  }
})

useEventListener('keydown', (e: KeyboardEvent) => {
  if (!disabledDefaultKeyboardEvent) return

  // 确保阻止浏览器默认行为
  e.preventDefault()
  e.stopPropagation()

  let shortcutKey = getShortcutKey(e)
  // console.log('e', e, e.keyCode, e.ctrlKey, e.altKey, e.shiftKey)
  // console.log('key', shortcutKey)

  // if (shortcutKey[shortcutKey.length-1] === '+') {
  //   settingStore.shortcutKeyMap[editShortcutKey] = DefaultShortcutKeyMap[editShortcutKey]
  //   return ElMessage.warning('设备失败！')
  // }

  if (editShortcutKey) {
    if (shortcutKey === 'Delete') {
      settingStore.shortcutKeyMap[editShortcutKey] = ''
    } else {
      // 忽略单独的修饰键
      if (shortcutKey === 'Ctrl+' || shortcutKey === 'Alt+' || shortcutKey === 'Shift+' ||
          e.key === 'Control' || e.key === 'Alt' || e.key === 'Shift') {
        return;
      }

      for (const [k, v] of Object.entries(settingStore.shortcutKeyMap)) {
        if (v === shortcutKey && k !== editShortcutKey) {
          settingStore.shortcutKeyMap[editShortcutKey] = DefaultShortcutKeyMap[editShortcutKey]
          return Toast.warning('快捷键重复！')
        }
      }
      settingStore.shortcutKeyMap[editShortcutKey] = shortcutKey
    }
  }
})

function handleInputBlur() {
  // 输入框失焦时结束编辑状态
  editShortcutKey = ''
}

function handleBodyClick() {
  if (editShortcutKey) {
    editShortcutKey = ''
  }
}

function focusShortcutInput() {
  // 找到当前正在编辑的快捷键输入框
  const inputElements = document.querySelectorAll('.set-key input')
  if (inputElements && inputElements.length > 0) {
    // 聚焦第一个找到的输入框
    const inputElement = inputElements[0] as HTMLInputElement
    inputElement.focus()
  }
}

// 快捷键中文名称映射
function getShortcutKeyName(key: string): string {
  const shortcutKeyNameMap = {
    'ShowWord': '显示单词',
    'TranslateWord': '单词翻译',
    'EditArticle': '编辑文章',
    'Next': '下一个',
    'Previous': '上一个',
    'ToggleSimple': '切换已掌握状态',
    'ToggleCollect': '切换收藏状态',
    'NextChapter': '下一组',
    'PreviousChapter': '上一组',
    'RepeatChapter': '重复本组',
    'DictationChapter': '默写本组',
    'PlayWordPronunciation': '播放发音',
    'ToggleShowTranslate': '切换显示翻译',
    'ToggleDictation': '切换默写模式',
    'ToggleTheme': '切换主题',
    'ToggleConciseMode': '切换简洁模式',
    'TogglePanel': '切换面板',
    'RandomWrite': '随机默写',
    'NextRandomWrite': '继续随机默写',
    'KnowWord': '认识单词',
    'UnknownWord': '不认识单词',
  }

  return shortcutKeyNameMap[key] || key
}

function resetShortcutKeyMap() {
  editShortcutKey = ''
  settingStore.shortcutKeyMap = cloneDeep(DefaultShortcutKeyMap)
  Toast.success('恢复成功')
}

let importLoading = $ref(false)

const { loading: exportLoading, exportData } = useExport()

function importJson(str: string, notice: boolean = true) {
  importLoading = true
  let obj = {
    version: -1,
    val: {
      setting: {},
      dict: {},
      [PracticeSaveWordKey.key]: {},
      [PracticeSaveArticleKey.key]: {},
      [APP_VERSION.key]: {},
    }
  }
  try {
    obj = JSON.parse(str)
    let data = obj.val
    let settingState = checkAndUpgradeSaveSetting(data.setting)
    settingState.load = true
    settingStore.setState(settingState)
    let baseState = checkAndUpgradeSaveDict(data.dict)
    baseState.load = true
    store.setState(baseState)
    if (obj.version >= 3) {
      try {
        let save: any = obj.val[PracticeSaveWordKey.key] || {}
        if (save.val && Object.keys(save.val).length > 0) {
          localStorage.setItem(PracticeSaveWordKey.key, JSON.stringify(obj.val[PracticeSaveWordKey.key]))
        }
      } catch (e) {
        //todo 上报
      }
    }
    if (obj.version >= 4) {
      try {
        let save: any = obj.val[PracticeSaveArticleKey.key] || {}
        if (save.val && Object.keys(save.val).length > 0) {
          localStorage.setItem(PracticeSaveArticleKey.key, JSON.stringify(obj.val[PracticeSaveArticleKey.key]))
        }
      } catch (e) {
        //todo 上报
      }
      try {
        let r: any = obj.val[APP_VERSION.key] || -1
        set(APP_VERSION.key, r)
        runtimeStore.isNew = r ? (APP_VERSION.version > Number(r)) : true
      } catch (e) {
        //todo 上报
      }
    }
    notice && Toast.success('导入成功！')
  } catch (err) {
    return Toast.error('导入失败！')
  } finally {
    importLoading = false
  }
}

async function importData(e) {
  importLoading = true
  await exportData('已自动备份数据', 'TypeWords数据备份.zip')
  await sleep(1500)
  let file = e.target.files[0]
  if (!file) return
  if (file.name.endsWith(".json")) {
    let reader = new FileReader();
    reader.onload = function (v) {
      let str: any = v.target.result;
      if (str) {
        importJson(str)
      }
    }
    reader.readAsText(file);
  } else if (file.name.endsWith(".zip")) {
    try {
      const JSZip = await loadJsLib('JSZip', LIB_JS_URL.JSZIP);
      const zip = await JSZip.loadAsync(file);

      const dataFile = zip.file("data.json");
      if (!dataFile) {
        return Toast.error("缺少 data.json，导入失败");
      }

      const mp3Folder = zip.folder("mp3");
      if (mp3Folder) {
        const records: { id: string; file: Blob }[] = [];
        for (const filename in zip.files) {
          if (filename.startsWith("mp3/") && filename.endsWith(".mp3")) {
            const entry = zip.file(filename);
            if (!entry) continue;
            const blob = await entry.async("blob");
            const id = filename.replace(/^mp3\//, "").replace(/\.mp3$/, "");
            records.push({ id, file: blob });
          }
        }
        await set(LOCAL_FILE_KEY, records);
      }

      const str = await dataFile.async("string");
      importJson(str, false)

      Toast.success("导入成功！");
    } catch (e) {
      Toast.error(e?.message || e || '导入失败')
    } finally {
      importLoading = false
    }
  } else {
    Toast.error("不支持的文件类型");
  }
  importLoading = false
}

let isNewHost = $ref(window.location.host === Host)

let showTransfer = $ref(false)

function transferOk() {
  setTimeout(() => {
    window.location.href = '/words'
  },                                                 1500)
}

// 退出登录
function handleLogout() {
  authStore.logout()
  Toast.success('已退出登录')
  router.push('/auth/password-login')
}

// 清除浏览器缓存
async function clearBrowserCache() {
  MessageBox.confirm(
    '确认清除所有浏览器缓存？这将清除 IndexedDB、localStorage、Service Worker 缓存和 HTTP 缓存，并刷新页面。',
    '清除缓存',
    async () => {
      try {
        // 1. 清除 Service Worker 缓存
        if ('serviceWorker' in navigator && 'caches' in window) {
          try {
            // 清除所有 Cache Storage
            const cacheNames = await caches.keys()
            await Promise.all(cacheNames.map(name => caches.delete(name)))
            
            // 注销所有 Service Worker
            const registrations = await navigator.serviceWorker.getRegistrations()
            await Promise.all(registrations.map(reg => reg.unregister()))
          } catch (error) {
            console.warn('清除 Service Worker 缓存失败:', error)
          }
        }
        
        // 2. 清除 IndexedDB - 删除整个数据库
        try {
          // 方法1: 删除所有键
          const idbKeys = await getIDBKeys()
          for (const key of idbKeys) {
            await del(key)
          }
          
          // 方法2: 删除整个数据库（更彻底）
          if ('indexedDB' in window) {
            const databases = await indexedDB.databases()
            for (const db of databases) {
              if (db.name) {
                const deleteReq = indexedDB.deleteDatabase(db.name)
                await new Promise((resolve, reject) => {
                  deleteReq.onsuccess = () => resolve(true)
                  deleteReq.onerror = () => reject(deleteReq.error)
                  deleteReq.onblocked = () => resolve(true) // 即使被阻塞也继续
                })
              }
            }
          }
        } catch (error) {
          console.warn('清除 IndexedDB 失败:', error)
        }
        
        // 3. 清除 localStorage（保留 token 和认证信息）
        const keysToKeep = ['token', 'auth']
        const allKeys = Object.keys(localStorage)
        for (const key of allKeys) {
          if (!keysToKeep.includes(key)) {
            localStorage.removeItem(key)
          }
        }
        
        // 4. 清除 sessionStorage
        sessionStorage.clear()
        
        // 5. 清除所有 Cookie（可选，但可能影响登录状态）
        // document.cookie.split(";").forEach(c => {
        //   document.cookie = c.replace(/^ +/, "").replace(/=.*/, "=;expires=" + new Date().toUTCString() + ";path=/");
        // });
        
        Toast.success('缓存已清除，页面即将刷新...')
        
        // 6. 强制刷新页面（使用时间戳参数清除 HTTP 缓存）
        setTimeout(() => {
          // 使用 location.href 而不是 reload，添加时间戳参数强制清除缓存
          const url = new URL(window.location.href)
          url.searchParams.set('_clear_cache', Date.now().toString())
          window.location.href = url.toString()
        }, 1000)
      } catch (error) {
        console.error('清除缓存失败:', error)
        Toast.error('清除缓存失败，请手动刷新页面')
      }
    }
  )
}
</script>

<template>
  <BasePage>
    <div class="setting text-md card flex flex-col" style="height: calc(100vh - 3rem);">
      <div class="page-title text-align-center">设置</div>
      <div class="flex flex-1 overflow-hidden gap-4">
        <div class="left">
          <div class="tabs">
            <div class="tab" :class="tabIndex === 4 && 'active'" @click="tabIndex = 4">
              <IconFluentDatabasePerson20Regular width="20"/>
              <span>数据管理</span>
            </div>

            <div class="tab" :class="tabIndex === 3 && 'active'" @click="tabIndex = 3">
              <IconFluentKeyboardLayoutFloat20Regular width="20"/>
              <span>快捷键设置</span>
            </div>

            <div class="tab" :class="tabIndex === 5 && 'active'" @click="()=>{
            tabIndex = 5
            runtimeStore.isNew = false
            set(APP_VERSION.key,APP_VERSION.version)
          }">
              <IconFluentTextBulletListSquare20Regular width="20"/>
              <span>更新日志</span>
              <div class="red-point" v-if="runtimeStore.isNew"></div>
            </div>
            <div class="tab" :class="tabIndex === 6 && 'active'" @click="tabIndex = 6">
              <IconFluentPerson20Regular width="20"/>
              <span>关于</span>
            </div>
          </div>
        </div>
        <div class="col-line"></div>
        <div class="flex-1  overflow-y-auto overflow-x-hidden pr-4 content">

          <div class="body" v-if="tabIndex === 3">
            <div class="row">
              <label class="main-title">功能</label>
              <div class="wrapper">快捷键(点击可修改)</div>
            </div>
            <div class="scroll">
              <div class="row" v-for="item of Object.entries(settingStore.shortcutKeyMap)">
                <label class="item-title">{{ getShortcutKeyName(item[0]) }}</label>
                <div class="wrapper" @click="editShortcutKey = item[0]">
                  <div class="set-key" v-if="editShortcutKey === item[0]">
                    <input ref="shortcutInput" :value="item[1]?item[1]:'未设置快捷键'" readonly type="text"
                           @blur="handleInputBlur">
                    <span @click.stop="editShortcutKey = ''">按键盘进行设置，<span
                        class="text-red!">设置完成点击这里</span></span>
                  </div>
                  <div v-else>
                    <div v-if="item[1]">{{ item[1] }}</div>
                    <span v-else>未设置快捷键</span>
                  </div>
                </div>
              </div>
            </div>
            <div class="row">
              <label class="item-title"></label>
              <div class="wrapper">
                <BaseButton @click="resetShortcutKeyMap">恢复默认</BaseButton>
              </div>
            </div>
          </div>

          <div v-if="tabIndex === 4">
            <div>
              所有用户数据
              <b class="text-red">保存在本地浏览器中</b>。如果您需要在不同的设备、浏览器上使用 {{ APP_NAME }}，
              您需要手动进行数据导出和导入
            </div>
            <BaseButton :loading="exportLoading" size="large" class="mt-3" @click="exportData()">导出数据备份(ZIP)</BaseButton>
            <div class="text-gray text-sm mt-2">💾 导出的ZIP文件包含所有学习数据，可在其他设备上导入恢复</div>

            <div class="line mt-15 mb-3"></div>

            <div>请注意，导入数据将<b class="text-red"> 完全覆盖 </b>当前所有数据，请谨慎操作。执行导入操作时，会先自动备份当前数据到您的电脑中，供您随时恢复
            </div>
            <div class="flex gap-space mt-3">
              <div class="import hvr-grow">
                <BaseButton size="large" :loading="importLoading">导入数据恢复</BaseButton>
                <input type="file"
                       accept="application/json,.zip,application/zip"
                       @change="importData">
              </div>
            </div>

            <template v-if="isNewHost">
              <div class="line my-3"></div>
              <div>请注意，如果本地已有使用记录，请先备份当前数据，迁移数据后将<b class="text-red"> 完全覆盖 </b>当前所有数据，请谨慎操作。
              </div>
              <div class="flex gap-space mt-3">
                <BaseButton @click="showTransfer = true">迁移 2study.top 网站数据</BaseButton>
              </div>
            </template>

            <div class="line mt-15 mb-3"></div>
            <div class="flex gap-space">
              <BaseButton type="orange" size="large" @click="clearBrowserCache">清除缓存</BaseButton>
              <BaseButton type="info" size="large" @click="handleLogout">退出登录</BaseButton>
            </div>
          </div>

          <!--          日志-->
          <Log v-if="tabIndex === 5"/>

          <div v-if="tabIndex === 6" class="center flex-col">
            <About/>
            <div class="text-md color-gray mt-10">
              Build {{ gitLastCommitHash }}
            </div>
          </div>
        </div>
      </div>
    </div>
  </BasePage>

  <MigrateDialog
      v-model="showTransfer"
      @ok="transferOk"
  />
</template>

<style scoped lang="scss">

.col-line {
  border-right: 2px solid gainsboro;
}

.setting {

  .left {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    align-items: center;

    .tabs {
      padding: .6rem 0;
      display: flex;
      flex-direction: column;
      gap: .6rem;

      .tab {
        @apply cursor-pointer flex items-center relative;
        padding: .6rem .9rem;
        border-radius: .5rem;
        width: 10rem;
        gap: .6rem;
        transition: all .5s;

        &:hover {
          background: var(--btn-primary);
          color: white;
        }

        &.active {
          background: var(--btn-primary);
          color: white;
        }
      }
    }
  }

  .content {
    .row {
      min-height: 2.6rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: calc(var(--space) * 5);

      .wrapper {
        height: 2rem;
        flex: 1;
        display: flex;
        justify-content: flex-end;
        gap: var(--space);

        span {
          text-align: right;
          //width: 30rem;
          font-size: .7rem;
          color: gray;
        }

        .set-key {
          align-items: center;

          input {
            width: 9rem;
            box-sizing: border-box;
            margin-right: .6rem;
            height: 1.8rem;
            outline: none;
            font-size: 1rem;
            border: 1px solid gray;
            border-radius: .2rem;
            padding: 0 .3rem;
            background: var(--color-second);
            color: var(--color-font-1);
          }

        }
      }

      .main-title {
        font-size: 1.1rem;
        font-weight: bold;
      }

      .item-title {
        font-size: 1rem;
      }

      .sub-title {
        font-size: .9rem;
      }
    }

    .body {
      height: 100%;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }

    .scroll {
      flex: 1;
      padding-right: .6rem;
      overflow: auto;
    }

    .line {
      border-bottom: 1px solid #c4c3c3;
    }
  }
}

.import {
  display: inline-flex;
  position: relative;

  input {
    position: absolute;
    height: 100%;
    width: 100%;
    opacity: 0;
  }
}

// 移动端适配
@media (max-width: 768px) {
  .setting {
    flex-direction: column;

    .left {
      width: 100%;
      border-right: none;
      border-bottom: 2px solid gainsboro;

      .tabs {
        flex-direction: row;
        overflow-x: auto;
        padding: 0.5rem;
        gap: 0.3rem;

        .tab {
          white-space: nowrap;
          padding: 0.4rem 0.6rem;
          font-size: 0.9rem;

          span {
            display: none;
          }
        }
      }
    }

    .content {
      padding: 0 1rem;

      .row {
        flex-direction: column;
        align-items: flex-start;
        gap: 0.5rem;
        min-height: auto;
        padding: 0.5rem 0;

        .wrapper {
          width: 100%;
          justify-content: flex-start;

          .set-key {
            width: 100%;

            input {
              width: 100%;
              max-width: 200px;
            }
          }

          // 补充：选择器和输入框优化
          .base-select, .base-input {
            width: 100% !important;
            max-width: none;
          }

          // 单选按钮组优化
          .radio-group {
            flex-direction: column;
            gap: 0.5rem;

            .radio {
              min-height: 44px;
              width: 100%;
            }
          }

          // 滑块优化
          .slider {
            width: 100%;
          }
        }

        .main-title {
          font-size: 1rem;
        }

        .item-title {
          font-size: 0.9rem;
        }
      }

      .body {
        height: auto;
        max-height: 60vh;
      }
    }
  }
}
</style>
