import {loadJsLib, shakeCommonDict} from "@/utils";
import {
  APP_NAME,
  APP_VERSION,
  EXPORT_DATA_KEY, LIB_JS_URL,
  LOCAL_FILE_KEY,
  Origin,
  PracticeSaveArticleKey,
  PracticeSaveWordKey,
  SAVE_DICT_KEY,
  SAVE_SETTING_KEY
} from "@/config/env.ts";
import {get} from "idb-keyval";
import {saveAs} from "file-saver";
import dayjs from "dayjs";
import Toast from "@/components/base/toast/Toast.ts";
import {useBaseStore} from "@/stores/base.ts";
import {useSettingStore} from "@/stores/setting.ts";
import {ref} from "vue";

export function useExport() {
  const store = useBaseStore()
  const settingStore = useSettingStore()

  let loading = ref(false)

  async function exportData(notice = '导出成功！', fileName = `${APP_NAME}-User-Data-${dayjs().format('YYYY-MM-DD HH-mm-ss')}.zip`) {
    if (loading.value) return
    loading.value = true
    try {
      const JSZip = await loadJsLib('JSZip', LIB_JS_URL.JSZIP);
      let data = {
        version: EXPORT_DATA_KEY.version,
        val: {
          setting: {
            version: SAVE_SETTING_KEY.version,
            val: settingStore.$state
          },
          dict: {
            version: SAVE_DICT_KEY.version,
            val: shakeCommonDict(store.$state)
          },
          [PracticeSaveWordKey.key]: {
            version: PracticeSaveWordKey.version,
            val: {}
          },
          [PracticeSaveArticleKey.key]: {
            version: PracticeSaveArticleKey.version,
            val: {}
          },
          [APP_VERSION.key]: -1
        }
      }
      let d = localStorage.getItem(PracticeSaveWordKey.key)
      if (d) {
        try {
          data.val[PracticeSaveWordKey.key] = JSON.parse(d)
        } catch (e) {
        }
      }
      let d1 = localStorage.getItem(PracticeSaveArticleKey.key)
      if (d1) {
        try {
          data.val[PracticeSaveArticleKey.key] = JSON.parse(d1)
        } catch (e) {
        }
      }
      let r = await get(APP_VERSION.key)
      data.val[APP_VERSION.key] = r

      const zip = new JSZip();
      zip.file("data.json", JSON.stringify(data));

      const mp3 = zip.folder("mp3");
      const allRecords = await get(LOCAL_FILE_KEY);
      for (const rec of allRecords ?? []) {
        mp3.file(rec.id + ".mp3", rec.file);
      }
      let content = await zip.generateAsync({type: "blob"})
      saveAs(content, fileName);
      notice && Toast.success(notice)
      return content
    } catch (e) {
      Toast.error(e?.message || e || '导出失败')
    } finally {
      loading.value = false
    }
  }

  return {
    loading,
    exportData,
  }
}