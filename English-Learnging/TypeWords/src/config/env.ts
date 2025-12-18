import { offset } from "@floating-ui/dom";

export const GITHUB = 'https://github.com/zyronon/TypeWords'
export const Host = 'typewords.cc'
export const EMAIL = 'zyronon@163.com'
export const Origin = `https://${Host}`
export const APP_NAME = 'Type Words'

const common = {
  word_dict_list_version: 1
}
const map = {
  DEV: {
    API: 'http://localhost/',
  }
}

export const ENV = Object.assign(map['DEV'], common)

export let AppEnv = {
  TOKEN: localStorage.getItem('token') ?? '',
  IS_OFFICIAL: false,
  IS_LOGIN: false,
  CAN_REQUEST: false
}

AppEnv.IS_LOGIN = !!AppEnv.TOKEN
AppEnv.CAN_REQUEST = AppEnv.IS_LOGIN && AppEnv.IS_OFFICIAL
// console.log('AppEnv.CAN_REQUEST',AppEnv.CAN_REQUEST)

export const RESOURCE_PATH = ENV.API + 'static'

export const DICT_LIST = {
  WORD: {
    ALL: `/list/word.json`,
    RECOMMENDED: `/list/recommend_word.json`,
  },
  ARTICLE: {
    ALL: `/list/article.json`,
    RECOMMENDED: `/list/article.json`,
  }
}

export const SoundFileOptions = [
  {value: '机械键盘', label: '机械键盘'},
  {value: '机械键盘1', label: '机械键盘1'},
  {value: '机械键盘2', label: '机械键盘2'},
  {value: '老式机械键盘', label: '老式机械键盘'},
  {value: '笔记本键盘', label: '笔记本键盘'},
]
export const APP_VERSION = {
  key: 'type-words-app-version',
  version: 2
}
export const SAVE_DICT_KEY = {
  key: 'typing-word-dict',
  version: 4
}
export const SAVE_SETTING_KEY = {
  key: 'typing-word-setting',
  version: 17
}
export const EXPORT_DATA_KEY = {
  key: 'typing-word-export',
  version: 4
}
export const LOCAL_FILE_KEY = 'typing-word-files'

export const PracticeSaveWordKey = {
  key: 'PracticeSaveWord',
  version: 1
}
export const PracticeSaveArticleKey = {
  key: 'PracticeSaveArticle',
  version: 1
}

export const TourConfig = {
  useModalOverlay: true,
  defaultStepOptions: {
    canClickTarget: false,
    classes: 'shadow-md bg-purple-dark',
    cancelIcon: {enabled: true},
    modalOverlayOpeningPadding: 10,
    modalOverlayOpeningRadius: 6,
    floatingUIOptions: {
      middleware: [offset({mainAxis: 30})]
    },
  },
  total: 7
}

export const LIB_JS_URL = {
  SHEPHERD: import.meta.env.MODE === 'development' ?
    'https://cdn.jsdelivr.net/npm/shepherd.js@14.5.1/dist/esm/shepherd.mjs'
    : Origin + '/libs/Shepherd.14.5.1.mjs',
  SNAPDOM: `${Origin}/libs/snapdom.min.js`,
  JSZIP: `${Origin}/libs/jszip.min.js`,
  XLSX: `${Origin}/libs/xlsx.full.min.js`,
}