# 添加文章的代码逻辑与存储位置分析

本文档详细分析了 TypeWords 项目中添加文章功能的代码逻辑和数据存储位置。

## 📋 目录

- [整体流程](#整体流程)
- [代码执行流程](#代码执行流程)
- [数据存储位置](#数据存储位置)
- [关键代码文件](#关键代码文件)
- [数据流转图](#数据流转图)

## 🔄 整体流程

添加文章的完整流程如下：

```
用户编辑文章 → 保存到运行时状态 → 同步到基础状态 → 自动持久化到 IndexedDB
```

## 📝 代码执行流程

### 1. 用户编辑文章

**文件**: `src/pages/article/components/EditArticle.vue`

- 用户在 `EditArticle` 组件中编辑文章内容
- 点击"保存"按钮触发 `save()` 函数（第125行）

```typescript
function save(option: 'save' | 'saveAndNext') {
  // 1. 验证必填字段（标题、正文）
  // 2. 处理音频时间轴数据（lrcPosition）
  // 3. 生成文章ID（如果没有）
  // 4. 触发 save 事件，将文章数据传递给父组件
  emit(option as any, editArticle)
}
```

### 2. 保存到运行时状态

**文件**: `src/pages/article/BatchEditArticlePage.vue`

- `saveArticle()` 函数（第85行）接收文章数据
- 将文章保存到 `runtimeStore.editDict.articles` 数组中

```typescript
function saveArticle(val: Article): boolean {
  if (val.id) {
    // 更新已存在的文章
    let rIndex = runtimeStore.editDict.articles.findIndex(v => v.id === val.id)
    if (rIndex > -1) {
      runtimeStore.editDict.articles[rIndex] = cloneDeep(val)
    }
  } else {
    // 添加新文章
    val.id = nanoid(6)  // 生成6位随机ID
    runtimeStore.editDict.articles.push(val)
  }
  // 同步到基础状态
  syncBookInMyStudyList()
  return true
}
```

**关键点**:
- `runtimeStore.editDict` 是运行时临时状态（`src/stores/runtime.ts`）
- 文章数据存储在 `runtimeStore.editDict.articles` 数组中
- 每个文章都有唯一的 `id`（使用 `nanoid(6)` 生成）

### 3. 同步到基础状态

**文件**: `src/hooks/article.ts`

- `syncBookInMyStudyList()` 函数（第362行）将运行时状态同步到基础状态

```typescript
export function syncBookInMyStudyList(study = false) {
  _nextTick(() => {
    const base = useBaseStore()
    const runtimeStore = useRuntimeStore()
    let temp = runtimeStore.editDict
    
    // 标记为自定义书籍（如果不是）
    if (!temp.custom && temp.id !== DictId.articleCollect) {
      temp.custom = true
      if (!temp.id.includes('_custom')) {
        temp.id += '_custom'
      }
    }
    
    // 更新文章数量
    temp.length = temp.articles.length
    
    // 同步到 baseStore.article.bookList
    let rIndex = base.article.bookList.findIndex((v) => v.id === temp.id)
    if (rIndex > -1) {
      base.article.bookList[rIndex] = getDefaultDict(temp)
      if (study) base.article.studyIndex = rIndex
    } else {
      base.article.bookList.push(getDefaultDict(temp))
      if (study) base.article.studyIndex = base.article.bookList.length - 1
    }
  }, 100)
}
```

**关键点**:
- 将 `runtimeStore.editDict` 同步到 `baseStore.article.bookList`
- 如果书籍已存在则更新，否则添加新书籍
- 自动标记自定义书籍（`custom: true`）

### 4. 自动持久化到 IndexedDB

**文件**: `src/App.vue`

- `watch` 监听器（第25行）监听 `store.$state` 的变化
- 自动将数据保存到 IndexedDB

```typescript
watch(store.$state, (n: BaseState) => {
  // 如果正在初始化，不保存数据，避免覆盖
  if (isInitializing) return
  
  // 1. 过滤数据（移除非自定义书籍的文章数据，节省空间）
  let data = shakeCommonDict(n)
  
  // 2. 保存到 IndexedDB
  set(SAVE_DICT_KEY.key, JSON.stringify({
    val: data, 
    version: SAVE_DICT_KEY.version
  }))
  
  // 3. 清理未使用的音频文件
  // ...
})
```

**关键点**:
- 使用 `idb-keyval` 库操作 IndexedDB
- 存储键名: `SAVE_DICT_KEY.key` = `'typing-word-dict'`
- 数据格式: `{ val: BaseState, version: 4 }`
- `shakeCommonDict()` 函数会过滤掉非自定义书籍的文章数据，只保留自定义和收藏的文章

## 💾 数据存储位置

### 1. 内存中的状态管理

#### Pinia Store 状态

**基础状态** (`src/stores/base.ts`):
```typescript
baseStore.article.bookList: Dict[]  // 所有书籍列表
```

**运行时状态** (`src/stores/runtime.ts`):
```typescript
runtimeStore.editDict: Dict  // 当前编辑的书籍
runtimeStore.editDict.articles: Article[]  // 当前书籍的文章列表
```

### 2. 持久化存储

#### IndexedDB 存储

**文章数据存储**:
- **键名**: `'typing-word-dict'` (`SAVE_DICT_KEY.key`)
- **存储库**: IndexedDB（通过 `idb-keyval` 库）
- **数据结构**:
  ```json
  {
    "version": 4,
    "val": {
      "article": {
        "bookList": [
          {
            "id": "xxx_custom",
            "name": "书籍名称",
            "custom": true,
            "articles": [
              {
                "id": "abc123",
                "title": "文章标题",
                "titleTranslate": "译文标题",
                "text": "原文内容",
                "textTranslate": "译文内容",
                "audioSrc": "音频URL",
                "audioFileId": "音频文件ID",
                "lrcPosition": [[0, 10], [10, 20]],
                "newWords": [],
                "questions": [],
                "nameList": []
              }
            ]
          }
        ]
      }
    }
  }
  ```

**音频文件存储**:
- **键名**: `'typing-word-files'` (`LOCAL_FILE_KEY`)
- **存储库**: IndexedDB
- **数据结构**: `Array<{ id: string, file: Blob }>`
- **说明**: 音频文件以 Blob 形式存储，通过 `audioFileId` 关联到文章

### 3. 数据过滤逻辑

**文件**: `src/utils/index.ts` - `shakeCommonDict()` 函数

```typescript
export function shakeCommonDict(n: BaseState): BaseState {
  let data: BaseState = cloneDeep(n)
  
  // 只保留自定义书籍和收藏书籍的文章数据
  data.article.bookList.map((v: Dict) => {
    if (!v.custom && ![DictId.articleCollect].includes(v.id)) {
      v.articles = []  // 清空非自定义书籍的文章
    } else {
      // 移除运行时生成的 sections 数据（节省空间）
      v.articles.map(a => {
        a.sections = []
      })
    }
  })
  
  return data
}
```

**关键点**:
- 非自定义书籍的文章数据不保存（使用时从服务器加载）
- 只保存自定义书籍（`custom: true`）和收藏书籍（`DictId.articleCollect`）的文章
- `sections` 字段不保存（运行时动态生成）

## 📁 关键代码文件

| 文件路径 | 作用 |
|---------|------|
| `src/pages/article/components/EditArticle.vue` | 文章编辑组件，处理用户输入和保存 |
| `src/pages/article/BatchEditArticlePage.vue` | 批量编辑页面，管理文章列表 |
| `src/stores/base.ts` | 基础状态管理，存储所有书籍和文章 |
| `src/stores/runtime.ts` | 运行时状态管理，存储当前编辑的书籍 |
| `src/hooks/article.ts` | 文章相关工具函数，包括同步函数 |
| `src/App.vue` | 应用入口，监听状态变化并自动保存 |
| `src/utils/index.ts` | 工具函数，包括数据过滤函数 |
| `src/config/env.ts` | 配置文件，定义存储键名和版本号 |

## 🔀 数据流转图

```
┌─────────────────────────────────────────────────────────────┐
│  用户操作：编辑文章并点击保存                                │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  EditArticle.vue                                            │
│  - save() 函数验证数据                                       │
│  - emit('save', editArticle)                                │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  BatchEditArticlePage.vue                                   │
│  - saveArticle() 函数                                       │
│  - 保存到 runtimeStore.editDict.articles[]                 │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  hooks/article.ts                                           │
│  - syncBookInMyStudyList() 函数                             │
│  - 同步到 baseStore.article.bookList[]                     │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  App.vue                                                    │
│  - watch(store.$state) 监听状态变化                         │
│  - shakeCommonDict() 过滤数据                               │
│  - set(SAVE_DICT_KEY.key, data) 保存到 IndexedDB          │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  IndexedDB                                                  │
│  - 键名: 'typing-word-dict'                                 │
│  - 存储: { version: 4, val: BaseState }                    │
│  - 音频文件: 'typing-word-files'                            │
└─────────────────────────────────────────────────────────────┘
```

## 🔍 关键数据结构

### Article 接口

```typescript
interface Article {
  id?: number | string;           // 文章ID（6位随机字符串）
  title: string;                  // 原文标题
  titleTranslate: string;          // 译文标题
  text: string;                    // 原文内容
  textTranslate: string;           // 译文内容
  newWords: Word[];               // 生词列表
  sections: Sentence[][];         // 段落句子（运行时生成，不保存）
  audioSrc: string;                // 在线音频URL
  audioFileId: string;             // 本地音频文件ID（IndexedDB）
  lrcPosition: number[][];         // 音频时间轴 [[start, end], ...]
  nameList: string[];              // 人名列表
  questions: Question[];           // 问题列表
}
```

### Dict 接口（书籍）

```typescript
interface Dict {
  id: string;                     // 书籍ID
  name: string;                   // 书籍名称
  custom: boolean;                 // 是否自定义
  articles: Article[];            // 文章列表
  length: number;                 // 文章数量
  // ... 其他字段
}
```

## 📌 重要注意事项

1. **数据过滤**: 只有自定义书籍和收藏书籍的文章会被保存到 IndexedDB
2. **sections 字段**: 运行时动态生成，不保存到 IndexedDB
3. **音频文件**: 单独存储在 IndexedDB，通过 `audioFileId` 关联
4. **自动保存**: 状态变化时自动保存，无需手动调用保存函数
5. **版本控制**: 数据包含版本号，用于数据迁移和升级
6. **初始化保护**: 应用初始化期间不会保存数据，避免覆盖

## 🔧 如何查看存储的数据

### 浏览器开发者工具

1. 打开浏览器开发者工具（F12）
2. 进入 **Application** 标签页
3. 左侧选择 **IndexedDB**
4. 找到数据库（通常是网站域名）
5. 查看 `typing-word-dict` 键的数据

### 代码中查看

```typescript
import { get } from 'idb-keyval'
import { SAVE_DICT_KEY } from '@/config/env.ts'

// 读取存储的数据
const data = await get(SAVE_DICT_KEY.key)
console.log(JSON.parse(data))
```

## 📚 相关文档

- [如何添加自定义文章](./ADD_CUSTOM_ARTICLE.md) - 用户使用指南
- [部署文档](./DEPLOYMENT.md) - 部署相关说明

---

**最后更新**: 2025-01-XX
**版本**: 基于代码分析生成
