# 如何访问和编辑 IndexedDB

## 📍 IndexedDB 的位置

IndexedDB 是浏览器内置的数据库存储，**不在文件系统中**，而是存储在浏览器的数据库里。

## 🔍 方法一：通过浏览器开发者工具查看（最简单）

### Chrome/Edge 浏览器

1. **打开开发者工具**
   - 按 `F12` 或 `Ctrl+Shift+I` (Windows/Linux)
   - 或 `Cmd+Option+I` (Mac)
   - 或右键页面 → "检查"

2. **进入 Application 标签页**
   - 点击顶部标签栏的 **Application**（应用）
   - 如果没有看到，可能需要点击 `>>` 展开更多标签

3. **查看 IndexedDB**
   - 左侧边栏找到 **Storage**（存储）→ **IndexedDB**
   - 展开后会看到数据库名称（通常是网站域名，如 `typewords.cc`）
   - 点击数据库名称，会看到所有的 Object Store（对象存储）

4. **查看数据**
   - 点击 Object Store（如 `keyval-store`）
   - 右侧会显示所有的键值对
   - 可以点击键名查看对应的值

### Firefox 浏览器

1. 打开开发者工具（F12）
2. 点击 **存储** 标签页
3. 展开 **IndexedDB**
4. 选择数据库和 Object Store
5. 查看数据

## 🛠️ 方法二：通过代码访问

### 在浏览器控制台中使用

打开浏览器控制台（Console），输入以下代码：

```javascript
// 需要先加载 idb-keyval 库（如果项目中有）
// 或者直接使用原生 IndexedDB API

// 方法1: 使用 idb-keyval（项目中使用的方式）
// 如果页面已经加载了 idb-keyval，可以直接使用：
import('idb-keyval').then(({ get, set, keys, clear }) => {
  // 获取所有键
  keys().then(keys => {
    console.log('所有键:', keys)
  })
  
  // 读取特定键的值
  get('typing-word-dict').then(data => {
    console.log('文章数据:', JSON.parse(data))
  })
  
  // 读取设置数据
  get('typing-word-setting').then(data => {
    console.log('设置数据:', JSON.parse(data))
  })
  
  // 读取音频文件列表
  get('typing-word-files').then(data => {
    console.log('音频文件:', data)
  })
})

// 方法2: 使用原生 IndexedDB API
const request = indexedDB.open('keyval-store', 1)
request.onsuccess = (event) => {
  const db = event.target.result
  const transaction = db.transaction(['keyval-store'], 'readonly')
  const store = transaction.objectStore('keyval-store')
  
  // 获取所有键
  const getAllKeys = store.getAllKeys()
  getAllKeys.onsuccess = () => {
    console.log('所有键:', getAllKeys.result)
  }
  
  // 读取特定值
  const getRequest = store.get('typing-word-dict')
  getRequest.onsuccess = () => {
    console.log('数据:', JSON.parse(getRequest.result))
  }
}
```

## 📝 项目中使用的 IndexedDB 键名

根据 `src/config/env.ts`，项目中使用以下键名：

| 键名 | 说明 | 数据类型 |
|------|------|---------|
| `typing-word-dict` | 文章和单词数据 | JSON 字符串 |
| `typing-word-setting` | 用户设置 | JSON 字符串 |
| `typing-word-files` | 音频文件 | Array<{id: string, file: Blob}> |
| `type-words-app-version` | 应用版本号 | 数字 |

## ✏️ 如何编辑 IndexedDB 数据

### 方法1: 通过浏览器开发者工具编辑

1. 在 Application → IndexedDB 中找到要编辑的键
2. 双击值字段，可以直接编辑 JSON
3. 修改后按 Enter 保存
4. **注意**: 修改后需要刷新页面才能生效

### 方法2: 通过代码编辑

在浏览器控制台中：

```javascript
import('idb-keyval').then(({ get, set }) => {
  // 读取数据
  get('typing-word-dict').then(data => {
    const obj = JSON.parse(data)
    
    // 修改数据
    obj.val.article.bookList[0].name = '新名称'
    
    // 保存回去
    set('typing-word-dict', JSON.stringify(obj))
      .then(() => {
        console.log('保存成功！')
        // 刷新页面使更改生效
        location.reload()
      })
  })
})
```

### 方法3: 创建编辑工具页面

我已经为你创建了一个 IndexedDB 编辑工具页面，访问 `/idb-editor` 即可使用。

## ⚠️ 注意事项

1. **数据格式**: 
   - `typing-word-dict` 和 `typing-word-setting` 存储的是 JSON 字符串
   - 需要先 `JSON.parse()` 解析，修改后再 `JSON.stringify()` 保存

2. **版本号**: 
   - 数据格式包含版本号，不要随意修改版本号
   - 格式: `{ version: 4, val: {...} }`

3. **备份**: 
   - 修改前建议先导出数据备份
   - 可以通过 Application → IndexedDB → 右键 → "Clear" 清空数据

4. **刷新页面**: 
   - 修改 IndexedDB 后，需要刷新页面才能看到效果
   - 应用会自动从 IndexedDB 读取数据

## 🔧 快速查看数据的代码片段

将以下代码保存为书签，点击即可查看所有 IndexedDB 数据：

```javascript
javascript:(function(){
  const keys = ['typing-word-dict', 'typing-word-setting', 'typing-word-files', 'type-words-app-version'];
  Promise.all(keys.map(key => 
    new Promise((resolve) => {
      const request = indexedDB.open('keyval-store');
      request.onsuccess = () => {
        const db = request.result;
        const tx = db.transaction(['keyval-store'], 'readonly');
        const store = tx.objectStore('keyval-store');
        const getReq = store.get(key);
        getReq.onsuccess = () => {
          resolve({key, value: getReq.result});
        };
      };
    })
  )).then(results => {
    console.table(results);
    results.forEach(r => {
      console.log(`\n${r.key}:`, r.value);
    });
  });
})();
```

## 📚 相关文件

- `src/config/env.ts` - 定义所有 IndexedDB 键名
- `src/App.vue` - 自动保存逻辑
- `src/stores/base.ts` - 读取 IndexedDB 数据
- `src/utils/index.ts` - 数据过滤和升级逻辑
