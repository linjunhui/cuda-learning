# TypeWords VSCode 插件

这是 TypeWords 英语学习工具的 VSCode 插件客户端，可以在 VSCode 中快速访问和使用 TypeWords 的功能。

## 功能特性

- 🚀 快速打开 TypeWords Web 应用
- 📚 直接打开单词练习页面
- 📖 直接打开文章练习页面
- 🔤 翻译选中的文本
- ⌨️ 快捷键支持（Ctrl+Shift+T / Cmd+Shift+T）
- 📊 状态栏快捷入口

## 安装

### 从源码安装

1. 进入插件目录：
```bash
cd vscode-extension
```

2. 安装依赖：
```bash
npm install
```

3. 编译插件：
```bash
npm run compile
```

4. 打包插件：
```bash
npm run package
```

5. 在 VSCode 中安装：
   - 按 `F1` 打开命令面板
   - 输入 `Extensions: Install from VSIX...`
   - 选择生成的 `.vsix` 文件

### 开发模式

1. 在 VSCode 中打开 `vscode-extension` 目录
2. 按 `F5` 启动调试，会打开一个新的 VSCode 窗口（Extension Development Host）
3. 在新窗口中测试插件功能

## 配置

在 VSCode 设置中可以配置以下选项：

- `typewords.serverUrl`: TypeWords 服务器地址（默认：`http://47.108.93.168:48372`）
- `typewords.autoOpen`: 启动时自动打开 TypeWords（默认：`false`）

## 使用方法

### 命令面板

1. 按 `Ctrl+Shift+P` (Mac: `Cmd+Shift+P`) 打开命令面板
2. 输入 `TypeWords` 查看所有可用命令：
   - `TypeWords: 打开 TypeWords`
   - `TypeWords: 练习单词`
   - `TypeWords: 练习文章`
   - `TypeWords: 翻译选中文本`

### 快捷键

- `Ctrl+Shift+T` (Mac: `Cmd+Shift+T`): 快速打开 TypeWords

### 右键菜单

在编辑器中选中文本后，右键菜单会出现"翻译选中文本"选项。

### 状态栏

点击状态栏右侧的 TypeWords 图标可以快速打开应用。

## 开发

### 项目结构

```
vscode-extension/
├── src/
│   └── extension.ts      # 插件主入口文件
├── out/                  # 编译输出目录（自动生成）
├── package.json          # 插件配置文件
├── tsconfig.json         # TypeScript 配置
└── README.md            # 说明文档
```

### 调试

1. 在 VSCode 中打开插件目录
2. 按 `F5` 启动调试
3. 在新打开的窗口中测试插件功能
4. 在 `extension.ts` 中设置断点进行调试

### 构建

```bash
# 编译 TypeScript
npm run compile

# 监听模式编译
npm run watch

# 打包插件
npm run package
```

## 许可证

与主项目保持一致。
