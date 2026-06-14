# 部署与启动指南

本文档说明如何修改开发端口以及如何部署和启动项目。

## 目录

- [修改开发端口](#修改开发端口)
- [本地开发启动](#本地开发启动)
- [项目构建](#项目构建)
- [部署方式](#部署方式)
  - [GitHub Pages 部署](#github-pages-部署)
  - [阿里云 OSS 部署](#阿里云-oss-部署)
  - [本地预览](#本地预览)

## 修改开发端口

项目使用 Vite 作为构建工具，开发服务器端口配置在 `vite.config.mts` 文件中。

### 方法一：修改配置文件（推荐）

1. 打开 `vite.config.mts` 文件
2. 找到 `server` 配置项（约第 114 行）
3. 修改 `port` 字段的值

```typescript
server: {
  port: 3000,  // 修改为你想要的端口号，例如 8080
  open: false,
  host: '0.0.0.0',
  proxy: {
    '/baidu': 'https://api.fanyi.baidu.com/api/trans/vip/translate'
  }
}
```

### 方法二：使用命令行参数

启动时通过命令行参数指定端口：

```bash
# npm
npm run dev -- --port 8080

# pnpm
pnpm dev -- --port 8080

# yarn
yarn dev --port 8080
```

### 方法三：使用环境变量

可以通过环境变量 `PORT` 来设置端口：

```bash
# Linux/Mac
PORT=8080 npm run dev

# Windows (PowerShell)
$env:PORT=8080; npm run dev

# Windows (CMD)
set PORT=8080 && npm run dev
```

**注意**：如果指定的端口已被占用，Vite 会自动尝试使用下一个可用端口。

## 本地开发启动

### 前置要求

- Node.js（推荐版本 18+）
- npm、pnpm 或 yarn 包管理器

### 安装依赖

```bash
# 使用 npm
npm install

# 使用 pnpm（推荐）
pnpm install

# 使用 yarn
yarn install
```

### 启动开发服务器

```bash
# 使用 npm
npm run dev

# 使用 pnpm
pnpm dev

# 使用 yarn
yarn dev
```

启动成功后，默认访问地址为：`http://localhost:3000`

如果修改了端口，请访问对应的端口地址。

## 项目构建

### 标准构建

构建生产版本：

```bash
# npm
npm run build

# pnpm
pnpm build

# yarn
yarn build
```

构建产物将输出到 `dist` 目录。

### OSS 构建

构建用于阿里云 OSS 部署的版本（会进行额外的 CDN 优化）：

```bash
# npm
npm run build-oss

# pnpm
pnpm build-oss

# yarn
yarn build-oss
```

### 预览构建产物

构建完成后，可以使用预览命令查看构建结果：

```bash
# npm
npm run preview

# pnpm
pnpm preview

# yarn
yarn preview
```

预览服务器默认端口为 4173，访问地址：`http://localhost:4173`

## 部署方式

### GitHub Pages 部署

项目已配置 GitHub Actions 工作流，推送到 `master` 分支时会自动部署到 GitHub Pages。

#### 自动部署

1. 确保 `.github/workflows/deploy-pages.yml` 文件存在
2. 推送代码到 `master` 分支
3. GitHub Actions 会自动执行构建和部署

#### 手动触发部署

1. 进入 GitHub 仓库页面
2. 点击 "Actions" 标签
3. 选择 "Deploy to GitHub Pages" 工作流
4. 点击 "Run workflow" 手动触发

#### 配置 GitHub Pages

1. 进入仓库 Settings
2. 找到 Pages 设置
3. 选择 Source 为 "GitHub Actions"

### 阿里云 OSS 部署

项目支持部署到阿里云 OSS，并自动刷新 CDN 缓存。

#### 前置配置

需要在 GitHub Secrets 中配置以下环境变量：

- `OSS_KEY_ID`: 阿里云 AccessKey ID
- `OSS_KEY_SECRET`: 阿里云 AccessKey Secret
- `OSS_BUCKET`: OSS 存储桶名称
- `OSS_REGION`: OSS 区域（如：oss-cn-hangzhou）
- `CDN_DOMAIN`: CDN 域名（可选）

#### 自动部署

1. 确保 `.github/workflows/deploy-aliyun-oss.yml` 文件存在
2. 配置好 GitHub Secrets
3. 推送代码到 `master` 分支
4. GitHub Actions 会自动执行构建和部署

#### 本地部署

如果需要手动部署到 OSS：

```bash
# 1. 构建 OSS 版本
pnpm run build-oss

# 2. 部署到 OSS（需要配置环境变量）
OSS_REGION=your-region \
OSS_KEY_ID=your-key-id \
OSS_KEY_SECRET=your-key-secret \
OSS_BUCKET=your-bucket \
pnpm run deploy-oss
```

#### 部署脚本说明

部署脚本 (`scripts/deploy-oss.js`) 会：

1. 上传 `dist` 目录下的所有文件到 OSS
2. 跳过以下目录（保留远程文件）：
   - `dicts`
   - `sound`
   - `libs`
   - `imgs`
3. 删除远程多余的文件（本地未上传的文件）
4. 刷新 CDN 缓存（如果配置了 `CDN_DOMAIN`）

### 本地预览

构建完成后，可以使用预览命令查看构建结果：

```bash
npm run preview
```

预览服务器默认运行在 `http://localhost:4173`，可以通过 `--port` 参数修改端口：

```bash
npm run preview -- --port 8080
```

### 其他部署方式

#### Vercel 部署

项目包含 `.vercel` 配置，可以直接部署到 Vercel：

1. 安装 Vercel CLI：`npm i -g vercel`
2. 在项目根目录运行：`vercel`
3. 按照提示完成部署

#### 静态服务器部署

将构建后的 `dist` 目录内容上传到任何静态文件服务器即可：

1. 执行 `npm run build` 构建项目
2. 将 `dist` 目录下的所有文件上传到服务器
3. 配置 Web 服务器（如 Nginx）指向上传的文件

**Nginx 配置示例**：

```nginx
server {
    listen 80;
    server_name your-domain.com;
    root /path/to/dist;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

## 常见问题

### 端口被占用

如果启动时提示端口被占用：

1. 修改 `vite.config.mts` 中的端口号
2. 或使用命令行参数指定其他端口：`npm run dev -- --port 8080`
3. 或关闭占用端口的进程

### 构建失败

如果构建失败，请检查：

1. Node.js 版本是否符合要求（推荐 18+）
2. 依赖是否正确安装：`npm install` 或 `pnpm install`
3. 查看构建错误信息，根据提示修复问题

### 部署后页面空白

如果部署后页面空白，请检查：

1. 构建是否成功完成
2. 服务器是否正确配置了路由（SPA 应用需要配置 fallback 到 index.html）
3. 检查浏览器控制台是否有错误信息

## 相关链接

- [Vite 官方文档](https://vitejs.dev/)
- [GitHub Actions 文档](https://docs.github.com/en/actions)
- [阿里云 OSS 文档](https://help.aliyun.com/product/31815.html)
