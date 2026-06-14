/**
 * 简单的代理服务器（使用 Node.js 内置模块，无需额外依赖）
 * 用于生产环境代理翻译接口
 * 使用方法: node scripts/proxy-server.js
 */

const http = require('http')
const https = require('https')
const fs = require('fs')
const path = require('path')
const url = require('url')

const PORT = 48372
const DIST_DIR = path.join(__dirname, '../dist')

// 获取文件 MIME 类型
function getMimeType(filePath) {
  const ext = path.extname(filePath).toLowerCase()
  const mimeTypes = {
    '.html': 'text/html',
    '.js': 'application/javascript',
    '.css': 'text/css',
    '.json': 'application/json',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.gif': 'image/gif',
    '.svg': 'image/svg+xml',
    '.ico': 'image/x-icon',
    '.woff': 'font/woff',
    '.woff2': 'font/woff2',
    '.ttf': 'font/ttf',
    '.eot': 'application/vnd.ms-fontobject',
    '.mp3': 'audio/mpeg',
    '.wav': 'audio/wav',
    '.ogg': 'audio/ogg',
    '.mp4': 'video/mp4',
    '.webm': 'video/webm'
  }
  return mimeTypes[ext] || 'application/octet-stream'
}

// 代理请求到 iciba.com
function proxyIciba(req, res) {
  // 解析 URL，正确提取路径和查询参数
  const parsedUrl = url.parse(req.url, true)
  // 移除 /iciba 前缀，获取目标路径
  const targetPath = parsedUrl.pathname.replace(/^\/iciba/, '')
  // 构建查询字符串
  const queryString = parsedUrl.search || ''
  
  // 根据路径判断使用哪个域名
  // 新 API 使用 dict.iciba.com，旧 API 使用 www.iciba.com
  let baseUrl = 'https://www.iciba.com'
  if (targetPath.startsWith('/dictionary/')) {
    baseUrl = 'https://dict.iciba.com'
  }
  
  // 构建完整的目标 URL
  const targetUrl = `${baseUrl}${targetPath}${queryString}`
  
  console.log(`[${new Date().toISOString()}] 代理请求: ${req.method} ${req.url}`)
  console.log(`  -> ${targetUrl}`)
  
  https.get(targetUrl, {
    headers: {
      'Accept': '*/*',
      'Accept-Language': 'zh-CN,zh;q=0.9',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
      'Pragma': 'no-cache',
      'Referer': 'https://www.iciba.com/',
      'Sec-Fetch-Dest': 'empty',
      'Sec-Fetch-Mode': 'cors',
      'Sec-Fetch-Site': 'cross-site',
      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.6261.95 Safari/537.36',
      'sec-ch-ua': '"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"',
      'sec-ch-ua-mobile': '?0',
      'sec-ch-ua-platform': '"Windows"',
    }
  }, (proxyRes) => {
    console.log(`  <- 响应状态: ${proxyRes.statusCode}`)
    
    // 复制响应头
    const responseHeaders = {}
    Object.keys(proxyRes.headers).forEach(key => {
      responseHeaders[key] = proxyRes.headers[key]
    })
    
    // 设置 CORS 头（覆盖原有的）
    responseHeaders['Access-Control-Allow-Origin'] = '*'
    responseHeaders['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    responseHeaders['Access-Control-Allow-Headers'] = 'Content-Type'
    
    res.writeHead(proxyRes.statusCode, responseHeaders)
    proxyRes.pipe(res)
  }).on('error', (err) => {
    console.error(`  <- 代理错误: ${err.message}`)
    res.writeHead(500, { 
      'Content-Type': 'text/plain',
      'Access-Control-Allow-Origin': '*'
    })
    res.end('代理请求失败: ' + err.message)
  })
}

// 处理静态文件请求
function serveStatic(req, res) {
  // 解析URL，分离路径和查询参数
  const parsedUrl = url.parse(req.url, true)
  let requestPath = parsedUrl.pathname || '/'
  
  // 解码URL路径（处理编码后的特殊字符，如 %26, %20 等）
  try {
    requestPath = decodeURIComponent(requestPath)
  } catch (e) {
    // 如果解码失败，使用原始路径
    console.warn('URL解码失败:', requestPath, e.message)
  }
  
  // 构建文件路径
  let filePath = path.join(DIST_DIR, requestPath === '/' ? 'index.html' : requestPath)
  
  // 安全检查：确保文件在 dist 目录内
  const resolvedPath = path.resolve(filePath)
  if (!resolvedPath.startsWith(path.resolve(DIST_DIR))) {
    res.writeHead(403)
    res.end('Forbidden')
    return
  }
  
  fs.stat(filePath, (err, stats) => {
    if (err || !stats.isFile()) {
      // 文件不存在，检查是否是静态资源（音频、图片等）
      const ext = path.extname(filePath).toLowerCase()
      const staticExtensions = ['.mp3', '.wav', '.ogg', '.mp4', '.webm', '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', '.woff', '.woff2', '.ttf', '.eot', '.json']
      
      if (staticExtensions.includes(ext)) {
        // 静态资源不存在，返回404
        res.writeHead(404, { 'Content-Type': 'text/plain' })
        res.end('File Not Found')
        return
      }
      
      // 其他情况返回 index.html（SPA 路由支持）
      filePath = path.join(DIST_DIR, 'index.html')
    }
    
    fs.readFile(filePath, (err, data) => {
      if (err) {
        res.writeHead(404)
        res.end('Not Found')
        return
      }
      
      res.writeHead(200, {
        'Content-Type': getMimeType(filePath),
        'Cache-Control': 'public, max-age=3600'
      })
      res.end(data)
    })
  })
}

// 创建服务器
const server = http.createServer((req, res) => {
  // 设置 CORS 头
  res.setHeader('Access-Control-Allow-Origin', '*')
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type')
  
  if (req.method === 'OPTIONS') {
    res.writeHead(200)
    res.end()
    return
  }
  
  // 解析 URL 路径（不包含查询参数）
  const parsedUrl = url.parse(req.url || '/', true)
  const pathname = parsedUrl.pathname || '/'
  
  // 代理翻译接口（检查路径是否以 /iciba 开头）
  if (pathname.startsWith('/iciba')) {
    proxyIciba(req, res)
    return
  }
  
  // 静态文件服务
  serveStatic(req, res)
})

server.listen(PORT, '0.0.0.0', () => {
  console.log(`🚀 服务器运行在 http://0.0.0.0:${PORT}`)
  console.log(`📦 静态文件目录: ${DIST_DIR}`)
  console.log(`🔄 代理接口: /iciba -> https://dict.iciba.com 或 https://www.iciba.com`)
})
