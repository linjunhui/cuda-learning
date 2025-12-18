import {defineConfig} from 'vite'
import Vue from '@vitejs/plugin-vue'
import VueJsx from "@vitejs/plugin-vue-jsx";
import {resolve} from 'path'
import {visualizer} from "rollup-plugin-visualizer";
import SlidePlugin from './src/components/slide/data.js';
import {getLastCommit} from "git-last-commit";
import UnoCSS from 'unocss/vite'
import VueMacros from 'unplugin-vue-macros/vite'
import Icons from 'unplugin-icons/vite'
import Components from 'unplugin-vue-components/vite'
import IconsResolver from 'unplugin-icons/resolver'
import {viteExternalsPlugin} from 'vite-plugin-externals'

function pathResolve(dir: string) {
  return resolve(__dirname, ".", dir)
}

const lifecycle = process.env.npm_lifecycle_event;
let isCdnBuild = ['build-oss', 'report-oss'].includes(lifecycle)
let isAnalyseBuild = ['report-oss', 'report'].includes(lifecycle)

export default defineConfig(() => {
  return new Promise(resolve => {
    let latestCommitHash = ''
    getLastCommit((err, commit) => {
      if (!err) latestCommitHash = commit.shortHash
      resolve({
        plugins: [
          Icons({
            autoInstall: true,
            compiler: 'vue3',
          }),
          Components({
            resolvers: [
              IconsResolver({
                prefix: 'Icon',
              }),
            ],
          }),
          VueMacros({
            plugins: {
              vue: Vue(),
              vueJsx: VueJsx(),
            },
          }),
          UnoCSS(),
          SlidePlugin(),
          isAnalyseBuild ?
            visualizer({
              gzipSize: true,
              brotliSize: true,
              emitFile: false,
              filename: "report.html",
              open: true
            }) : null,
          isCdnBuild ? [
            {
              name: 'inject-cdn-head',
              enforce: 'pre',
              transformIndexHtml(html) {
                const scripts = `
<script src="./libs/vue.global.prod.min.js" crossorigin="anonymous"></script>
<script src="./libs/vue-router.global.prod.min.js" crossorigin="anonymous"></script>
<script src="./libs/axios.min.js" crossorigin="anonymous"></script>
`
                return html.replace('<head>', `<head>${scripts}`)
              },
            },
            viteExternalsPlugin({
              vue: 'Vue',
              'vue-router': 'VueRouter',
              axios: 'axios',
            })
          ] : null,
        ],
        build: {
          rollupOptions: {
            output: {
              manualChunks(id) {
                if (id.includes('node_modules/@iconify') || id.includes('~icons')) {
                  return 'icons';
                }
                if (id.includes('utils')
                  || id.includes('hooks')
                ) {
                  return 'utils'
                }
                if (!isCdnBuild) return
                if (id.includes('dialog')) {
                  return 'dialog'
                }
              }
            }
          }
        },
        define: {
          LATEST_COMMIT_HASH: JSON.stringify(latestCommitHash + (process.env.NODE_ENV === 'production' ? '' : ' (dev)')),
        },
        base: './',
        resolve: {
          alias: {
            "@": pathResolve("src"),
          },
          extensions: ['.mjs', '.js', '.ts', '.jsx', '.tsx', '.json', '.vue']
        },
        css: {
          preprocessorOptions: {
            scss: {
              api: "modern-compiler"
            }
          }
        },
        server: {
          port: 48372,
          open: false,
          host: '0.0.0.0',
          proxy: {
            '/baidu': 'https://api.fanyi.baidu.com/api/trans/vip/translate',
            '/iciba': {
              target: 'https://www.iciba.com',
              changeOrigin: true,
              rewrite: (path) => path.replace(/^\/iciba/, ''),
              configure: (proxy, options) => {
                proxy.on('proxyReq', (proxyReq, req, res) => {
                  // 设置必要的请求头
                  proxyReq.setHeader('Accept', '*/*')
                  proxyReq.setHeader('Accept-Language', 'zh-CN,zh;q=0.9,en;q=0.8')
                  proxyReq.setHeader('Connection', 'keep-alive')
                  proxyReq.setHeader('Referer', 'https://www.iciba.com/')
                  proxyReq.setHeader('Sec-Fetch-Dest', 'empty')
                  proxyReq.setHeader('Sec-Fetch-Mode', 'cors')
                  proxyReq.setHeader('Sec-Fetch-Site', 'same-origin')
                  proxyReq.setHeader('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36')
                  proxyReq.setHeader('sec-ch-ua', '"Google Chrome";v="143", "Chromium";v="143", "Not A(Brand";v="24"')
                  proxyReq.setHeader('sec-ch-ua-mobile', '?0')
                  proxyReq.setHeader('sec-ch-ua-platform', '"Windows"')
                  proxyReq.setHeader('x-nextjs-data', '1')
                })
              }
            }
          }
        }
      })
    })
  })
})
