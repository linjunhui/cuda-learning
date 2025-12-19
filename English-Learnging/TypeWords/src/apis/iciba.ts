/**
 * 金山词霸 API
 * 用于获取单词的音标和中文意思
 * 使用官方 API: https://dict.iciba.com/dictionary/word/query/web
 */

import md5 from 'md5'

export interface IcibaWordInfo {
  word_name: string
  symbols: Array<{
    ph_en: string  // 英式音标
    ph_am: string  // 美式音标
    parts: Array<{
      part: string  // 词性，如 "int.", "n."
      means: string[]  // 中文意思数组
    }>
  }>
}

export interface IcibaApiResponse {
  message?: {
    baesInfo?: IcibaWordInfo  // API 返回的拼写错误字段
    baseInfo?: IcibaWordInfo  // 修复后的字段
    bidce?: any  // API 返回的拼写错误字段
    bidec?: any  // 修复后的字段
    [key: string]: any
  }
  [key: string]: any
}

/**
 * 生成 API 签名
 * @param word 单词
 * @param timestamp 时间戳
 * @returns MD5 签名
 */
function generateSignature(word: string, timestamp: number): string {
  // hard code in http://www.iciba.com/_next/static/chunks/8caea17ae752a5965491f530aed3596fce3ca5a9.f4f0c70d4f1b9d4253e3.js
  const hashKey = '7ece94d9f9c202b0d2ec557dg4r9bc'
  const hashMessageBody = `61000006${timestamp}${word}`
  const hashMessage = `/dictionary/word/query/web${hashMessageBody}${hashKey}`
  return md5(hashMessage)
}

/**
 * 获取单词翻译信息
 * @param word 要查询的单词
 * @returns 返回单词信息，包含音标和中文意思
 */
export async function getWordTranslation(word: string): Promise<IcibaWordInfo | null> {
  try {
    const rawWord = word.trim()
    if (!rawWord) {
      return null
    }

    // 处理单词大小写（首字母保持原样，其余转小写）
    const wordCapital = rawWord.substring(0, 1)
    const wordLower = rawWord.substring(1).toLowerCase()
    const processedWord = `${wordCapital}${wordLower}`

    // 生成签名参数
    const now = Date.now()
    const signature = generateSignature(processedWord, now)

    // 构建查询参数
    const queryParams = [
      'client=6',
      'key=1000006',
      `timestamp=${now}`,
      `word=${encodeURIComponent(processedWord)}`,
      `signature=${signature}`,
    ]
    const queryString = queryParams.join('&')

    // 使用代理路径（直接调用会被 CORS 阻止）
    const url = `/iciba/dictionary/word/query/web?${queryString}`

    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Accept': '*/*',
        'Accept-Language': 'zh-CN,zh;q=0.9',
      }
    })

    if (!response.ok) {
      console.error(`代理请求失败: ${response.status} ${response.statusText}`)
      console.error('请确保代理服务器正在运行 (node scripts/proxy-server.js)')
      return null
    }

    // 解析响应数据
    let data: IcibaApiResponse
    try {
      data = await response.json()
    } catch (jsonError) {
      console.error('解析响应 JSON 失败:', jsonError)
      return null
    }

    // 处理 API 返回的拼写错误字段
    if (data.message) {
      // 修复 baesInfo -> baseInfo
      if ('baesInfo' in data.message && !('baseInfo' in data.message)) {
        data.message.baseInfo = data.message.baesInfo
        delete data.message.baesInfo
      }

      // 修复 bidce -> bidec
      if ('bidce' in data.message && !('bidec' in data.message)) {
        data.message.bidec = data.message.bidce
        delete data.message.bidce
      }

      // 优先使用 baseInfo，如果没有则使用 baesInfo
      const wordInfo = data.message.baseInfo || data.message.baesInfo

      if (!wordInfo) {
        console.warn('未找到单词信息:', word, '响应数据:', data)
        return null
      }

      return wordInfo
    }

    console.warn('响应数据格式不正确:', data)
    return null
  } catch (error) {
    console.error('获取单词翻译失败:', error)
    if (error instanceof TypeError && error.message.includes('Failed to fetch')) {
      console.error('网络请求失败，请检查代理服务器是否正在运行')
    }
    return null
  }
}
