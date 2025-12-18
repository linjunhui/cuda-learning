/**
 * 金山词霸 API
 * 用于获取单词的音标和中文意思
 */

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

export interface IcibaResponse {
  pageProps: {
    query: {
      w: string
    }
    initialReduxState: {
      word: {
        wordInfo: {
          baesInfo: IcibaWordInfo
        }
      }
    }
  }
}

/**
 * 获取单词翻译信息
 * @param word 要查询的单词
 * @returns 返回单词信息，包含音标和中文意思
 */
export async function getWordTranslation(word: string): Promise<IcibaWordInfo | null> {
  try {
    // 优先使用代理路径，如果失败则尝试直接调用
    const isDev = import.meta.env.DEV
    let baseUrl = '/iciba'
    let url = `${baseUrl}/_next/data/4TJ-osf4lpT-hPtwIOdfu/word.json?w=${encodeURIComponent(word)}`
    
    let response = await fetch(url, {
      method: 'GET',
      headers: {
        'Accept': '*/*',
      }
    })
    
    // 如果代理失败（404 或网络错误），尝试直接调用
    if (!response.ok && !isDev) {
      console.warn('代理请求失败，尝试直接调用:', response.status)
      baseUrl = 'https://www.iciba.com'
      url = `${baseUrl}/_next/data/4TJ-osf4lpT-hPtwIOdfu/word.json?w=${encodeURIComponent(word)}`
      response = await fetch(url, {
        method: 'GET',
        headers: {
          'Accept': '*/*',
          'Referer': `https://www.iciba.com/word?w=${encodeURIComponent(word)}`,
        },
        mode: 'cors',
      })
    }

    if (!response.ok) {
      console.error('金山词霸 API 请求失败:', response.status, response.statusText)
      return null
    }

    const data: IcibaResponse = await response.json()
    
    // 从响应中提取 baesInfo
    const baesInfo = data?.pageProps?.initialReduxState?.word?.wordInfo?.baesInfo
    
    if (!baesInfo) {
      console.warn('未找到单词信息:', word)
      return null
    }

    return baesInfo
  } catch (error) {
    console.error('获取单词翻译失败:', error)
    return null
  }
}
