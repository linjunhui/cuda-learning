import http, {axiosInstance, AxiosResponse} from "@/utils/http.ts";
import { Dict } from "@/types/types.ts";
import { cloneDeep } from "@/utils";

function remove(data?: any) {
  if (data) {
    let s = cloneDeep(data)
    delete s.words
    delete s.articles
    delete s.statistics
    return s;
  }
}

export function dictListVersion() {
  return http<number>('dict/dictListVersion', null, null, 'get')
}

export function myDictList(params?) {
  return http('dict/myDictList', null, params, 'get')
}

export function add2MyDict(data) {
  return http('dict/add2MyDict', remove(data), null, 'post')
}

export function addStat(data) {
  return http('dict/addStat', data, null, 'post')
}

export function detail(params?, data?) {
  // 添加时间戳参数破坏缓存，确保获取最新数据
  const cacheBustParams = {
    ...params,
    _t: Date.now(), // 时间戳参数
  }
  // 使用 axiosInstance 直接调用，添加 no-cache 头
  return axiosInstance({
    url: 'dict/detail',
    method: 'get',
    params: cacheBustParams,
    data,
    headers: {
      'Cache-Control': 'no-cache, no-store, must-revalidate',
      'Pragma': 'no-cache',
      'Expires': '0',
    },
  }).then(response => {
    // 处理响应，保持与 http 函数相同的格式
    const {data} = response
    if (response.status !== 200) {
      return Promise.resolve({
        code: response.status,
        msg: response.statusText,
        data: null,
        success: false,
      })
    }
    if (data === null) {
      return Promise.resolve({
        code: 500,
        msg: '系统出现错误',
        data: {},
        success: false,
      })
    }
    if (typeof data !== 'object') {
      return Promise.resolve({
        data,
        success: true,
        code: 200
      })
    }
    return Promise.resolve(data)
  }).catch(error => {
    // 错误处理，保持与 http 函数相同的格式
    if (error.response === undefined && error.status === undefined) {
      return Promise.resolve({
        code: 500,
        msg: '服务器响应超时',
        data: null,
        success: false,
      })
    }
    if (error.response && error.response.status >= 500) {
      return Promise.resolve({
        code: 500,
        msg: '服务器出现错误',
        data: null,
        success: false,
      })
    }
    if (error.response && error.response.status === 401) {
      return Promise.resolve({
        code: 500,
        msg: '用户名或密码不正确',
        data: null,
        success: false,
      })
    }
    const {data} = error.response || {}
    if (data && data.code !== undefined) {
      return Promise.resolve({
        code: data.code,
        msg: data.msg,
        success: false,
      })
    }
    return Promise.resolve({
      code: 500,
      success: false,
      msg: data?.msg || '请求失败',
      data: null,
    })
  })
}

export function setDictProp(params?, data?) {
  return http<Dict>('dict/setDictProp', remove(data), remove(params), 'post')
}

export function syncSetting(params?, data?) {
  return http<Dict>('dict/syncSetting', remove(data), remove(params), 'post')
}

export function getSetting(params?, data?) {
  return http<Dict>('dict/getSetting', remove(data), remove(params), 'get')
}

export function addDict(params?, data?) {
  return http<Dict>('dict/addDict', remove(data), remove(params), 'post')
}

export function uploadImportData<T>(data, onUploadProgress): Promise<AxiosResponse<T>> {
  return axiosInstance({
    url: 'dict/uploadImportData',
    method: 'post',
    headers: {
      contentType: 'formdata',
    },
    timeout: 1000000000,
    data,
    onUploadProgress
  })
}

export function upload(data, onUploadProgress) {
  return axiosInstance({
    url: 'file/upload',
    method: 'post',
    headers: {
      contentType: 'formdata',
    },
    data,
    onUploadProgress
  })
}

export function getProgress() {
  return http<{ status: number; reason: string }>('dict/getProgress', null, null, 'get')
}
