<script setup lang="ts">
import { useBaseStore } from "@/stores/base.ts";
import { useRouter } from "vue-router";
import BaseIcon from "@/components/BaseIcon.vue";
import {
  _getAccomplishDate,
  _getDictDataByUrl, _getStudyProgress,
  _nextTick,
  isMobile,
  loadJsLib,
  resourceWrap,
  shuffle,
  useNav
} from "@/utils";
import BasePage from "@/components/BasePage.vue";
import { DictResource, WordPracticeMode } from "@/types/types.ts";
import { watch } from "vue";
import { getCurrentStudyWord } from "@/hooks/dict.ts";
import { useRuntimeStore } from "@/stores/runtime.ts";
import Book from "@/components/Book.vue";
import PopConfirm from "@/components/PopConfirm.vue";
import Progress from '@/components/base/Progress.vue';
import Toast from '@/components/base/toast/Toast.ts';
import BaseButton from "@/components/BaseButton.vue";
import { getDefaultDict } from "@/types/func.ts";
import DeleteIcon from "@/components/icon/DeleteIcon.vue";
import PracticeSettingDialog from "@/pages/word/components/PracticeSettingDialog.vue";
import ChangeLastPracticeIndexDialog from "@/pages/word/components/ChangeLastPracticeIndexDialog.vue";
import { useSettingStore } from "@/stores/setting.ts";
import { useFetch } from "@vueuse/core";
import { AppEnv, DICT_LIST, Host, LIB_JS_URL, PracticeSaveWordKey, TourConfig } from "@/config/env.ts";
import { myDictList } from "@/apis";
import PracticeWordListDialog from "@/pages/word/components/PracticeWordListDialog.vue";
import ShufflePracticeSettingDialog from "@/pages/word/components/ShufflePracticeSettingDialog.vue";


const store = useBaseStore()
const settingStore = useSettingStore()
const router = useRouter()
const {nav} = useNav()
const runtimeStore = useRuntimeStore()
let loading = $ref(true)
let isSaveData = $ref(false)

let currentStudy = $ref({
  new: [],
  review: [],
  write: [],
  shuffle: [],
})

watch(() => store.load, n => {
  if (n) {
    init()
    _nextTick(async () => {
      const Shepherd = await loadJsLib('Shepherd', LIB_JS_URL.SHEPHERD);
      const tour = new Shepherd.Tour(TourConfig);
      tour.on('cancel', () => {
        localStorage.setItem('tour-guide', '1');
      });
      tour.addStep({
        id: 'step1',
        text: '点击这里选择一本词典开始学习',
        attachTo: {
          element: '#step1',
          on: 'bottom'
        },
        buttons: [
          {
            text: `下一步（1/${TourConfig.total}）`,
            action() {
              tour.next()
              router.push('/dict-list')
            }
          }
        ]
      });
      const r = localStorage.getItem('tour-guide');
      if (settingStore.first && !r && !isMobile()) tour.start();
    }, 500)
  }
}, {immediate: true})

async function init() {
  if (AppEnv.CAN_REQUEST) {
    let res = await myDictList({type: "word"})
    if (res.success) {
      store.setState(Object.assign(store.$state, res.data))
    }
  }
  if (store.word.studyIndex >= 3) {
    if (!store.sdict.custom && !store.sdict.words.length) {
      store.word.bookList[store.word.studyIndex] = await _getDictDataByUrl(store.sdict)
    }
  }
  if (!currentStudy.new.length && store.sdict.words.length) {
    let d = localStorage.getItem(PracticeSaveWordKey.key)
    if (d) {
      try {
        let obj = JSON.parse(d)
        currentStudy = obj.val.taskWords
        isSaveData = true
      } catch (e) {
        localStorage.removeItem(PracticeSaveWordKey.key)
        currentStudy = getCurrentStudyWord()
      }
    } else {
      currentStudy = getCurrentStudyWord()
    }
  }
  loading = false
}

function startPractice() {
  if (store.sdict.id) {
    if (!store.sdict.words.length) {
      return Toast.warning('没有单词可学习！')
    }
    window.umami?.track('startStudyWord', {
      name: store.sdict.name,
      index: store.sdict.lastLearnIndex,
      perDayStudyNumber: store.sdict.perDayStudyNumber,
      custom: store.sdict.custom,
      complete: store.sdict.complete,
      wordPracticeMode: settingStore.wordPracticeMode
    })
    //把是否是第一次设置为false
    settingStore.first = false
    nav('practice-words/' + store.sdict.id, {}, {taskWords: currentStudy})
  } else {
    window.umami?.track('no-dict')
    Toast.warning('请先选择一本词典')
  }
}

let showPracticeSettingDialog = $ref(false)
let showShufflePracticeSettingDialog = $ref(false)
let showChangeLastPracticeIndexDialog = $ref(false)
let showPracticeWordListDialog = $ref(false)

async function goDictDetail(val: DictResource) {
  if (!val.id) return nav('dict-list')
  runtimeStore.editDict = getDefaultDict(val)
  nav('dict-detail', {})
}

let isManageDict = $ref(false)
let selectIds = $ref([])

function handleBatchDel() {
  selectIds.forEach(id => {
    let r = store.word.bookList.findIndex(v => v.id === id)
    if (r !== -1) {
      if (store.word.studyIndex === r) {
        store.word.studyIndex = -1
      }
      if (store.word.studyIndex > r) {
        store.word.studyIndex--
      }
      store.word.bookList.splice(r, 1)
    }
  })
  selectIds = []
  Toast.success("删除成功！")
}

function toggleSelect(item) {
  let rIndex = selectIds.findIndex(v => v === item.id)
  if (rIndex > -1) {
    selectIds.splice(rIndex, 1)
  } else {
    selectIds.push(item.id)
  }
}

const progressTextLeft = $computed(() => {
  if (store.sdict.complete) return '已学完，进入总复习阶段'
  return '已学习' + store.currentStudyProgress + '%'
})
const progressTextRight = $computed(() => {
  // if (store.sdict.complete) return store.sdict?.length
  return store.sdict?.lastLearnIndex
})

function check(cb: Function) {
  if (!store.sdict.id) {
    Toast.warning('请先选择一本词典')
  } else {
    runtimeStore.editDict = getDefaultDict(store.sdict)
    cb()
  }
}

async function savePracticeSetting() {
  Toast.success('修改成功')
  isSaveData = false
  localStorage.removeItem(PracticeSaveWordKey.key)
  await store.changeDict(runtimeStore.editDict)
  currentStudy = getCurrentStudyWord()
}

async function onShufflePracticeSettingOk(total) {
  window.umami?.track('startShuffleStudyWord', {
    name: store.sdict.name,
    index: store.sdict.lastLearnIndex,
    perDayStudyNumber: store.sdict.perDayStudyNumber,
    total,
    custom: store.sdict.custom,
    complete: store.sdict.complete,
  })
  isSaveData = false
  localStorage.removeItem(PracticeSaveWordKey.key)

  let ignoreList = [store.allIgnoreWords, store.knownWords][settingStore.ignoreSimpleWord ? 0 : 1]
  currentStudy.shuffle = shuffle(store.sdict.words.slice(0, store.sdict.lastLearnIndex).filter(v => !ignoreList.includes(v.word))).slice(0, total)
  nav('practice-words/' + store.sdict.id, {}, {
    taskWords: currentStudy,
    total //用于再来一组时，随机出正确的长度，因为练习中可能会点击已掌握，导致重学一遍之后长度变少，如果再来一组，此时长度就不正确
  })
}

async function saveLastPracticeIndex(e) {
  Toast.success('修改成功')
  runtimeStore.editDict.lastLearnIndex = e
  showChangeLastPracticeIndexDialog = false
  isSaveData = false
  localStorage.removeItem(PracticeSaveWordKey.key)
  await store.changeDict(runtimeStore.editDict)
  currentStudy = getCurrentStudyWord()
}

const {
  data: recommendDictList,
  isFetching
} = useFetch(resourceWrap(DICT_LIST.WORD.RECOMMENDED)).json()

let isNewHost = $ref(window.location.host === Host)

</script>

<template>
  <BasePage>
    <div class="card flex flex-col md:flex-row gap-4">
      <div class="flex-1 w-full flex flex-col justify-between">
        <div class="flex gap-3">
          <div class="p-1 center rounded-full bg-white">
            <IconFluentBookNumber20Filled class="text-xl color-link"/>
          </div>
          <div
              @click="goDictDetail(store.sdict)"
              class="text-2xl font-bold cursor-pointer">
            {{ store.sdict.name || '当前无正在学习的词典' }}
          </div>
        </div>

        <template v-if="store.sdict.id">
          <div class="mt-4 flex flex-col gap-2">
            <div class="">当前进度：{{ progressTextLeft }}</div>
            <Progress size="large" :percentage="store.currentStudyProgress" :show-text="false"></Progress>
            <div class="text-sm flex justify-between">
              <span>已完成 {{ progressTextRight }} 词 / 共 {{ store.sdict.words.length }} 词</span>
              <span v-if="store.sdict.id">
              预计完成日期：{{ _getAccomplishDate(store.sdict.words.length, store.sdict.perDayStudyNumber) }}
            </span>
            </div>
          </div>
          <div class="flex items-center mt-4 gap-4">
            <BaseButton type="info"
                        size="small"
                        @click="router.push('/dict-list')">
              <div class="center gap-1">
                <IconFluentArrowSwap20Regular/>
                <span>选择词典</span>
              </div>
            </BaseButton>
            <PopConfirm
                :disabled="!isSaveData"
                title="当前存在未完成的学习任务，修改会重新生成学习任务，是否继续？"
                @confirm="check(()=>showChangeLastPracticeIndexDialog = true)">
              <BaseButton type="info"
                          size="small"
                          v-if="store.sdict.id"
              >
                <div class="center gap-1">
                  <IconFluentSlideTextTitleEdit20Regular/>
                  <span>更改进度</span>
                </div>
              </BaseButton>
            </PopConfirm>
          </div>
        </template>

        <div class="flex items-center gap-4 mt-2 flex-1" v-else>
          <div class="title">请选择一本词典开始学习</div>
          <BaseButton id="step1" type="primary" size="large" @click="router.push('/dict-list')">
            <div class="center gap-1">
              <IconFluentAdd16Regular/>
              <span>选择词典</span>
            </div>
          </BaseButton>
        </div>
      </div>

      <div class="flex-1 w-full mt-4 md:mt-0" :class="!store.sdict.id && 'opacity-30 cursor-not-allowed'">
        <div class="flex justify-between">
          <div class="flex items-center gap-2">
            <div class="p-2 center rounded-full bg-white ">
              <IconFluentStar20Filled class="text-lg color-amber"/>
            </div>
            <div class="text-xl font-bold">
              {{ isSaveData ? '上次任务' : '今日任务' }}
            </div>
            <span class="color-link cursor-pointer"
                  v-if="store.sdict.id"
                  @click="showPracticeWordListDialog = true">词表</span>

          </div>
          <div class="flex gap-1 items-center"
               v-if="store.sdict.id"
          >
            每日目标
            <div style="color:#ac6ed1;"
                 class="bg-third px-2 h-10 flex center text-2xl rounded">
              {{ store.sdict.id ? store.sdict.perDayStudyNumber : 0 }}
            </div>
            个单词
            <PopConfirm
                :disabled="!isSaveData"
                title="当前存在未完成的学习任务，修改会重新生成学习任务，是否继续？"
                @confirm="check(()=>showPracticeSettingDialog = true)">
              <BaseButton
                  type="info" size="small">更改
              </BaseButton>
            </PopConfirm>
          </div>
        </div>
        <div class="flex mt-4 justify-between">
          <div class="stat">
            <div class="num">{{ currentStudy.new.length }}</div>
            <div class="txt">新词数</div>
          </div>
          <template v-if="settingStore.wordPracticeMode === WordPracticeMode.System">
            <div class="stat">
              <div class="num">{{ currentStudy.review.length }}</div>
              <div class="txt">复习上次</div>
            </div>
            <div class="stat">
              <div class="num">{{ currentStudy.write.length }}</div>
              <div class="txt">复习之前</div>
            </div>
          </template>
        </div>
        <div class="flex items-end mt-4">
          <BaseButton size="large"
                      class="flex-1"
                      :disabled="!store.sdict.id"
                      :loading="loading"
                      @click="startPractice">
            <div class="flex items-center gap-2">
              <span class="line-height-[2]">{{ isSaveData ? '继续学习' : '开始学习' }}</span>
              <IconFluentArrowCircleRight16Regular class="text-xl"/>
            </div>
          </BaseButton>

          <div
              v-if="false"
              class="w-full flex box-border  cp  color-white">
            <div
                @click="startPractice"
                class="flex-1 rounded-l-lg center gap-2 py-1 bg-[var(--btn-primary)]  hover:opacity-50">
              <span class="line-height-[2]">{{ isSaveData ? '继续学习' : '开始学习' }}</span>
              <IconFluentArrowCircleRight16Regular class="text-xl"/>
            </div>

            <div class="relative group">
              <div
                  class="w-10 rounded-r-lg h-full center bg-[var(--btn-primary)] hover:bg-gray border-solid border-2 border-l-gray border-transparent box-border">
                <IconFluentChevronDown20Regular/>
              </div>

              <div
                  class="space-y-2 pt-2 absolute z-2 right-0 border rounded  opacity-0 scale-95
           group-hover:opacity-100 group-hover:scale-100
           transition-all duration-150 pointer-events-none group-hover:pointer-events-auto"
              >
                <div>
                  <BaseButton
                      size="large" type="orange"
                      :loading="loading"
                      @click="check(()=>showShufflePracticeSettingDialog = true)">
                    <div class="flex items-center gap-2">
                      <span class="line-height-[2]">随机复习</span>
                      <IconFluentArrowShuffle20Filled class="text-xl"/>
                    </div>
                  </BaseButton>
                </div>
                <div>
                  <BaseButton
                      size="large" type="orange"
                      :loading="loading"
                      @click="check(()=>showShufflePracticeSettingDialog = true)">
                    <div class="flex items-center gap-2">
                      <span class="line-height-[2]">重新学习</span>
                      <IconFluentArrowShuffle20Filled class="text-xl"/>
                    </div>
                  </BaseButton>
                </div>
              </div>
            </div>
          </div>

          <BaseButton
              v-if="store.sdict.id && store.sdict.lastLearnIndex"
              size="large" type="orange"
              :loading="loading"
              @click="check(()=>showShufflePracticeSettingDialog = true)">
            <div class="flex items-center gap-2">
              <span class="line-height-[2]">随机复习</span>
              <IconFluentArrowShuffle20Filled class="text-xl"/>
            </div>
          </BaseButton>
        </div>
      </div>
    </div>

    <div class="card flex flex-col">
      <div class="flex justify-between">
        <div class="title">我的词典</div>
        <div class="flex gap-4 items-center">
          <PopConfirm title="确认删除所有选中词典？" @confirm="handleBatchDel" v-if="selectIds.length">
            <BaseIcon class="del" title="删除">
              <DeleteIcon/>
            </BaseIcon>
          </PopConfirm>

          <div class="color-link cursor-pointer" v-if="store.word.bookList.length > 3"
               @click="isManageDict = !isManageDict; selectIds = []">{{ isManageDict ? '取消' : '管理词典' }}
          </div>
          <div class="color-link cursor-pointer" @click="nav('dict-detail', { isAdd: true })">创建个人词典</div>
        </div>
      </div>
      <div class="flex gap-4 flex-wrap  mt-4">
        <Book :is-add="false"
              quantifier="个词"
              :item="item"
              :checked="selectIds.includes(item.id)"
              @check="() => toggleSelect(item)"
              :show-checkbox="isManageDict && j >= 3"
              v-for="(item, j) in store.word.bookList"
              @click="goDictDetail(item)"/>
        <Book :is-add="true" @click="router.push('/dict-list')"/>
      </div>
    </div>

    <div class="card flex flex-col overflow-hidden" v-loading="isFetching">
      <div class="flex justify-between">
        <div class="title">推荐</div>
        <div class="flex gap-4 items-center">
          <div class="color-link cursor-pointer" @click="router.push('/dict-list')">更多</div>
        </div>
      </div>

      <div class="flex gap-4 flex-wrap  mt-4 min-h-50">
        <Book :is-add="false"
              quantifier="个词"
              :item="item as any"
              v-for="(item, j) in recommendDictList" @click="goDictDetail(item as any)"/>
      </div>
    </div>
  </BasePage>

  <PracticeSettingDialog
      :show-left-option="false"
      v-model="showPracticeSettingDialog"
      @ok="savePracticeSetting"/>

  <ChangeLastPracticeIndexDialog
      v-model="showChangeLastPracticeIndexDialog"
      @ok="saveLastPracticeIndex"
  />

  <PracticeWordListDialog
      :data="currentStudy"
      v-model="showPracticeWordListDialog"
  />

  <ShufflePracticeSettingDialog
      v-model="showShufflePracticeSettingDialog"
      @ok="onShufflePracticeSettingOk"/>

</template>

<style scoped lang="scss">
.stat {
  @apply w-31% box-border flex flex-col items-center justify-center rounded-xl p-2 bg-[var(--bg-history)];
  border: 1px solid gainsboro;

  .num {
    @apply color-[#409eff] text-4xl font-bold;
  }

  .txt {
    @apply color-gray-500;
  }
}
</style>
