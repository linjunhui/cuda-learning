<script setup lang="ts">
import {onMounted, onUnmounted, provide, ref, watch} from "vue";

import Statistics from "@/pages/word/Statistics.vue";
import { emitter, EventKey, useEvents } from "@/utils/eventBus.ts";
import { useSettingStore } from "@/stores/setting.ts";
import { useRuntimeStore } from "@/stores/runtime.ts";
import { Dict, PracticeData, ShortcutKey, TaskWords, Word, WordPracticeMode, WordPracticeType } from "@/types/types.ts";
import { useDisableEventListener, useOnKeyboardEventListener, useStartKeyboardEventListener } from "@/hooks/event.ts";
import useTheme from "@/hooks/theme.ts";
import { getCurrentStudyWord, useWordOptions } from "@/hooks/dict.ts";
import { _getDictDataByUrl, _nextTick, cloneDeep, isMobile, loadJsLib, resourceWrap, shuffle } from "@/utils";
import { useRoute, useRouter } from "vue-router";
import Footer from "@/pages/word/components/Footer.vue";
import Panel from "@/components/Panel.vue";
import BaseIcon from "@/components/BaseIcon.vue";
import Tooltip from "@/components/base/Tooltip.vue";
import WordList from "@/components/list/WordList.vue";
import TypeWord from "@/pages/word/components/TypeWord.vue";
import Empty from "@/components/Empty.vue";
import { useBaseStore } from "@/stores/base.ts";
import { usePracticeStore } from "@/stores/practice.ts";
import Toast from '@/components/base/toast/Toast.ts'
import { getDefaultDict, getDefaultWord } from "@/types/func.ts";
import ConflictNotice from "@/components/ConflictNotice.vue";
import PracticeLayout from "@/components/PracticeLayout.vue";

import { DICT_LIST, LIB_JS_URL, PracticeSaveWordKey, TourConfig } from "@/config/env.ts";
import { ToastInstance } from "@/components/base/toast/type.ts";
import { watchOnce } from "@vueuse/core";

const {
  isWordCollect,
  toggleWordCollect,
  isWordSimple,
  toggleWordSimple
} = useWordOptions()
const settingStore = useSettingStore()
const runtimeStore = useRuntimeStore()
const {toggleTheme} = useTheme()
const router = useRouter()
const route = useRoute()
const store = useBaseStore()
const statStore = usePracticeStore()
const typingRef: any = $ref()
let showConflictNotice = $ref(false)
let allWrongWords = new Set()
let showStatDialog = $ref(false)
let loading = $ref(false)
let timer = $ref(0)
let isFocus = true
let taskWords = $ref<TaskWords>({
  new: [],
  review: [],
  write: [],
  shuffle: [],
})

let data = $ref<PracticeData>({
  index: 0,
  words: [],
  wrongWords: [],
  excludeWords: [],
})
let isTypingWrongWord = ref(false)

provide('isTypingWrongWord', isTypingWrongWord)
provide('practiceData', data)
provide('practiceTaskWords', taskWords)

async function loadDict() {
  // console.log('load好了开始加载')
  let dict = getDefaultDict()
  let dictId = route.params.id
  if (dictId) {
    //先在自己的词典列表里面找，如果没有再在资源列表里面找
    dict = store.word.bookList.find(v => v.id === dictId)
    let r = await fetch(resourceWrap(DICT_LIST.WORD.ALL))
    let dict_list = await r.json()
    if (!dict) dict = dict_list.flat().find(v => v.id === dictId) as Dict
    if (dict && dict.id) {
      //如果是不是自定义词典，就请求数据
      if (!dict.custom) dict = await _getDictDataByUrl(dict)
      if (!dict.words.length) {
        router.push('/word')
        return Toast.warning('没有单词可学习！')
      }
      store.changeDict(dict)
      initData(getCurrentStudyWord(), true)
      loading = false
    } else {
      router.push('/word')
    }
  } else {
    router.push('/word')
  }
}

watch(() => store.load, (n) => {
  if (n && loading) loadDict()
}, {immediate: true})

onMounted(() => {
  //如果是从单词学习主页过来的，就直接使用；否则等待加载
  if (runtimeStore.routeData) {
    initData(runtimeStore.routeData.taskWords, true)
  } else {
    loading = true
  }
  if (route.query.guide) {
    showConflictNotice = false
  } else {
    showConflictNotice = true
  }
  document.addEventListener('visibilitychange', () => {
    isFocus = !document.hidden
  })
})

onUnmounted(() => {
  timer && clearInterval(timer)
})

watchOnce(() => data.words.length, (newVal, oldVal) => {
  //如果是从无值变有值，代表是开始
  if (!oldVal && newVal) {
    _nextTick(async () => {
      const Shepherd = await loadJsLib('Shepherd', LIB_JS_URL.SHEPHERD);
      const tour = new Shepherd.Tour(TourConfig);
      tour.on('cancel', () => {
        localStorage.setItem('tour-guide', '1');
      });
      tour.addStep({
        id: 'step5',
        text: '这里可以练习拼写单词，只需要按下键盘上对应的按键即可，没有输入框！',
        attachTo: {element: '#word', on: 'bottom'},
        buttons: [
          {
            text: `下一步（5/${TourConfig.total}）`,
            action: tour.next
          }
        ]
      });

      tour.addStep({
        id: 'step6',
        text: '这里是文章练习',
        attachTo: {element: '#article', on: 'top'},
        buttons: [
          {
            text: `下一步（6/${TourConfig.total}）`,
            action() {
              tour.next()
              router.push('/articles')
            }
          }
        ]
      });

      const r = localStorage.getItem('tour-guide');
      if (settingStore.first && !r && !isMobile()) {
        tour.start();
      }
    }, 500)
  }
})

useStartKeyboardEventListener()
useDisableEventListener(() => loading)

function initData(initVal: TaskWords, init: boolean = false) {
  let d = localStorage.getItem(PracticeSaveWordKey.key)
  if (d && init) {
    try {
      let obj = JSON.parse(d)
      let s = obj.val
      taskWords = Object.assign(taskWords, s.taskWords)
      //这里直接赋值的话，provide后的inject获取不到最新值
      data = Object.assign(data, s.practiceData)
      statStore.$patch(s.statStoreData)
    } catch (e) {
      localStorage.removeItem(PracticeSaveWordKey.key)
      initData(initVal, true)
    }
  } else {
    // taskWords = initVal
    //不能直接赋值，会导致 inject 的数据为默认值
    taskWords = Object.assign(taskWords, initVal)
    //如果 shuffle 数组不为空，就说明是复习
    if (taskWords.shuffle.length === 0) {
      if (taskWords.new.length === 0) {
        if (taskWords.review.length) {
          settingStore.wordPracticeType = WordPracticeType.Identify
          statStore.step = 3
          data.words = taskWords.review
        } else {
          if (taskWords.write.length) {
            settingStore.wordPracticeType = WordPracticeType.Identify
            data.words = taskWords.write
            statStore.step = 6
          } else {
            Toast.warning('没有可学习的单词！')
            router.push('/word')
          }
        }
      } else {
        settingStore.wordPracticeType = WordPracticeType.FollowWrite
        data.words = taskWords.new
        statStore.step = 0
      }
      statStore.total = taskWords.review.length + taskWords.new.length + taskWords.write.length
      statStore.newWordNumber = taskWords.new.length
      statStore.reviewWordNumber = taskWords.review.length
      statStore.writeWordNumber = taskWords.write.length
    } else {
      settingStore.wordPracticeType = WordPracticeType.Dictation
      data.words = taskWords.shuffle
      statStore.step = 10
      statStore.total = taskWords.shuffle.length
      statStore.newWordNumber = 0
      statStore.reviewWordNumber = 0
      statStore.writeWordNumber = statStore.total
    }

    data.index = 0
    data.wrongWords = []
    data.excludeWords = []
    allWrongWords.clear()
    statStore.startDate = Date.now()
    statStore.inputWordNumber = 0
    statStore.wrong = 0
    statStore.spend = 0
    isTypingWrongWord.value = false
  }
  clearInterval(timer)
  timer = setInterval(() => {
    if (isFocus) {
      statStore.spend += 1000
      savePracticeData()
    }
  }, 1000)

}

const word = $computed<Word>(() => {
  return data.words[data.index] ?? getDefaultWord()
})
const prevWord: Word = $computed(() => {
  return data.words?.[data.index - 1] ?? undefined
})
const nextWord: Word = $computed(() => {
  return data.words?.[data.index + 1] ?? undefined
})

watch(() => settingStore.wordPracticeType, (n) => {
  if (settingStore.wordPracticeMode === WordPracticeMode.Free) return
  switch (n) {
    case WordPracticeType.Spell:
    case WordPracticeType.Dictation:
      settingStore.dictation = true;
      settingStore.translate = true;
      break
    case WordPracticeType.Listen:
      settingStore.dictation = true;
      settingStore.translate = false;
      break
    case WordPracticeType.FollowWrite:
      settingStore.dictation = false;
      settingStore.translate = true;
      break
    case WordPracticeType.Identify:
      settingStore.dictation = false;
      settingStore.translate = false;
      break
  }
}, {immediate: true})

const groupSize = 7

function wordLoop() {
  // 学习模式
  if (settingStore.wordPracticeType === WordPracticeType.FollowWrite) {
    data.index++
    // 到达一个组末尾，就切换到拼写模式
    if (data.index % groupSize === 0) {
      settingStore.wordPracticeType = WordPracticeType.Spell
      data.index -= groupSize // 回到刚学单词开头
    }
  } else {
    // 拼写模式
    data.index++
    // 拼写走完一组，切回跟写模式
    if (data.index % groupSize === 0) {
      settingStore.wordPracticeType = WordPracticeType.FollowWrite
    }
  }
}

let toastInstance: ToastInstance = null

function goNextStep(originList, mode, msg) {
  //每次都判断，因为每次都可能新增已掌握的单词
  let list = originList.filter(v => (!data.excludeWords.includes(v.word)))
  console.log(msg)
  if (list.length) {
    if (toastInstance) toastInstance.close()
    toastInstance = Toast.info('输入完成后按空格键切换下一个', {duration: 5000})
    data.words = list
    settingStore.wordPracticeType = mode
    data.index = 0
    statStore.step++
  } else {
    console.log(msg + ':无单词略过')
    statStore.step += 3
    next()
  }
}

async function next(isTyping: boolean = true) {
  if (isTyping) statStore.inputWordNumber++
  if (settingStore.wordPracticeMode === WordPracticeMode.Free) {
    if (data.index === data.words.length - 1) {
      data.wrongWords = data.wrongWords.filter(v => (!data.excludeWords.includes(v.word)))
      if (data.wrongWords.length) {
        isTypingWrongWord.value = true
        settingStore.wordPracticeType = WordPracticeType.FollowWrite
        console.log('当前学完了，但还有错词')
        data.words = shuffle(cloneDeep(data.wrongWords))
        data.index = 0
        data.wrongWords = []
      } else {
        console.log('自由模式，全完学完了')
        showStatDialog = true
        setTimeout(() => localStorage.removeItem(PracticeSaveWordKey.key), 300)
      }
    } else {
      data.index++
    }
  } else {
    if (data.index === data.words.length - 1) {
      if (statStore.step === 0 || isTypingWrongWord.value) {
        if (settingStore.wordPracticeType !== WordPracticeType.Spell) {
          //回到最后一组的开始位置
          data.index = Math.floor(data.index / groupSize) * groupSize
          emitter.emit(EventKey.resetWord)
          settingStore.wordPracticeType = WordPracticeType.Spell
          return
        }
      }
      data.wrongWords = data.wrongWords.filter(v => (!data.excludeWords.includes(v.word)))
      if (data.wrongWords.length) {
        isTypingWrongWord.value = true
        settingStore.wordPracticeType = WordPracticeType.FollowWrite
        console.log('当前学完了，但还有错词')
        data.words = shuffle(cloneDeep(data.wrongWords))
        data.index = 0
        data.wrongWords = []
      } else {
        isTypingWrongWord.value = false
        console.log('当前学完了，没错词', statStore.total, statStore.step, data.index)
        //学完了，这里第 7 步如果无单词，加 3 就是 9 了
        if (statStore.step >= 8) {
          statStore.spend = Date.now() - statStore.startDate
          console.log('全完学完了')
          showStatDialog = true
          setTimeout(() => localStorage.removeItem(PracticeSaveWordKey.key), 300)
          return;
        }

        //开始默写之前
        if (statStore.step === 7) {
          return goNextStep(shuffle(taskWords.write), WordPracticeType.Dictation, '开始默写之前')
        }

        //开始听写之前
        if (statStore.step === 6) {
          return goNextStep(shuffle(taskWords.write), WordPracticeType.Listen, '开始听写之前')
        }

        //开始自测之前
        if (statStore.step === 5) {
          return goNextStep(taskWords.write, WordPracticeType.Identify, '开始自测之前')
        }

        //开始默写上次
        if (statStore.step === 4) {
          return goNextStep(shuffle(taskWords.review), WordPracticeType.Dictation, '开始默写上次')
        }

        //开始听写上次
        if (statStore.step === 3) {
          return goNextStep(shuffle(taskWords.review), WordPracticeType.Listen, '开始听写上次')
        }

        //开始自测昨日
        if (statStore.step === 2) {
          return goNextStep(taskWords.review, WordPracticeType.Identify, '开始自测昨日')
        }

        //开始默写新词
        if (statStore.step === 1) {
          return goNextStep(shuffle(taskWords.new), WordPracticeType.Dictation, '开始默写新词')
        }

        //开始听写新词
        if (statStore.step === 0) {
          return goNextStep(shuffle(taskWords.new), WordPracticeType.Listen, '开始听写新词')
        }
      }
    } else {
      if (statStore.step === 0) {
        wordLoop()
      } else {
        if (isTypingWrongWord.value) wordLoop()
        else data.index++
      }
    }
  }
  savePracticeData()
}

function skipStep() {
  data.index = data.words.length - 1
  settingStore.wordPracticeType = WordPracticeType.Spell
  data.wrongWords = []
  next(false)
}

function onWordKnow() {
  //标记模式时，用户认识的单词加入到排除里面，后续不再复习
  let rIndex = data.excludeWords.findIndex(v => v === word.word)
  if (rIndex < 0) {
    data.excludeWords.push(word.word)
  }
}

function onTypeWrong() {
  let temp = word.word.toLowerCase()
  if (!allWrongWords.has(word.word.toLowerCase())) {
    allWrongWords.add(word.word.toLowerCase())
    statStore.wrong++
  }
  if (!store.wrong.words.find((v: Word) => v.word.toLowerCase() === temp)) {
    store.wrong.words.push(word)
    store.wrong.length = store.wrong.words.length
  }
  if (!data.wrongWords.find((v: Word) => v.word.toLowerCase() === temp)) {
    data.wrongWords.push(word)
  }
  savePracticeData()
}

function savePracticeData() {
  localStorage.setItem(PracticeSaveWordKey.key, JSON.stringify({
    version: PracticeSaveWordKey.version,
    val: {
      taskWords,
      practiceData: data,
      statStoreData: statStore.$state,
    }
  }))
}

watch(() => data.index, savePracticeData)

function onKeyUp(e: KeyboardEvent) {
  // console.log('onKeyUp', e)
  typingRef.hideWord()
}

function onKeyDown(e: KeyboardEvent) {
  // console.log('onKeyDown', e)
  switch (e.key) {
    case 'Backspace':
      typingRef.del()
      break
  }
}

useOnKeyboardEventListener(onKeyDown, onKeyUp)

function repeat() {
  console.log('重学一遍')
  let temp = cloneDeep(taskWords)
  let ignoreList = [store.allIgnoreWords, store.knownWords][settingStore.ignoreSimpleWord ? 0 : 1]
  //随机练习单独处理
  if (taskWords.shuffle.length) {
    temp.shuffle = shuffle(temp.shuffle.filter(v => !ignoreList.includes(v.word)))
  } else {
    if (settingStore.wordPracticeMode === WordPracticeMode.System) settingStore.dictation = false
    if (store.sdict.lastLearnIndex === 0 && store.sdict.complete) {
      //如果是刚刚完成，那么学习进度要从length减回去，因为lastLearnIndex为0了，同时改complete为false
      store.sdict.lastLearnIndex = store.sdict.length - statStore.newWordNumber
      store.sdict.complete = false
    } else {
      //将学习进度减回去
      store.sdict.lastLearnIndex = store.sdict.lastLearnIndex - statStore.newWordNumber
    }
    //排除已掌握单词
    temp.new = temp.new.filter(v => !ignoreList.includes(v.word))
    temp.review = temp.review.filter(v => !ignoreList.includes(v.word))
    temp.write = temp.write.filter(v => !ignoreList.includes(v.word))
  }
  emitter.emit(EventKey.resetWord)
  initData(temp)
}

function prev() {
  if (data.index === 0) {
    Toast.warning('已经是第一个了~')
  } else {
    data.index--
  }
}

function skip(e: KeyboardEvent) {
  next(false)
  // e.preventDefault()
}

function show(e: KeyboardEvent) {
  if (![WordPracticeType.FollowWrite].includes(settingStore.wordPracticeType)) onTypeWrong()
  typingRef.showWord()
}

function collect(e: KeyboardEvent) {
  toggleWordCollect(word)
}

function play() {
  typingRef.play()
}

function toggleWordSimpleWrapper() {
  if (!isWordSimple(word)) {
    //延迟一下，不知道为什么不延迟会导致当前条目不自动定位到列表中间
    setTimeout(() => next(false))
  }
  let rIndex = data.excludeWords.findIndex(v => v === word.word)
  if (rIndex > -1) {
    data.excludeWords.splice(rIndex, 1)
  } else {
    data.excludeWords.push(word.word)
  }
  toggleWordSimple(word)
}

function toggleTranslate() {
  settingStore.translate = !settingStore.translate
}

function toggleDictation() {
  settingStore.dictation = !settingStore.dictation
}

function toggleConciseMode() {
  settingStore.showToolbar = !settingStore.showToolbar
  settingStore.showPanel = settingStore.showToolbar
}

function togglePanel() {
  settingStore.showPanel = !settingStore.showPanel
}

function continueStudy() {
  let temp = cloneDeep(taskWords)
  //随机练习单独处理
  if (taskWords.shuffle.length) {
    let ignoreList = [store.allIgnoreWords, store.knownWords][settingStore.ignoreSimpleWord ? 0 : 1]
    temp.shuffle = shuffle(store.sdict.words.filter(v => !ignoreList.includes(v.word))).slice(0, runtimeStore.routeData.total)
    if (showStatDialog) showStatDialog = false
  } else {
    if (settingStore.wordPracticeMode === WordPracticeMode.System) settingStore.dictation = false
    //这里判断是否显示结算弹框，如果显示了结算弹框的话，就不用加进度了
    if (!showStatDialog) {
      console.log('没学完，强行跳过')
      store.sdict.lastLearnIndex = store.sdict.lastLearnIndex + statStore.newWordNumber
    } else {
      console.log('学完了，正常下一组')
      showStatDialog = false
    }
    temp = getCurrentStudyWord()
  }
  emitter.emit(EventKey.resetWord)
  initData(temp)
}

function randomWrite() {
  console.log('随机默写')
  data.words = shuffle(data.words);
  data.index = 0
  settingStore.dictation = true
}

function nextRandomWrite() {
  console.log('继续随机默写')
  initData(getCurrentStudyWord())
  randomWrite();
  showStatDialog = false
}

useEvents([
  [EventKey.repeatStudy, repeat],
  [EventKey.continueStudy, continueStudy],
  [EventKey.randomWrite, nextRandomWrite],
  [EventKey.changeDict, () => {
    initData(getCurrentStudyWord())
  }],

  [ShortcutKey.ShowWord, show],
  [ShortcutKey.Previous, prev],
  [ShortcutKey.Next, skip],
  [ShortcutKey.ToggleCollect, collect],
  [ShortcutKey.ToggleSimple, toggleWordSimpleWrapper],
  [ShortcutKey.PlayWordPronunciation, play],

  [ShortcutKey.RepeatChapter, repeat],
  [ShortcutKey.NextChapter, continueStudy],
  [ShortcutKey.ToggleShowTranslate, toggleTranslate],
  [ShortcutKey.ToggleDictation, toggleDictation],
  [ShortcutKey.ToggleTheme, toggleTheme],
  [ShortcutKey.ToggleConciseMode, toggleConciseMode],
  [ShortcutKey.TogglePanel, togglePanel],
  [ShortcutKey.RandomWrite, randomWrite],
  [ShortcutKey.NextRandomWrite, nextRandomWrite],
])

</script>

<template>
  <PracticeLayout
      v-loading="loading"
      panelLeft="var(--word-panel-margin-left)">
    <template v-slot:practice>
      <div class="practice-word">
        <div class="absolute z-1 top-4   w-full" v-if="settingStore.showNearWord">
          <div class="center gap-2 cursor-pointer float-left"
               @click="prev"
               v-if="prevWord">
            <IconFluentArrowLeft16Regular class="arrow" width="22"/>
            <Tooltip
                :title="`上一个(${settingStore.shortcutKeyMap[ShortcutKey.Previous]})`"
            >
              <div class="word">{{ prevWord.word }}</div>
            </Tooltip>
          </div>
          <div class="center gap-2 cursor-pointer float-right mr-3"
               @click="next(false)"
               v-if="nextWord">
            <Tooltip
                :title="`下一个(${settingStore.shortcutKeyMap[ShortcutKey.Next]})`"
            >
              <div class="word" :class="settingStore.dictation && 'word-shadow'">{{ nextWord.word }}</div>
            </Tooltip>
            <IconFluentArrowRight16Regular class="arrow" width="22"/>
          </div>
        </div>
        <TypeWord
            ref="typingRef"
            :word="word"
            @wrong="onTypeWrong"
            @complete="next"
            @know="onWordKnow"
        />
      </div>
    </template>
    <template v-slot:panel>
      <Panel>
        <template v-slot:title>
          <!--          <span>{{ store.sdict.name }} ({{ data.index + 1 }} / {{ data.words.length }})</span>-->
          <div class="center gap-space">
            <span>{{ store.sdict.name }} ({{ store.sdict.lastLearnIndex }} / {{ store.sdict.length }})</span>

            <BaseIcon
                @click="continueStudy"
                :title="`下一组(${settingStore.shortcutKeyMap[ShortcutKey.NextChapter]})`">
              <IconFluentArrowRight16Regular class="arrow" width="22"/>
            </BaseIcon>
            <BaseIcon
                @click="randomWrite"
                :title="`随机默写(${settingStore.shortcutKeyMap[ShortcutKey.RandomWrite]})`">
              <IconFluentArrowShuffle16Regular class="arrow" width="22"/>
            </BaseIcon>
          </div>
        </template>
        <div class="panel-page-item pl-4">
          <WordList
              v-if="data.words.length"
              :is-active="settingStore.showPanel"
              :static="false"
              :show-word="!settingStore.dictation"
              :show-translate="settingStore.translate"
              :list="data.words"
              :activeIndex="data.index"
              @click="(val:any) => data.index = val.index"
          >
            <template v-slot:suffix="{item,index}">
              <BaseIcon
                  :class="!isWordCollect(item)?'collect':'fill'"
                  @click.stop="toggleWordCollect(item)"
                  :title="!isWordCollect(item) ? '收藏' : '取消收藏'">
                <IconFluentStar16Regular v-if="!isWordCollect(item)"/>
                <IconFluentStar16Filled v-else/>
              </BaseIcon>

              <BaseIcon
                  :class="!isWordSimple(item)?'collect':'fill'"
                  @click.stop="toggleWordSimple(item)"
                  :title="!isWordSimple(item) ? '标记为已掌握' : '取消标记已掌握'">
                <IconFluentCheckmarkCircle16Regular v-if="!isWordSimple(item)"/>
                <IconFluentCheckmarkCircle16Filled v-else/>
              </BaseIcon>
            </template>
          </WordList>
          <Empty v-else/>
        </div>
      </Panel>
    </template>
    <template v-slot:footer>
      <Footer
          :is-simple="isWordSimple(word)"
          @toggle-simple="toggleWordSimpleWrapper"
          :is-collect="isWordCollect(word)"
          @toggle-collect="toggleWordCollect(word)"
          @skip="next(false)"
          @skipStep="skipStep"
      />
    </template>
  </PracticeLayout>
  <Statistics v-model="showStatDialog"/>
  <ConflictNotice v-if="showConflictNotice"/>
</template>

<style scoped lang="scss">

.practice-wrapper {
  width: 100%;
  height: 100vh;
  display: flex;
  justify-content: center;
  overflow: hidden;
}

.practice-word {
  height: 100%;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  align-items: center;
  position: relative;
  width: var(--toolbar-width);
}

// 移动端适配
@media (max-width: 768px) {
  .practice-word {
    width: 100%;

    .absolute.z-1.top-4 {
      z-index: 100; // 提高层级，确保不被遮挡

      .center.gap-2.cursor-pointer {
        min-height: 44px;
        min-width: 44px;
        padding: 0.5rem;
        display: flex;
        align-items: center;
        justify-content: center;

        .word {
          pointer-events: none; // 文字不拦截点击
        }

        .arrow {
          pointer-events: none; // 箭头图标不拦截点击
        }
      }
    }
  }
}

.word-panel-wrapper {
  position: absolute;
  left: var(--panel-margin-left);
  //left: 0;
  top: .8rem;
  z-index: 1;
  height: calc(100% - 1.5rem);
}
</style>
