<script setup lang="ts">
import {inject, onMounted, onUnmounted, watch} from "vue"
import {Article, ArticleWord, PracticeArticleWordType, Sentence, ShortcutKey, Word} from "@/types/types.ts";
import {useBaseStore} from "@/stores/base.ts";
import {useSettingStore} from "@/stores/setting.ts";
import {usePlayBeep, usePlayKeyboardAudio, usePlayWordAudio} from "@/hooks/sound.ts";
import {emitter, EventKey, useEvents} from "@/utils/eventBus.ts";
import {useEventListener} from "@/hooks/event.ts";
import {_dateFormat, _nextTick, isMobile, msToHourMinute, total} from "@/utils";
import '@imengyu/vue3-context-menu/lib/vue3-context-menu.css'
import ContextMenu from '@imengyu/vue3-context-menu'
import BaseButton from "@/components/BaseButton.vue";
import QuestionForm from "@/pages/article/components/QuestionForm.vue";
import {getDefaultArticle, getDefaultWord} from "@/types/func.ts";
import Toast from '@/components/base/toast/Toast.ts'
import TypingWord from "@/pages/article/components/TypingWord.vue";
import Space from "@/pages/article/components/Space.vue";
import {useWordOptions} from "@/hooks/dict.ts";
import nlp from "compromise/three";
import {nanoid} from "nanoid";
import {usePracticeStore} from "@/stores/practice.ts";
import {PracticeSaveArticleKey} from "@/config/env.ts";
import WordTranslationDialog from "@/components/WordTranslationDialog.vue";

interface IProps {
  article: Article,
  sectionIndex?: number,
  sentenceIndex?: number,
  wordIndex?: number,
  stringIndex?: number,
}

const props = withDefaults(defineProps<IProps>(), {
  article: () => getDefaultArticle(),
  sectionIndex: 0,
  sentenceIndex: 0,
  wordIndex: 0,
  stringIndex: 0,
})

const emit = defineEmits<{
  ignore: [],
  wrong: [val: Word],
  play: [val: {
    sentence: Sentence,
    handle: boolean
  }],
  nextWord: [val: ArticleWord],
  complete: [],
  next: [],
  replay: [],
}>()

let typeArticleRef = $ref<HTMLInputElement>()
let mobileInputRef = $ref<HTMLInputElement>()
let articleWrapperRef = $ref<HTMLInputElement>(null)
let sectionIndex = $ref(0)
let sentenceIndex = $ref(0)
let wordIndex = $ref(0)
let stringIndex = $ref(0)
let input = $ref('')
let wrong = $ref('')
//是否是输入空格
let isSpace = $ref(false)
let isEnd = $ref(false)
let hoverIndex = $ref({
  sectionIndex: -1,
  sentenceIndex: -1,
  wordIndex: -1,
})
let cursor = $ref({
  top: 0,
  left: 0,
})

const currentIndex = $computed(() => {
  return `${sectionIndex}${sentenceIndex}${wordIndex}`
})

const playBeep = usePlayBeep()
const playKeyboardAudio = usePlayKeyboardAudio()
const playWordAudio = usePlayWordAudio()

const {
  toggleWordCollect,
} = useWordOptions()

const store = useBaseStore()
const settingStore = useSettingStore()
const statStore = usePracticeStore()
const isMob = isMobile()

watch([() => sectionIndex, () => sentenceIndex, () => wordIndex, () => stringIndex], ([a, b, c,]) => {
  localStorage.setItem(PracticeSaveArticleKey.key, JSON.stringify({
    version: PracticeSaveArticleKey.version,
    val: {
      practiceData: {
        sectionIndex,
        sentenceIndex,
        wordIndex,
        stringIndex,
        id: props.article.id
      },
      statStoreData: statStore.$state,
    }
  }))
  checkCursorPosition(a, b, c)
})

// watch(() => props.article.id, init, {immediate: true})

watch(() => settingStore.translate, () => {
  checkTranslateLocation().then(() => checkCursorPosition())
})

watch(() => isEnd, n => {
  if (n) {
    _nextTick(() => {
      typeArticleRef?.scrollTo({top: typeArticleRef.scrollHeight, behavior: "smooth"})
    })
  } else {
    typeArticleRef?.scrollTo({top: 0, behavior: "smooth"})
  }
})

function init() {
  if (!props.article.id) return
  isSpace = isEnd = false
  let d = localStorage.getItem(PracticeSaveArticleKey.key)
  if (d) {
    try {
      let obj = JSON.parse(d)
      let data = obj.val
      statStore.$patch(data.statStoreData)
      jump(data.practiceData.sectionIndex, data.practiceData.sentenceIndex, data.practiceData.wordIndex)
    } catch (e) {
      localStorage.removeItem(PracticeSaveArticleKey.key)
      init()
    }
  } else {
    wrong = input = ''
    sectionIndex = 0
    sentenceIndex = 0
    wordIndex = 0
    stringIndex = 0
    //todo 这在直接修改不太合理
    props.article.sections.map((v) => {
      v.map((w) => {
        w.words.map(s => {
          s.input = ''
        })
      })
    })
    typeArticleRef?.scrollTo({top: 0, behavior: "smooth"})
  }
  _nextTick(() => {
    emit('play', {sentence: props.article.sections[sectionIndex][sentenceIndex], handle: false})
    if (isNameWord()) next()
  })
  checkTranslateLocation().then(() => checkCursorPosition())
  focusMobileInput()
}

function checkCursorPosition(a = sectionIndex, b = sentenceIndex, c = wordIndex) {
  // console.log('checkCursorPosition')
  _nextTick(() => {
    // 选中目标元素
    const currentWord = document.querySelector(`.section:nth-of-type(${a + 1}) .sentence:nth-of-type(${b + 1}) .word:nth-of-type(${c + 1})`);
    if (currentWord) {
      // 在 currentWord 内找 .word-end
      const end = currentWord.querySelector('.word-end');
      if (end) {
        // 获取 articleWrapper 的位置
        const articleRect = articleWrapperRef.getBoundingClientRect();
        const endRect = end.getBoundingClientRect();
        //如果当前输入位置大于屏幕的0.7高度，就滚动屏幕的1/3
        if (endRect.y > window.innerHeight * 0.7) {
          typeArticleRef?.scrollTo({top: typeArticleRef.scrollTop + window.innerHeight * 0.3, behavior: "smooth"})
        }
        // 计算相对位置
        cursor = {
          top: endRect.top - articleRect.top,
          left: endRect.left - articleRect.left,
        };
      }
    }
  },)
}

function checkTranslateLocation() {
  // console.log('checkTranslateLocation')
  return new Promise<void>(resolve => {
    if (isMob) {
      resolve()
      return
    }
    _nextTick(() => {
      let articleRect = articleWrapperRef.getBoundingClientRect()
      props.article.sections.map((v, i) => {
        v.map((w, j) => {
          let location = i + '-' + j
          let wordClassName = `.word${location}`
          let word = document.querySelector(wordClassName)
          let wordRect = word.getBoundingClientRect()
          let translateClassName = `.translate${location}`
          let translate: HTMLDivElement = document.querySelector(translateClassName)

          translate.style.opacity = '1'
          translate.style.top = wordRect.top - articleRect.top + 24 + 'px'
          // @ts-ignore
          translate.firstChild.style.width = wordRect.left - articleRect.left + 'px'
          // console.log(word, wordRect.left - articleRect.left)
          // console.log('word-wordRect', wordRect)
        })
      })
      resolve()
    }, 300)
  })
}

function focusMobileInput() {
  if (!isMob) return
  mobileInputRef?.focus()
}

function processMobileCharacter(char: string) {
  if (!char) return
  const code = char === ' ' ? 'Space' : char === '\n' ? 'Enter' : `Key${char.toUpperCase()}`
  const fakeEvent = {
    key: char,
    code,
    preventDefault() {
    },
    stopPropagation() {
    },
  } as unknown as KeyboardEvent
  onTyping(fakeEvent)
}

function handleMobileInput(event: Event) {
  if (!isMob) return
  const target = event.target as HTMLInputElement
  const value = target?.value ?? ''
  if (!value) return
  for (const char of value) {
    processMobileCharacter(char)
  }
  target.value = ''
}

function handleMobileBeforeInput(event: InputEvent) {
  if (!isMob) return
  if (event.inputType === 'deleteContentBackward') {
    event.preventDefault()
    del()
  }
}


const normalize = (s: string) => s.toLowerCase().trim()
const namePatterns = $computed(() => {
  return Array.from(new Set((props.article?.nameList ?? []).map(normalize).filter(Boolean).map(s => s.split(/\s+/).filter(Boolean)).flat().concat([
    'Mr', 'Mrs', 'Ms', 'Dr', 'Miss',
  ].map(normalize))))
})

const isNameWord = () => {
  let currentSection = props.article.sections[sectionIndex]
  let currentSentence = currentSection[sentenceIndex]
  let w: ArticleWord = currentSentence.words[wordIndex]
  return w?.type === PracticeArticleWordType.Word && namePatterns.length > 0 && namePatterns.includes(normalize(w.word))
}

let isTyping = false
//专用锁，因为这个方法父级要调用
let lock = false

function nextSentence() {
  if (lock || isEnd) return
  checkTranslateLocation()
  lock = true
  let currentSection = props.article.sections[sectionIndex]
  let currentSentence = currentSection[sentenceIndex]
  //这里把未输入的单词补全，因为删除时会用到input
  currentSentence.words.forEach((word, i) => {
    word.input = word.input + word.word.slice(word.input?.length ?? 0)
  })

  //todo 计得把略过的单词加上统计里面去
  // if (!store.allIgnoreWords.includes(currentWord.word.toLowerCase()) && currentWord.type === PracticeArticleWordType.Word) {
  //   statisticsStore.inputNumber++
  // }
  isSpace = false;
  input = wrong = ''
  stringIndex = 0;
  wordIndex = 0
  sentenceIndex++
  if (!currentSection[sentenceIndex]) {
    sentenceIndex = 0
    sectionIndex++
    if (!props.article.sections[sectionIndex]) {
      console.log('打完了')
      isEnd = true
      emit('complete')
    } else {
      if (isNameWord()) next()
      emit('play', {sentence: props.article.sections[sectionIndex][0], handle: false})
    }
  } else {
    if (isNameWord()) next()
    emit('play', {sentence: currentSection[sentenceIndex], handle: false})

  }
  lock = false
  focusMobileInput()
}

const next = () => {
  isSpace = false;
  input = wrong = ''
  stringIndex = 0;

  let currentSection = props.article.sections[sectionIndex]
  let currentSentence = currentSection[sentenceIndex]
  let currentWord: ArticleWord = currentSentence.words[wordIndex]

  // 检查下一个单词是否存在
  if (wordIndex + 1 < currentSentence.words.length) {
    wordIndex++;
    currentWord = currentSentence.words[wordIndex]
    //这里把未输入的单词补全，因为删除时会用到input
    currentSentence.words.slice(0, wordIndex).forEach((word, i) => {
      word.input = word.input + word.word.slice(word.input?.length ?? 0)
    })
    if ([PracticeArticleWordType.Symbol, PracticeArticleWordType.Number].includes(currentWord.type) && settingStore.ignoreSymbol) {
      next()
    } else if (isNameWord()) {
      next()
    } else {
      emit('nextWord', currentWord);
    }
  } else {
    nextSentence()
  }
}

function onTyping(e: KeyboardEvent) {
  debugger
  if (!props.article.sections.length) return
  if (isTyping || isEnd) return;
  isTyping = true;
  // console.log('keyDown', e.key, e.code, e.keyCode)
  try {
    let currentSection = props.article.sections[sectionIndex]
    let currentSentence = currentSection[sentenceIndex]
    let currentWord: ArticleWord = currentSentence.words[wordIndex]
    wrong = ''

    if (isSpace) {
      if (e.code === 'Space') {
        next()
      } else {
        // 如果在第一个单词的最后一位上， 不按空格的直接输入下一个字母的话
        next()
        isTyping = false
        onTyping(e)
        // wrong = ' '
        // playBeep()
        // setTimeout(() => {
        //   wrong = ''
        //   wrong = input = ''
        // }, 500)
      }
    } else {

      // if (isNameWord(currentWord)) {
      //   isSpace = false
      //   next()
      //   isTyping = false
      //   return onTyping(e)
      // }
      let letter = e.key
      let key = currentWord.word[stringIndex]
      // console.log('key', key,)

      let isRight = false
      if (settingStore.ignoreCase) {
        isRight = key.toLowerCase() === letter.toLowerCase()
      } else {
        isRight = key === letter
      }
      if (!isRight) {
        if (currentWord.type === PracticeArticleWordType.Word) {
          emit('wrong', currentWord)
        }
        playBeep()
      }

      input += letter
      currentWord.input = input
      stringIndex++
      //单词输入完毕
      if (!currentWord.word[stringIndex]) {
        input = ''
        if (currentWord.nextSpace) {
          isSpace = true
        } else {
          next()
        }
      }
    }
    playKeyboardAudio()
    e.preventDefault()
  } catch (e) {
    //todo 上报
    localStorage.removeItem(PracticeSaveArticleKey.key)
    init()
  } finally {
    isTyping = false
  }
}

function play() {
  let currentSection = props.article.sections[sectionIndex]
  emit('play', {sentence: currentSection[sentenceIndex], handle: true})
}

function del() {
  if (wrong) {
    wrong = ''
  } else {
    if (isEnd) return;
    if (isSpace) {
      isSpace = false
    }
    let endSentence = false
    let endWord = false
    let endString = false
    if (stringIndex === 0) {
      if (wordIndex === 0) {
        if (sentenceIndex === 0) {
          if (sectionIndex === 0) {
            return
          } else {
            endSentence = endString = endWord = true
            sectionIndex--
          }
        } else {
          endString = endWord = true
          sentenceIndex--
        }
      } else {
        endString = true
        wordIndex--
      }
    } else stringIndex--
    let currentSection = props.article.sections[sectionIndex]
    if (endSentence) sentenceIndex = currentSection.length - 1
    let currentSentence = currentSection[sentenceIndex]
    if (endWord) wordIndex = currentSentence.words.length - 1
    let currentWord: ArticleWord = currentSentence.words[wordIndex]
    if (endString) {
      checkTranslateLocation()
      if (currentWord.nextSpace) {
        isSpace = true
        stringIndex = currentWord.word.length
      } else {
        stringIndex = currentWord.word.length - 1
      }
    }
    input = currentWord.input = currentWord.input.slice(0, stringIndex)
    checkCursorPosition()
    focusMobileInput()
  }
}

function showSentence(i1: number = sectionIndex, i2: number = sentenceIndex, i3: number = wordIndex) {
  hoverIndex = {sectionIndex: i1, sentenceIndex: i2, wordIndex: i3}
}

function hideSentence() {
  hoverIndex = {sectionIndex: -1, sentenceIndex: -1, wordIndex: -1}
}

function jump(i, j, w, sentence?) {
  sectionIndex = i
  sentenceIndex = j
  //todo 这里有可能是符号，要处理下
  wordIndex = w
  stringIndex = 0
  input = wrong = ''
  isEnd = isSpace = false
  props.article.sections.map((v, i) => {
    v.map((w, j) => {
      w.words.map((v, k) => {
        if (i <= sectionIndex && j <= sentenceIndex && k < wordIndex) {
          v.input = v.word
        } else {
          v.input = ''
        }
      })
    })
  })
  if (sentence) {
    emit('play', {sentence: sentence, handle: false})
  }
}

function onContextMenu(e: MouseEvent, sentence: Sentence, i, j, w) {
  const selectedText = window.getSelection().toString();
  console.log(selectedText);
  //prevent the browser's default menu
  e.preventDefault();
  //show your menu
  ContextMenu.showContextMenu({
    x: e.x,
    y: e.y,
    items: [
      {
        label: "收藏单词",
        onClick: () => {
          let word = props.article.sections[i][j].words[w]
          let text = word.word
          let doc = nlp(text)
          // 优先判断是不是动词
          if (doc.verbs().found) {
            text = doc.verbs().toInfinitive().text()
          }
          // 如果是名词（复数 → 单数）
          if (doc.nouns().found) {
            text = doc.nouns().toSingular().text()
          }
          if (!text.length) text = word.word
          console.log('text', text)
          toggleWordCollect(getDefaultWord({word: text, id: nanoid()}))
          Toast.success(text + ' 添加成功')
        }
      },
      {
        label: "复制",
        children: [
          {
            label: "复制句子",
            onClick: () => {
              navigator.clipboard.writeText(sentence.text).then(r => {
                Toast.success('已复制')
              })
            }
          },
          {
            label: "复制单词",
            onClick: () => {
              let word = props.article.sections[i][j].words[w]
              navigator.clipboard.writeText(word.word).then(r => {
                Toast.success('已复制')
              })
            }
          }
        ]
      },
      {
        label: "从这开始",
        onClick: () => {
          jump(i, j, w + 1, sentence)
        }
      },
      {
        label: "播放句子",
        onClick: () => {
          emit('play', {sentence: sentence, handle: true})
        }
      },
      {
        label: "语法分析",
        onClick: () => {
          navigator.clipboard.writeText(sentence.text).then(r => {
            Toast.success('已复制！随后将打开语法分析网站！')
            setTimeout(() => {
              window.open('https://enpuz.com/')
            }, 1000)
          })
        }
      },
      {
        label: "有道词典翻译",
        children: [
          {
            label: "翻译单词",
            onClick: () => {
              let word = props.article.sections[i][j].words[w]
              window.open(`https://www.youdao.com/result?word=${word.word}&lang=en`, '_blank')
            }
          },
          {
            label: "翻译句子",
            onClick: () => {
              window.open(`https://www.youdao.com/result?word=${sentence.text}&lang=en`, '_blank')
            }
          },
        ]
      },
    ]
  });
}

onMounted(() => {
  emitter.on(EventKey.resetWord, () => {
    wrong = input = ''
  })
  emitter.on(EventKey.onTyping, onTyping)
  if (isMob) {
    focusMobileInput()
  }
})

onUnmounted(() => {
  emitter.off(EventKey.resetWord,)
  emitter.off(EventKey.onTyping, onTyping)
  // 清理定时器
  if (translationTimer) {
    clearTimeout(translationTimer)
    translationTimer = null
  }
})

useEvents([
  [ShortcutKey.KnowWord, onTyping],
  [ShortcutKey.UnknownWord, onTyping],
  [ShortcutKey.TranslateWord, showWordTranslation],
])

defineExpose({showSentence, play, del, hideSentence, nextSentence, init})

function isCurrent(i: number, j: number, w: number) {
  return `${i}${j}${w}` === currentIndex
}

let showQuestions = $ref(false)

const currentPractice = inject('currentPractice', [])

// 单词翻译弹窗
let showWordTranslationDialog = $ref(false)
let currentWordForTranslation = $ref('')
let translationTimer: ReturnType<typeof setTimeout> | null = null

/**
 * 获取当前正在输入的单词
 */
function getCurrentWord(): string {
  if (!props.article.sections.length) return ''
  
  let currentSection = props.article.sections[sectionIndex]
  if (!currentSection) return ''
  
  let currentSentence = currentSection[sentenceIndex]
  if (!currentSentence) return ''
  
  let currentWord: ArticleWord = currentSentence.words[wordIndex]
  if (!currentWord) return ''
  
  // 如果是单词类型，返回单词文本
  if (currentWord.type === PracticeArticleWordType.Word) {
    return currentWord.word || ''
  }
  
  return ''
}

/**
 * 显示单词翻译弹窗（3秒后自动关闭）
 */
function showWordTranslation() {
  const word = getCurrentWord()
  if (!word || !word.trim()) {
    Toast.warning('请先选择一个单词')
    return
  }
  
  // 清除之前的定时器
  if (translationTimer) {
    clearTimeout(translationTimer)
    translationTimer = null
  }
  
  currentWordForTranslation = word.trim()
  showWordTranslationDialog = true
  
  // 3秒后自动关闭
  translationTimer = setTimeout(() => {
    showWordTranslationDialog = false
    translationTimer = null
  }, 3000)
}

</script>

<template>
  <div class="typing-article" ref="typeArticleRef" @click="focusMobileInput">
    <input
      v-if="isMob"
      ref="mobileInputRef"
      class="mobile-input"
      type="text"
      inputmode="text"
      autocomplete="off"
      autocorrect="off"
      autocapitalize="none"
      @beforeinput="handleMobileBeforeInput"
      @input="handleMobileInput"
    />
    <header class="mb-4">
      <div class="title"><span class="font-family text-3xl">{{
          store.sbook.lastLearnIndex + 1
        }}.</span>{{ props.article.title }}
      </div>
      <div class="titleTranslate" v-if="settingStore.translate">{{ props.article.titleTranslate }}</div>
    </header>

    <div id="article-content" class="article-content"
         :class="[
          settingStore.translate && 'tall',
          settingStore.dictation && 'dictation',
      ]"
         ref="articleWrapperRef">
      <article>
        <div class="section" v-for="(section,indexI) in props.article.sections">
                <span class="sentence"
                      v-for="(sentence,indexJ) in section">
                  <span
                    v-for="(word,indexW) in sentence.words"
                    @contextmenu="e=>onContextMenu(e,sentence,indexI,indexJ,indexW)"
                    class="word"
                    :class="[(sectionIndex>indexI
                        ?'wrote':
                        (sectionIndex>=indexI &&sentenceIndex>indexJ)
                        ?'wrote' :
                        (sectionIndex>=indexI &&sentenceIndex>=indexJ && wordIndex>indexW)
                        ?'wrote':
                         (sectionIndex>=indexI &&sentenceIndex>=indexJ && wordIndex>=indexW && stringIndex>=word.word.length)
                        ?'wrote':
                        ''),
                        indexW === 0 && `word${indexI}-${indexJ}`,
                        ]">
                    <span class="word-wrap"
                          @mouseenter="settingStore.allowWordTip && showSentence(indexI,indexJ,indexW)"
                          @mouseleave="hideSentence"
                          :class="[
                           hoverIndex.sectionIndex === indexI && hoverIndex.sentenceIndex === indexJ && hoverIndex.wordIndex === indexW
                          &&'hover-show',
                          word.type === PracticeArticleWordType.Number && 'font-family text-xl'
                          ]"
                          @click="playWordAudio(word.word)"
                    >
                      <TypingWord :word="word"
                                  :is-typing="true"
                                  v-if="isCurrent(indexI,indexJ,indexW) && !isSpace"/>
                      <TypingWord :word="word" :is-typing="false" v-else/>
                      <span class="border-bottom" v-if="settingStore.dictation"></span>
                    </span>
                   <Space
                     v-if="word.nextSpace"
                     class="word-end"
                     :is-wrong="false"
                     :is-wait="isCurrent(indexI,indexJ,indexW) && isSpace"
                     :is-shake="isCurrent(indexI,indexJ,indexW) && isSpace && wrong !== ''"
                   />
                  </span>
                  <span
                    class="sentence-translate-mobile"
                    v-if="isMob && settingStore.translate && sentence.translate">
                    {{ sentence.translate }}
                  </span>
                </span>
        </div>
      </article>
      <div class="translate" v-show="settingStore.translate">
        <template v-for="(v,indexI) in props.article.sections">
          <div class="row"
               :class="[
                   `translate${indexI+'-'+indexJ}`,
                   (sectionIndex>indexI
                        ?'wrote':
                        (sectionIndex>=indexI &&sentenceIndex>indexJ)
                        ?'wrote' :
                        ''),
                        ]"
               v-for="(item,indexJ) in v">
            <span class="space"></span>
            <Transition name="fade">
              <span class="text" v-if="item.translate">{{ item.translate }}</span>
            </Transition>
          </div>
        </template>
      </div>
      <div class="cursor" v-if="!isEnd" :style="{top:cursor.top+'px',left:cursor.left+'px'}"></div>
    </div>

    <div class="options flex justify-center" v-if="isEnd">
      <BaseButton
        @click="emit('replay')">重新练习
      </BaseButton>
      <BaseButton
        v-if="store.sbook.lastLearnIndex < store.sbook.articles.length - 1"
        @click="emit('next')">下一篇
      </BaseButton>
    </div>

    <div class="font-family text-base pr-2 mb-50 mt-10" v-if="currentPractice.length && isEnd">
      <div class="text-2xl font-bold">学习记录</div>
      <div class="mt-1 mb-3">总学习时长：{{ msToHourMinute(total(currentPractice, 'spend')) }}</div>
      <div class="item border border-item border-solid mt-2 p-2 bg-[var(--bg-history)] rounded-md flex justify-between"
           :class="i === currentPractice.length-1 && 'color-red!'"
           v-for="(item,i) in currentPractice">
        <span :class="i === currentPractice.length-1 ? 'color-red':'color-gray'"
        >{{
            i === currentPractice.length - 1 ? '当前' : i + 1
          }}.&nbsp;&nbsp;{{ _dateFormat(item.startDate) }}</span>
        <span>{{ msToHourMinute(item.spend) }}</span>
      </div>
    </div>

    <template v-if="false">
      <div class="center">
        <BaseButton @click="showQuestions =! showQuestions">显示题目</BaseButton>
      </div>
      <div class="toggle" v-if="showQuestions">
        <QuestionForm :questions="article.questions"
                      :duration="300"
                      :immediateFeedback="false"
                      :randomize="true"
        />
      </div>
    </template>

    <!-- 单词翻译弹窗 -->
    <WordTranslationDialog
      v-model="showWordTranslationDialog"
      :word="currentWordForTranslation"
    />
  </div>
</template>

<style scoped lang="scss">

.wrote {
  color: grey;
}

$translate-lh: 3.2;
$article-lh: 2.4;

.typing-article {
  height: 100%;
  overflow: auto;
  color: var(--color-article);
  width: var(--article-width);
  font-size: 1.6rem;

  header {
    word-wrap: break-word;
    position: relative;
    padding-top: 3rem;

    .title {
      text-align: center;
      font-size: 2.2rem;
      font-family: var(--en-article-family);
    }

    .titleTranslate {
      @extend .title;
      font-size: 1.2rem;
      margin-top: 0.5rem;
      font-family: var(--zh-article-family);
      font-weight: bold;
    }
  }

  .mobile-input {
    position: absolute;
    opacity: 0;
    pointer-events: none;
    height: 0;
    width: 0;
  }

  .article-content {
    position: relative;
  }

  .dictation {
    .border-bottom {
      display: inline-block !important;
    }
    .translate{
      color: var(--color-reverse-black);
    }
  }

  .tall {
    article {
      line-height: $article-lh;
    }
  }

  article {
    word-break: keep-all;
    word-wrap: break-word;
    white-space: pre-wrap;
    font-family: var(--en-article-family);

    .wrote, .hover-show {
      :deep(.hide) {
        opacity: 1 !important;
      }

      .border-bottom {
        display: none !important;
      }
    }

    .hover-show {
      border-radius: 0.2rem;
      //background: var(--color-select-bg);
      @apply bg-green!;

      :deep(.hide) {
        opacity: 1 !important;
      }
    }

    .section {
      margin-bottom: 1.5rem;

      .sentence {
        transition: all .3s;
      }

      .word {
        display: inline-block;

        .word-wrap {
          position: relative;
          transition: background-color .3s;
        }

        .border-bottom {
          position: absolute;
          width: 100%;
          height: 100%;
          left: 0;
          top: 0;
          border-bottom: 2px solid var(--color-article);
          display: none;
          transform: translateY(-0.2rem);
        }
      }
    }
  }

  .translate {
    pointer-events: none;
    position: absolute;
    top: 0;
    left: 0;
    height: 100%;
    width: 100%;
    font-size: 1.2rem;
    line-height: $translate-lh;
    letter-spacing: .2rem;
    font-family: var(--zh-article-family);
    font-weight: bold;
    color: #818181;

    .row {
      position: absolute;
      left: 0;
      width: 100%;
      opacity: 0;
      transition: all .3s;

      .space {
        transition: all .3s;
        display: inline-block;
      }
    }
  }
}

.sentence-translate-mobile {
  display: none;
}

// 移动端适配
@media (max-width: 768px) {
  .typing-article {
    width: 100vw;
    max-width: 100%;
    padding: 1rem 0.5rem;

    // 标题优化
    header {
      .title {
        font-size: 1.2rem;
        line-height: 1.4;
        word-break: break-word;
        margin-bottom: 1rem;

        .font-family {
          font-size: 1rem;
        }
      }

      .titleTranslate {
        font-size: 0.9rem;
        margin-top: 0.5rem;
        opacity: 0.8;
      }
    }

    // 句子显示优化
    .article-content {
      article {
        .section {
          margin-bottom: 1rem;

          .sentence {
            font-size: 1rem;
            line-height: 1.6;
            word-break: break-word;
            margin-bottom: 0.5rem;

            .word {
              .word-wrap {
                padding: 0.1rem 0.05rem;
                min-height: 24px;
                display: inline-flex;
                align-items: center;
              }
            }
          }
        }
      }
    }

    .sentence-translate-mobile {
      display: block;
      margin-top: 0.4rem;
      font-size: 0.9rem;
      line-height: 1.4;
      color: var(--color-font-3);
      font-family: var(--zh-article-family);
      word-break: break-word;
    }

    // 翻译区域优化
    .translate {
      display: none;
    }

    // 问答表单优化
    .question-form {
      padding: 0.5rem;

      .base-button {
        width: 100%;
        min-height: 48px;
        margin-top: 0.5rem;
      }
    }
  }
}

@media (max-width: 480px) {
  .typing-article {
    padding: 0.5rem 0.3rem;

    header {
      .title {
        font-size: 1rem;

        .font-family {
          font-size: 0.9rem;
        }
      }

      .titleTranslate {
        font-size: 0.8rem;
      }
    }

    .article-content {
      article {
        .section {
          .sentence {
            font-size: 0.9rem;
            line-height: 1.5;
          }
        }
      }
    }

    .sentence-translate-mobile {
      font-size: 0.85rem;
      line-height: 1.35;
    }
  }
}
</style>
