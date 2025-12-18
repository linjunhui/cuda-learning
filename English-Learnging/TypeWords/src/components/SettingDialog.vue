<script setup lang="ts">
import {useSettingStore} from "@/stores/setting.ts";
import {getAudioFileUrl, usePlayAudio} from "@/hooks/sound.ts";
import {ShortcutKey} from "@/types/types.ts";
import VolumeIcon from "@/components/icon/VolumeIcon.vue";
import {useBaseStore} from "@/stores/base.ts";
import {SoundFileOptions} from "@/config/env.ts";
import {Option, Select} from "@/components/base/select";
import Switch from "@/components/base/Switch.vue";
import Slider from "@/components/base/Slider.vue";
import RadioGroup from "@/components/base/radio/RadioGroup.vue";
import Radio from "@/components/base/radio/Radio.vue";
import InputNumber from "@/components/base/InputNumber.vue";
import Textarea from "@/components/base/Textarea.vue";
import SettingItem from "@/pages/setting/SettingItem.vue";
import {defineAsyncComponent} from "vue";
import BaseIcon from "@/components/BaseIcon.vue";

const Dialog = defineAsyncComponent(() => import('@/components/dialog/Dialog.vue'))

const props = defineProps<{
  type: 'article' | 'word'
}>()

const tabIndex = $ref(props.type === 'word' ? 1 : 2)
const settingStore = useSettingStore()
const store = useBaseStore()
let show = $ref(false)

const simpleWords = $computed({
  get: () => store.simpleWords.join(','),
  set: v => {
    try {
      store.simpleWords = v.split(',');
    } catch (e) {

    }
  }
})

</script>

<template>
  <Dialog v-model="show" title="设置">
    <div class="setting text-lg w-200 h-[60vh] text-md flex flex-col">
      <div class="flex flex-1 overflow-hidden">
        <div class="left">
          <div class="tabs">
            <div class="tab" :class="tabIndex === 1 && 'active'" @click="tabIndex = 1" v-if="type === 'word'">
              <IconFluentTextUnderlineDouble20Regular width="20"/>
              <span>单词</span>
            </div>
            <div class="tab" :class="tabIndex === 2 && 'active'" @click="tabIndex = 2" v-if="type === 'article'">
              <IconFluentBookLetter20Regular width="20"/>
              <span>文章</span>
            </div>
            <div class="tab" :class="tabIndex === 0 && 'active'" @click="tabIndex = 0">
              <IconFluentSettings20Regular width="20"/>
              <span>通用</span>
            </div>
          </div>
        </div>
        <div class="content">
          <!--        通用练习设置-->
          <!--        通用练习设置-->
          <!--        通用练习设置-->
          <div v-if="tabIndex === 0">
            <SettingItem title="忽略大小写"
                         desc="开启后，输入时不区分大小写，如输入“hello”和“Hello”都会被认为是正确的"
            >
              <Switch v-model="settingStore.ignoreCase"/>
            </SettingItem>

            <SettingItem title="允许默写模式下显示提示"
                         :desc="`开启后，可以通过将鼠标移动到单词上或者按快捷键 ${settingStore.shortcutKeyMap[ShortcutKey.ShowWord]} 显示正确答案`"
            >
              <Switch v-model="settingStore.allowWordTip"/>
            </SettingItem>

            <div class="line"></div>
            <SettingItem title="简单词过滤"
                         desc="开启后，练习的单词中不会包含简单词；文章统计的总词数中不会包含简单词"
            >
              <Switch v-model="settingStore.ignoreSimpleWord"/>
            </SettingItem>

            <SettingItem title="简单词列表"
                         class="items-start!"
                         v-if="settingStore.ignoreSimpleWord"
            >
            <Textarea
              placeholder="多个单词用英文逗号隔号"
              v-model="simpleWords" :autosize="{minRows: 6, maxRows: 10}"/>
            </SettingItem>

            <!--          音效-->
            <!--          音效-->
            <!--          音效-->
            <div class="line"></div>
            <SettingItem main-title="音效"/>
            <SettingItem title="单词/句子发音口音"
                         desc="仅单词生效，文章固定美音"
            >
              <Select v-model="settingStore.soundType"
                      placeholder="请选择"
                      class="w-50!"
              >
                <Option label="美音" value="us"/>
                <Option label="英音" value="uk"/>
              </Select>
            </SettingItem>

            <div class="line"></div>
            <SettingItem title="按键音">
              <Switch v-model="settingStore.keyboardSound"/>
            </SettingItem>
            <SettingItem title="按键音效">
              <Select v-model="settingStore.keyboardSoundFile"
                      placeholder="请选择"
                      class="w-50!"
              >
                <Option
                  v-for="item in SoundFileOptions"
                  :key="item.value"
                  :label="item.label"
                  :value="item.value"
                >
                  <div class="flex justify-between items-center w-full">
                    <span>{{ item.label }}</span>
                    <VolumeIcon
                      :time="100"
                      @click="usePlayAudio(getAudioFileUrl(item.value)[0])"/>
                  </div>
                </Option>
              </Select>
            </SettingItem>
            <SettingItem title="音量">
              <Slider v-model="settingStore.keyboardSoundVolume" showText showValue unit="%"/>
            </SettingItem>

          </div>


          <!--        单词练习设置-->
          <!--        单词练习设置-->
          <!--        单词练习设置-->
          <div v-if="tabIndex === 1">
<!--            <SettingItem title="练习模式">-->
<!--              <RadioGroup v-model="settingStore.wordPracticeMode" class="flex-col gap-0!">-->
<!--                <Radio :value="WordPracticeMode.System" label="智能模式:自动规划学习、复习、听写、默写"/>-->
<!--                <Radio :value="WordPracticeMode.Free" label="自由模式:系统不强制复习与默写"/>-->
<!--              </RadioGroup>-->
<!--            </SettingItem>-->

            <SettingItem title="显示上一个/下一个单词"
                         desc="开启后，练习中会在上方显示上一个/下一个单词"
            >
              <Switch v-model="settingStore.showNearWord"/>
            </SettingItem>

            <SettingItem title="不默认显示练习设置弹框"
                         desc="在词典详情页面，点击学习按钮后，是否显示练习设置弹框"
            >
              <Switch v-model="settingStore.disableShowPracticeSettingDialog"/>
            </SettingItem>

            <SettingItem title="输入错误时，清空已输入内容"
            >
              <Switch v-model="settingStore.inputWrongClear"/>
            </SettingItem>


            <SettingItem title="单词循环设置" class="gap-0!">
              <RadioGroup v-model="settingStore.repeatCount">
                <Radio :value="1" size="default">1</Radio>
                <Radio :value="2" size="default">2</Radio>
                <Radio :value="3" size="default">3</Radio>
                <Radio :value="5" size="default">5</Radio>
                <Radio :value="100" size="default">自定义</Radio>
              </RadioGroup>
              <div class="ml-2 center gap-space" v-if="settingStore.repeatCount === 100">
                <span>循环次数</span>
                <InputNumber v-model="settingStore.repeatCustomCount"
                             :min="6"
                             :max="15"
                             type="number"
                />
              </div>
            </SettingItem>

            <SettingItem title="复习比"
                         desc="复习词与新词的比例，修改后下次学习生效"
            >
              <InputNumber :min="0" :max="10" v-model="settingStore.wordReviewRatio"/>
            </SettingItem>

            <!--          发音-->
            <!--          发音-->
            <!--          发音-->
            <div class="line"></div>
            <SettingItem mainTitle="音效"/>
            <SettingItem title="单词自动发音">
              <Switch v-model="settingStore.wordSound"/>
            </SettingItem>
            <SettingItem title="音量">
              <Slider v-model="settingStore.wordSoundVolume" showText showValue unit="%"/>
            </SettingItem>
            <SettingItem title="倍速">
              <Slider v-model="settingStore.wordSoundSpeed" :step="0.1" :min="0.5" :max="3" showText showValue/>
            </SettingItem>
            <div class="line"></div>
            <SettingItem title="效果音（输入错误、完成时的音效）">
              <Switch v-model="settingStore.effectSound"/>
            </SettingItem>
            <SettingItem title="音量">
              <Slider v-model="settingStore.effectSoundVolume" showText showValue unit="%"/>
            </SettingItem>

            <!--          自动切换-->
            <!--          自动切换-->
            <!--          自动切换-->
            <div class="line"></div>
            <SettingItem mainTitle="自动切换"/>
            <SettingItem title="自动切换下一个单词"
                         desc="仅在 **跟写** 时生效，听写、自测、默写均不会自动切换，需要手动按 **空格键** 切换"
            >
              <Switch v-model="settingStore.autoNextWord"/>
            </SettingItem>

            <SettingItem title="自动切换下一个单词时间"
                         desc="正确输入单词后，自动跳转下一个单词的时间"
            >
              <InputNumber v-model="settingStore.waitTimeForChangeWord"
                           :disabled="!settingStore.autoNextWord"
                           :min="0"
                           :max="10000"
                           :step="100"
                           type="number"
              />
              <span class="ml-4">毫秒</span>
            </SettingItem>


            <!--          字体设置-->
            <!--          字体设置-->
            <!--          字体设置-->
            <div class="line"></div>
            <SettingItem mainTitle="字体设置"/>
            <SettingItem title="外语字体">
              <Slider
                :min="10"
                :max="100"
                v-model="settingStore.fontSize.wordForeignFontSize" showText showValue unit="px"/>
            </SettingItem>
            <SettingItem title="中文字体">
              <Slider
                :min="10"
                :max="100"
                v-model="settingStore.fontSize.wordTranslateFontSize" showText showValue unit="px"/>
            </SettingItem>
          </div>


          <!--        文章练习设置-->
          <!--        文章练习设置-->
          <!--        文章练习设置-->
          <div v-if="tabIndex === 2">
            <!--          发音-->
            <!--          发音-->
            <!--          发音-->
            <SettingItem mainTitle="音效"/>
            <SettingItem title="自动播放句子">
              <Switch v-model="settingStore.articleSound"/>
            </SettingItem>
            <SettingItem title="自动播放下一篇">
              <Switch v-model="settingStore.articleAutoPlayNext"/>
            </SettingItem>
            <SettingItem title="音量">
              <Slider v-model="settingStore.articleSoundVolume" showText showValue unit="%"/>
            </SettingItem>
            <SettingItem title="倍速">
              <Slider v-model="settingStore.articleSoundSpeed" :step="0.1" :min="0.5" :max="3" showText showValue/>
            </SettingItem>

            <div class="line"></div>
            <SettingItem title="输入时忽略符号/数字/人名">
              <Switch v-model="settingStore.ignoreSymbol"/>
            </SettingItem>
          </div>
        </div>
      </div>
    </div>
  </Dialog>
  <BaseIcon title="设置" @click="show = true;tabIndex = props.type === 'word' ? 1 : 2">
    <IconFluentSettings20Regular/>
  </BaseIcon>
</template>

<style scoped lang="scss">

.setting {

  .left {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    align-items: center;
    border-right: 1px solid gainsboro;

    .tabs {
      padding: 1rem;
      display: flex;
      flex-direction: column;
      gap: .6rem;
      //color: #0C8CE9;

      .tab {
        @apply cursor-pointer flex items-center relative;
        padding: .6rem .9rem;
        border-radius: .5rem;
        width: 8rem;
        gap: .6rem;
        transition: all .5s;

        &:hover {
          background: var(--btn-primary);
          color: white;
        }

        &.active {
          background: var(--btn-primary);
          color: white;
        }
      }
    }
  }

  .content {
    flex: 1;
    height: 100%;
    overflow: auto;
    padding: 0 1.6rem;

    .line {
      border-bottom: 1px solid #c4c3c3;
    }
  }
}
</style>
