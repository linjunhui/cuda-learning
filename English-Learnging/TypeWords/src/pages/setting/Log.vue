<script setup lang="ts">

</script>

<template>
  <div>
    <div class="log-item">
      <div class="mb-2">
        <div>
          <div>日期：2025/12/11</div>
          <div>内容：修复音标显示错误问题，优化反馈页面</div>
        </div>
      </div>
    </div>
    <div class="log-item">
      <div class="mb-2">
        <div>
          <div>日期：2025/12/10</div>
          <div>内容：新增选项：复习比(单词练习时，复习词与新词的比例)</div>
        </div>
      </div>
    </div>
    <div class="log-item">
      <div class="mb-2">
        <div>
          <div>日期：2025/12/5</div>
          <div>内容：解决练习界面无法复制、全选的问题</div>
        </div>
      </div>
    </div>
    <div class="log-item">
      <div class="mb-2">
        <div>
          <div>日期：2025/12/3</div>
          <div>内容：单词、文章设置修改为弹框，更方便</div>
        </div>
      </div>
    </div>
    <div class="log-item">
      <div class="mb-2">
        <div>
          <div>日期：2025/12/3</div>
          <div>内容：录入新概念（三、四）部分音频，优化文章相关功能</div>
        </div>
      </div>
    </div>
    <div class="log-item">
      <div class="mb-2">
        <div>
          <div>日期：2025/12/2</div>
          <div>内容：完成新概念（一）音频，优化文章管理页面</div>
        </div>
      </div>
    </div>
    <div class="log-item">
      <div class="mb-2">
        <div>
          <div>日期：2025/11/30</div>
          <div>内容：文章里的单词可点击播放</div>
        </div>
      </div>
    </div>
    <div class="log-item">
      <div class="mb-2">
        <div>
          <div>日期：2025/11/29</div>
          <div>内容：修改 Slider 组件显示bug，新增 IE 浏览器检测提示</div>
        </div>
      </div>
    </div>
    <div class="log-item">
      <div class="mb-2">
        <div>
          <div>日期：2025/11/28</div>
          <div>内容：新增引导框、 新增<a href="https://github.com/zyronon/TypeWords/pull/175" target="_blank">词典测试模式（由大佬
            hebeihang 开发）</a></div>
        </div>
      </div>
    </div>
    <div class="log-item">
      <div class="mb-2">
        <div>
          <div>日期：2025/11/25</div>
          <div>内容：文章练习新增人名忽略功能（新概念一已全部适配），上传了新概念（一）1-18 音频</div>
        </div>
      </div>
    </div>
    <div class="log-item">
      <div class="mb-2">
        <div>
          <div>日期：2025/11/23</div>
          <div>内容：优化练习完成结算界面，新增分享功能</div>
        </div>
      </div>
    </div>
    <div class="log-item">
      <div class="mb-2">
        <div>
          <div>日期：2025/11/22</div>
          <div>内容：适配移动端</div>
        </div>
      </div>
    </div>
    <div class="log-item">
      <div class="mb-2">
        <div>
          <div>日期：2025/11/16</div>
          <div>内容：自测单词时，不认识单词可以直接输入，自动标识为错误单词，无需按2</div>
        </div>
      </div>
    </div>
    <div class="log-item">
      <div class="mb-2">
        <div>
          <div>日期：2025/11/15</div>
          <div>内容：练习单词时，底部工具栏新增“跳到下一阶段”按钮</div>
        </div>
      </div>
    </div>
    <div class="log-item">
      <div class="mb-2">
        <div>
          <div>日期：2025/11/14</div>
          <div>内容：新增文章练习时可跳过空格：如果在单词的最后一位上，不按空格直接输入下一个字母的话，自动跳下一个单词，
            按空格也自动跳下一个单词
          </div>
        </div>
      </div>
    </div>
    <div class="log-item">
      <div class="mb-2">
        <div>
          <div>日期：2025/11/13</div>
          <div>内容：新增文章练习时“输入时忽略符号/数字”选项</div>
        </div>
      </div>
    </div>
    <div class="log-item">
      <div class="mb-2">
        <div>
          <div>日期：2025/11/6</div>
          <div>内容：新增随机复习功能</div>
        </div>
      </div>
    </div>
    <div class="log-item">
      <div class="mb-2">
        <div>
          <div>日期：2025/10/30</div>
          <div>内容：集成PWA基础配置，支持用户以类App形式打开项目</div>
        </div>
      </div>
    </div>
    <div class="log-item">
      <div class="mb-2">
        <div>
          <div>日期：2025/10/26</div>
          <div>内容：进一步完善单词练习，解决复习数量太多的问题</div>
        </div>
        <div class="text-base mt-1">
          <ol>
            <li>
              <div class="title"><b>智能模式优化</b></div>
              <div class="desc">练习时新增四种练习模式：学习、自测、听写、默写。</div>
            </li>
            <li>
              <div class="title"><b>学习模式</b></div>
              <div class="desc">
                <ul>
                  <li>仅在练习新词时出现。</li>
                  <li>采用「跟写 / 拼写」方式进行学习。</li>
                  <li>每 7 个单词会 <b>强制进行听写</b>，解决原来“一次练太多，听写时已忘记”的问题。</li>
                </ul>
              </div>
            </li>
            <li>
              <div class="title"><b>自测模式（新增）</b></div>
              <div class="desc">
                <ul>
                  <li>仅在复习已学单词时出现。</li>
                  <li>不再强制拼写，提供「我认识」与「不认识」选项。</li>
                  <li>选择「我认识」后，该单词在后续听写或默写中将不再出现，<b>显著减少复习数量</b>。</li>
                </ul>
              </div>
            </li>
            <li>
              <div class="title"><b>听写模式</b></div>
              <div class="desc">原有逻辑保持不变。</div>
            </li>
            <li>
              <div class="title"><b>默写模式（新增）</b></div>
              <div class="desc">
                <ul>
                  <li>仅显示释义，不自动发音，不显示单词长度。</li>
                  <li>适合强化拼写记忆的场景。</li>
                </ul>
              </div>
            </li>
          </ol>
          <b>说明：</b>
          <div>本次更新重点解决了“复习单词数量过多、效率偏低”的问题。</div>
          <div>通过引入「复习」与「默写」两种模式，使复习流程更加灵活、高效。</div>
        </div>
      </div>
    </div>
    <div class="log-item">
      <div class="mb-2">
        <div>
          <div>日期：2025/10/8</div>
          <div>内容：文章支持自动播放下一篇</div>
        </div>
      </div>
    </div>
    <div class="log-item">
      <div class="mb-2">
        <div>
          <div>日期：2025/9/14</div>
          <div>内容：完善文章编辑、导入、导出等功能</div>
        </div>
        <div class="text-base mt-1">
          <div>1、文章的音频管理功能，目前已可添加音频、设置句子与音频的对应位置</div>
          <div>2、文章可导入、导出</div>
          <div>3、单词可导入、导出</div>
        </div>
      </div>
    </div>
    <div class="log-item">
      <div class="mb-2">
        <div>
          <div>日期：2025/8/10</div>
          <div>内容：2.0版本发布，全新UI，全新逻辑，新增短语、例句、近义词等功能</div>
        </div>
      </div>
    </div>
    <div class="log-item">
      <div class="mb-2">
        <div>
          <div>日期：2025/7/19</div>
          <div>内容：1.0版本发布</div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped lang="scss">

.log-item {
  border-bottom: 1px solid var(--color-input-border);
  margin-bottom: 1rem;
}

</style>