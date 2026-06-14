# Level 4: 专家级模块 (Expert)

本级别包含最复杂的模块，涉及分布式系统和高级优化技术。

## 📚 模块列表

### 4.1 MOE 专家混合
- **文件位置**：`csrc/moe/`
- **内容概述**：
  - 专家路由（Top-K 选择）
  - 数据对齐和重排
  - 融合门控计算
  - 分组矩阵乘法
  - 多种量化格式支持（FP8, NVFP4, W4A8）
  - CUTLASS 和 Marlin 格式支持

**主要算子**：
- `moe_align_block_size` - 对齐数据到块大小
- `topk_softmax` / `topk_sigmoid` - Top-K 门控
- `moe_fused_gate` - 融合门控
- `fp8_blockwise_scaled_grouped_mm` - FP8 分组矩阵乘法

### 4.2 Speculative 推测性解码
- **文件位置**：`csrc/speculative/`
- **内容概述**：
  - 树状推测解码
  - Eagle 算法实现
  - N-gram 验证
  - 位打包优化

**主要算子**：
- `tree_speculative_sampling_target_only` - 树状推测采样
- `verify_tree_greedy` - 贪婪树验证
- `build_tree_kernel_efficient` - 高效构建解码树

### 4.3 KVCacheIO KV缓存I/O
- **文件位置**：`csrc/kvcacheio/`
- **内容概述**：
  - 不同布局之间的 KV 缓存传输
  - PagedAttention 格式支持
  - MLA 格式支持
  - 批量传输优化

**主要算子**：
- `transfer_kv_per_layer` - 单层 KV 传输
- `transfer_kv_all_layer` - 所有层批量传输
- `transfer_kv_per_layer_mla` - MLA 格式单层传输

### 4.4 AllReduce 分布式通信
- **文件位置**：`csrc/allreduce/`
- **内容概述**：
  - 自定义 AllReduce 实现
  - MSCL++ 库集成
  - 多 GPU 通信优化
  - IPC 缓冲区管理

**主要算子**：
- `init_custom_ar` - 初始化自定义 AllReduce
- `all_reduce` - 执行 AllReduce 操作
- `mscclpp_allreduce` - MSCL++ AllReduce

## 🎯 学习目标

- ✅ 理解复杂的分支调度算法（MOE）
- ✅ 掌握高级解码策略（Speculative Decoding）
- ✅ 理解复杂的内存管理（KVCacheIO）
- ✅ 掌握分布式系统通信（AllReduce）

## 📖 学习路径

这些模块都非常复杂，建议：
1. 先理解理论基础（相关论文）
2. 阅读源代码和注释
3. 运行测试用例理解行为
4. 分析性能特征

---

**详细文档待完善，请参考源代码和测试用例**

