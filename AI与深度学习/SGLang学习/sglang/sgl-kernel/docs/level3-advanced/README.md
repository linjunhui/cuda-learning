# Level 3: 高级模块 (Advanced)

本级别包含需要深入理解 Transformer 架构和序列模型的高级模块。

## 📚 模块列表

### 3.1 Attention 注意力机制
- **文件位置**：`csrc/attention/`
- **内容概述**：
  - Lightning Attention 解码实现
  - CUTLASS MLA（Multi-head Latent Attention）内核
  - 注意力状态合并（用于并行解码）
  - Cascade 注意力
  - 垂直/斜线索引转换（稀疏注意力）

**主要算子**：
- `lightning_attention_decode` - Lightning 注意力解码
- `cutlass_mla_decode` - CUTLASS MLA 解码
- `merge_state` / `merge_state_v2` - 合并注意力状态
- `convert_vertical_slash_indexes` - 索引格式转换

### 3.2 Mamba 状态空间模型
- **文件位置**：`csrc/mamba/`
- **内容概述**：
  - Mamba 模型架构
  - 因果一维卷积实现
  - 状态空间模型的高效计算

**主要算子**：
- `causal_conv1d` - 因果一维卷积

## 🎯 学习目标

- ✅ 深入理解 Transformer 注意力机制
- ✅ 掌握并行解码和状态合并技术
- ✅ 理解 Mamba 状态空间模型
- ✅ 掌握复杂模型的 GPU 优化

## 📖 学习路径

建议先学习 Attention 机制，再学习 Mamba 模型。

---

**详细文档待完善，请参考源代码和测试用例**

