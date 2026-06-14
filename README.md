# CUDA 学习项目

> 这是一个专注于 CUDA 编程、GPU 计算和 AI 基础设施学习的项目仓库

## 根目录一览（已分组）

根目录按主题合并为少数一级目录，便于浏览；具体课程与项目仍在各子目录内，路径仅多一级前缀。

| 目录 | 内容 |
|------|------|
| **CUDA/** | CUDA 系统学习、编程基础、进阶、项目实战、CudaSteps |
| **AI与深度学习/** | AI-Agent、Flash-Attention、SGLang、PyTorch、Transformers、LeetGPU |
| **职业与通识/** | 英语、TSE、成长笔记、职业发展、文档收集（含同步副本） |
| **文档与工具/** | 仓库说明类 Markdown、Git/Gitee 指南、学习计划与记录、MD 同步脚本 |
| **C++学习/** | C++ 系统学习与练习 |
| **Python加强/** | Python 进阶与练习 |

### CUDA/

- `CUDA/CUDA学习/`：CUDA 编程系统学习（阶段化讲义与练习）
- `CUDA/CUDA编程基础/`：入门示例与性能分析材料
- `CUDA/CUDA进阶/`：高级主题
- `CUDA/CUDA项目实战/`：GEMM、MemoryPool 等实战工程
- `CUDA/CudaSteps/`：分步教程与示例代码

### AI与深度学习/

- `AI与深度学习/AI-Agent/`：Agent 与 RAG 等资料
- `AI与深度学习/Flash-Attention学习/`：Flash Attention 与 CUDA 内核相关
- `AI与深度学习/SGLang学习/`：SGLang 框架与算子练习
- `AI与深度学习/Pytorch学习/`：PyTorch 与自定义算子
- `AI与深度学习/Transformers/`：Transformer 相关笔记
- `AI与深度学习/LeetGPU/`：GPU 编程挑战与题解

### 职业与通识/

- `职业与通识/English-Learnging/`：英语与技术阅读材料
- `职业与通识/TSE学习/`：测试工程师能力模型
- `职业与通识/成长-改变/`：个人成长类笔记
- `职业与通识/职业发展/`：JD、面试与职业规划
- `职业与通识/文档收集/`：由 `文档与工具/sync_md_files.sh` 同步的 Markdown 汇总（勿手改为主源）

### 文档与工具/

- 各类指南与计划：`Gitee仓库创建和推送指南.md`、`OpenCL与CUDA对比分析.md`、`学习计划执行指南.md`、`面试准备.md` 等
- `sync_md_files.sh`：将全仓库 `.md` 同步到 `职业与通识/文档收集/`（日志在同目录 `sync_md_files.log`）

## 根目录文件

- `README.md`：本说明
- `目录索引.md`：更细的目录说明与链接（路径已按本次整理更新）

## 项目特点

1. **系统性学习**：从语言基础到 CUDA、再到推理框架形成主线
2. **实战导向**：CUDA 项目实战与 LeetGPU 等动手目录
3. **多领域覆盖**：GPU、C++、Python、AI 工程与软技能材料分区存放
4. **持续更新**：内容随学习进度增补，结构以根目录分组为主轴

## 快速导航

- [目录索引](目录索引.md)：按条目的详细索引
- [目录整理记录](文档与工具/目录整理.md)：历史整理说明
- [CUDA 学习入口](CUDA/CUDA学习/README.md)
- [C++ 学习入口](C++学习/README.md)
- [职业发展](职业与通识/职业发展/README.md)

## 使用说明

1. 先读本 README，再按需打开 `目录索引.md` 或各子目录 README
2. CUDA 相关统一从 `CUDA/` 进入，避免在根目录寻找旧路径
3. 若使用 MD 同步脚本，请在仓库根执行：`文档与工具/sync_md_files.sh`

---

**最后更新**：2026年4月  
**项目状态**：持续学习中
