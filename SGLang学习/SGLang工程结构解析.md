# SGLang 工程结构解析

## 目录

1. [项目概述](#项目概述)
2. [整体架构](#整体架构)
3. [核心组件详解](#核心组件详解)
4. [目录结构说明](#目录结构说明)
5. [关键模块分析](#关键模块分析)
6. [构建与部署](#构建与部署)

---

## 项目概述

SGLang (Structured Generation Language) 是一个高性能的大语言模型和视觉语言模型服务框架，旨在提供低延迟、高吞吐量的推理服务。项目支持从单GPU到大规模分布式集群的多种部署场景。

### 核心特性

- **快速后端运行时**: RadixAttention前缀缓存、零开销CPU调度器、Prefill-Decode分离、推测解码、连续批处理、分页注意力等
- **广泛模型支持**: Llama、Qwen、DeepSeek、Kimi、GLM、GPT、Gemma、Mistral等
- **多硬件支持**: NVIDIA GPU (GB200/B300/H100/A100)、AMD GPU (MI355/MI300)、Intel Xeon CPU、Google TPU、Ascend NPU等
- **灵活前端语言**: 支持链式生成调用、高级提示、控制流、多模态输入、并行和外部交互

---

## 整体架构

SGLang采用分层架构设计，主要包含三个核心组件：

```
┌─────────────────────────────────────────────────────────┐
│                    SGLang 整体架构                        │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  sgl-router  │  │   python/    │  │  sgl-kernel  │  │
│  │  (Rust)      │  │  sglang/     │  │  (CUDA/C++) │  │
│  │  路由网关     │  │  Python运行时 │  │  内核库      │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                           │
│  ┌──────────────────────────────────────────────────┐   │
│  │         SRT (SGLang Runtime) 核心运行时          │   │
│  │  - 模型执行器 (model_executor)                  │   │
│  │  - 内存缓存 (mem_cache)                         │   │
│  │  - 批处理调度 (managers)                        │   │
│  │  - 分布式通信 (distributed)                     │   │
│  └──────────────────────────────────────────────────┘   │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

---

## 核心组件详解

### 1. python/sglang/ - Python运行时层

**位置**: `/sglang/python/sglang/`

这是SGLang的主要Python代码库，包含：

#### 1.1 SRT (SGLang Runtime) 核心运行时
**位置**: `python/sglang/srt/`

SRT是SGLang的核心运行时系统，负责模型推理的执行。

**主要子模块**:

- **`entrypoints/`** - 服务入口点
  - `engine.py` - 核心推理引擎实现 (SRT引擎)
  - `EngineBase.py` - 引擎基类，定义统一API接口
  - `http_server.py` - HTTP服务器入口
  - `http_server_engine.py` - HTTP服务器引擎
  - `grpc_server.py` - gRPC服务器入口
  - `context.py` - 上下文管理
  - `openai/` - OpenAI兼容API实现
    - `serving_chat.py` - Chat Completions API
    - `serving_completions.py` - Text Completions API
    - `serving_embedding.py` - Embeddings API
    - `serving_rerank.py` - Rerank API
    - `serving_responses.py` - Responses API
    - `serving_classify.py` - Classification API
    - `serving_score.py` - Scoring API
    - `tool_server.py` - 工具调用服务器
    - `protocol.py` - OpenAI协议实现
  - `tool.py` - 工具调用支持

- **`model_executor/`** - 模型执行器
  - `model_runner.py` - 基础模型运行器
  - `cuda_graph_runner.py` - CUDA Graph优化执行器 (捕获CUDA图以提高性能)
  - `piecewise_cuda_graph_runner.py` - 分段CUDA图运行器
  - `cpu_graph_runner.py` - CPU执行器
  - `npu_graph_runner.py` - NPU执行器
  - `mindspore_runner.py` - MindSpore后端支持
  - `forward_batch_info.py` - 前向传播批信息
  - `hook_manager.py` - Hook管理器

- **`mem_cache/`** - 内存缓存系统
  - `radix_cache.py` - Radix树缓存实现
  - `radix_cache_cpp.py` - C++ Radix树绑定
  - `base_prefix_cache.py` - 前缀缓存基类
  - `storage/` - 存储后端
    - `hf3fs/` - HuggingFace 3FS存储
    - `lmcache/` - LMCache存储
    - `eic/` - EIC存储
    - `aibrix_kvcache/` - Aibrix KV缓存

- **`managers/`** - 批处理和资源管理
  - `scheduler.py` - 核心调度器，管理请求调度和批处理
  - `schedule_batch.py` - 批处理调度逻辑
  - `schedule_policy.py` - 调度策略
  - `cache_controller.py` - 缓存控制器
  - `tokenizer_manager.py` - Tokenizer管理器
  - `template_manager.py` - 模板管理器 (Chat模板)
  - `detokenizer_manager.py` - Detokenizer管理器
  - `data_parallel_controller.py` - 数据并行控制器
  - `session_controller.py` - 会话控制器
  - `multimodal_processor.py` - 多模态处理器
  - `request_metrics_exporter.py` - 请求指标导出器

- **`layers/`** - 神经网络层实现
  - 包含各种Transformer层的CUDA优化实现
  - 支持多种量化格式 (FP4/FP8/INT4/AWQ/GPTQ)
  - MoE (Mixture of Experts) 层实现

- **`models/`** - 模型定义
  - 支持多种模型架构 (Llama、Qwen、DeepSeek等)
  - 每个模型包含特定的配置和实现

- **`distributed/`** - 分布式通信
  - `device_communicators/` - 设备间通信器
  - 支持Tensor/Pipeline/Expert/Data并行

- **`disaggregation/`** - Prefill-Decode分离
  - 实现Prefill和Decode阶段的分离部署

- **`sampling/`** - 采样策略
  - `sampling_params.py` - 采样参数
  - `penaltylib/` - 惩罚库 (频率/存在惩罚)

- **`speculative/`** - 推测解码
  - 支持N-gram和模型辅助的推测解码

- **`lora/`** - LoRA支持
  - 多LoRA批处理支持

- **`multimodal/`** - 多模态处理
  - `processors/` - 各种视觉/音频处理器
  - 支持LLaVA、Qwen-VL、DeepSeek-VL等

#### 1.2 Frontend Language (前端语言)
**位置**: `python/sglang/lang/`

SGLang的前端编程语言实现，提供结构化生成能力。

- **`api.py`** - 前端API
- **`interpreter.py`** - 解释器
- **`ir.py`** - 中间表示
- **`tracer.py`** - 追踪器
- **`backend/`** - 后端实现

#### 1.3 CLI工具
**位置**: `python/sglang/cli/`

命令行工具：

- **`main.py`** - 主入口
- **`serve.py`** - 服务启动
- **`generate.py`** - 生成工具

#### 1.4 多模态生成
**位置**: `python/sglang/multimodal_gen/`

支持图像和视频生成：

- **`runtime/`** - 运行时
  - `pipelines/` - 生成管道
  - `models/` - 生成模型
  - `layers/` - 生成层
- **`configs/`** - 配置文件

### 2. sgl-kernel/ - CUDA内核库

**位置**: `/sglang/sgl-kernel/`

SGLang的底层CUDA内核实现，提供高性能计算原语。

#### 2.1 目录结构

```
sgl-kernel/
├── csrc/                    # C++/CUDA源代码
│   ├── attention/           # Attention内核
│   │   ├── cascade.cu       # Cascade attention
│   │   ├── cutlass_mla_kernel.cu  # CUTLASS MLA内核
│   │   └── lightning_attention_decode_kernel.cu  # Lightning attention
│   ├── gemm/               # GEMM (矩阵乘法) 内核
│   │   ├── 各种量化GEMM实现
│   │   └── FP4/FP8/INT8等格式支持
│   ├── elementwise/        # 逐元素操作
│   │   ├── activation.cu   # 激活函数
│   │   └── rmsnorm.cu       # RMS归一化
│   ├── moe/                # MoE (专家混合) 内核
│   │   ├── moe_fused_gate.cu
│   │   ├── moe_topk_softmax_kernels.cu
│   │   └── moe_sum.cu
│   ├── quantization/       # 量化相关
│   │   └── gguf/           # GGUF格式支持
│   ├── memory/             # 内存操作
│   ├── allreduce/          # AllReduce通信
│   ├── speculative/        # 推测解码
│   └── common_extension.cc # PyTorch扩展绑定
├── include/                 # 头文件
│   └── sgl_kernel_ops.h    # 内核操作接口
├── python/                  # Python绑定
│   └── sgl_kernel/
├── benchmark/              # 性能基准测试
└── tests/                  # 单元测试
```

#### 2.2 主要内核类型

1. **Attention内核**
   - Cascade Attention
   - CUTLASS MLA (Multi-head Latent Attention)
   - Lightning Attention Decode

2. **GEMM内核**
   - 支持多种精度: FP16, BF16, FP8, FP4, INT8, INT4
   - 量化GEMM: AWQ, GPTQ, GGUF
   - Blockwise GEMM for MoE

3. **MoE内核**
   - Fused Gate计算
   - TopK选择
   - Expert路由和聚合

4. **量化支持**
   - FP8 Blockwise量化
   - Per-token量化
   - Per-tensor量化

### 3. sgl-router/ - 路由网关

**位置**: `/sglang/sgl-router/`

用Rust编写的高性能路由网关，负责请求路由、负载均衡和流量管理。

#### 3.1 架构特点

- **控制平面**: Worker管理、服务发现、健康检查
- **数据平面**: HTTP/gRPC路由、负载均衡、缓存感知路由
- **可靠性**: 重试、熔断器、速率限制、队列管理

#### 3.2 主要模块

```
sgl-router/
├── src/
│   ├── main.rs                    # 主入口
│   ├── routers/                   # 路由器实现
│   │   ├── http_router.rs        # HTTP路由
│   │   ├── grpc_router.rs        # gRPC路由
│   │   └── openai_router.rs      # OpenAI兼容路由
│   ├── policies/                  # 负载均衡策略
│   │   ├── cache_aware.rs        # 缓存感知路由
│   │   ├── power_of_two.rs       # Power-of-Two选择
│   │   ├── round_robin.rs        # 轮询
│   │   └── random.rs             # 随机选择
│   ├── tokenizer/                 # Tokenizer实现
│   │   ├── huggingface.rs        # HuggingFace tokenizer
│   │   ├── tiktoken.rs           # Tiktoken支持
│   │   └── cache/                # Tokenizer缓存
│   ├── reasoning_parser/          # 推理解析器
│   │   └── 支持DeepSeek-R1、Qwen3等
│   ├── tool_parser/              # 工具调用解析器
│   ├── protocols/                # 协议实现
│   ├── mcp/                      # MCP (Model Context Protocol) 支持
│   └── data_connector/           # 数据连接器
│       └── oracle.rs             # Oracle数据库连接
├── bindings/                     # 语言绑定
│   ├── python/                   # Python绑定
│   └── go/                       # Go绑定
└── tests/                        # 测试
```

#### 3.3 核心功能

1. **负载均衡策略**
   - `random`: 随机选择
   - `round_robin`: 轮询
   - `cache_aware`: 缓存感知路由 (Radix树)
   - `power_of_two`: 从两个候选中选择负载较低的

2. **Prefill-Decode分离支持**
   - 支持Prefill和Decode worker分离部署
   - 自动路由到合适的worker

3. **gRPC路由**
   - 完全Rust实现的tokenizer
   - 推理解析器 (DeepSeek-R1、Qwen3等)
   - 工具调用解析器

4. **OpenAI兼容API**
   - `/v1/chat/completions`
   - `/v1/completions`
   - `/v1/embeddings`
   - `/v1/responses`
   - `/v1/conversations`

---

## 目录结构说明

### 根目录结构

```
sglang/
├── python/                    # Python运行时 (主要代码)
│   └── sglang/               # SGLang Python包
│       ├── srt/              # SGLang Runtime核心
│       ├── lang/             # 前端语言
│       ├── cli/              # CLI工具
│       └── multimodal_gen/   # 多模态生成
├── sgl-kernel/               # CUDA内核库
│   ├── csrc/                 # C++/CUDA源码
│   ├── include/              # 头文件
│   └── python/              # Python绑定
├── sgl-router/               # Rust路由网关
│   ├── src/                  # Rust源码
│   └── bindings/             # 语言绑定
├── benchmark/                # 基准测试
│   ├── kernels/              # 内核基准测试
│   └── 各种模型/任务基准测试
├── examples/                 # 示例代码
│   ├── frontend_language/    # 前端语言示例
│   ├── runtime/              # 运行时示例
│   └── profiler/             # 性能分析示例
├── docs/                     # 文档
├── test/                     # 测试套件
├── scripts/                  # 构建和工具脚本
├── docker/                   # Docker配置
└── 3rdparty/                 # 第三方依赖
```

### 关键配置文件

- **`python/pyproject.toml`** - Python包配置
- **`sgl-kernel/pyproject.toml`** - 内核库配置
- **`sgl-kernel/CMakeLists.txt`** - CMake构建配置
- **`sgl-router/Cargo.toml`** - Rust项目配置

---

## 关键模块分析

### 1. 内存缓存系统 (RadixAttention)

**位置**: `python/sglang/srt/mem_cache/`

RadixAttention是SGLang的核心优化之一，通过Radix树实现前缀缓存。

**关键组件**:
- `radix_cache.py` - Python实现
- `radix_cache_cpp.py` - C++绑定 (高性能)
- `cpp_radix_tree/` - C++ Radix树实现

**工作原理**:
1. 将输入提示构建为Radix树
2. 共享相同前缀的请求复用KV缓存
3. 支持动态插入和删除
4. LRU eviction策略

### 2. 批处理调度系统

**位置**: `python/sglang/srt/managers/`

**核心管理器**:

- **Scheduler** (`scheduler.py`): 核心调度器
  - 管理所有请求的生命周期
  - 协调Prefill和Decode阶段
  - 实现连续批处理 (Continuous Batching)
  - 动态批大小调整
  - 支持多种调度策略

- **CacheController** (`cache_controller.py`): 缓存控制器
  - Radix树维护
  - 缓存分配和释放
  - 前缀匹配和复用
  - LRU eviction策略

- **TokenizerManager** (`tokenizer_manager.py`): Tokenizer管理
  - 多Tokenizer支持
  - Token编码/解码
  - 与调度器通信

- **TemplateManager** (`template_manager.py`): 模板管理
  - Chat模板处理
  - 支持多种模板格式

- **DetokenizerManager** (`detokenizer_manager.py`): Detokenizer管理
  - Token到文本的转换
  - 流式输出处理

- **DataParallelController** (`data_parallel_controller.py`): 数据并行控制
  - 协调多个DP worker
  - 请求分发和结果聚合

- **SessionController** (`session_controller.py`): 会话控制
  - 多轮对话管理
  - 会话状态维护

### 3. 分布式通信

**位置**: `python/sglang/srt/distributed/`

支持多种并行策略：

- **Tensor Parallelism (TP)**: 模型并行
- **Pipeline Parallelism (PP)**: 流水线并行
- **Expert Parallelism (EP)**: 专家并行 (MoE)
- **Data Parallelism (DP)**: 数据并行

**通信后端**:
- NCCL (NVIDIA)
- RCCL (AMD)
- 自定义AllReduce实现

### 4. Prefill-Decode分离

**位置**: `python/sglang/srt/disaggregation/`

将Prefill和Decode阶段分离到不同的worker，提高资源利用率。

**优势**:
- Prefill worker专注于计算密集型任务
- Decode worker专注于内存带宽密集型任务
- 独立扩展和优化

### 5. 量化支持

**位置**: `python/sglang/srt/layers/quantization/`

支持多种量化格式：

- **FP8**: 8位浮点量化
- **FP4**: 4位浮点量化
- **INT8/INT4**: 整数量化
- **AWQ/GPTQ**: 权重量化
- **GGUF**: GGUF格式支持

### 6. MoE (Mixture of Experts) 支持

**位置**: `python/sglang/srt/layers/moe/`

**关键组件**:
- Expert路由
- TopK专家选择
- Expert并行通信
- Fused MoE内核

---

## 构建与部署

### 1. Python包构建

```bash
# 安装SGLang
pip install sglang

# 从源码构建
cd python
pip install -e .
```

### 2. sgl-kernel构建

```bash
cd sgl-kernel
make build

# 或使用CMake
cmake -B build
cmake --build build
```

### 3. sgl-router构建

```bash
cd sgl-router
cargo build --release

# Python绑定
cd bindings/python
maturin develop  # 开发模式
maturin build --release  # 生产构建
```

### 4. 服务启动

```bash
# HTTP模式
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --tp-size 1

# gRPC模式
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --grpc-mode \
    --tp-size 1

# 使用Router
python -m sglang_router.launch_router \
    --worker-urls http://worker1:8000 http://worker2:8000 \
    --policy cache_aware
```

### 5. Docker部署

```bash
# 构建镜像
docker build -f docker/b300.Dockerfile -t sglang:latest .

# 运行容器
docker run --gpus all sglang:latest
```

---

## 数据流和请求处理流程

### HTTP请求处理流程

```
客户端请求
    ↓
HTTP Server (http_server.py)
    ↓
OpenAI API Handler (serving_chat.py等)
    ↓
Engine.generate() (engine.py)
    ↓
Scheduler.schedule() (scheduler.py)
    ↓
CacheController (查找/创建缓存)
    ↓
ModelRunner.forward() (model_executor/)
    ↓
CUDA Kernels (sgl-kernel)
    ↓
返回结果给客户端
```

### gRPC请求处理流程

```
客户端请求
    ↓
gRPC Server (grpc_server.py)
    ↓
Tokenizer (Rust tokenizer in sgl-router)
    ↓
Engine.generate()
    ↓
(后续流程同HTTP)
```

### 批处理调度流程

1. **请求接收**: Scheduler接收新请求
2. **缓存查找**: CacheController在Radix树中查找前缀匹配
3. **批构建**: Scheduler将请求组织成批次
4. **Prefill阶段**: 处理新token的前向传播
5. **Decode阶段**: 迭代生成新token
6. **采样**: 根据采样参数选择下一个token
7. **结果返回**: 流式或批量返回结果

---

## 关键设计模式

### 1. 引擎模式 (Engine Pattern)

- **EngineBase**: 抽象基类，定义统一接口
- **Engine**: 具体实现，包含完整的推理逻辑
- 支持多种后端: CUDA、CPU、NPU、MindSpore

### 2. 管理器模式 (Manager Pattern)

- 每个资源类型有对应的Manager
- Manager负责资源的生命周期管理
- 通过Mixin实现功能组合

### 3. 调度器模式 (Scheduler Pattern)

- 核心调度器协调所有组件
- 支持多种调度策略
- 可扩展的调度策略接口

### 4. 缓存模式 (Cache Pattern)

- Radix树实现前缀缓存
- 多级缓存策略
- 可插拔的存储后端

---

## 性能优化技术

### 1. RadixAttention

- **原理**: 使用Radix树存储和共享KV缓存
- **优势**: 相同前缀的请求共享缓存，减少重复计算
- **实现**: C++ Radix树 + Python绑定

### 2. CUDA Graph

- **原理**: 捕获CUDA操作序列，减少CPU-GPU同步
- **优势**: 降低启动开销，提高吞吐量
- **实现**: `cuda_graph_runner.py`

### 3. 连续批处理 (Continuous Batching)

- **原理**: 动态调整批次，新请求可立即加入
- **优势**: 提高GPU利用率，降低延迟
- **实现**: `scheduler.py`中的动态批管理

### 4. Prefill-Decode分离

- **原理**: 将计算密集的Prefill和内存密集的Decode分离
- **优势**: 独立扩展，优化资源利用
- **实现**: `disaggregation/`模块

### 5. 量化优化

- **支持格式**: FP8、FP4、INT8、INT4
- **实现**: 专门的量化GEMM内核
- **优势**: 减少内存占用，提高速度

---

## 总结

SGLang是一个设计精良的高性能LLM服务框架，具有以下特点：

1. **分层架构**: Python运行时、CUDA内核、Rust路由网关清晰分离
2. **高性能优化**: RadixAttention、连续批处理、Prefill-Decode分离、CUDA Graph等
3. **广泛支持**: 多种模型、硬件平台、量化格式
4. **生产就绪**: 完整的监控、日志、分布式支持
5. **可扩展性**: 模块化设计，易于扩展新功能

### 核心优势

- **低延迟**: 通过RadixAttention和CUDA Graph优化
- **高吞吐**: 连续批处理和Prefill-Decode分离
- **易用性**: OpenAI兼容API，简单易用
- **可扩展**: 支持从单GPU到大规模集群

通过理解这些组件和模块，可以更好地使用和扩展SGLang来满足特定的推理需求。

---

## 参考资料

- [SGLang官方文档](https://docs.sglang.io/)
- [GitHub仓库](https://github.com/sgl-project/sglang)
- [SGLang博客](https://lmsys.org/blog/)
