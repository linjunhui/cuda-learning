# 工程化方案
## 工程结构
- test_softmax.py 
- bindings.cpp
- setup.py
- softmax_kernel.cu

## Step1 test_softmax.py 编写
题目分析：输入序列长度为(N, D)
- N 序列 长度, Row
- D dim 维度, Col
- 测试数据设计 (N, D) -> (512, 512), (1024, 512), (1024, 1024), (1024, 2048), (2048, 2048), (1, 131072)
- torch测试代码
```
torch.testing.assert_close(
    actual,           # 你自己算子算出的结果 (Tensor)
    expected,         # 官方/基准算子的结果 (Tensor)
    atol=1e-3,        # 绝对误差 (Absolute tolerance): |a - b| <= atol
    rtol=1e-3,        # 相对误差 (Relative tolerance): |a - b| <= rtol * |b|
    check_device=True,# 是否检查两个张量都在同一块显卡上（推荐开启）
    check_dtype=True, # 是否检查数据类型一致（如都是 float16）
    msg="Softmax Error" # 自定义报错前缀
)
```

## Step2 float4向量化 基础知识
### 类型转换 float -> float4
- 1. 用 reinterpret_cast来重解释指针
- 1.1  `#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])`  value 是数组的一个元素，取value的地址，再重新解释, 这里最后还有个[0] 表示取第0个元素，那么最终的结果是一个float4 而不是 float4 *
- 1.2  取元素value时跨步(stride=4)取
### float4 元素操作
- 1. 和普通类型一样 float4 reg_a = FLOAT4(input[idx])
- 2. float4  元素读取(reg_a.x, reg_a.y, reg_a.z, reg_a.w)

### 输入数据对齐问题
- 输入 数据形状 (N, D), 当D不是4的倍数时，可能跨行访问，也可能访问非法内存
- 判断D的大小，进行填充 torch.nn.functional.constant_pad_nd(input, pad, value=0.0)
- pad = (左，右，上，下), 表示在每个方向填充的个数， value 设置为-FLT_MAX，避免影响softmax分母计算

## Step2 核函数设计
- 这里考虑的场景是 (N, D) 的矩阵，按行计算，一个Block计算一行的softmax
### online softmax的原理
- 1. 不求解全局的最大值，计算局部(区间)的分母和最大值
- 2. 区间逐渐合并
- 2.1 两个区间合并 ，会更新最大值，要对原区间求得的分母进行缩放
### online softmax实现思路
- 1. 按照warp 32个线程来划分区间
- 1.1 32个元素，起始一个元素算一个区间，元素两两合并区间，求得 改区间的分母和最大值
- 1.2 warp reduce完成就得到了，warp_num个 分母和最大值
- 1.3 考虑一个Block的线程数不超过1024，那么warp_num也不会超过32，所以这里可以复用 warp reduce来计算，整个block的分母和最大值
- 2. 求解 softmax的值
- 2.1 用shared memory 将分母和最大值共享给block中的所有线程

### 具体实现
#### warp reduce
- 1. 定义 struct MD; 分母(exp_sum) 和 最大值(max_value)
- 2. warp reduce 要循环5次
- 3. __shfl_xor_sync 传入的值是 MD
- 4. 计算逻辑
- 4.1 比较当前线程的MD1和传过来的MD2, max_value比较大的，MD不用动, 这里假设MD2的max_value较大
- 4.2 缩放max_value较小区间的 exp_sum, 简单推理(e^-max1 -> e^-max2) 那么 原分母exp_sum 要乘上一个因子 e^(-max2+max1)
- 4.3 两个区间分母相加 MD1.exp_sum * e^(MD1.max_value - MD2.max_value) + MD2.exp_sum, max_value 用MD2的，返回新的MD

#### online softmax kernel
- 1. 首先输入 input 是一个 (N, D) 的矩阵，当前block只处理一行，所以先获取blockIdx.x 来读取当前block要处理的行
- 2. 获取threadIdx.x ，从当前行取出当前线程要处理的数据，考虑 D > BLOCK_SIZE，一行元素个数比线程数量多
- 2.0 初始化 shared memory , 元素个数warp_num就行，用于后面的block内规约，这里for循环的计算用寄存器就行
- 2.1 计算每个线程要处理的元素数量， elements_per_thread = (D + BLOCK_SIZE - 1) / BLOCK_SIZE
- 2.2 这里使用 stride_size = BLOCK_SIZE 来进行跨步处理数据，带宽利用率高，一个取数据把一个warp的数据都取了
- 2.3 这里直接在for循环里面进行计算，原因：BLOCK_SIZE = 1024, 即使D=1024*1024 也就循环1024次，对于GPU来说非常快
- 2.4 当计算次数超 10^5次，考虑block并行，这里记录，先不做
- 2.5 构建MD，边界内 exp_sum 和 max_value可以用当前元素来设置(1, x), 边界外(0, -FLT_MAX); 这里的1可以看作区间就一个元素x, 最大值也是x ,e^(x-x) = 1; 边界外的按道理是不参与softmax就算的将max_value设置为很大的负数，再进行比较时 不影响两个区间合并时的max_value, exp_sum = 0 也不影响分母的结果
- 2.6 做完以上计算之后，数据个数变成 BLOCK_SIZE个， 注意边界 D < BLOCK_SIZE, 超出边界的数据处理
- 3. warp reduce 计算出每个warp的 MD, 存储到shared memory
- 4. warp_num 不会超过32(一个block最多1024个线程，warp_size=32), 再执行一次warp reduce, 得到一行的MD，存储到shared memory给所有线程用
- 5. 计算每个元素的softmax值，for循环，逻辑同2.2~2.3，跨步循环，计算每

## Step x 包装函数
- 1. 使用dispatch思想，当前只处理 D 是4的倍数的情况
- 2. vec.data_ptr<float>() 转 float4 *, ``

## Step3 Profiling
你的 Step3 规划已经非常专业了，这三条指令分别代表了从**宏观执行流**到**微观硬件瓶颈**，再到**核心性能指标**的递进分析。

以下是补全后的标题及简要说明，你可以直接更新到你的工程化方案中：

---

## Step3 Profiling (性能分析)

### 3.1 系统级执行流追踪 (Timeline Analysis)

`nsys profile -t cuda,nvtx --stats=true python test_softmax.py`

* **说明**：使用 **Nsight Systems** 观测程序大局。主要检查 Python 调用 CUDA Kernel 的开销（Overhead）、Kernel 之间是否存在不必要的空隙（Gaps），以及是否有意外的同步（Sync）阻塞了流水线。它能帮你确认 GPU 是否在“持续干活”而没有在等 CPU 指令。

### 3.2 算子级深度特征刻画 (Full Kernel Analysis)

`ncu --set full --target-processes all -o ncu_report python test_softmax.py`

* **说明**：使用 **Nsight Compute** 对 Kernel 进行全量指标抓取。它会生成一份详尽的报告，包含寄存器使用量、共享内存配置、Warp 状态统计等。这是定位** Bank Conflict** 或**计算/访存停顿（Stalls）**最权威的手段。生成的 `.ncu-report` 文件可以下载到本地用图形化界面打开。

### 3.3 核心瓶颈极限测试 (Speed-of-Light / SOL Analysis)

`ncu --metrics gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_sol_limit_sustained_elapsed.avg.pct_of_peak_sustained_elapsed python test_softmax.py`

* **说明**：**直击痛点**。通过这两个关键指标，直接计算算子达到了硬件理论带宽极限的百分之多少。
* `throughput`：反映实际有效利用的带宽。
* `sol_limit`：反映硬件层面的繁忙程度。
* 对于 Softmax 这种访存密集型任务，如果这两个百分比接近（且在高位），说明你的算子已经完全榨干了 1660 Super 的显存带宽。
