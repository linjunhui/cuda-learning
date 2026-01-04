### 3.1 系统级执行流追踪 (Timeline Analysis)

`nsys profile -t cuda,nvtx --stats=true python test_softmax.py`
主要是统计 CUDA API 的执行情况
* **说明**：使用 **Nsight Systems** 观测程序大局。主要检查 Python 调用 CUDA Kernel 的开销（Overhead）、Kernel 之间是否存在不必要的空隙（Gaps），以及是否有意外的同步（Sync）阻塞了流水线。它能帮你确认 GPU 是否在“持续干活”而没有在等 CPU 指令。

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)    Med (ns)   Min (ns)  Max (ns)   StdDev (ns)              Name            
 --------  ---------------  ---------  ----------  ----------  --------  ---------  -----------  ----------------------------
     69.8        395262022          6  65877003.7  34602284.0   1495371  221676813   86591553.1  cudaDeviceSynchronize       
     27.5        155572857       3111     50007.3     12628.0      6878   23836428     752277.9  cudaLaunchKernel            
      1.3          7160350         15    477356.7    462241.0    374820     736095      87666.7  cudaMalloc                  
      0.6          3260290          3   1086763.3    714459.0     54616    2491215    1260242.7  cudaStreamSynchronize       
      0.5          2919899          1   2919899.0   2919899.0   2919899    2919899          0.0  cudaHostAlloc               
      0.1           742213         21     35343.5       799.0       340     717994     156416.7  cudaStreamIsCapturing_v10000
      0.1           482226          3    160742.0     30757.0     20755     430714     233856.1  cudaMemsetAsync             
      0.1           479299          6     79883.2      3793.5      3003     461196     186805.6  cudaEventCreateWithFlags    
      0.0           112439          3     37479.7     32337.0     30628      49474      10422.5  cudaMemcpyAsync             
      0.0            93426          6     15571.0     15263.5     11305      21031       3133.7  cudaEventRecord             
      0.0             8007          6      1334.5      1308.5       422       2305        980.8  cudaEventDestroy            
      0.0             1322          1      1322.0      1322.0      1322       1322          0.0  cuModuleGetLoadingMode   
通过 Nsight Systems UI (图形界面) 打开, 可以看到timeline,、kernel 开始结束时间、CPU和GPU重叠时间、kernel直接的间隔
    /home/jonson/cuda-learning/CUDA项目实战/softmax/report1.nsys-rep
    /home/jonson/cuda-learning/CUDA项目实战/softmax/report1.sqlite

### 3.2 算子级深度特征刻画 (Full Kernel Analysis)

`ncu --set full --target-processes all -c 100 -o ncu_report4 python test_softmax.py`

* **说明**：使用 **Nsight Compute** 对 Kernel 进行全量指标抓取。它会生成一份详尽的报告，包含寄存器使用量、共享内存配置、Warp 状态统计等。这是定位** Bank Conflict** 或**计算/访存停顿（Stalls）**最权威的手段。生成的 `.ncu-report` 文件可以下载到本地用图形化界面打开。

#### 1. 核心性能列
* **Estimated Speedup (%)**:
* **含义**：这是 NCU 的专家系统根据当前瓶颈给出的“优化潜力股”。
* **解读**：如果这个百分比较高（比如 > 10%），说明该算子存在明显的低效点（如访存未合并或 Bank Conflict）。点击算子详情，NCU 会告诉你具体怎么改能拿到这部分收益。

* **Duration (us)**:
* **含义**：核函数在 GPU 上运行的**净时间**。
* **解读**：这是最直观的性能指标。你的 `online_safe_softmax` (序号 2) 耗时 97.76µs。注意对比序号 1 的 `softmax_warp_forward` 耗时 31.74µs，说明你的实战版本还有较大的提升空间（约 3 倍差距）。


### 2. 吞吐量列（决定你是计算密集还是访存密集）
* **Compute Throughput (%)**:
* **含义**：计算单元（SM）的利用率。
* **解读**：代表 GPU 里的计算核心有多忙。你的 `online_safe_softmax` 是 **43.93%**，而 `softmax_warp_forward` 只有 **33.83%**。

* **Memory Throughput (%)**:
* **含义**：显存带宽的利用率。
* **解读**：**这是 Softmax 最关键的指标。** 你的算子只有 **33.83%**，而优化过的 `warp_forward` 达到了 **85.71%**。
* **结论**：你的算子目前没能喂饱显存带宽。优化目标是让这个值冲向 80% 以上。

### 3. 资源开销列（决定并发度）
* **# Registers**:
* **含义**：每个线程占用的寄存器数量。
* **解读**：你的算子用了 **51** 个。1660 Super 每个 SM 的寄存器总量有限，寄存器用得越多，能够同时并行的线程块（Occupancy）就越少。50+ 个属于中等偏高，可能限制了并发。
