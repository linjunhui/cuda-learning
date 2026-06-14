/**
 * ALiBi (Attention with Linear Biases) CUDA 实现
 * 
 * ALiBi 是一种位置编码方法，通过在注意力分数中添加线性偏置来替代传统的位置编码。
 * 核心思想：对于位置 i 和 j，偏置为 -alpha * |i - j|，其中 alpha 是学习到的斜率参数。
 * 
 * 算法流程：
 * 1. 计算注意力分数：S = Q * K^T + ALiBi_Bias
 * 2. 对每行应用 Softmax：P = softmax(S)
 * 3. 加权求和：Output = P * V
 * 
 * 并行化策略：
 * - 每个 CUDA Block 处理输出矩阵的一行（对应一个 query）
 * - 使用 Shared Memory 存储该行的注意力分数，加速 Softmax 计算
 * - 使用 Warp-level Reduce + Shared Memory 混合归约算法完成归约操作（最大值、求和）
 *   - 先在 warp 内使用 shuffle 指令进行归约（无需共享内存和同步）
 *   - 再在 warp 间使用共享内存进行归约
 *   - 相比原子操作和纯共享内存归约，性能更好，内存占用更少
 * 
 * 设计合理性（从 Attention 公式角度）：
 * 
 * 标准的 Attention 公式：Output = softmax(QK^T / √d) · V
 * 
 * 对于输出矩阵的第 i 行，展开为：
 *   Output[i] = softmax(Q[i] · K^T) · V
 * 
 * 计算步骤：
 * 1. 注意力分数：S[i] = Q[i] · K^T
 *    - Q[i] 是 1×d 向量（Q 的第 i 行）
 *    - K^T 是 d×N 矩阵（K 的转置）
 *    - 结果 S[i] 是 1×N 向量：S[i][j] = Q[i] · K[j]
 * 
 * 2. Softmax 归一化：P[i] = softmax(S[i])
 *    - 对 S[i]（1×N 向量）逐行做 softmax
 *    - 结果 P[i] 是 1×N 向量（归一化的注意力权重）
 * 
 * 3. 加权求和：Output[i] = P[i] · V
 *    - P[i] 是 1×N 向量
 *    - V 是 N×d 矩阵
 *    - 结果 Output[i] 是 1×d 向量：Output[i][k] = Σ(j) P[i][j] * V[j][k]
 * 
 * 关键观察：行独立性
 * - Output[i] 只依赖 Q[i]（Q 的一行），不需要其他 Q[j]（j ≠ i）
 * - Output[i] 需要所有 K[0..N-1] 和 V[0..N-1]（因为要计算 Q[i]·K^T 和 P[i]·V）
 * - 不同行之间完全独立：
 *     Output[0] = f(Q[0], K[0..N-1], V[0..N-1])
 *     Output[1] = f(Q[1], K[0..N-1], V[0..N-1])
 *     Output[2] = f(Q[2], K[0..N-1], V[0..N-1])
 *   - Output[i] 和 Output[j]（i ≠ j）互不依赖，可以完全并行计算
 * 
 * 因此，一个 Block 处理一行的设计是合理的：
 * 1. 数学独立性：每行输出是独立的函数计算，符合公式的数学结构
 * 2. 数据访问模式：每个 Block 读取 1 行 Q + N 行 K + N 行 V，不同 Block 共享 K 和 V
 * 3. Softmax 局部性：Softmax 是逐行操作，只需要该行的 S[i]（N 个元素），适合放在 Shared Memory
 */

#include <iostream>      // 标准输入输出流
#include <vector>        // 动态数组容器
#include <cmath>         // 数学函数（expf 等）
#include <cuda_runtime.h> // CUDA 运行时 API
#include <float.h>        // 浮点数常量（FLT_MAX）

/**
 * ALiBi 注意力计算的 CUDA Kernel
 * 
 * 参数说明：
 * @param Q: Query 矩阵，形状为 [M, d]，存储在全局内存中
 * @param K: Key 矩阵，形状为 [N, d]，存储在全局内存中
 * @param V: Value 矩阵，形状为 [N, d]，存储在全局内存中
 * @param output: 输出矩阵，形状为 [M, d]，存储计算结果
 * @param M: Query 的数量（输出矩阵的行数）
 * @param N: Key/Value 的数量（序列长度）
 * @param d: 特征维度（每个 Query/Key/Value 向量的长度）
 * @param alpha: ALiBi 偏置的斜率参数，控制位置偏置的强度
 * 
 * 线程组织：
 * - Grid 维度：M 个 Blocks（每个 Block 处理一行输出）
 * - Block 维度：256 个线程（可配置）
 * - Shared Memory：每个 Block 分配 N * sizeof(float) 字节，存储该行的注意力分数
 * 
 * 为什么一个 Block 处理一行？
 * 从 Attention 公式 Output[i] = softmax(Q[i] · K^T) · V 可以看出：
 * - 每行输出只依赖 Q 的一行（Q[i]），不同行之间完全独立
 * - Softmax 是逐行操作，只需要该行的注意力分数（N 个元素）
 * - 这种设计充分利用了行独立性，最大化并行度，同时利用 Shared Memory 加速 Softmax
 */
__global__ void alibi_kernel(const float* Q, const float* K, const float* V, float* output, 
                             int M, int N, int d, float alpha) {
    // ========== 第一步：确定当前 Block 处理的行 ==========
    // blockIdx.x 是当前 Block 在 Grid 中的 x 方向索引（0 到 M-1）
    // 每个 Block 负责计算输出矩阵的一行，对应一个 Query 向量
    int m = blockIdx.x;  // Q的行 index
    
    // 边界检查：防止 Block 索引超出有效范围（当 M 不是线程块大小的整数倍时）
    if (m >= M) return;

    // ========== 第二步：声明 Shared Memory ==========
    // extern __shared__ 表示动态分配的共享内存，大小在 kernel 启动时指定
    // S[] 用于存储当前行（第 m 行）的注意力分数 S[m][0..N-1]
    // 大小 = N * sizeof(float)，在调用 kernel 时通过第三个参数指定
    // Shared Memory 的优势：
    // 1. 访问速度比全局内存快约 100 倍
    // 2. 同一 Block 内的所有线程可以共享数据
    // 3. 适合存储需要多次访问的中间结果
    extern __shared__ float S[]; 

    // ========== 第三步：获取线程信息 ==========
    // threadIdx.x: 当前线程在 Block 内的 x 方向索引（0 到 blockDim.x-1）
    // blockDim.x: Block 内的线程总数（本例中为 256）
    int tid = threadIdx.x;  // 当前的行内索引
    int num_threads = blockDim.x; // 当前block的线程数量

    // ========== 第四步：计算注意力分数 S = Q * K^T + ALiBi_Bias ==========
    // 对于输出矩阵的第 m 行，需要计算：
    // S[m][j] = Q[m] · K[j] + alpha * (m - j)
    // 其中 Q[m] · K[j] 是向量点积，alpha * (m - j) 是 ALiBi 线性偏置
    // 
    // 并行化策略：使用循环步进（stride loop）模式
    // 每个线程处理多个列（j），线程 0 处理 j=0, 256, 512...
    // 线程 1 处理 j=1, 257, 513...，以此类推
    // 这样可以充分利用所有线程，避免线程空闲
    /*
     输出的形状 是 M x d
     Q * K^T -> 形状 M x N
     当前线程计算的元素坐标是 (m, j)
     Q: 形状 M x d
     K^T: 形状 d * N
     设计的是当前m行的计算，就是Q的m行，与 K^T的每一列都要计算一次，输出N个元素
     那么现在要考虑的就是： 当前block 是num_threads个线程 处理 N个元素的计算输出

     0 <= tid < num_threads, 跨步处理数据stride num_threads

     在tid上累加num_threads 直到N为止
     
     现在就是 计算 (m, j)  Q的m行 和 K 的j行
    */
    for (int j = tid; j < N; j += num_threads) {
        // 计算 Q[m] 和 K[j] 的点积
        // Q[m] 在全局内存中的位置：Q[m * d + k]，k 从 0 到 d-1
        // K[j] 在全局内存中的位置：K[j * d + k]，k 从 0 到 d-1
        float sum = 0.0f;
        for (int k = 0; k < d; ++k) {
            // 向量点积：sum += Q[m][k] * K[j][k]
            sum += Q[m * d + k] * K[j * d + k];
        }
        
        // 添加 ALiBi 线性偏置
        // 偏置公式：-alpha * |m - j|
        // 当 m >= j 时（当前位置在目标位置之后或相同），偏置为 -alpha * (m - j)
        // 这会让模型更关注较近的位置，较远的位置注意力分数会被降低
        // 注意：这里使用 (m - j) 而不是 |m - j|，因为通常 m >= j（因果注意力）
        S[j] = sum + alpha * (float)(m - j);
    }
    
    // 同步屏障：确保所有线程都完成了注意力分数的计算
    // 在访问 Shared Memory 之前，必须确保所有写入操作都已完成
    // __syncthreads() 会阻塞当前线程，直到同一 Block 内的所有线程都到达此点
    __syncthreads();

    // ========== 第五步：Row-wise Softmax 计算 ==========
    // Softmax 公式：P[i] = exp(S[i] - max(S)) / sum(exp(S[j] - max(S)))
    // 为了数值稳定性，使用 "safe softmax" 方法：
    // 1. 先找到最大值 max(S)
    // 2. 计算 exp(S[i] - max(S))，避免 exp 溢出
    // 3. 归一化：除以 sum(exp(S[j] - max(S)))

    // ---------- 5a. 找最大值（Safe Softmax 的第一步）----------
    // 每个线程先在自己的工作范围内找局部最大值
    float local_max = -FLT_MAX;  // FLT_MAX 是 float 类型的最大正值
    for (int j = tid; j < N; j += num_threads) {
        if (S[j] > local_max) local_max = S[j];
    }
    
    // 使用 Warp-level Reduce + Shared Memory 混合归约算法
    // 优势：
    // 1. Warp shuffle 在同一个 warp（32个线程）内进行，无需共享内存和同步
    // 2. 减少共享内存访问：只有每个 warp 的第一个线程写入共享内存
    // 3. 减少同步次数：只需要一次跨 warp 的同步
    // 4. 性能更好：warp shuffle 是硬件加速的，延迟极低
    // 
    // 算法流程：
    // 1. 在每个 warp 内使用 shuffle 指令进行归约（32个线程 -> 1个值）
    // 2. 每个 warp 的第一个线程将结果写入共享内存
    // 3. 在共享内存中进行跨 warp 的归约（8个 warp -> 1个值）
    
    // 第一步：Warp 内归约（使用 shuffle 指令）
    // warpSize = 32，是 CUDA 的编译时常量
    // 对于 256 个线程，有 8 个 warp（256 / 32 = 8）
    // 
    // __shfl_down_sync 是 CUDA 9.0+ 引入的 warp shuffle 指令
    // 功能：从当前线程向下偏移 offset 个位置的线程获取值
    // 语法：__shfl_down_sync(mask, var, offset)
    // - mask: warp 内参与 shuffle 的线程掩码（0xFFFFFFFF 表示所有 32 个线程）
    // - var: 要 shuffle 的变量
    // - offset: 偏移量（1, 2, 4, 8, 16）
    const unsigned int warp_mask = 0xFFFFFFFF;  // 所有 32 个线程都参与
    const int lane_id = tid % 32;  // 当前线程在 warp 内的索引（0-31）
    const int warp_id = tid / 32;  // 当前线程所属的 warp 索引（0-7）
    
    // 树形归约：在 warp 内进行 5 次 shuffle（32 -> 16 -> 8 -> 4 -> 2 -> 1）
    // 每次将距离为 offset 的两个值进行比较，取较大者
    for (int offset = 16; offset > 0; offset >>= 1) {
        // 从距离 offset 的位置获取值
        float val = __shfl_down_sync(warp_mask, local_max, offset);
        // 比较并更新最大值
        if (val > local_max) local_max = val;
    }
    
    // 第二步：将每个 warp 的归约结果写入共享内存
    // 只有每个 warp 的第一个线程（lane_id == 0）需要写入
    // 共享内存大小只需要存储 warp 数量（8个），而不是所有线程（256个）
    __shared__ float warp_max[8];  // 8 个 warp，每个存储一个最大值
    
    if (lane_id == 0) {
        warp_max[warp_id] = local_max;
    }
    
    // 同步：确保所有 warp 都写入了结果
    __syncthreads();
    
    // 第三步：跨 warp 归约（在共享内存中进行）
    // 只有第一个 warp 的线程参与（warp_id == 0）
    if (warp_id == 0) {
        // 从共享内存读取所有 warp 的归约结果
        local_max = (lane_id < (num_threads / 32)) ? warp_max[lane_id] : -FLT_MAX;
        
        // 在第一个 warp 内再次进行 shuffle 归约
        for (int offset = 16; offset > 0; offset >>= 1) {
            float val = __shfl_down_sync(warp_mask, local_max, offset);
            if (val > local_max) local_max = val;
        }
        
        // 将最终结果写回共享内存的第一个位置，供所有线程读取
        if (lane_id == 0) {
            warp_max[0] = local_max;
        }
    }
    
    // 同步：确保跨 warp 归约完成
    __syncthreads();
    
    // 第四步：所有线程从共享内存读取最终的最大值
    float final_max = warp_max[0];

    // ---------- 5b. 计算 Exp 和归一化分母（Softmax 的第二步和第三步）----------
    // 计算 exp(S[j] - final_max) 并求和
    float local_sum = 0.0f;
    for (int j = tid; j < N; j += num_threads) {
        // 计算 exp(S[j] - final_max) 并直接更新 S[j]
        // 这样做的好处是：S[j] 现在存储的是归一化前的概率值
        // 数值稳定性：减去最大值后，exp 的输入是负数或零，避免溢出
        S[j] = expf(S[j] - final_max);
        
        // 累加局部和，用于后续计算归一化分母
        local_sum += S[j];
    }
    
    // 使用 Warp-level Reduce + Shared Memory 混合归约算法进行求和
    // 与最大值归约使用相同的策略，但操作是累加而不是比较
    
    // 第一步：Warp 内归约（使用 shuffle 指令）
    // 在 warp 内进行树形归约求和
    for (int offset = 16; offset > 0; offset >>= 1) {
        // 从距离 offset 的位置获取值并累加
        float val = __shfl_down_sync(warp_mask, local_sum, offset);
        local_sum += val;
    }
    
    // 第二步：将每个 warp 的归约结果写入共享内存
    // 复用之前用于最大值归约的共享内存数组
    __shared__ float warp_sum[8];  // 8 个 warp，每个存储一个和
    
    if (lane_id == 0) {
        warp_sum[warp_id] = local_sum;
    }
    
    // 同步：确保所有 warp 都写入了结果
    __syncthreads();
    
    // 第三步：跨 warp 归约（在共享内存中进行）
    // 只有第一个 warp 的线程参与（warp_id == 0）
    if (warp_id == 0) {
        // 从共享内存读取所有 warp 的归约结果
        local_sum = (lane_id < (num_threads / 32)) ? warp_sum[lane_id] : 0.0f;
        
        // 在第一个 warp 内再次进行 shuffle 归约求和
        for (int offset = 16; offset > 0; offset >>= 1) {
            float val = __shfl_down_sync(warp_mask, local_sum, offset);
            local_sum += val;
        }
        
        // 将最终结果写回共享内存的第一个位置，供所有线程读取
        if (lane_id == 0) {
            warp_sum[0] = local_sum;
        }
    }
    
    // 同步：确保跨 warp 归约完成
    __syncthreads();
    
    // 第四步：所有线程从共享内存读取最终的归一化分母
    float denom = warp_sum[0];

    // ========== 第六步：计算 Softmax(S) * V ==========
    // 这是注意力机制的最终步骤：使用归一化后的注意力权重对 Value 向量进行加权求和
    // Output[m][k] = sum_j(P[m][j] * V[j][k])
    // 其中 P[m][j] = S[j] / denom（归一化后的注意力权重）
    // 
    // 并行化策略：每个线程计算输出向量的若干维度
    // 线程 0 处理 k=0, 256, 512...
    // 线程 1 处理 k=1, 257, 513...
    for (int k = tid; k < d; k += num_threads) {
        float res = 0.0f;
        
        // 对所有的 Value 向量进行加权求和
        // S[j] / denom 是归一化后的注意力权重 P[m][j]
        // V[j * d + k] 是第 j 个 Value 向量的第 k 维
        for (int j = 0; j < N; ++j) {
            res += (S[j] / denom) * V[j * d + k];
        }
        
        // 将结果写入全局内存
        // output[m * d + k] 是输出矩阵第 m 行第 k 列的元素
        output[m * d + k] = res;
    }
    
    // 注意：这里不需要最后的 __syncthreads()
    // 因为每个线程写入的是不同的全局内存位置，不存在竞争条件
}

/**
 * 封装好的 solve 函数
 * 
 * 这是一个 C 接口函数，用于从外部调用 ALiBi 注意力计算
 * extern "C" 表示使用 C 链接约定，避免 C++ 名称修饰（name mangling）
 * 这样可以从 C 代码或其他语言（如 Python）中调用此函数
 * 
 * 参数说明：
 * @param Q: Query 矩阵，形状为 [M, d]，存储在 GPU 全局内存中
 * @param K: Key 矩阵，形状为 [N, d]，存储在 GPU 全局内存中
 * @param V: Value 矩阵，形状为 [N, d]，存储在 GPU 全局内存中
 * @param output: 输出矩阵，形状为 [M, d]，用于存储计算结果
 * @param M: Query 的数量（输出矩阵的行数）
 * @param N: Key/Value 的数量（序列长度）
 * @param d: 特征维度
 * @param alpha: ALiBi 偏置的斜率参数
 * 
 * 注意：此函数假设所有输入数据已经在 GPU 内存中，不会进行内存拷贝
 */
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, 
                      int M, int N, int d, float alpha) {
    // ========== 配置 Kernel 启动参数 ==========
    
    // threads: 每个 Block 的线程数
    // 选择 256 的原因：
    // 1. 256 是 CUDA 架构中常见的线程块大小（32 的倍数，符合 warp 大小）
    // 2. 足够多的线程可以隐藏内存访问延迟
    // 3. 不会超过大多数 GPU 的每个 Block 最大线程数限制（通常是 1024）
    // 4. 为 Shared Memory 使用留出空间（每个 Block 的 Shared Memory 有限）
    int threads = 256; 
    
    // blocks: Grid 中的 Block 数量
    // 每个 Block 处理输出矩阵的一行，所以需要 M 个 Blocks
    int blocks = M;
    
    // shared_mem_size: 每个 Block 需要的动态 Shared Memory 大小（字节）
    // 用于存储该行的注意力分数 S[m][0..N-1]
    // 
    // 注意：总共享内存使用量 = 动态共享内存 + 静态共享内存
    // - 动态共享内存：N * sizeof(float) 字节（用于 S[]）
    // - 静态共享内存：8 * sizeof(float) = 32 字节（用于 warp 归约结果的 warp_max[] 和 warp_sum[]）
    // - 总共享内存：N * 4 + 32 字节
    // 
    // 优化说明：使用 warp-level reduce 后，静态共享内存从 256 * 4 = 1024 字节减少到 8 * 4 = 32 字节
    // 这是因为只需要存储每个 warp 的归约结果（8 个 warp），而不是所有线程的结果（256 个线程）
    // 
    // 限制：总共享内存不能超过每个 Block 的 Shared Memory 限制（通常为 48KB 或 96KB）
    // 例如：如果 N = 1024，则总共享内存 = 1024 * 4 + 32 = 4128 字节（约 4KB），这在大多数 GPU 上都可以接受
    size_t shared_mem_size = N * sizeof(float);

    // ========== 启动 CUDA Kernel ==========
    // Kernel 启动语法：kernel_name<<<grid_size, block_size, shared_mem_size>>>(参数列表)
    // 
    // <<<blocks, threads, shared_mem_size>>> 是 CUDA 的执行配置（execution configuration）
    // - 第一个参数：Grid 维度（Block 数量）
    // - 第二个参数：Block 维度（每个 Block 的线程数）
    // - 第三个参数：每个 Block 的 Shared Memory 大小（可选，字节数）
    // 
    // 执行流程：
    // 1. GPU 调度器会创建 M 个 Blocks
    // 2. 每个 Block 包含 256 个线程
    // 3. 每个 Block 分配 shared_mem_size 字节的 Shared Memory
    // 4. 所有 Blocks 并行执行（受 GPU 硬件资源限制）
    alibi_kernel<<<blocks, threads, shared_mem_size>>>(Q, K, V, output, M, N, d, alpha);
    
    // ========== 同步等待 Kernel 完成 ==========
    // cudaDeviceSynchronize() 会阻塞 CPU 线程，直到所有 GPU 操作完成
    // 
    // 为什么需要同步？
    // 1. CUDA Kernel 启动是异步的，CPU 不会等待 GPU 完成就继续执行
    // 2. 如果后续代码需要使用计算结果，必须等待 Kernel 完成
    // 3. 对于测试程序，同步可以确保错误检查的准确性
    // 
    // 注意：在生产代码中，如果不需要立即使用结果，可以省略同步以提高性能
    // 可以使用 CUDA Streams 来管理异步操作
    cudaDeviceSynchronize();
}

/**
 * Main 测试程序
 * 
 * 这个函数演示了如何使用 ALiBi Kernel 进行注意力计算
 * 完整的 CUDA 程序流程包括：
 * 1. 在 Host（CPU）内存中准备数据
 * 2. 在 Device（GPU）内存中分配空间
 * 3. 将数据从 Host 拷贝到 Device
 * 4. 启动 Kernel 进行计算
 * 5. 将结果从 Device 拷贝回 Host
 * 6. 释放 GPU 内存资源
 */
int main() {
    // ========== 第一步：定义测试参数 ==========
    // 设定矩阵维度（符合约束条件）
    // M: Query 数量，对应输出矩阵的行数
    // N: Key/Value 数量，对应序列长度
    // d: 特征维度，每个 Query/Key/Value 向量的长度
    // alpha: ALiBi 偏置斜率，控制位置偏置的强度
    // 
    // 注意：这些值的选择需要考虑：
    // 1. Shared Memory 限制：N * sizeof(float) 不能超过每个 Block 的 Shared Memory 限制
    // 2. 全局内存限制：确保 GPU 有足够的内存存储所有矩阵
    // 3. 计算复杂度：M * N * d 决定了计算量
    const int M = 32;
    const int N = 32;
    const int d = 64;
    const float alpha = 0.5f;

    // 计算每个矩阵需要的内存大小（字节）
    // 使用 size_t 类型，因为内存大小可能很大
    size_t size_q = M * d * sizeof(float);  // Query 矩阵：M 行 × d 列 × 4 字节
    size_t size_k = N * d * sizeof(float);  // Key 矩阵：N 行 × d 列 × 4 字节
    size_t size_v = N * d * sizeof(float);  // Value 矩阵：N 行 × d 列 × 4 字节
    size_t size_out = M * d * sizeof(float); // 输出矩阵：M 行 × d 列 × 4 字节

    // ========== 第二步：分配 Host 内存并初始化数据 ==========
    // Host 内存：CPU 可访问的内存（RAM）
    // 使用 std::vector 自动管理内存，避免手动 new/delete
    // 
    // 命名约定：
    // - h_ 前缀表示 Host 内存（如 h_Q, h_K）
    // - d_ 前缀表示 Device 内存（如 d_Q, d_K）
    std::vector<float> h_Q(M * d), h_K(N * d), h_V(N * d), h_O(M * d);
    
    // 使用随机数初始化输入矩阵
    // rand() 返回 0 到 RAND_MAX 之间的整数
    // (float)rand() / RAND_MAX 将其归一化到 [0, 1] 区间
    // 
    // 注意：这里没有设置随机种子，每次运行结果可能不同
    // 在实际应用中，应该使用更好的随机数生成器（如 std::mt19937）
    for(int i=0; i < M*d; ++i) h_Q[i] = (float)rand() / RAND_MAX;
    for(int i=0; i < N*d; ++i) h_K[i] = (float)rand() / RAND_MAX;
    for(int i=0; i < N*d; ++i) h_V[i] = (float)rand() / RAND_MAX;

    // ========== 第三步：分配 Device 内存 ==========
    // Device 内存：GPU 可访问的内存（显存）
    // cudaMalloc 在 GPU 全局内存中分配空间
    // 
    // 函数签名：cudaError_t cudaMalloc(void** devPtr, size_t size)
    // - devPtr: 指向指针的指针，函数会修改这个指针指向分配的内存地址
    // - size: 要分配的字节数
    // - 返回值：cudaError_t，表示操作是否成功
    // 
    // 注意：cudaMalloc 分配的内存是未初始化的，包含随机数据
    float *d_Q, *d_K, *d_V, *d_O;
    
    // 为每个矩阵分配 GPU 内存
    // 如果分配失败，cudaMalloc 会返回错误码，但这里没有检查（简化代码）
    // 生产代码中应该检查每个 cudaMalloc 的返回值
    cudaMalloc(&d_Q, size_q);
    cudaMalloc(&d_K, size_k);
    cudaMalloc(&d_V, size_v);
    cudaMalloc(&d_O, size_out);

    // ========== 第四步：将数据从 Host 拷贝到 Device ==========
    // cudaMemcpy 用于在 Host 和 Device 之间拷贝数据
    // 
    // 函数签名：cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind)
    // - dst: 目标地址
    // - src: 源地址
    // - count: 要拷贝的字节数
    // - kind: 拷贝方向，cudaMemcpyHostToDevice 表示从 Host 到 Device
    // 
    // 注意：cudaMemcpy 是同步操作，会阻塞 CPU 直到拷贝完成
    // 可以使用 cudaMemcpyAsync 进行异步拷贝，配合 CUDA Streams 提高性能
    cudaMemcpy(d_Q, h_Q.data(), size_q, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), size_k, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), size_v, cudaMemcpyHostToDevice);

    // ========== 第五步：执行 GPU 计算 ==========
    std::cout << "Running ALiBi kernel..." << std::endl;
    
    // 调用 solve 函数启动 Kernel
    // 此时所有数据都在 GPU 内存中，可以直接进行计算
    solve(d_Q, d_K, d_V, d_O, M, N, d, alpha);

    // ========== 第六步：检查 CUDA 错误 ==========
    // cudaGetLastError 返回最后一个 CUDA 运行时错误
    // 
    // 为什么需要检查错误？
    // 1. Kernel 启动是异步的，错误可能不会立即报告
    // 2. 某些错误（如配置错误）在启动时不会检测到
    // 3. 在同步点（如 cudaDeviceSynchronize）检查可以捕获之前的错误
    // 
    // 注意：cudaGetLastError 会清除错误状态，所以应该只调用一次
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // cudaGetErrorString 将错误码转换为可读的错误信息
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // ========== 第七步：将结果从 Device 拷贝回 Host ==========
    // cudaMemcpyDeviceToHost 表示从 Device 到 Host 的拷贝方向
    // 将计算结果从 GPU 内存拷贝到 CPU 内存，以便后续处理或验证
    cudaMemcpy(h_O.data(), d_O, size_out, cudaMemcpyDeviceToHost);

    // ========== 第八步：验证结果 ==========
    // 打印输出矩阵的第一行的前 5 个元素
    // 用于快速验证程序是否正常运行
    // 
    // 注意：这只是简单的验证，实际应用中应该：
    // 1. 与参考实现（如 CPU 版本）进行对比
    // 2. 检查数值精度
    // 3. 使用单元测试框架进行系统化测试
    std::cout << "Output[0][0..4]: ";
    for(int i=0; i<5; ++i) std::cout << h_O[i] << " ";
    std::cout << "\nSuccess!" << std::endl;

    // ========== 第九步：释放 GPU 内存资源 ==========
    // cudaFree 释放通过 cudaMalloc 分配的内存
    // 
    // 为什么需要释放？
    // 1. GPU 内存是有限的资源，应该及时释放
    // 2. 避免内存泄漏，特别是在长时间运行的程序中
    // 3. 良好的编程习惯
    // 
    // 注意：
    // 1. 不能释放未通过 cudaMalloc 分配的内存
    // 2. 不能重复释放同一块内存
    // 3. 释放后不应该再访问该内存
    cudaFree(d_Q); 
    cudaFree(d_K); 
    cudaFree(d_V); 
    cudaFree(d_O);

    // std::vector 会自动管理内存，不需要手动释放
    // 当 main 函数返回时，h_Q, h_K, h_V, h_O 会自动析构并释放内存

    return 0;
}