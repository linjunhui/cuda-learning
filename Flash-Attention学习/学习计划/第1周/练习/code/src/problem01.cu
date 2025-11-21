/*
题目 1：主机-设备架构理解
知识点：主机-设备架构
任务：判断给出的典型 CUDA 操作分别在主机还是设备执行，并说明原因。
提示：梳理内存分配、内核调用、数据传输等流程，区分“在主机发起”与“在设备实际执行”。

回答（保留原理解并附修正）：
- 原始表述：`cuda*` 操作由主机端发起，交由 GPU 流执行，对主机来说是异步的。
- 修正说明：
  - A. `malloc` → 主机执行。
  - B. `cudaMalloc` → 主机发起 API 调用，驱动在主机侧向设备申请显存（归类为“主机调用，作用于设备”）。
  - C. `kernel<<<...>>>` → 主机发起内核启动，真正的计算在设备执行。
  - D. `cudaMemcpy` → 主机调用的数据传输，驱动协调主机/设备之间拷贝。
  ➜ 结论：示例中的 API 调用都在主机线程执行，但其作用对象可能位于设备；只有内核代码本身在 GPU 上运行。

点评：
- ✅ 已按要求给出 A/B/C/D 的主机/设备归属，并在代码中补充了 `free(h_data)`、`cudaDeviceSynchronize()` 的错误检查，逻辑更完整。
- ⚠️ 仍建议在 `printf_error_string` 中区分成功与失败（例如成功时不打印或打印 OK，失败时打印错误并返回），避免大量重复的 "Status: cudaSuccess" 输出。
- ⚠️ 若想进一步提升可读性，可在注释或日志中补充每个步骤对应的判定理由，帮助读者快速对照题目中的四个操作。
*/

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}

void printf_error_string(char *msg, cudaError_t err) {
    printf("CUDA Operator : %s Status: %s\n", msg, cudaGetErrorString(err));
}

int main() {

    int N = 1000;
    float* h_data;
    float* d_data;

    int byte_size = sizeof(float) * N;

    h_data = (float *)malloc(byte_size);

    cudaMalloc(&d_data, byte_size);
    cudaError_t err = cudaGetLastError();
    printf_error_string("Malloc CUDA MEM", err);

    // 初始数据
    for(int i = 0; i < N; i++) {
        h_data[i] = i * 1.0f;
    }

    // 输出传输
    err = cudaMemcpy(d_data, h_data, byte_size, cudaMemcpyHostToDevice);
    printf_error_string("Data Transfer from Host to Device", err);

    // 准备 cuda线程尺寸
    dim3 block_size(32, 1, 1);
    int grid_size_x = (N + 32 - 1) / 32;
    dim3 grid_size(grid_size_x, 1, 1);
    kernel<<<grid_size, block_size>>>(d_data, N);
    err = cudaGetLastError();
    printf_error_string("Boot Kernel", err);

    err = cudaMemcpy(h_data, d_data, byte_size, cudaMemcpyDeviceToHost);
    printf_error_string("Data Transfer from Device to Host", err);

    cudaFree(d_data);
    err = cudaDeviceSynchronize();
    printf_error_string("Host Synchronize Device", err);


    free(h_data);

    return 0;
}