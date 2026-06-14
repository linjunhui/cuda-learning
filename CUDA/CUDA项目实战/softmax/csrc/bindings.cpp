#include <torch/extension.h>
#include <vector>

// 1. 声明你在 .cu 文件里写的 C++ 函数包装器
// 注意：输入通常是 torch::Tensor
torch::Tensor dispatch_softmax(torch::Tensor input);
torch::Tensor dispatch_softmax_vec(torch::Tensor input);

// 2. 使用 PYBIND11 宏进行绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("online_softmax", &dispatch_softmax, "Online Safe Softmax (CUDA)");
    m.def("online_softmax_vec", &dispatch_softmax_vec, "Online Safe Softmax (CUDA) with float4");
}