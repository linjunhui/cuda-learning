# T6 CMake 构建配置

**依赖**: T1 BlockHeader  
**产出**: CMakeLists.txt（可并行于 T2~T5）  
**参考**: [01_工程目录建立与设计](../StringProcessingLib/docs/01_工程目录建立与设计.md)

---

## 任务目标
配置 CMake 构建系统，使项目可编译，并集成 GoogleTest。

## 检查清单
- [ ] 根目录 `CMakeLists.txt` 配置
- [ ] 包含 GoogleTest（FetchContent）
- [ ] `include_directories` 设置正确
- [ ] 库目标与测试目标配置
- [ ] 编译无警告

## 完成标准
- `cd build && cmake .. && make` 成功
- 测试可被编译（即使尚未实现全部用例）

## 状态
- [ ] 未开始
- [ ] 进行中
- [ ] 已完成
