# T2 FixedSizePool 构造与 init

**依赖**: T1 BlockHeader  
**产出**: `fixed_size_pool.h`、`fixed_size_pool.cpp`（骨架）  
**参考**: [03_详细设计.md](../开发管理/03_详细设计.md)

---

## 任务目标
实现 FixedSizePool 的构造逻辑与 `initialize_blocks`，将预分配的大块内存串成空闲链表。

## 设计要求

### 成员变量
| 成员 | 类型 | 说明 |
|------|------|------|
| memory_block_ | void* | 预分配的大块内存 |
| block_size_ | size_t | 单个 block 大小（含 header）|
| block_count_ | size_t | block 数量 |
| free_list_ | BlockHeader* | 空闲链表头 |
| mutex_ | std::mutex | 互斥锁 |

### 初始化算法
1. 计算 `actual_block_size = align_up(block_size + sizeof(BlockHeader))`
2. `memory_block_ = operator new(actual_block_size * block_count)`
3. 遍历每个 block，用 `next` 串成 free_list

## 检查清单
- [ ] 构造函数 `FixedSizePool(size_t block_size, size_t block_count)`
- [ ] `initialize_blocks()` 实现
- [ ] 成员变量声明正确

## 完成标准
- 构造完成后 free_list 包含 block_count 个块
- 头文件与源文件结构正确

## 状态
- [ ] 未开始
- [ ] 进行中
- [ ] 已完成
