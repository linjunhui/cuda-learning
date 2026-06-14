# T4 reset

**依赖**: T3 allocate/deallocate  
**产出**: 重置逻辑  
**参考**: [03_详细设计.md](../开发管理/03_详细设计.md)

---

## 任务目标
实现 `reset()`，将池内所有已分配块回收至 free_list。

## 设计要求

### reset 算法
1. lock(mutex_)
2. 重新初始化 free_list（同初始化逻辑：遍历每个 block 串成链表）
3. unlock(mutex_)

### 说明
- 不释放 `memory_block_`，只重建 free_list
- 所有之前 allocate 出的 ptr 在 reset 后失效，用户不可再使用

## 检查清单
- [ ] `reset()` 实现
- [ ] 重置后可再次 allocate block_count 块
- [ ] 线程安全

## 完成标准
- reset 后 free_list 包含 block_count 个块
- 可再次分配直至耗尽

## 状态
- [ ] 未开始
- [ ] 进行中
- [ ] 已完成
