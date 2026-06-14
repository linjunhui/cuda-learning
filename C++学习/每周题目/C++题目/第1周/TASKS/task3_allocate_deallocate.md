# T3 allocate / deallocate

**依赖**: T2 FixedSizePool 构造  
**产出**: 核心分配逻辑  
**参考**: [03_详细设计.md](../开发管理/03_详细设计.md)

---

## 任务目标
实现 `allocate()` 与 `deallocate()`，支持从空闲链表取块、归还块。

## 设计要求

### allocate
1. lock(mutex_)
2. if (free_list_ == nullptr) return nullptr
3. BlockHeader* block = free_list_
4. free_list_ = block->next
5. unlock(mutex_)
6. return (char*)block + sizeof(BlockHeader)

### deallocate
1. void* ptr → BlockHeader* block = (BlockHeader*)((char*)ptr - sizeof(BlockHeader))
2. 可选：校验 ptr 是否属于本池
3. lock(mutex_)
4. block->next = free_list_; free_list_ = block
5. unlock(mutex_)

### 约束
- `allocate()` 失败返回 nullptr
- `deallocate(nullptr)` 不操作

## 检查清单
- [ ] `allocate()` 实现
- [ ] `deallocate()` 实现
- [ ] 线程安全（mutex 保护）
- [ ] 返回给用户的是「用户区域」指针（跳过 header）

## 完成标准
- 单线程多次 alloc/dealloc 正确
- 分配的块可写入、可正确释放并 reuse

## 状态
- [ ] 未开始
- [ ] 进行中
- [ ] 已完成
