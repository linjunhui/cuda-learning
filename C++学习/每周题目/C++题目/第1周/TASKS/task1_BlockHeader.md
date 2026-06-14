# T1 BlockHeader 定义

**依赖**: 无  
**产出**: `include/StringProcessingLib/memory_pool/block_header.h`  
**参考**: [03_详细设计.md](../开发管理/03_详细设计.md)

---

## 任务目标
定义内存块头部结构 `BlockHeader`，供 FixedSizePool 使用。

## 设计要求（来自详细设计）

### 数据结构
```cpp
struct BlockHeader {
    BlockHeader* next;  // 指向下一个空闲块
};
```

### 约束
- 位于每个 block 的起始位置
- 用户可用空间 = block_size - sizeof(BlockHeader)
- 需考虑内存对齐（alignof(BlockHeader)）

## 检查清单
- [ ] `block_header.h` 定义完成
- [ ] 头文件保护 `#ifndef` / `#define` / `#endif`
- [ ] 内存对齐考虑
- [ ] 命名空间 `StringProcessingLib::MemoryPool`

## 完成标准
- 结构体定义符合设计
- 头文件可被正确包含，无编译错误

## 状态
- [ ] 未开始
- [ ] 进行中
- [ ] 已完成
