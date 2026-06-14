# T5 析构、禁拷贝

**依赖**: T4 reset  
**产出**: 完整类实现  
**参考**: [03_详细设计.md](../开发管理/03_详细设计.md)

---

## 任务目标
实现析构函数释放 `memory_block_`，禁止拷贝和赋值。

## 设计要求

### 析构
- 释放 `memory_block_`（operator delete）
- 不依赖用户是否已 deallocate（reset 或逐个 deallocate 均可）

### 禁拷贝
```cpp
FixedSizePool(const FixedSizePool&) = delete;
FixedSizePool& operator=(const FixedSizePool&) = delete;
```

## 检查清单
- [ ] 析构函数释放 memory_block_
- [ ] 拷贝构造 = delete
- [ ] 拷贝赋值 = delete
- [ ] 无双重释放、无泄漏（valgrind）

## 完成标准
- 析构后无内存泄漏
- 拷贝/赋值被编译期禁止

## 状态
- [ ] 未开始
- [ ] 进行中
- [ ] 已完成
