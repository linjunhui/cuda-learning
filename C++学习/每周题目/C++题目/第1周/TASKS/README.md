# TASKS - 开发任务

> 本目录存放开发计划中的任务卡片，每个任务对应一个独立文档，便于跟踪与执行。

## 任务索引

| 任务 | 文档 | 依赖 | 产出 | 状态 |
|------|------|------|------|------|
| T1 | [task1_BlockHeader.md](task1_BlockHeader.md) | - | block_header.h | - |
| T2 | [task2_FixedSizePool构造.md](task2_FixedSizePool构造.md) | T1 | fixed_size_pool.h/cpp（骨架）| - |
| T3 | [task3_allocate_deallocate.md](task3_allocate_deallocate.md) | T2 | 核心分配逻辑 | - |
| T4 | [task4_reset.md](task4_reset.md) | T3 | 重置逻辑 | - |
| T5 | [task5_析构与禁拷贝.md](task5_析构与禁拷贝.md) | T4 | 完整类实现 | - |
| T6 | [task6_CMake构建.md](task6_CMake构建.md) | T1 | CMakeLists.txt | - |
| T7 | [task7_单元测试.md](task7_单元测试.md) | T5 | test_fixed_size_pool.cpp | - |

## 执行顺序

```
T1 → T2 → T3 → T4 → T5 → T7
  ↓
T6 可并行于 T2~T5
```

## 状态说明

- `-` 未开始
- `进行中` 开发中
- `已完成` 已验收
