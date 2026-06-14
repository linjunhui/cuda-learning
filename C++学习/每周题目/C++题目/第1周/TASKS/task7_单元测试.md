# T7 单元测试

**依赖**: T5 完整类实现  
**产出**: `test_fixed_size_pool.cpp`  
**参考**: [05_测试验证.md](../开发管理/05_测试验证.md)

---

## 任务目标
编写 FixedSizePool 的单元测试，覆盖基础、边界、多线程场景。

## 测试用例（来自 05_测试验证）

### 基础
| 用例 | 描述 | 预期 |
|------|------|------|
| 构造 | 创建 FixedSizePool(64, 100) | 无异常 |
| allocate | 分配一块内存 | 非 nullptr，可写入 |
| deallocate | 释放后再次分配 | 可重复使用 |
| 耗尽 | 分配 block_count 次 | 第 block_count+1 次返回 nullptr |
| reset | 分配若干后 reset | 可再次分配 block_count 块 |

### 边界
| 用例 | 描述 | 预期 |
|------|------|------|
| deallocate(nullptr) | 传空指针 | 不崩溃 |

### 多线程（可选）
| 用例 | 描述 | 预期 |
|------|------|------|
| 并发 | 多线程 alloc/dealloc | 无数据竞争、无泄漏 |

## 检查清单
- [ ] test_fixed_size_pool.cpp 创建
- [ ] 基础用例全部通过
- [ ] 边界用例通过
- [ ] ctest 全部通过
- [ ] valgrind 无泄漏

## 完成标准
- `ctest --output-on-failure` 100% 通过
- valgrind 检查通过

## 状态
- [ ] 未开始
- [ ] 进行中
- [ ] 已完成
