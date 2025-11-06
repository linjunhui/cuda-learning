# Google Test (GTest) 基础教程

## 1. Google Test 简介

### 1.1 什么是 Google Test

Google Test（GTest）是 Google 开发的 C++ 测试框架，用于编写单元测试。

**特点**：
- ✅ 丰富的断言宏
- ✅ 测试用例自动发现
- ✅ 测试夹具（Test Fixtures）
- ✅ 参数化测试
- ✅ Mock 支持（GMock）

### 1.2 为什么使用 Google Test

- **标准化**：C++ 测试的事实标准
- **易用性**：简洁的 API，易于学习
- **功能强大**：支持各种测试场景
- **社区支持**：广泛使用，文档完善

## 2. 安装和集成

### 2.1 使用 FetchContent（推荐）

```cmake
include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)

FetchContent_MakeAvailable(googletest)

target_link_libraries(your_test_target
    PRIVATE
        GTest::gtest_main
)
```

### 2.2 使用 find_package

```cmake
find_package(GTest REQUIRED)

target_link_libraries(your_test_target
    PRIVATE
        GTest::GTest
        GTest::Main
)
```

### 2.3 基本测试文件结构

```cpp
#include "your_header.h"
#include <gtest/gtest.h>

namespace YourNamespace {

TEST(TestSuiteName, TestCaseName) {
    // 测试代码
    EXPECT_EQ(1 + 1, 2);
}

} // namespace YourNamespace
```

## 3. 基本断言

### 3.1 成功/失败断言

```cpp
// 成功断言：测试失败时继续执行
EXPECT_TRUE(condition);   // 期望条件为真
EXPECT_FALSE(condition);  // 期望条件为假

// 失败断言：测试失败时立即终止
ASSERT_TRUE(condition);
ASSERT_FALSE(condition);
```

**区别**：
- `EXPECT_*`：测试失败后继续执行后续测试
- `ASSERT_*`：测试失败后立即终止当前测试

### 3.2 相等性断言

```cpp
// 相等
EXPECT_EQ(expected, actual);  // ==
EXPECT_NE(val1, val2);        // !=

// 严格相等（指针）
EXPECT_STREQ(str1, str2);     // C 字符串相等
EXPECT_STRNE(str1, str2);     // C 字符串不等
EXPECT_STRCASEEQ(str1, str2); // 忽略大小写的字符串相等
```

**示例**：
```cpp
TEST(ComparisonTest, Basic) {
    EXPECT_EQ(2, 1 + 1);
    EXPECT_NE(2, 3);
    
    const char* str1 = "hello";
    const char* str2 = "hello";
    EXPECT_STREQ(str1, str2);
}
```

### 3.3 关系断言

```cpp
EXPECT_LT(val1, val2);  // <  (Less Than)
EXPECT_LE(val1, val2);  // <= (Less or Equal)
EXPECT_GT(val1, val2);  // >  (Greater Than)
EXPECT_GE(val1, val2);  // >= (Greater or Equal)
```

### 3.4 浮点数断言

```cpp
EXPECT_FLOAT_EQ(expected, actual);     // 近似相等
EXPECT_DOUBLE_EQ(expected, actual);   // 近似相等
EXPECT_NEAR(val1, val2, abs_error);   // 误差范围内相等
```

**示例**：
```cpp
TEST(FloatTest, NearEquality) {
    EXPECT_NEAR(0.1 + 0.2, 0.3, 0.0001);
    EXPECT_FLOAT_EQ(1.0f / 3.0f, 0.333333f);
}
```

## 4. 测试用例组织

### 4.1 TEST 宏

```cpp
TEST(TestSuiteName, TestCaseName) {
    // 测试代码
}
```

**命名规则**：
- `TestSuiteName`：测试套件名称（通常是类名）
- `TestCaseName`：测试用例名称（描述测试场景）

**示例**：
```cpp
TEST(FixedSizePoolTest, Initialization) {
    FixedSizePool pool(1024, 100);
    EXPECT_EQ(pool.get_block_size(), 1024);
    EXPECT_EQ(pool.get_block_count(), 100);
}

TEST(FixedSizePoolTest, Allocation) {
    FixedSizePool pool(32, 10);
    void* ptr = pool.allocate();
    EXPECT_NE(ptr, nullptr);
}
```

### 4.2 测试夹具（Test Fixtures）

**目的**：多个测试共享相同的设置和清理代码

```cpp
class FixedSizePoolTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 每个测试前执行
        pool = std::make_unique<FixedSizePool>(1024, 100);
    }
    
    void TearDown() override {
        // 每个测试后执行
        pool.reset();
    }
    
    std::unique_ptr<FixedSizePool> pool;
};

// 使用 TEST_F 而不是 TEST
TEST_F(FixedSizePoolTest, Initialization) {
    EXPECT_EQ(pool->get_block_size(), 1024);
}

TEST_F(FixedSizePoolTest, Allocation) {
    void* ptr = pool->allocate();
    EXPECT_NE(ptr, nullptr);
}
```

### 4.3 SetUp 和 TearDown

```cpp
class MyTestFixture : public ::testing::Test {
protected:
    void SetUp() override {
        // 在每个测试前调用
        // 初始化资源
    }
    
    void TearDown() override {
        // 在每个测试后调用
        // 清理资源
    }
    
    // 共享的测试数据
    int value = 0;
};
```

## 5. 常见测试模式

### 5.1 测试构造函数

```cpp
TEST(FixedSizePoolTest, Constructor) {
    FixedSizePool pool(1024, 100);
    
    EXPECT_EQ(pool.get_block_size(), 1024);
    EXPECT_EQ(pool.get_block_count(), 100);
    EXPECT_EQ(pool.get_use_count(), 0);
}
```

### 5.2 测试正常流程

```cpp
TEST(FixedSizePoolTest, AllocateAndDeallocate) {
    FixedSizePool pool(32, 10);
    
    // 分配
    void* ptr1 = pool.allocate();
    EXPECT_NE(ptr1, nullptr);
    EXPECT_EQ(pool.get_use_count(), 1);
    
    // 再分配
    void* ptr2 = pool.allocate();
    EXPECT_NE(ptr2, nullptr);
    EXPECT_EQ(pool.get_use_count(), 2);
    
    // 释放
    pool.deallocate(ptr1);
    EXPECT_EQ(pool.get_use_count(), 1);
}
```

### 5.3 测试边界条件

```cpp
TEST(FixedSizePoolTest, EmptyPool) {
    FixedSizePool pool(32, 0);
    void* ptr = pool.allocate();
    EXPECT_EQ(ptr, nullptr);
}

TEST(FixedSizePoolTest, FullPool) {
    FixedSizePool pool(32, 2);
    
    void* ptr1 = pool.allocate();
    void* ptr2 = pool.allocate();
    void* ptr3 = pool.allocate();  // 应该失败
    
    EXPECT_NE(ptr1, nullptr);
    EXPECT_NE(ptr2, nullptr);
    EXPECT_EQ(ptr3, nullptr);  // 池已满
}
```

### 5.4 测试异常情况

```cpp
TEST(FixedSizePoolTest, InvalidDeallocation) {
    FixedSizePool pool(32, 10);
    
    // 释放空指针（如果不支持，应该检查）
    // pool.deallocate(nullptr);  // 可能导致断言或异常
    
    // 释放不属于池的指针
    int* invalid_ptr = new int(42);
    // 这可能导致未定义行为，需要设计时考虑
}
```

## 6. 参数化测试

### 6.1 基本参数化测试

```cpp
class FixedSizePoolParamTest : public ::testing::TestWithParam<std::tuple<size_t, size_t>> {
};

TEST_P(FixedSizePoolParamTest, DifferentSizes) {
    auto [block_size, block_count] = GetParam();
    FixedSizePool pool(block_size, block_count);
    
    EXPECT_EQ(pool.get_block_size(), block_size);
    EXPECT_EQ(pool.get_block_count(), block_count);
}

INSTANTIATE_TEST_SUITE_P(
    PoolSizes,
    FixedSizePoolParamTest,
    ::testing::Values(
        std::make_tuple(32, 10),
        std::make_tuple(64, 20),
        std::make_tuple(128, 5)
    )
);
```

### 6.2 布尔参数化

```cpp
class PoolTest : public ::testing::TestWithParam<bool> {};

TEST_P(PoolTest, TestWithFlag) {
    bool flag = GetParam();
    // 使用 flag 进行测试
}

INSTANTIATE_TEST_SUITE_P(
    WithAndWithoutFlag,
    PoolTest,
    ::testing::Bool()
);
```

## 7. 死亡测试（Death Tests）

**用途**：测试程序是否按预期终止（如断言失败）

```cpp
TEST(DeathTest, InvalidInput) {
    FixedSizePool pool(0, 10);  // 无效的块大小
    
    // 期望程序终止（如果构造函数中有断言）
    EXPECT_DEATH({
        FixedSizePool invalid_pool(0, 10);
    }, ".*");
}
```

## 8. 运行测试

### 8.1 命令行运行

```bash
# 运行所有测试
./test_fixed_size_pool

# 运行特定测试套件
./test_fixed_size_pool --gtest_filter=FixedSizePoolTest.*

# 运行特定测试用例
./test_fixed_size_pool --gtest_filter=FixedSizePoolTest.Initialization

# 列出所有测试
./test_fixed_size_pool --gtest_list_tests
```

### 8.2 使用 CTest

```bash
# 运行所有测试
ctest

# 显示详细输出
ctest --output-on-failure

# 更详细的信息
ctest -V

# 并行运行
ctest -j 4
```

### 8.3 常用 GTest 选项

```bash
# 过滤测试
--gtest_filter=TestSuite.*           # 运行指定测试套件的所有测试
--gtest_filter=TestSuite.TestCase    # 运行特定测试用例
--gtest_filter=-TestSuite.*          # 排除测试套件

# 重复运行
--gtest_repeat=10                    # 重复运行 10 次

# 随机顺序
--gtest_shuffle                      # 随机顺序运行测试

# 输出控制
--gtest_color=yes                    # 彩色输出
--gtest_brief=1                      # 只显示失败的详细信息
--gtest_output=xml:report.xml       # 输出 XML 报告
```

## 9. 测试最佳实践

### 9.1 测试命名

```cpp
// ✅ 好的命名
TEST(FixedSizePoolTest, Initialization)          // 清晰
TEST(FixedSizePoolTest, AllocationWhenEmpty)      // 描述场景
TEST(FixedSizePoolTest, DeallocationReturnsToPool) // 描述行为

// ❌ 不好的命名
TEST(Test1, test)                                 // 不清晰
TEST(FixedSizePoolTest, test_1)                   // 没有描述性
```

### 9.2 测试独立性

```cpp
// ✅ 好的：每个测试独立
TEST_F(FixedSizePoolTest, Test1) {
    FixedSizePool pool(32, 10);
    // 测试代码
}

TEST_F(FixedSizePoolTest, Test2) {
    FixedSizePool pool(32, 10);  // 重新创建，不依赖 Test1
    // 测试代码
}

// ❌ 不好：测试之间依赖
TEST_F(FixedSizePoolTest, Test1) {
    pool->allocate();  // 修改共享状态
}

TEST_F(FixedSizePoolTest, Test2) {
    // 依赖 Test1 的状态，可能失败
    EXPECT_EQ(pool->get_use_count(), 1);
}
```

### 9.3 测试覆盖

- **正常路径**：测试正常使用场景
- **边界条件**：测试边界值（0、最大值等）
- **错误处理**：测试错误输入和异常情况
- **状态验证**：测试对象的状态变化

### 9.4 使用测试夹具

```cpp
// ✅ 推荐：使用夹具减少重复代码
class FixedSizePoolTest : public ::testing::Test {
protected:
    void SetUp() override {
        pool = std::make_unique<FixedSizePool>(1024, 100);
    }
    
    std::unique_ptr<FixedSizePool> pool;
};

TEST_F(FixedSizePoolTest, Test1) {
    // 使用 pool
}

// ❌ 不推荐：每个测试都重复创建
TEST(FixedSizePoolTest, Test1) {
    FixedSizePool pool(1024, 100);
    // ...
}
```

## 10. 实际示例

### 10.1 完整测试文件示例

```cpp
#include "StringProcessingLib/memory_pool/fixed_size_pool.h"
#include <gtest/gtest.h>

namespace StringProcessingLib {
namespace MemoryPool {

class FixedSizePoolTest : public ::testing::Test {
protected:
    void SetUp() override {
        pool = std::make_unique<FixedSizePool>(1024, 100);
    }
    
    std::unique_ptr<FixedSizePool> pool;
};

TEST_F(FixedSizePoolTest, Initialization) {
    EXPECT_EQ(pool->get_block_size(), 1024);
    EXPECT_EQ(pool->get_block_count(), 100);
    EXPECT_EQ(pool->get_use_count(), 0);
}

TEST_F(FixedSizePoolTest, Allocate) {
    void* ptr = pool->allocate();
    EXPECT_NE(ptr, nullptr);
    EXPECT_EQ(pool->get_use_count(), 1);
}

TEST_F(FixedSizePoolTest, Deallocate) {
    void* ptr = pool->allocate();
    pool->deallocate(ptr);
    EXPECT_EQ(pool->get_use_count(), 0);
}

TEST_F(FixedSizePoolTest, FullPool) {
    // 分配所有块
    std::vector<void*> ptrs;
    for (size_t i = 0; i < 100; ++i) {
        void* ptr = pool->allocate();
        EXPECT_NE(ptr, nullptr);
        ptrs.push_back(ptr);
    }
    
    // 应该满了
    void* ptr = pool->allocate();
    EXPECT_EQ(ptr, nullptr);
}

} // namespace MemoryPool
} // namespace StringProcessingLib
```

## 11. 常见问题

### 11.1 链接错误

**错误**：
```
undefined reference to `testing::InitGoogleTest(...)`
```

**解决**：
```cmake
target_link_libraries(your_test
    PRIVATE
        GTest::gtest_main  # 包含 main() 函数
)
```

### 11.2 测试不运行

**问题**：测试注册了但不执行

**检查**：
1. 确保链接了 `gtest_main`
2. 确保测试宏 `TEST()` 或 `TEST_F()` 正确使用
3. 检查命名空间是否正确

### 11.3 测试输出不清晰

**解决**：
```bash
# 使用详细输出
./test_fixed_size_pool --gtest_brief=0

# 或使用 CTest
ctest --output-on-failure -V
```

## 12. 总结

### Google Test 核心概念

1. **断言**：`EXPECT_*` 和 `ASSERT_*` 宏
2. **测试用例**：`TEST()` 宏定义测试
3. **测试夹具**：`TEST_F()` 与 `::testing::Test` 子类
4. **参数化测试**：`TEST_P()` 与 `INSTANTIATE_TEST_SUITE_P()`

### 测试编写原则

- ✅ **独立性**：每个测试应该独立运行
- ✅ **可重复性**：多次运行结果一致
- ✅ **快速**：测试应该快速执行
- ✅ **清晰**：测试名称和代码应该清晰表达意图
- ✅ **覆盖性**：覆盖正常路径、边界条件和错误情况

掌握这些基础，你就能编写高质量的单元测试了！
