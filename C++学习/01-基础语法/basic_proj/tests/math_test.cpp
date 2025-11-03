#include "math.h"
#include <gtest/gtest.h>
#include <sstream>
#include <iostream>

// 由于 math::add 只输出结果，我们需要测试输出
// 这里创建一个测试来验证 add 函数的正确性
// 注意：实际项目中，更好的做法是让函数返回值而不是直接输出

namespace math {
namespace test {

// 辅助函数：捕获 cout 输出
std::string capture_output(void (*func)(int, int), int a, int b) {
    std::ostringstream oss;
    std::streambuf* old_cout = std::cout.rdbuf();
    std::cout.rdbuf(oss.rdbuf());
    
    func(a, b);
    
    std::cout.rdbuf(old_cout);
    return oss.str();
}

} // namespace test
} // namespace math

// 测试套件
class MathTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 测试前初始化（如果需要）
    }
    
    void TearDown() override {
        // 测试后清理（如果需要）
    }
};

// 测试用例：基本加法功能
TEST(MathTest, BasicAddition) {
    // 测试 1 + 2 = 3
    std::string output = math::test::capture_output(math::add, 1, 2);
    EXPECT_EQ(output, "3\n");
}

// 测试用例：负数加法
TEST(MathTest, NegativeNumbers) {
    // 测试 -1 + 2 = 1
    std::string output = math::test::capture_output(math::add, -1, 2);
    EXPECT_EQ(output, "1\n");
}

// 测试用例：两个负数
TEST(MathTest, TwoNegativeNumbers) {
    // 测试 -3 + (-5) = -8
    std::string output = math::test::capture_output(math::add, -3, -5);
    EXPECT_EQ(output, "-8\n");
}

// 测试用例：零值
TEST(MathTest, ZeroValues) {
    // 测试 0 + 0 = 0
    std::string output = math::test::capture_output(math::add, 0, 0);
    EXPECT_EQ(output, "0\n");
    
    // 测试 5 + 0 = 5
    output = math::test::capture_output(math::add, 5, 0);
    EXPECT_EQ(output, "5\n");
}

// 测试用例：大数值
TEST(MathTest, LargeNumbers) {
    // 测试 1000 + 2000 = 3000
    std::string output = math::test::capture_output(math::add, 1000, 2000);
    EXPECT_EQ(output, "3000\n");
}

// 测试用例：边界值
TEST(MathTest, BoundaryValues) {
    // 测试最大整数值（假设是典型系统的最大值）
    int max_int = 2147483647;
    std::string output = math::test::capture_output(math::add, max_int, 0);
    std::string expected = std::to_string(max_int) + "\n";
    EXPECT_EQ(output, expected);
}
