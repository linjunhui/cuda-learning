/**
 * @file test_block_header.cpp
 * @brief BlockHeader单元测试
 */

#include <cassert>
#include <iostream>
#include "StringProcessingLib/memory_pool/block_header.h"

using namespace StringProcessingLib::MemoryPool;

// 测试BlockHeader默认构造
void test_default_constructor() {
    std::cout << "测试1：默认构造函数\n";
    BlockHeader header;
    
    assert(header.size == 0);
    assert(header.is_free == true);
    assert(header.next == nullptr);
    assert(header.prev == nullptr);
    assert(header.allocation_count == 0);
    assert(header.last_alloc_time == 0);
    assert(header.is_valid() == true); // 魔数应该正确
    // 注意：当size为0时，get_user_size()会因为无符号整数下溢而返回一个很大的值
    // 这是边界情况，实际使用中应该确保size >= sizeof(BlockHeader)
    assert(header.get_user_size() == static_cast<size_t>(0) - sizeof(BlockHeader));
    
    std::cout << "  ✓ 默认构造函数测试通过\n";
}

// 测试BlockHeader参数构造
void test_parameter_constructor() {
    std::cout << "测试2：参数构造函数\n";
    const size_t TEST_SIZE = 256;
    BlockHeader header(TEST_SIZE, false);
    
    assert(header.size == TEST_SIZE);
    assert(header.is_free == false);
    assert(header.next == nullptr);
    assert(header.is_valid() == true);
    
    std::cout << "  ✓ 参数构造函数测试通过\n";
}

// 测试is_valid方法
void test_is_valid() {
    std::cout << "测试3：魔数验证\n";
    BlockHeader header;
    
    // 正常情况应该是有效的
    assert(header.is_valid() == true);
    
    // 破坏魔数
    header.magic = 0x12345678;
    assert(header.is_valid() == false);
    
    // 恢复魔数
    header.magic = 0xDEADBEEF;
    assert(header.is_valid() == true);
    
    std::cout << "  ✓ 魔数验证测试通过\n";
}

// 测试get_user_size方法
void test_get_user_size() {
    std::cout << "测试4：获取用户空间大小\n";
    const size_t BLOCK_SIZE = 256;
    BlockHeader header(BLOCK_SIZE);
    
    size_t user_size = header.get_user_size();
    size_t expected = BLOCK_SIZE - sizeof(BlockHeader);
    
    assert(user_size == expected);
    std::cout << "  ✓ 用户空间大小计算正确: " << user_size << " 字节\n";
}

// 测试get_user_ptr方法
void test_get_user_ptr() {
    std::cout << "测试5：获取用户指针\n";
    BlockHeader header(256);
    void* user_ptr = header.get_user_ptr();
    
    // 用户指针应该指向BlockHeader之后
    char* header_ptr = reinterpret_cast<char*>(&header);
    char* expected_ptr = header_ptr + sizeof(BlockHeader);
    
    assert(user_ptr == expected_ptr);
    std::cout << "  ✓ 用户指针计算正确\n";
}

// 测试mark_allocated方法
void test_mark_allocated() {
    std::cout << "测试6：标记为已分配\n";
    BlockHeader header;
    const uint64_t TEST_TIME = 1234567890;
    
    header.mark_allocated(TEST_TIME);
    
    assert(header.is_free == false);
    assert(header.allocation_count == 1);
    assert(header.last_alloc_time == TEST_TIME);
    
    // 再次调用应该增加计数
    header.mark_allocated();
    assert(header.allocation_count == 2);
    
    std::cout << "  ✓ 标记为已分配测试通过\n";
}

// 测试mark_free方法
void test_mark_free() {
    std::cout << "测试7：标记为空闲\n";
    BlockHeader header;
    header.is_free = false;
    
    header.mark_free();
    assert(header.is_free == true);
    
    std::cout << "  ✓ 标记为空闲测试通过\n";
}

// 测试reset方法
void test_reset() {
    std::cout << "测试8：重置块\n";
    BlockHeader header(256, false);
    header.next = reinterpret_cast<BlockHeader*>(0x1234);
    header.prev = reinterpret_cast<BlockHeader*>(0x5678);
    header.allocation_count = 5;
    header.last_alloc_time = 999;
    header.magic = 0xBAD; // 破坏魔数
    
    header.reset();
    
    assert(header.is_free == true);
    assert(header.next == nullptr);
    assert(header.prev == nullptr);
    assert(header.allocation_count == 0);
    assert(header.last_alloc_time == 0);
    assert(header.is_valid() == true); // 魔数应该被恢复
    
    std::cout << "  ✓ 重置块测试通过\n";
}

// 测试get_header_from_user_ptr辅助函数
void test_get_header_from_user_ptr() {
    std::cout << "测试9：从用户指针获取头部\n";
    BlockHeader header(256);
    void* user_ptr = header.get_user_ptr();
    
    // 从用户指针反推应该得到原头部
    BlockHeader* recovered = get_header_from_user_ptr(user_ptr);
    assert(recovered == &header);
    
    std::cout << "  ✓ 从用户指针获取头部测试通过\n";
}

// 测试calculate_aligned_block_size
void test_calculate_aligned_block_size() {
    std::cout << "测试10：计算对齐的块大小\n";
    
    size_t user_size = 100;
    size_t aligned_size = calculate_aligned_block_size(user_size);
    
    // 对齐后的大小应该包含头部并且对齐
    assert(aligned_size >= sizeof(BlockHeader) + user_size);
    
    // 应该是对齐值的倍数
    size_t alignment = alignof(BlockHeader);
    assert(aligned_size % alignment == 0);
    
    std::cout << "  ✓ 计算对齐大小: " << user_size << " -> " << aligned_size << "\n";
}

// 测试validate_block_header
void test_validate_block_header() {
    std::cout << "测试11：验证块头部\n";
    
    BlockHeader valid_header(256);
    assert(validate_block_header(&valid_header) == true);
    
    // 无效指针
    assert(validate_block_header(nullptr) == false);
    
    // 无效魔数
    BlockHeader invalid_magic(256);
    invalid_magic.magic = 0xBAD;
    assert(validate_block_header(&invalid_magic) == false);
    
    // 大小太小
    BlockHeader too_small;
    too_small.size = sizeof(BlockHeader) - 1;
    assert(validate_block_header(&too_small) == false);
    
    std::cout << "  ✓ 块头部验证测试通过\n";
}

// 主测试函数
int main() {
    std::cout << "========== BlockHeader单元测试开始 ==========\n\n";
    
    try {
        test_default_constructor();
        test_parameter_constructor();
        test_is_valid();
        test_get_user_size();
        test_get_user_ptr();
        test_mark_allocated();
        test_mark_free();
        test_reset();
        test_get_header_from_user_ptr();
        test_calculate_aligned_block_size();
        test_validate_block_header();
        
        std::cout << "\n========== 所有测试通过！==========\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n测试失败: " << e.what() << "\n";
        return 1;
    }
}

