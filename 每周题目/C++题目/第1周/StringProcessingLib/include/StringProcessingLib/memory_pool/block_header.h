/**
 * @file block_header.h
 * @brief 内存块头部结构定义
 * @author StringProcessingLib Team
 * @date 2024
 * 
 * 本文件定义了内存池中内存块的头部结构，用于管理内存块的生命周期、
 * 状态跟踪和错误检测。每个内存块都包含一个BlockHeader结构体，
 * 提供内存管理所需的关键信息。
 */

#ifndef STRINGPROCESSINGLIB_MEMORY_POOL_BLOCK_HEADER_H
#define STRINGPROCESSINGLIB_MEMORY_POOL_BLOCK_HEADER_H

#include <cstddef>
#include <cstdint>

namespace StringProcessingLib {
namespace MemoryPool {

/**
 * @brief 内存块头部结构
 * 
 * 每个内存块都包含一个BlockHeader结构体，位于内存块的开头。
 * 该结构体包含管理内存块所需的所有元数据信息。
 * 
 * 内存布局：
 * +------------------+
 * | BlockHeader      | <- 块头部信息
 * +------------------+
 * | User Data        | <- 用户数据区域
 * +------------------+
 * 
 * 设计特点：
 * - 紧凑设计：最小化元数据开销
 * - 错误检测：使用魔数检测内存损坏
 * - 链表支持：支持空闲块链表管理
 * - 状态跟踪：跟踪块的使用状态
 */
struct BlockHeader {
    // 基本信息
    size_t size;           ///< 块大小（字节），包含头部大小
    bool is_free;          ///< 块状态：true=空闲，false=已分配
    
    // 链表管理
    BlockHeader* next;     ///< 下一个块指针（用于空闲块链表）
    BlockHeader* prev;     ///< 上一个块指针（用于双向链表）
    
    // 错误检测
    uint32_t magic;        ///< 魔数，用于检测内存损坏
    
    // 统计信息（可选）
    uint32_t allocation_count; ///< 分配次数统计
    uint64_t last_alloc_time; ///< 最后分配时间戳
    
    /**
     * @brief 默认构造函数
     * 
     * 初始化所有成员变量为默认值
     */
    BlockHeader() noexcept
        : size(0)
        , is_free(true)
        , next(nullptr)
        , prev(nullptr)
        , magic(MAGIC_NUMBER)
        , allocation_count(0)
        , last_alloc_time(0)
    {}
    
    /**
     * @brief 参数化构造函数
     * 
     * @param block_size 块大小
     * @param free_state 空闲状态
     */
    BlockHeader(size_t block_size, bool free_state = true) noexcept
        : size(block_size)
        , is_free(free_state)
        , next(nullptr)
        , prev(nullptr)
        , magic(MAGIC_NUMBER)
        , allocation_count(0)
        , last_alloc_time(0)
    {}
    
    /**
     * @brief 验证魔数是否正确
     * 
     * @return true 如果魔数正确
     * @return false 如果魔数错误（可能内存损坏）
     */
    bool is_valid() const noexcept {
        return magic == MAGIC_NUMBER;
    }
    
    /**
     * @brief 获取用户数据区域大小
     * 
     * @return 用户可用的数据大小
     */
    size_t get_user_size() const noexcept {
        return size - sizeof(BlockHeader);
    }
    
    /**
     * @brief 获取用户数据区域指针
     * 
     * @return 指向用户数据区域的指针
     */
    void* get_user_ptr() noexcept {
        return reinterpret_cast<void*>(this + 1);
    }
    
    /**
     * @brief 获取用户数据区域指针（const版本）
     * 
     * @return 指向用户数据区域的指针
     */
    const void* get_user_ptr() const noexcept {
        return reinterpret_cast<const void*>(this + 1);
    }
    
    /**
     * @brief 标记块为已分配
     * 
     * @param alloc_time 分配时间戳
     */
    void mark_allocated(uint64_t alloc_time = 0) noexcept {
        is_free = false;
        allocation_count++;
        last_alloc_time = alloc_time;
    }
    
    /**
     * @brief 标记块为空闲
     */
    void mark_free() noexcept {
        is_free = true;
    }
    
    /**
     * @brief 重置块状态
     */
    void reset() noexcept {
        is_free = true;
        next = nullptr;
        prev = nullptr;
        allocation_count = 0;
        last_alloc_time = 0;
        magic = MAGIC_NUMBER; // 重新设置魔数
    }

private:
    static constexpr uint32_t MAGIC_NUMBER = 0xDEADBEEF; ///< 魔数常量
};

/**
 * @brief 从用户指针获取块头部
 * 
 * @param user_ptr 用户数据指针
 * @return 对应的块头部指针
 */
inline BlockHeader* get_header_from_user_ptr(void* user_ptr) noexcept {
    return reinterpret_cast<BlockHeader*>(user_ptr) - 1;
}

/**
 * @brief 从用户指针获取块头部（const版本）
 * 
 * @param user_ptr 用户数据指针
 * @return 对应的块头部指针
 */
inline const BlockHeader* get_header_from_user_ptr(const void* user_ptr) noexcept {
    return reinterpret_cast<const BlockHeader*>(user_ptr) - 1;
}

/**
 * @brief 计算对齐后的块大小
 * 
 * @param user_size 用户请求的大小
 * @param alignment 对齐要求
 * @return 对齐后的总块大小（包含头部）
 */
inline size_t calculate_aligned_block_size(size_t user_size, size_t alignment = alignof(BlockHeader)) noexcept {
    size_t total_size = sizeof(BlockHeader) + user_size;
    return (total_size + alignment - 1) & ~(alignment - 1);
}

/**
 * @brief 验证块头部完整性
 * 
 * @param header 块头部指针
 * @return true 如果块头部完整
 * @return false 如果块头部损坏
 */
inline bool validate_block_header(const BlockHeader* header) noexcept {
    if (!header) return false;
    if (!header->is_valid()) return false;
    if (header->size < sizeof(BlockHeader)) return false;
    return true;
}

} // namespace MemoryPool
} // namespace StringProcessingLib

#endif // STRINGPROCESSINGLIB_MEMORY_POOL_BLOCK_HEADER_H
