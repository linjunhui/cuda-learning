/**
 * @file block_header.cpp
 * @brief BlockHeader结构体的方法实现
 * @author StringProcessingLib Team
 * @date 2024
 * 
 * 设计说明：
 * BlockHeader 的所有方法都采用内联实现，定义在 block_header.h 中。
 * 这样设计的原因：
 * 1. 性能优化：BlockHeader 的方法都是极简的（1-3行代码），内联可以消除函数调用开销
 * 2. 内存池是性能关键路径，频繁调用这些函数，内联带来的性能提升明显
 * 3. 简单性：所有实现在一处，便于维护和理解
 * 4. 避免链接问题：不需要在多个编译单元中链接相同的实现
 * 
 * 如果未来需要优化编译时间（减少依赖），可以考虑将相对复杂的函数
 * （如 validate_block_header）移到实现文件。但对于当前的设计目标（性能优先），
 * 保持内联是最佳选择。
 */

#include "StringProcessingLib/memory_pool/block_header.h"

namespace StringProcessingLib {
namespace MemoryPool {

// 注意：当前所有方法都在头文件中内联实现，这是有意为之的设计选择。
// 如果需要将某些方法移到实现文件，可以在这里添加实现。
// 
// 示例（如果需要移动 validate_block_header）：
// bool validate_block_header(const BlockHeader* header) noexcept {
//     if (!header) return false;
//     if (!header->is_valid()) return false;
//     if (header->size < sizeof(BlockHeader)) return false;
//     return true;
// }

} // namespace MemoryPool
} // namespace StringProcessingLib

