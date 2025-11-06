
#include <mutex>
#include "StringProcessingLib/memory_pool/block_header.h"
// 为了使用 size_t，需要包含 <cstddef>
#include <cstddef>
namespace StringProcessingLib {
namespace MemoryPool {

class FixedSizePool {

    private:
        void* memory_block;
        size_t block_size;
        size_t block_count;
        BlockHeader* free_list;
        std::mutex mutex_;

        size_t use_count;
        size_t total_allocations;
        size_t total_deallocations;

    public:
        // explicit 防止隐式转换
        void initialize_blocks();
        explicit FixedSizePool(size_t block_size, size_t block_count);
        size_t get_block_size() const;
        size_t get_block_count() const;
};
} // namespace MemoryPool
} // namespace StringProcessingLib
