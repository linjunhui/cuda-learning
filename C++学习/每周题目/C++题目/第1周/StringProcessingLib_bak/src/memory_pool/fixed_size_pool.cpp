#include "StringProcessingLib/memory_pool/fixed_size_pool.h"
#include "StringProcessingLib/memory_pool/block_header.h"
#include <new>
#include <cstdlib>

namespace StringProcessingLib {
namespace MemoryPool {

FixedSizePool::FixedSizePool(size_t block_size, size_t block_count)
    : block_size(block_size), block_count(block_count), free_list(nullptr), memory_block(nullptr), use_count(0), total_allocations(0), total_deallocations(0), mutex_() {
        this->block_count = block_count;
        this->block_size = block_size;

        size_t total_size = block_count * block_size;
        this->memory_block = std::aligned_alloc(alignof(BlockHeader), total_size);

        if(!this->memory_block) {
            throw std::bad_alloc();
        }

        this->use_count = 0;
        this->total_allocations = 0;
        this->total_deallocations = 0;
        // 初始化互斥锁
}

void FixedSizePool::initialize_blocks() {
    char *ptr = static_cast<char *>(this->memory_block);
    this->free_list = nullptr;

    for(size_t i = 0; i < this->block_count; ++i) {
        char *block_ptr = ptr + i * this->block_size;
        BlockHeader *header = reinterpret_cast<BlockHeader *>(block_ptr);
        header->reset();
        
        header->next = this->free_list;
        this->free_list = header;
    }
}


size_t FixedSizePool::get_block_size() const {
    return block_size;
}

size_t FixedSizePool::get_block_count() const {
    return block_count;
}
} // namespace MemoryPool
} // namespace StringProcessingLib

