#include "block_header.hpp"

#include <cstddef>
#include <cstdlib>

namespace MemoryPool {
    class FixedMemoryPool {
        public:
            BlockHeader *free_list;
            size_t block_size;
            size_t block_count;

        public:
            FixedMemoryPool(size_t block_size, size_t block_count): block_size(block_size), block_count(block_count) {

            }

            void init_pool() {
                size_t total_bytes = block_size * block_count;

                BlockHeader* current_block = reinterpret_cast<BlockHeader *>(std::aligned_alloc(alignof(BlockHeader), total_bytes));
                free_list = current_block;
                current_block->is_free = true;

                for(int i = 0; i < block_count - 1; i++) {
                    BlockHeader* next_block = reinterpret_cast<BlockHeader *>(reinterpret_cast<char *>(current_block) + block_size);
                    next_block->is_free = true;
                    current_block->next_block = next_block;
                    current_block = next_block;
                }
                current_block->next_block = nullptr;
            }
    };
}