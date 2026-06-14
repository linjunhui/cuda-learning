#include "memory_pool.hpp"

#include <cassert>
#include <cstdint>


namespace MemoryPool {

static void test_single_block_chain() {
    size_t block_size = 64, block_count = 15;
    FixedMemoryPool pool(block_size, block_count);
    pool.init_pool();

    assert(pool.free_list != nullptr);
    assert(pool.free_list->is_free);
    assert(pool.free_list->next_block != nullptr);
}

}  // namespace MemoryPool

int main() {
    using namespace MemoryPool;
    test_single_block_chain();
    return 0;
}
