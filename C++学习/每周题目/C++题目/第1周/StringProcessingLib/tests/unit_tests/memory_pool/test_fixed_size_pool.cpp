#include "StringProcessingLib/memory_pool/fixed_size_pool.h"

#include <gtest/gtest.h>

namespace StringProcessingLib {
namespace MemoryPool {

TEST(FixedSizePoolTest, Initialization) {
    FixedSizePool pool(1024, 100);
    EXPECT_EQ(pool.get_block_size(), 1024);
    EXPECT_EQ(pool.get_block_count(), 100);
}

} // namespace MemoryPool
} // namespace StringProcessingLib