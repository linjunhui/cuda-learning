#pragma once

namespace MemoryPool {

class BlockHeader {
public:
    bool is_free;
    BlockHeader* next_block;
};

}  // namespace MemoryPool
