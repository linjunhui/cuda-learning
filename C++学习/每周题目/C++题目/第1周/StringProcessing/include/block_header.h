#ifndef BLOCK_HEADER_H
#define BLOCK_HEADER_H

namespace StringProcessingLib::MemoryPool {
    struct BlockHeader {
        BlockHeader* next;
    };
}

#endif