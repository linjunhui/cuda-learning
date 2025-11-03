# BlockHeader源文件
set(BLOCK_HEADER_SOURCES
    block_header.cpp
)

# FixedSizePool源文件（第二阶段添加）
set(FIXED_SIZE_POOL_SOURCES
    # fixed_size_pool.cpp  # 将在第二阶段添加
)

# 所有源文件
set(ALL_SOURCES
    ${BLOCK_HEADER_SOURCES}
    ${FIXED_SIZE_POOL_SOURCES}
)

# 创建库
add_library(stringprocessinglib STATIC ${ALL_SOURCES})

# 包含头文件目录
target_include_directories(stringprocessinglib PUBLIC
    ${CMAKE_SOURCE_DIR}/../include
)

# 链接pthread库（用于mutex，将在第二阶段需要）
target_link_libraries(stringprocessinglib PUBLIC pthread)

