
# CUDA矩阵库 Makefile

# 编译器设置
NVCC = nvcc
CXX = g++

# 编译选项
NVCC_FLAGS = -std=c++11 -O2
CXX_FLAGS = -std=c++11 -O2 -Wall -Wextra -Wpedantic

# 包含路径 - 使用更清晰的路径管理
INCLUDE_DIRS = include include/cuda_matrix_lib
INCLUDES = $(addprefix -I, $(INCLUDE_DIRS))

# 库链接
LIBS = -lcudart

# 源文件
CORE_SOURCES = src/core/error_handler.cpp
TEST_SOURCES = tests/unit_tests/test_error_handler.cpp

# 目标文件
CORE_OBJECTS = $(CORE_SOURCES:.cpp=.o)
TEST_OBJECTS = $(TEST_SOURCES:.cpp=.o)

# 可执行文件
TEST_EXECUTABLE = tests/unit_tests/test_error_handler

# 目录创建
MKDIR = mkdir -p

# 默认目标
all: $(TEST_EXECUTABLE)

# 编译核心库对象文件
%.o: %.cpp
	@echo "编译 $<..."
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

# 编译测试可执行文件
$(TEST_EXECUTABLE): $(CORE_OBJECTS) $(TEST_OBJECTS)
	@echo "链接测试可执行文件..."
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -o $@ $^ $(LIBS)

# 运行测试
test: $(TEST_EXECUTABLE)
	@echo "运行单元测试..."
	./$(TEST_EXECUTABLE)

# 清理
clean:
	rm -f $(CORE_OBJECTS) $(TEST_OBJECTS) $(TEST_EXECUTABLE)

# 帮助信息
help:
	@echo "可用目标:"
	@echo "  all     - 编译所有目标"
	@echo "  test    - 编译并运行测试"
	@echo "  clean   - 清理编译文件"
	@echo "  help    - 显示此帮助信息"

.PHONY: all test clean help
