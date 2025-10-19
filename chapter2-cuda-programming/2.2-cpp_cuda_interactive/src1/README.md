## #define
- #define 宏是用在预处理阶段的，它将替换掉宏定义的文本。

## 手动编译链接
- 用g++ 编译main1.cpp; g++ -c  main1.cpp -o main1.o
- 用nvcc 编译 hello.cu; /usr/local/cuda-12.6/bin/nvcc -c hello.cu -o hello.o
- 链接两个编译好的文件，并生成可执行文件; /usr/local/cuda-12.6/bin/nvcc main1.o hello.o -o main1.out