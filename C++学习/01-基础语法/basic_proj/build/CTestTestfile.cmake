# CMake generated Testfile for 
# Source directory: /home/jonson/cuda-learning/C++学习/01-基础语法/basic_proj
# Build directory: /home/jonson/cuda-learning/C++学习/01-基础语法/basic_proj/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(MathTest "/home/jonson/cuda-learning/C++学习/01-基础语法/basic_proj/build/math_test")
set_tests_properties(MathTest PROPERTIES  _BACKTRACE_TRIPLES "/home/jonson/cuda-learning/C++学习/01-基础语法/basic_proj/CMakeLists.txt;78;add_test;/home/jonson/cuda-learning/C++学习/01-基础语法/basic_proj/CMakeLists.txt;0;")
subdirs("_deps/googletest-build")
