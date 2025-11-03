# CMake generated Testfile for 
# Source directory: /home/jonson/cuda-learning/每周题目/C++题目/第1周/StringProcessingLib
# Build directory: /home/jonson/cuda-learning/每周题目/C++题目/第1周/StringProcessingLib/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(FixedSizePoolTest "/home/jonson/cuda-learning/每周题目/C++题目/第1周/StringProcessingLib/build/test_fixed_size_pool")
set_tests_properties(FixedSizePoolTest PROPERTIES  WORKING_DIRECTORY "/home/jonson/cuda-learning/每周题目/C++题目/第1周/StringProcessingLib/build" _BACKTRACE_TRIPLES "/home/jonson/cuda-learning/每周题目/C++题目/第1周/StringProcessingLib/CMakeLists.txt;64;add_test;/home/jonson/cuda-learning/每周题目/C++题目/第1周/StringProcessingLib/CMakeLists.txt;0;")
subdirs("_deps/googletest-build")
