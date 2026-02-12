cmake_minimum_required(VERSION 3.10)
project(Simpleinfer LANGUAGES CXX CUDA)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 编译选项：开启所有警告（对自己狠一点，代码质量才高）
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 设置输出目录：让编译出的 .so 和可执行文件都去 build/lib 和 build/bin
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 包含头文件路径
include_directories(${CMAKE_SOURCE_DIR}/include)

# 添加子目录
add_subdirectory(src)

