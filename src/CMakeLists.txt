# src/CMakeLists.txt

# 1. 自动查找当前目录下所有的 .cpp 文件 (包括 main.cpp)
#    这样以后你添加新的 .cpp 文件，不用每次都改 CMakeLists
file(GLOB_RECURSE SRC_FILES *.cpp)

# 2. 告诉 CMake 这里的代码要编译成一个可执行文件
#    名字叫 MyInfer_Demo，源文件使用上面找到的那些
add_executable(Simpleinfer_Demo ${SRC_FILES}
        base/alloc.cpp)

# 3. 指定头文件路径
#    这样我们在 cpp 里写 #include "tensor.h" 时，编译器能找到 include/ 目录
target_include_directories(Simpleinfer_Demo PUBLIC ${CMAKE_SOURCE_DIR}/include)

# 4. (未来预留) 如果以后有 CUDA 代码，可以在这里链接 CUDA 库
# target_link_libraries(MyInfer_Demo PRIVATE cudart)