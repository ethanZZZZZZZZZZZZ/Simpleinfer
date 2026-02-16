// 头文件保护符（防止被重复include报错）
#ifndef MY_INFER_BASE_ALLOC_H_
#define MY_INFER_BASE_ALLOC_H_

#include <memory>
#include <map>
#include <vector>
#include "base.h"

namespace base {
// 内存拷贝方向枚举
enum class MemcpyKind {
    kMemcpyCPU2CPU = 0,
    kMemcpyCPU2CUDA = 1,
    kMemcpyCUDA2CPU = 2,
    kMemcpyCUDA2CUDA = 3,
};

// 抽象基类：设备内存分配器
class DeviceAllocator {
    public:
     // explicit关键字防止cpp编译器在背后偷偷做类型转换
     explicit DeviceAllocator(DeviceType device_type) : device_type_(device_type) {}

     virtual DeviceType device_type() const {return device_type_;}

     // 纯虚函数（=0）：逼着子类必须实现具体的“释放”和“申请”逻辑ds
     virtual void release(void* ptr) const = 0;
     virtual void* allocate(size_t byte_size) const = 0;

     // 虚函数：拷贝内存
     virtual void memcpy(const void* src_ptr, void* dest_prt, size_t byte_size,
                         MemcpyKind memcpy_kind = MemcpyKind::kMemcpyCPU2CPU, 
                         void* stream = nullptr, 
                         bool need_sync = false) const;

     // 虚函数：把内存全部清零。在创建新的tensor时经常用到
     virtual void memset_zero(void* ptr, size_t byte_size, void* stream, bool need_sync = false);                

    private:
     DeviceType device_type_ = DeviceType::kDeviceUnknown;
};

// 具体实现类：CPU内存分配器
class CPUDeviceAllocator : public DeviceAllocator {
    public:
     explicit CPUDeviceAllocator();

     //overide
     void* allocate(size_t byte_size) const override;
     void release(void* ptr) const override;
};
// CUDA内存块描述
struct CudaMemoryBuffer {
    void* data;
    size_t byte_size;
    bool busy;

    CudaMemoryBuffer() = default;

    CudaMemoryBuffer(void* data, size_t byte_size, bool busy)
        : data(data), byte_size(byte_size), busy(busy) {}
};

// 具体类实现：CUDA显存分配器，带内存池功能
class CUDADeviceAllocator : public DeviceAllocator {
    public:
     explicit CUDADeviceAllocator();

     void* allocate(size_t byte_size) const override;
     void release(void* ptr) const override;

    private:
    // mutable允许这些变量在const函数中被修改
    // ket(int)是GPU设备的ID，因为一台机器可能有多卡
    mutable std::map<int, size_t> no_busy_cnt_;
    // 存放和复用几MB的零散内存
    mutable std::map<int, std::vector<CudaMemoryBuffer>> big_buffers_map_;
    // 存放复用超大块连续内存
    mutable std::map<int, std::vector<CudaMemoryBuffer>> cuda_buffers_map_;
};

// 单例工厂：用于获取全局唯一的CUDA分配器
class CUDADeviceAllocator:


}