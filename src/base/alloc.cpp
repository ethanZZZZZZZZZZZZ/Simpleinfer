#include "base/alloc.h"
#include <cuda_runtime_api.h>

namespace base {

// ============================================================================
// memcpy：中央物流调度中心，负责将数据从源头搬运到目标地
// ============================================================================
void DeviceAllocator::memcpy(const void* src_ptr, void* dest_ptr, size_t byte_size,
                              MemcpyKind memcpy_kind, void* stream, bool need_sync) const {
  // 1. 防御性编程：发车前检查，如果货物（src）或仓库（dest）指针是空的，立刻报错拦截
  CHECK_NE(src_ptr, nullptr);
  CHECK_NE(dest_ptr, nullptr);

  // 2. 如果要搬运的字节数是 0，直接下班，什么都不干
  if (!byte_size) {
    return;
  }

  // 3. 把对外的void* stream还原成真正的CUDA流指针
  cudaStream_t stream_ = nullptr;
  if (stream) {
    stream_ = static_cast<CUstream_st*>(stream);
  }

  // 4. 物流路由分发：根据搬运方向，调用不同的底层API
  if (memcpy_kind == MemcpyKind::kMemcpyCPU2CPU) {
    // 纯CPU搬运，直接cpp标准库
    std::memcpy(dest_ptr, src_ptr, byte_size);
  } else if (memcpy_kind == MemcpyKind::kMemcpyCPU2CUDA) {
    // CPU搬运到GPU
    if (!stream_) {
      cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyHostToDevice);
    } else {
      cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyHostToDevice, stream_);
    }
  } else if (memcpy_kind == MemcpyKind::kMemcpyCUDA2CPU) {
    // GPU搬运到CPU
    if (!stream_) {
      cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToHost);
    } else {
      cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToHost, stream_);
    }
  } else if (memcpy_kind == MemcpyKind::kMemcpyCUDA2CUDA) {
    // GPU 内部不同显存区域之间搬运 (Device To Device)
    if (!stream_) {
      cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToDevice);
    } else {
      cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToDevice, stream_);
    }
  } else {
    // 5. 兜底保护：万一有人乱传了一个不存在的枚举值，直接原地引爆程序
    LOG(FATAL) << "Unknown memcpy kind: " << int(memcpy_kind);
  }

  // 6. 全局同步要求：如果上层要求必须等搬运彻底结束才走，就强行阻塞CPU
  if (need_sync) {
    cudaDeviceSynchronize();
  }
}

// ============================================================================
// memset_zero：快速将指定的内存区域全部刷成 0 (通常用于初始化 Tensor)
// ============================================================================
void DeviceAllocator::memset_zero(void* ptr, size_t byte_size, void* stream,
                                  bool need_sync) {
  // 1. 检查自己的身份：作为基类方法，它必须知道当前到底是 CPU 还是 GPU 在干活
  CHECK(device_type_ != base::DeviceType::kDeviceUnknown);

  // 2. 路由分支：如果你是 CPU 分配器
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    // 直接用 C 标准库的 memset 把物理内存刷成 0
    std::memset(ptr, 0, byte_size);
  }
  // 3. 路由分支：如果你是 CUDA 分配器
  else {
    // 同样处理流的还原
    if (stream) {
      cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
      // 让 GPU 去异步刷 0
      cudaMemsetAsync(ptr, 0, byte_size, stream_);
    } else {
      // 让 GPU 去同步刷 0
      cudaMemset(ptr, 0, byte_size);
    }

    // 如果上层要求同步，就在这儿卡住等 GPU 刷完
    if (need_sync) {
      cudaDeviceSynchronize();
    }
  }
}
}