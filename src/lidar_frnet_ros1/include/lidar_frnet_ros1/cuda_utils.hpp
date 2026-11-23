// Copyright 2026
//
// Licensed under the Apache License, Version 2.0.

#ifndef LIDAR_FRNET_ROS1__CUDA_UTILS_HPP_
#define LIDAR_FRNET_ROS1__CUDA_UTILS_HPP_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace lidar_frnet_ros1::cuda_utils
{

inline void check_cuda_error(cudaError_t status, const char * expr, const char * file, int line)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(
      std::string("CUDA error at ") + file + ":" + std::to_string(line) + " for " + expr + ": " +
      cudaGetErrorString(status));
  }
}

template <typename T>
struct CudaDeleter
{
  void operator()(T * ptr) const
  {
    if (ptr != nullptr) {
      cudaFree(ptr);
    }
  }
};

template <typename T>
using CudaUniquePtr = std::unique_ptr<T, CudaDeleter<T>>;

template <typename T>
CudaUniquePtr<T> make_unique(std::size_t count)
{
  T * ptr = nullptr;
  check_cuda_error(cudaMalloc(reinterpret_cast<void **>(&ptr), sizeof(T) * count), "cudaMalloc", __FILE__, __LINE__);
  return CudaUniquePtr<T>(ptr);
}

struct CudaStreamDeleter
{
  void operator()(cudaStream_t stream) const { cudaStreamDestroy(stream); }
};

using CudaStream = std::unique_ptr<std::remove_pointer_t<cudaStream_t>, CudaStreamDeleter>;

inline CudaStream make_stream()
{
  cudaStream_t stream = nullptr;
  check_cuda_error(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags", __FILE__, __LINE__);
  return CudaStream(stream);
}

template <typename T>
void clear_async(T * ptr, std::size_t count, cudaStream_t stream)
{
  check_cuda_error(cudaMemsetAsync(ptr, 0, sizeof(T) * count, stream), "cudaMemsetAsync", __FILE__, __LINE__);
}

}  // namespace lidar_frnet_ros1::cuda_utils

#define CHECK_CUDA_ERROR(expr) \
  ::lidar_frnet_ros1::cuda_utils::check_cuda_error((expr), #expr, __FILE__, __LINE__)

#endif  // LIDAR_FRNET_ROS1__CUDA_UTILS_HPP_
