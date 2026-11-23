// Copyright 2025 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lidar_frnet_ros1/preprocess_kernel.hpp"
#include "lidar_frnet_ros1/utils.hpp"

#include "lidar_frnet_ros1/cuda_utils.hpp"
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <algorithm>
#include <cstdint>

namespace lidar_frnet_ros1
{
constexpr float PI_F = 3.14159265358979323846f;

PreprocessCuda::PreprocessCuda(const utils::NetworkParams & params, cudaStream_t stream)
: projection_{params.fov, params.interpolation, params.frustum, params.crop_box_enabled,
    params.crop_box_bounds},
  max_points_(params.num_points_profile.max),
  max_voxels_(params.num_unique_coors_profile.max),
  interpolation_stride_(static_cast<uint32_t>(params.interpolation_stride)),
  stream_(stream),
  sorted_keys_(cuda_utils::make_unique<int64_t>(max_points_)),
  sorted_indices_(cuda_utils::make_unique<int64_t>(max_points_)),
  labels_(cuda_utils::make_unique<int64_t>(max_points_))
{
  size_t sort_bytes = 0;
  size_t scan_bytes = 0;
  CHECK_CUDA_ERROR(cub::DeviceRadixSort::SortPairs(
    nullptr, sort_bytes, static_cast<const int64_t *>(nullptr), sorted_keys_.get(), labels_.get(),
    sorted_indices_.get(), max_points_, 0, sizeof(int64_t) * 8, stream_));
  CHECK_CUDA_ERROR(cub::DeviceScan::InclusiveSum(
    nullptr, scan_bytes, labels_.get(), labels_.get(), max_points_, stream_));
  cub_storage_bytes_ = std::max(sort_bytes, scan_bytes);
  cub_storage_ = cuda_utils::make_unique<uint8_t>(cub_storage_bytes_);
}

void PreprocessCuda::setCropBoxTransform(const float * T_sensor_to_ref_12)
{
  if (T_sensor_to_ref_12) {
    std::copy_n(T_sensor_to_ref_12, 12, projection_.sensor_to_crop.begin());
  } else {
    projection_.sensor_to_crop = {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0};
  }
}

/** Packed xyzi in half precision for atomic projection buffer. */
struct alignas(8) half4
{
  half4() = default;
  __device__ half4(__half x, __half y, __half z, __half w) : x(x), y(y), z(z), w(w) {}

  __half x, y, z, w;
};

/**
 * @brief Pack four half values into one 64-bit word (device).
 */
__device__ inline unsigned long long pack_half4(const half4 & h)
{
  unsigned long long packed = 0;
  auto p = reinterpret_cast<uint16_t *>(&packed);
  p[0] = __half_as_ushort(h.x);
  p[1] = __half_as_ushort(h.y);
  p[2] = __half_as_ushort(h.z);
  p[3] = __half_as_ushort(h.w);
  return packed;
}

/**
 * @brief Unpack 64-bit word into four half values (device).
 */
__device__ inline void unpack_half4(unsigned long long packed, half4 & h)
{
  auto p = reinterpret_cast<uint16_t *>(&packed);
  h.x = __ushort_as_half(p[0]);
  h.y = __ushort_as_half(p[1]);
  h.z = __ushort_as_half(p[2]);
  h.w = __ushort_as_half(p[3]);
}

/**
 * @brief Atomic max-by-depth: store new_val at address only if its depth (x^2+y^2+z^2) is greater
 *        than the current stored value (device).
 */
__device__ void atomicMaxByDepth_half4(unsigned long long * address, const half4 & new_val)
{
  half4 old_val;
  unsigned long long old_packed = atomicCAS(address, 0ULL, 0ULL);
  const float depth_new = __half2float(new_val.x) * __half2float(new_val.x) +
                          __half2float(new_val.y) * __half2float(new_val.y) +
                          __half2float(new_val.z) * __half2float(new_val.z);
  const auto new_packed = pack_half4(new_val);

  while (true) {
    unpack_half4(old_packed, old_val);

    float depth_old = __half2float(old_val.x) * __half2float(old_val.x) +
                      __half2float(old_val.y) * __half2float(old_val.y) +
                      __half2float(old_val.z) * __half2float(old_val.z);

    if (old_packed != 0 && depth_new <= depth_old) return;

    unsigned long long prev = atomicCAS(address, old_packed, new_packed);

    if (prev == old_packed) break;

    old_packed = prev;  // retry
  }
}

/**
 * @brief Project 3D point to 2D image coordinates (yaw/pitch to pixel x,y). Clamped to [0,
 * scale-1].
 * @param point 3D point in sensor frame
 * @param grid Dimensions of the projection grid
 * @param fov Vertical field of view
 * @return int2 (pixel x, pixel y)
 */
__device__ int2 project2d(
  const float3 point, const utils::Dims2d & grid, const utils::FieldOfView & fov)
{
  float depth = sqrtf(point.x * point.x + point.y * point.y + point.z * point.z);
  float yaw = -atan2f(point.y, point.x);
  float pitch = depth > 0.0f ? asinf(fminf(1.0f, fmaxf(-1.0f, point.z / depth))) : 0.0f;
  float proj_x = 0.5f * (yaw / PI_F + 1.0f);
  float proj_y = 1.0f - (pitch - fov.down) / fov.total;
  proj_x *= static_cast<float>(grid.w);
  proj_y *= static_cast<float>(grid.h);
  return make_int2(
    static_cast<int32_t>(fminf(fmaxf(proj_x, 0.0f), static_cast<float>(grid.w) - 1.0f)),
    static_cast<int32_t>(fminf(fmaxf(proj_y, 0.0f), static_cast<float>(grid.h) - 1.0f)));
}

/** @brief Trait: get intensity as float from point type (device). */
template <typename PointT>
__device__ inline float get_intensity(const PointT & point);

template <>
__device__ inline float get_intensity(const CloudPointTypeXYZI & point)
{
  return point.intensity;
}

template <>
__device__ inline float get_intensity(const CloudPointTypeXYZIRC & point)
{
  return static_cast<float>(point.intensity);
}

template <>
__device__ inline float get_intensity(const CloudPointTypeXYZIRT & point)
{
  return point.intensity;
}

template <>
__device__ inline float get_intensity(const CloudPointTypeXYZIRADRT & point)
{
  return point.intensity;
}

template <>
__device__ inline float get_intensity(const CloudPointTypeXYZIRCAEDT & point)
{
  return static_cast<float>(point.intensity);
}

/**
 * @brief Kernel: project points to 2D, filter by ego crop box, write xyzi/coors/keys and optional
 *        compact copy; keep the farthest point in each projection pixel.
 * @tparam PointT Input point type (e.g. CloudPointTypeXYZIRCAEDT)
 */
template <typename PointT>
__global__ void projectPoints_kernel(
  const PointT * cloud, const uint32_t num_points, uint32_t * output_num_points,
  float * output_points, int64_t * output_coors, int64_t * output_coors_keys,
  uint32_t * output_proj_idxs, uint64_t * output_proj_2d, PointT * output_cloud_compact,
  ProjectionParams projection)
{
  uint32_t point_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (point_idx >= num_points) return;

  const PointT & point = cloud[point_idx];
  const auto point3d = make_float3(point.x, point.y, point.z);
  const float intensity = get_intensity(point);
  if (!isfinite(point.x) || !isfinite(point.y) || !isfinite(point.z) || !isfinite(intensity)) {
    return;
  }

  // Ego crop box in reference frame: transform point to reference, then test AABB
  if (projection.crop_box_enabled) {
    const auto & tf = projection.sensor_to_crop;
    const auto & bounds = projection.crop_box_bounds;
    const float px = point3d.x, py = point3d.y, pz = point3d.z;
    const float p_ref_x = tf[0] * px + tf[1] * py + tf[2] * pz + tf[9];
    const float p_ref_y = tf[3] * px + tf[4] * py + tf[5] * pz + tf[10];
    const float p_ref_z = tf[6] * px + tf[7] * py + tf[8] * pz + tf[11];
    if (
      p_ref_x > bounds[0] && p_ref_x < bounds[3] && p_ref_y > bounds[1] && p_ref_y < bounds[4] &&
      p_ref_z > bounds[2] && p_ref_z < bounds[5]) {
      return;
    }
  }

  const uint32_t append_idx = atomicAdd(output_num_points, 1);
  const auto proj_coor = project2d(point3d, projection.frustum, projection.fov);

  output_points[append_idx * 4 + 0] = point.x;
  output_points[append_idx * 4 + 1] = point.y;
  output_points[append_idx * 4 + 2] = point.z;
  output_points[append_idx * 4 + 3] = intensity;

  output_coors[append_idx * 3 + 0] = 0;
  output_coors[append_idx * 3 + 1] = proj_coor.y;
  output_coors[append_idx * 3 + 2] = proj_coor.x;

  output_coors_keys[append_idx] = proj_coor.y * projection.frustum.w + proj_coor.x;

  if (output_cloud_compact != nullptr) {
    output_cloud_compact[append_idx] = point;
  }

  if (output_proj_idxs != nullptr && output_proj_2d != nullptr) {
    const auto proj_point = project2d(point3d, projection.interpolation, projection.fov);
    const auto proj_idx = proj_point.y * projection.interpolation.w + proj_point.x;
    auto output_proj_2d_address = reinterpret_cast<unsigned long long *>(&output_proj_2d[proj_idx]);
    atomicMaxByDepth_half4(output_proj_2d_address, half4(point.x, point.y, point.z, intensity));
    atomicExch(&output_proj_idxs[proj_idx], 1);
  }
}

/**
 * @brief Launch projectPoints_kernel for the given point type.
 */
template <typename PointT>
cudaError_t PreprocessCuda::projectPoints_launch_impl(
  const PointT * cloud, const uint32_t num_points, uint32_t * output_num_points,
  float * output_points, int64_t * output_coors, int64_t * output_coors_keys,
  uint32_t * output_proj_idxs, uint64_t * output_proj_2d, void * output_cloud_compact)
{
  dim3 block(utils::divup(num_points, utils::kernel_1d_size));
  dim3 threads(utils::kernel_1d_size);

  projectPoints_kernel<<<block, threads, 0, stream_>>>(
    cloud, num_points, output_num_points, output_points, output_coors, output_coors_keys,
    output_proj_idxs, output_proj_2d, static_cast<PointT *>(output_cloud_compact), projection_);

  return cudaGetLastError();
}

// Explicit instantiations
template cudaError_t PreprocessCuda::projectPoints_launch_impl<CloudPointTypeXYZI>(
  const CloudPointTypeXYZI *, const uint32_t, uint32_t *, float *, int64_t *, int64_t *, uint32_t *,
  uint64_t *, void *);
template cudaError_t PreprocessCuda::projectPoints_launch_impl<CloudPointTypeXYZIRC>(
  const CloudPointTypeXYZIRC *, const uint32_t, uint32_t *, float *, int64_t *, int64_t *,
  uint32_t *, uint64_t *, void *);
template cudaError_t PreprocessCuda::projectPoints_launch_impl<CloudPointTypeXYZIRT>(
  const CloudPointTypeXYZIRT *, const uint32_t, uint32_t *, float *, int64_t *, int64_t *,
  uint32_t *, uint64_t *, void *);
template cudaError_t PreprocessCuda::projectPoints_launch_impl<CloudPointTypeXYZIRADRT>(
  const CloudPointTypeXYZIRADRT *, const uint32_t, uint32_t *, float *, int64_t *, int64_t *,
  uint32_t *, uint64_t *, void *);
template cudaError_t PreprocessCuda::projectPoints_launch_impl<CloudPointTypeXYZIRCAEDT>(
  const CloudPointTypeXYZIRCAEDT *, const uint32_t, uint32_t *, float *, int64_t *, int64_t *,
  uint32_t *, uint64_t *, void *);

/**
 * @brief Dispatch projectPoints by input format to templated implementation.
 */
cudaError_t PreprocessCuda::projectPoints_launch(
  const void * cloud, const uint32_t num_points, CloudFormat format, uint32_t * output_num_points,
  float * output_points, int64_t * output_coors, int64_t * output_coors_keys,
  uint32_t * output_proj_idxs, uint64_t * output_proj_2d, void * output_cloud_compact)
{
  if (num_points == 0 || num_points > max_points_) {
    return cudaErrorInvalidValue;
  }
  switch (format) {
    case CloudFormat::XYZIRCAEDT:
      return projectPoints_launch_impl(
        static_cast<const CloudPointTypeXYZIRCAEDT *>(cloud), num_points, output_num_points,
        output_points, output_coors, output_coors_keys, output_proj_idxs, output_proj_2d,
        output_cloud_compact);
    case CloudFormat::XYZIRADRT:
      return projectPoints_launch_impl(
        static_cast<const CloudPointTypeXYZIRADRT *>(cloud), num_points, output_num_points,
        output_points, output_coors, output_coors_keys, output_proj_idxs, output_proj_2d,
        output_cloud_compact);
    case CloudFormat::XYZIRC:
      return projectPoints_launch_impl(
        static_cast<const CloudPointTypeXYZIRC *>(cloud), num_points, output_num_points,
        output_points, output_coors, output_coors_keys, output_proj_idxs, output_proj_2d,
        output_cloud_compact);
    case CloudFormat::XYZIRT:
      return projectPoints_launch_impl(
        static_cast<const CloudPointTypeXYZIRT *>(cloud), num_points, output_num_points,
        output_points, output_coors, output_coors_keys, output_proj_idxs, output_proj_2d,
        output_cloud_compact);
    case CloudFormat::XYZI:
      return projectPoints_launch_impl(
        static_cast<const CloudPointTypeXYZI *>(cloud), num_points, output_num_points,
        output_points, output_coors, output_coors_keys, output_proj_idxs, output_proj_2d,
        output_cloud_compact);
    default:
      return cudaErrorInvalidValue;
  }
}

/**
 * @brief Kernel: for empty pixels with valid left and right neighbors, interpolate xyzi and coors,
 *        append to output and update output_num_points atomically.
 */
__global__ void interpolatePoints_kernel(
  const uint32_t * proj_idxs, const uint64_t * proj_2d, uint32_t * output_num_points,
  float * output_points, int64_t * output_coors, int64_t * output_coors_keys,
  ProjectionParams projection, uint32_t max_points, uint32_t interpolation_stride)
{
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= projection.interpolation.w || y >= projection.interpolation.h) return;
  // Shift the sampled columns between rows so nearby rings use different columns.
  if (interpolation_stride > 1 && (x + y) % interpolation_stride != 0) return;

  uint32_t idx = y * projection.interpolation.w + x;

  if (proj_idxs[idx]) {
    return;  // Skip if the pixel is already filled
  }

  // Interpolate from left and right neighbors
  if (x > 0 && x < projection.interpolation.w - 1) {
    if (proj_idxs[idx - 1] && proj_idxs[idx + 1]) {
      // Interpolate points
      half4 left, right;
      unpack_half4(proj_2d[idx - 1], left);
      unpack_half4(proj_2d[idx + 1], right);

      // Compute the mean of the valid neighbors
      const auto interpolated_point = make_float4(
        (__half2float(left.x + right.x)) * 0.5f, (__half2float(left.y + right.y)) * 0.5f,
        (__half2float(left.z + right.z)) * 0.5f, (__half2float(left.w + right.w)) * 0.5f);

      const auto append_idx = atomicAdd(output_num_points, 1);
      if (append_idx >= max_points) return;
      output_points[append_idx * 4 + 0] = interpolated_point.x;
      output_points[append_idx * 4 + 1] = interpolated_point.y;
      output_points[append_idx * 4 + 2] = interpolated_point.z;
      output_points[append_idx * 4 + 3] = interpolated_point.w;

      const auto proj_coor = project2d(
        make_float3(interpolated_point.x, interpolated_point.y, interpolated_point.z),
        projection.frustum, projection.fov);

      output_coors[append_idx * 3 + 0] = 0;
      output_coors[append_idx * 3 + 1] = proj_coor.y;
      output_coors[append_idx * 3 + 2] = proj_coor.x;

      output_coors_keys[append_idx] = proj_coor.y * projection.frustum.w + proj_coor.x;
    }
  }
}

/**
 * @brief Launch interpolatePoints_kernel on 2D grid.
 */
cudaError_t PreprocessCuda::interpolatePoints_launch(
  uint32_t * proj_idxs, uint64_t * proj_2d, uint32_t * output_num_points, float * output_points,
  int64_t * output_coors, int64_t * output_coors_keys)
{
  if (interpolation_stride_ == 0) return cudaSuccess;

  dim3 block(
    utils::divup(projection_.interpolation.w, utils::kernel_2d_size),
    utils::divup(projection_.interpolation.h, utils::kernel_2d_size));
  dim3 threads(utils::kernel_2d_size, utils::kernel_2d_size);

  interpolatePoints_kernel<<<block, threads, 0, stream_>>>(
    proj_idxs, proj_2d, output_num_points, output_points, output_coors, output_coors_keys,
    projection_, max_points_, interpolation_stride_);

  return cudaGetLastError();
}

__global__ void markNewVoxels_kernel(const int64_t * keys, uint32_t num_points, int64_t * labels)
{
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_points) {
    labels[i] = i > 0 && keys[i] != keys[i - 1];
  }
}

__global__ void mapVoxels_kernel(
  const int64_t * coors, const int64_t * keys, const int64_t * sorted_indices,
  const int64_t * labels, uint32_t num_points, uint32_t max_voxels, int64_t * voxel_coors,
  int64_t * inverse_map)
{
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_points) return;

  const auto source = sorted_indices[i];
  const auto label = labels[i];
  inverse_map[source] = label;
  if (label < max_voxels && (i == 0 || keys[i] != keys[i - 1])) {
    voxel_coors[3 * label] = coors[3 * source];
    voxel_coors[3 * label + 1] = coors[3 * source + 1];
    voxel_coors[3 * label + 2] = coors[3 * source + 2];
  }
}

uint32_t PreprocessCuda::generateUniqueCoors(
  const uint32_t num_points, const int64_t * coors, const int64_t * coors_keys,
  int64_t * output_voxel_coors, int64_t * output_inverse_map)
{
  if (num_points == 0 || num_points > max_points_) {
    return 0;
  }

  auto * labels = labels_.get();
  thrust::sequence(thrust::cuda::par.on(stream_), labels, labels + num_points);
  size_t storage_bytes = cub_storage_bytes_;
  CHECK_CUDA_ERROR(cub::DeviceRadixSort::SortPairs(
    cub_storage_.get(), storage_bytes, coors_keys, sorted_keys_.get(), labels,
    sorted_indices_.get(), num_points, 0, sizeof(int64_t) * 8, stream_));

  const auto blocks = utils::divup(num_points, utils::kernel_1d_size);
  markNewVoxels_kernel<<<blocks, utils::kernel_1d_size, 0, stream_>>>(
    sorted_keys_.get(), num_points, labels);
  CHECK_CUDA_ERROR(cudaGetLastError());
  storage_bytes = cub_storage_bytes_;
  CHECK_CUDA_ERROR(cub::DeviceScan::InclusiveSum(
    cub_storage_.get(), storage_bytes, labels, labels, num_points, stream_));
  mapVoxels_kernel<<<blocks, utils::kernel_1d_size, 0, stream_>>>(
    coors, sorted_keys_.get(), sorted_indices_.get(), labels, num_points, max_voxels_,
    output_voxel_coors, output_inverse_map);
  CHECK_CUDA_ERROR(cudaGetLastError());

  int64_t last_label = 0;
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    &last_label, labels + num_points - 1, sizeof(last_label), cudaMemcpyDeviceToHost, stream_));
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));
  return static_cast<uint32_t>(last_label + 1);
}

}  // namespace lidar_frnet_ros1
