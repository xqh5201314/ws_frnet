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

#ifndef LIDAR_FRNET_ROS1__PREPROCESS_KERNEL_HPP_
#define LIDAR_FRNET_ROS1__PREPROCESS_KERNEL_HPP_

#include "lidar_frnet_ros1/cuda_utils.hpp"
#include "lidar_frnet_ros1/point_type.hpp"
#include "lidar_frnet_ros1/utils.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>

namespace lidar_frnet_ros1
{

struct ProjectionParams
{
  utils::FieldOfView fov;
  utils::Dims2d interpolation;
  utils::Dims2d frustum;
  bool crop_box_enabled;
  std::array<float, 6> crop_box_bounds;
  std::array<float, 12> sensor_to_crop{1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0};
};

/**
 * @brief CUDA preprocess: project points to 2D frustum, interpolate, generate unique voxel coors.
 */
class PreprocessCuda
{
public:
  PreprocessCuda(const utils::NetworkParams & params, cudaStream_t stream);

  /**
   * @brief Set per-frame transform from sensor to reference frame (12 floats: row-major 3x3 R then
   *        tx, ty, tz). Call before projectPoints_launch when crop box is enabled.
   * @param T_sensor_to_ref_12 Pointer to 12 floats, or nullptr for identity
   */
  void setCropBoxTransform(const float * T_sensor_to_ref_12);

  /**
   * @brief Sort coors by key, compute unique voxel coors and inverse map (point index -> voxel id).
   * @param num_points Number of points (coors/coors_keys length)
   * @param coors Point coordinates (3 per point)
   * @param coors_keys Sort key per point
   * @param output_voxel_coors Output: unique voxel coors (3 per voxel)
   * @param output_inverse_map Output: per-point voxel index
   * @return Number of unique voxels, which may exceed the output profile capacity
   */
  uint32_t generateUniqueCoors(
    const uint32_t num_points, const int64_t * coors, const int64_t * coors_keys,
    int64_t * output_voxel_coors, int64_t * output_inverse_map);

  /**
   * @brief Project points to 2D frustum coordinates and optionally write compact point copy.
   *
   * Filters points inside ego crop box when enabled. Dispatches to templated implementation by
   * input format.
   *
   * @param cloud Input point cloud (device)
   * @param num_points Number of points
   * @param format Input point format
   * @param output_num_points Output: number of points after crop (written by kernel)
   * @param output_points Output: xyzi (4 floats per point)
   * @param output_coors Output: (0, y, x) per point
   * @param output_coors_keys Output: sort key per point
   * @param output_proj_idxs Output: per-pixel filled flag, or nullptr to skip interpolation projection
   * @param output_proj_2d Output: per-pixel packed point, or nullptr to skip interpolation projection
   * @param output_cloud_compact Optional compact copy of input points for filtered output; may be
   *        nullptr
   * @return cudaError_t
   */
  cudaError_t projectPoints_launch(
    const void * cloud, const uint32_t num_points, CloudFormat format, uint32_t * output_num_points,
    float * output_points, int64_t * output_coors, int64_t * output_coors_keys,
    uint32_t * output_proj_idxs, uint64_t * output_proj_2d, void * output_cloud_compact = nullptr);

  /**
   * @brief Interpolate empty pixels from neighbors; append interpolated points and update coors.
   * @param proj_idxs Per-pixel filled flags (1 = has point)
   * @param proj_2d Per-pixel packed xyzi (from projectPoints)
   * @param output_num_points Input/output: current count, incremented by interpolated points
   * @param output_points Append: xyzi for interpolated points
   * @param output_coors Append: (0, y, x) for interpolated points
   * @param output_coors_keys Append: sort key for interpolated points
   * @return cudaError_t
   */
  cudaError_t interpolatePoints_launch(
    uint32_t * proj_idxs, uint64_t * proj_2d, uint32_t * output_num_points, float * output_points,
    int64_t * output_coors, int64_t * output_coors_keys);

private:
  /** Templated implementation for projectPoints_launch (one implementation per point type). */
  template <typename PointT>
  cudaError_t projectPoints_launch_impl(
    const PointT * cloud, const uint32_t num_points, uint32_t * output_num_points,
    float * output_points, int64_t * output_coors, int64_t * output_coors_keys,
    uint32_t * output_proj_idxs, uint64_t * output_proj_2d, void * output_cloud_compact);

  ProjectionParams projection_;
  const uint32_t max_points_;
  const uint32_t max_voxels_;
  const uint32_t interpolation_stride_;
  cudaStream_t stream_;
  cuda_utils::CudaUniquePtr<int64_t> sorted_keys_;
  cuda_utils::CudaUniquePtr<int64_t> sorted_indices_;
  cuda_utils::CudaUniquePtr<int64_t> labels_;
  cuda_utils::CudaUniquePtr<uint8_t> cub_storage_;
  size_t cub_storage_bytes_{0};
};

}  // namespace lidar_frnet_ros1

#endif  // LIDAR_FRNET_ROS1__PREPROCESS_KERNEL_HPP_
