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

#ifndef LIDAR_FRNET_ROS1__LIDAR_FRNET_HPP_
#define LIDAR_FRNET_ROS1__LIDAR_FRNET_HPP_

#include "lidar_frnet_ros1/cuda_utils.hpp"
#include "lidar_frnet_ros1/point_type.hpp"
#include "lidar_frnet_ros1/postprocess_kernel.hpp"
#include "lidar_frnet_ros1/preprocess_kernel.hpp"
#include "lidar_frnet_ros1/tensorrt_wrapper.hpp"
#include "lidar_frnet_ros1/utils.hpp"

#include <sensor_msgs/PointCloud2.h>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

namespace lidar_frnet_ros1
{
using cuda_utils::CudaUniquePtr;

/**
 * @brief Lidar FRNet pipeline: preprocess (project + interpolate), TensorRT inference, postprocess.
 *
 * Supports multiple cloud formats (XYZIRCAEDT, XYZIRADRT, XYZIRT, XYZIRC, XYZI). Outputs segmentation,
 * visualization, and optionally filtered point clouds. Ego crop box is applied in preprocess when
 * a static transform is provided.
 */
class LidarFRNet
{
public:
  /**
   * @brief Construct and initialize TensorRT engine, preprocess/postprocess CUDA, and GPU buffers.
   * @param trt_config TensorRT config (e.g. ONNX path, precision)
   * @param network_params Network and preprocessing params (profiles, FOV, crop box)
   * @param postprocessing_params Filter classes, palette, crop box bounds
   * @param build_only Build or load the engine without allocating point-cloud buffers
   */
  LidarFRNet(
    const TrtCommonConfig & trt_config, const utils::NetworkParams & network_params,
    const utils::PostprocessingParams & postprocessing_params, bool build_only = false);
  ~LidarFRNet() = default;

  /**
   * @brief Run full pipeline: copy input, preprocess, inference, postprocess; fill output clouds.
   * @param cloud_in Input point cloud (host)
   * @param cloud_seg_out Segmentation output (x,y,z,class_id,probability); size set by pipeline
   * @param cloud_viz_out Visualization output (x,y,z,rgb); size set by pipeline
   * @param cloud_filtered Filtered point cloud (filtered_output_format); width/row_step set by
   * pipeline
   * @param filtered_output_format Requested output format, or UNKNOWN to keep the input format
   * @param active_comm Which outputs have subscribers (skip work when false)
   * @param proc_timing Map filled with per-stage timings (e.g. preprocess_ms, inference_ms)
   * @param crop_sensor_to_ref Optional 12-float transform (sensor to ref) for ego crop box
   * @return true on success, false when the frame cannot be processed
   */
  bool process(
    const sensor_msgs::PointCloud2ConstPtr & cloud_in, sensor_msgs::PointCloud2 & cloud_seg_out,
    sensor_msgs::PointCloud2 & cloud_viz_out, sensor_msgs::PointCloud2 & cloud_filtered,
    CloudFormat filtered_output_format,
    const utils::ActiveComm & active_comm, std::unordered_map<std::string, double> & proc_timing,
    const std::array<float, 12> * crop_sensor_to_ref = nullptr);

private:
  /** Project points, interpolate, generate unique coors; set num_points_, num_points_raw_. */
  bool preprocess(uint32_t input_num_points, CloudFormat input_format, bool copy_compact);
  /** Run TensorRT enqueue. */
  bool inference();
  /** Fill seg/viz/filtered clouds from network output; set cloud width/row_step. */
  bool postprocess(
    const sensor_msgs::PointCloud2 & input, CloudFormat input_format, CloudFormat filtered_output_format,
    const utils::ActiveComm & active_comm, sensor_msgs::PointCloud2 & cloud_seg_out,
    sensor_msgs::PointCloud2 & cloud_viz_out, sensor_msgs::PointCloud2 & cloud_filtered);
  /** Allocate all GPU buffers to max profile sizes (called once in constructor). */
  void initTensors();

  cuda_utils::CudaStream stream_;
  std::unique_ptr<TrtCommon> network_ptr_{nullptr};
  std::unique_ptr<PreprocessCuda> preprocess_ptr_{nullptr};
  std::unique_ptr<PostprocessCuda> postprocess_ptr_{nullptr};
  utils::NetworkParams network_params_;

  // Distinct number of points before and after interpolation
  uint32_t num_points_raw_{0};
  uint32_t num_points_{0};

  // Inference
  CudaUniquePtr<float> points_d_{nullptr};         // N x 4 (x, y, z, intensity)
  CudaUniquePtr<int64_t> coors_d_{nullptr};        // N x 3 (0, y, x)
  CudaUniquePtr<int64_t> voxel_coors_d_{nullptr};  // M x 3 (0, y, x)
  CudaUniquePtr<int64_t> inverse_map_d_{nullptr};  // N
  CudaUniquePtr<float> pred_probs_d_{nullptr};     // NUM_CLASSES x 1
  // Preprocess & Postprocess
  CudaUniquePtr<std::uint8_t> cloud_in_d_{nullptr};
  CudaUniquePtr<std::uint8_t> cloud_compact_d_{nullptr};  // compact input points for filtered
  CudaUniquePtr<int64_t> coors_keys_d_{nullptr};
  CudaUniquePtr<uint32_t> num_points_d_{nullptr};
  CudaUniquePtr<uint32_t> proj_idxs_d_{nullptr};
  CudaUniquePtr<uint64_t> proj_2d_d_{nullptr};
  CudaUniquePtr<OutputSegmentationPointType> seg_data_d_{nullptr};
  CudaUniquePtr<OutputVisualizationPointType> viz_data_d_{nullptr};
  CudaUniquePtr<std::uint8_t> cloud_filtered_d_{nullptr};
  CudaUniquePtr<uint32_t> num_points_filtered_d_{nullptr};
};

}  // namespace lidar_frnet_ros1

#endif  // LIDAR_FRNET_ROS1__LIDAR_FRNET_HPP_
