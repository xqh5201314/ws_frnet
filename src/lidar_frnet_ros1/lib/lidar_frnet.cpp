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

#include "lidar_frnet_ros1/lidar_frnet.hpp"

#include "lidar_frnet_ros1/ros_utils.hpp"

#include <ros/console.h>

#include <chrono>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace lidar_frnet_ros1
{

/**
 * @brief Construct pipeline: set up TensorRT profiles, engine, CUDA stream, pre/post process, and
 *        allocate GPU tensors. Throws on setup failure.
 */
LidarFRNet::LidarFRNet(
  const TrtCommonConfig & trt_config, const utils::NetworkParams & network_params,
  const utils::PostprocessingParams & postprocessing_params, bool build_only)
: network_params_(network_params)
{
  constexpr auto max_row_points = static_cast<int64_t>(
    std::numeric_limits<uint32_t>::max() / max_point_step);
  if (network_params_.num_points_profile.max > max_row_points) {
    throw std::runtime_error("num_points max exceeds the PointCloud2 row size limit.");
  }
  if (postprocessing_params.palette.size() != network_params_.num_classes) {
    throw std::runtime_error("The postprocessing palette must match the network class count.");
  }

  const std::vector<ProfileDims> profile_dims{
    {
      "points", {2, {network_params.num_points_profile.min, 4}},
      {2, {network_params.num_points_profile.opt, 4}},
      {2, {network_params.num_points_profile.max, 4}}},
    {
      "coors", {2, {network_params.num_points_profile.min, 3}},
      {2, {network_params.num_points_profile.opt, 3}},
      {2, {network_params.num_points_profile.max, 3}}},
    {
      "voxel_coors", {2, {network_params.num_unique_coors_profile.min, 3}},
      {2, {network_params.num_unique_coors_profile.opt, 3}},
      {2, {network_params.num_unique_coors_profile.max, 3}}},
    {
      "inverse_map", {1, {network_params.num_points_profile.min}},
      {1, {network_params.num_points_profile.opt}}, {1, {network_params.num_points_profile.max}}}};

  using nvinfer1::DataType;
  using nvinfer1::TensorIOMode;
  const std::vector<NetworkIO> network_io{
    {"points", {2, {-1, 4}}, DataType::kFLOAT, TensorIOMode::kINPUT},
    {"coors", {2, {-1, 3}}, DataType::kINT64, TensorIOMode::kINPUT},
    {"voxel_coors", {2, {-1, 3}}, DataType::kINT64, TensorIOMode::kINPUT},
    {"inverse_map", {1, {-1}}, DataType::kINT64, TensorIOMode::kINPUT},
    {"pred_probs", {2, {-1, network_params_.num_classes}}, DataType::kFLOAT, TensorIOMode::kOUTPUT}};

  network_ptr_ = std::make_unique<TrtCommon>(trt_config);
  if (!network_ptr_->setup(profile_dims, network_io)) {
    throw std::runtime_error("Failed to set up the FRNet TensorRT engine.");
  }
  if (build_only) {
    return;
  }

  stream_ = cuda_utils::make_stream();
  preprocess_ptr_ = std::make_unique<PreprocessCuda>(network_params, stream_.get());
  postprocess_ptr_ = std::make_unique<PostprocessCuda>(
    postprocessing_params, stream_.get(), network_ptr_->outputTensorName() == "seg_logit");

  initTensors();
}

/**
 * @brief Run full pipeline: validate input count, copy to device, set crop transform if given,
 *        preprocess, inference, postprocess; fill output clouds and proc_timing.
 */
bool LidarFRNet::process(
  const sensor_msgs::PointCloud2ConstPtr & cloud_in, sensor_msgs::PointCloud2 & cloud_seg_out,
  sensor_msgs::PointCloud2 & cloud_viz_out, sensor_msgs::PointCloud2 & cloud_filtered,
  CloudFormat filtered_output_format,
  const utils::ActiveComm & active_comm, std::unordered_map<std::string, double> & proc_timing,
  const std::array<float, 12> * crop_sensor_to_ref)
{
  auto stage_start = std::chrono::steady_clock::now();
  if (!cloud_in || !stream_ || !active_comm) {
    return false;
  }
  const auto input_format = ros_utils::detectCloudFormat(*cloud_in);
  if (input_format == CloudFormat::UNKNOWN || !ros_utils::hasValidCloudData(*cloud_in)) {
    ROS_WARN_THROTTLE(5.0, "Dropping point cloud with unsupported fields, stride, or data size.");
    return false;
  }
  if (!active_comm.filtered || filtered_output_format == CloudFormat::UNKNOWN) {
    filtered_output_format = input_format;
  }
  if (!can_convert_format(input_format, filtered_output_format)) {
    ROS_WARN_THROTTLE(
      5.0, "Cannot convert filtered point cloud from %s to %s.", to_string(input_format),
      to_string(filtered_output_format));
    return false;
  }

  const auto input_num_points = static_cast<uint64_t>(cloud_in->width) * cloud_in->height;
  if (input_num_points > static_cast<uint64_t>(network_params_.num_points_profile.max)) {
    ROS_WARN_STREAM_THROTTLE(
      5.0, "Input cloud has " << input_num_points << " points, above num_points max "
                              << network_params_.num_points_profile.max << '.');
    return false;
  }
  const auto packed_row_size = static_cast<size_t>(cloud_in->width) * cloud_in->point_step;
  const auto stream = stream_.get();
  if (cloud_in->row_step == packed_row_size) {
    CHECK_CUDA_ERROR(cudaMemcpyAsync(
      cloud_in_d_.get(), cloud_in->data.data(), cloud_in->data.size(),
      cudaMemcpyHostToDevice, stream));
  } else {
    CHECK_CUDA_ERROR(cudaMemcpy2DAsync(
      cloud_in_d_.get(), packed_row_size, cloud_in->data.data(), cloud_in->row_step,
      packed_row_size, cloud_in->height, cudaMemcpyHostToDevice, stream));
  }
  preprocess_ptr_->setCropBoxTransform(crop_sensor_to_ref ? crop_sensor_to_ref->data() : nullptr);
  if (!preprocess(static_cast<uint32_t>(input_num_points), input_format, active_comm.filtered)) {
    return false;
  }
  auto now = std::chrono::steady_clock::now();
  proc_timing.emplace(
    "debug/processing_time/preprocess_ms",
    std::chrono::duration<double, std::milli>(now - stage_start).count());
  stage_start = now;

  if (!inference()) {
    return false;
  }
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
  now = std::chrono::steady_clock::now();
  proc_timing.emplace(
    "debug/processing_time/inference_ms",
    std::chrono::duration<double, std::milli>(now - stage_start).count());
  stage_start = now;

  if (!postprocess(
        *cloud_in, input_format, filtered_output_format, active_comm, cloud_seg_out,
        cloud_viz_out, cloud_filtered)) {
    return false;
  }
  now = std::chrono::steady_clock::now();
  proc_timing.emplace(
    "debug/processing_time/postprocess_ms",
    std::chrono::duration<double, std::milli>(now - stage_start).count());

  return true;
}

bool LidarFRNet::preprocess(uint32_t input_num_points, CloudFormat input_format, bool copy_compact)
{
  const auto stream = stream_.get();
  cuda_utils::clear_async(num_points_d_.get(), 1, stream);
  if (network_params_.interpolation_stride != 0) {
    const auto pixel_count = network_params_.interpolation.w * network_params_.interpolation.h;
    cuda_utils::clear_async(proj_idxs_d_.get(), pixel_count, stream);
    cuda_utils::clear_async(proj_2d_d_.get(), pixel_count, stream);
  }

  CHECK_CUDA_ERROR(preprocess_ptr_->projectPoints_launch(
    cloud_in_d_.get(), input_num_points, input_format, num_points_d_.get(), points_d_.get(),
    coors_d_.get(), coors_keys_d_.get(), proj_idxs_d_.get(), proj_2d_d_.get(),
    copy_compact ? cloud_compact_d_.get() : nullptr));
  if (copy_compact) {
    CHECK_CUDA_ERROR(cudaMemcpyAsync(
      &num_points_raw_, num_points_d_.get(), sizeof(num_points_raw_), cudaMemcpyDeviceToHost, stream));
  }
  CHECK_CUDA_ERROR(preprocess_ptr_->interpolatePoints_launch(
    proj_idxs_d_.get(), proj_2d_d_.get(), num_points_d_.get(), points_d_.get(), coors_d_.get(),
    coors_keys_d_.get()));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    &num_points_, num_points_d_.get(), sizeof(num_points_), cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

  const auto & points = network_params_.num_points_profile;
  if (num_points_ < points.min || num_points_ > points.max) {
    ROS_WARN_STREAM_THROTTLE(
      5.0, "Dropping cloud with " << num_points_ << " points after preprocessing (num_points: "
                                  << points.min << " to " << points.max << ").");
    return false;
  }

  const auto num_unique_coors = preprocess_ptr_->generateUniqueCoors(
    num_points_, coors_d_.get(), coors_keys_d_.get(), voxel_coors_d_.get(), inverse_map_d_.get());
  const auto & voxels = network_params_.num_unique_coors_profile;
  if (num_unique_coors < voxels.min || num_unique_coors > voxels.max) {
    ROS_WARN_STREAM_THROTTLE(
      5.0, "Dropping cloud with " << num_unique_coors << " unique coordinates (num_unique_coors: "
                                  << voxels.min << " to " << voxels.max << ").");
    return false;
  }

  if (!network_ptr_->setInputShape("points", {2, {num_points_, 4}}) ||
    !network_ptr_->setInputShape("coors", {2, {num_points_, 3}}) ||
    !network_ptr_->setInputShape("voxel_coors", {2, {num_unique_coors, 3}}) ||
    !network_ptr_->setInputShape("inverse_map", {1, {num_points_}})) {
    ROS_ERROR_THROTTLE(5.0, "TensorRT rejected the FRNet input shapes.");
    return false;
  }
  const auto output_shape = network_ptr_->outputShape();
  if (output_shape.nbDims != 2 || output_shape.d[0] != num_points_ ||
    output_shape.d[1] != network_params_.num_classes) {
    ROS_ERROR_THROTTLE(5.0, "TensorRT output shape does not match the frame and class count.");
    return false;
  }
  return true;
}

bool LidarFRNet::inference()
{
  if (!network_ptr_->enqueueV3(stream_.get())) {
    ROS_ERROR_THROTTLE(5.0, "TensorRT enqueueV3 failed.");
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_.get()));
    return false;
  }
  return true;
}

bool LidarFRNet::postprocess(
  const sensor_msgs::PointCloud2 & input, CloudFormat input_format, CloudFormat filtered_output_format,
  const utils::ActiveComm & active_comm, sensor_msgs::PointCloud2 & cloud_seg_out,
  sensor_msgs::PointCloud2 & cloud_viz_out, sensor_msgs::PointCloud2 & cloud_filtered)
{
  const auto stream = stream_.get();
  if (active_comm.filtered) {
    cuda_utils::clear_async(num_points_filtered_d_.get(), 1, stream);
  }
  CHECK_CUDA_ERROR(postprocess_ptr_->fillCloud_launch(
    points_d_.get(), active_comm.filtered ? cloud_compact_d_.get() : nullptr, num_points_raw_,
    pred_probs_d_.get(), num_points_, input_format, filtered_output_format, active_comm,
    num_points_filtered_d_.get(), seg_data_d_.get(), viz_data_d_.get(), cloud_filtered_d_.get()));

  if (active_comm.seg) {
    static const auto layout = ros_utils::generateSegmentationPointCloudLayout();
    ros_utils::resizeOutputCloud(input, layout, num_points_, cloud_seg_out);
    CHECK_CUDA_ERROR(cudaMemcpyAsync(
      cloud_seg_out.data.data(), seg_data_d_.get(), cloud_seg_out.data.size(),
      cudaMemcpyDeviceToHost, stream));
  }
  if (active_comm.viz) {
    static const auto layout = ros_utils::generateVisualizationPointCloudLayout();
    ros_utils::resizeOutputCloud(input, layout, num_points_, cloud_viz_out);
    CHECK_CUDA_ERROR(cudaMemcpyAsync(
      cloud_viz_out.data.data(), viz_data_d_.get(), cloud_viz_out.data.size(),
      cudaMemcpyDeviceToHost, stream));
  }
  if (active_comm.filtered) {
    uint32_t num_points_filtered = 0;
    CHECK_CUDA_ERROR(cudaMemcpyAsync(
      &num_points_filtered, num_points_filtered_d_.get(), sizeof(num_points_filtered),
      cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    if (num_points_filtered > num_points_) {
      ROS_ERROR_THROTTLE(5.0, "Filtered output exceeds the number of processed points.");
      return false;
    }
    const auto & layout = ros_utils::filteredPointCloudLayout(filtered_output_format);
    ros_utils::resizeOutputCloud(input, layout, num_points_filtered, cloud_filtered);
    if (num_points_filtered == 0) {
      return true;
    }
    CHECK_CUDA_ERROR(cudaMemcpyAsync(
      cloud_filtered.data.data(), cloud_filtered_d_.get(), cloud_filtered.data.size(),
      cudaMemcpyDeviceToHost, stream));
  }
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
  return true;
}

void LidarFRNet::initTensors()
{
  const auto max_input_buffer_size = network_params_.num_points_profile.max * max_point_step;

  cloud_in_d_ = cuda_utils::make_unique<std::uint8_t>(max_input_buffer_size);
  cloud_compact_d_ = cuda_utils::make_unique<std::uint8_t>(max_input_buffer_size);
  coors_keys_d_ = cuda_utils::make_unique<int64_t>(network_params_.num_points_profile.max);
  num_points_d_ = cuda_utils::make_unique<uint32_t>(1);
  if (network_params_.interpolation_stride != 0) {
    const auto pixel_count = network_params_.interpolation.w * network_params_.interpolation.h;
    proj_idxs_d_ = cuda_utils::make_unique<uint32_t>(pixel_count);
    proj_2d_d_ = cuda_utils::make_unique<uint64_t>(pixel_count);
  }
  seg_data_d_ =
    cuda_utils::make_unique<OutputSegmentationPointType>(network_params_.num_points_profile.max);
  viz_data_d_ =
    cuda_utils::make_unique<OutputVisualizationPointType>(network_params_.num_points_profile.max);
  cloud_filtered_d_ = cuda_utils::make_unique<std::uint8_t>(max_input_buffer_size);
  num_points_filtered_d_ = cuda_utils::make_unique<uint32_t>(1);

  points_d_ = cuda_utils::make_unique<float>(network_params_.num_points_profile.max * 4);
  coors_d_ = cuda_utils::make_unique<int64_t>(network_params_.num_points_profile.max * 3);
  voxel_coors_d_ =
    cuda_utils::make_unique<int64_t>(network_params_.num_unique_coors_profile.max * 3);
  inverse_map_d_ = cuda_utils::make_unique<int64_t>(network_params_.num_points_profile.max);
  pred_probs_d_ = cuda_utils::make_unique<float>(
    network_params_.num_points_profile.max * network_params_.num_classes);

  if (!network_ptr_->setTensorAddress("points", points_d_.get())) {
    throw std::runtime_error("Failed to bind TensorRT tensor: points");
  }
  if (!network_ptr_->setTensorAddress("coors", coors_d_.get())) {
    throw std::runtime_error("Failed to bind TensorRT tensor: coors");
  }
  if (!network_ptr_->setTensorAddress("voxel_coors", voxel_coors_d_.get())) {
    throw std::runtime_error("Failed to bind TensorRT tensor: voxel_coors");
  }
  if (!network_ptr_->setTensorAddress("inverse_map", inverse_map_d_.get())) {
    throw std::runtime_error("Failed to bind TensorRT tensor: inverse_map");
  }
  const auto & output_name = network_ptr_->outputTensorName();
  if (!network_ptr_->setTensorAddress(output_name, pred_probs_d_.get())) {
    throw std::runtime_error("Failed to bind the TensorRT output tensor: " + output_name);
  }
}

}  // namespace lidar_frnet_ros1
