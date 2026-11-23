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

#include "lidar_frnet_ros1/point_type.hpp"
#include "lidar_frnet_ros1/postprocess_kernel.hpp"
#include "lidar_frnet_ros1/utils.hpp"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cstdint>

namespace lidar_frnet_ros1
{

PostprocessCuda::PostprocessCuda(
  const utils::PostprocessingParams & params, cudaStream_t stream, bool output_is_logits)
: stream_(stream)
{
  params_.num_classes = params.palette.size();
  params_.num_filter_classes = params.filter_class_indices.size();
  params_.threshold = params.filter_class_probability_threshold;
  params_.output_is_logits = output_is_logits;
  std::copy(params.palette.begin(), params.palette.end(), params_.palette.begin());
  std::copy(
    params.filter_class_indices.begin(), params.filter_class_indices.end(),
    params_.filter_classes.begin());
}

/**
 * @brief Fill point from compact xyzi buffer (for interpolated points without original PointT).
 *        Device helper; specialized per point type.
 */
template <typename PointT>
__device__ void set_point_from_xyzi(PointT & point, const float * points_xyzi, uint32_t point_idx);

template <>
__device__ void set_point_from_xyzi<CloudPointTypeXYZI>(
  CloudPointTypeXYZI & point, const float * points_xyzi, uint32_t point_idx)
{
  const uint32_t base = point_idx * 4;
  point.x = points_xyzi[base + 0];
  point.y = points_xyzi[base + 1];
  point.z = points_xyzi[base + 2];
  point.intensity = points_xyzi[base + 3];
}

template <>
__device__ void set_point_from_xyzi<CloudPointTypeXYZIRC>(
  CloudPointTypeXYZIRC & point, const float * points_xyzi, uint32_t point_idx)
{
  const uint32_t base = point_idx * 4;
  point.x = points_xyzi[base + 0];
  point.y = points_xyzi[base + 1];
  point.z = points_xyzi[base + 2];
  point.intensity = static_cast<std::uint8_t>(points_xyzi[base + 3]);
  point.return_type = 0;
  point.channel = 0;
}

template <>
__device__ void set_point_from_xyzi<CloudPointTypeXYZIRT>(
  CloudPointTypeXYZIRT & point, const float * points_xyzi, uint32_t point_idx)
{
  const uint32_t base = point_idx * 4;
  point.x = points_xyzi[base + 0];
  point.y = points_xyzi[base + 1];
  point.z = points_xyzi[base + 2];
  point.intensity = points_xyzi[base + 3];
  point.ring = 0;
  point.timestamp = 0.0;
}

template <>
__device__ void set_point_from_xyzi<CloudPointTypeXYZIRADRT>(
  CloudPointTypeXYZIRADRT & point, const float * points_xyzi, uint32_t point_idx)
{
  const uint32_t base = point_idx * 4;
  point.x = points_xyzi[base + 0];
  point.y = points_xyzi[base + 1];
  point.z = points_xyzi[base + 2];
  point.intensity = points_xyzi[base + 3];
  point.ring = 0;
  point.azimuth = 0.0f;
  point.distance = 0.0f;
  point.return_type = 0;
  point.time_stamp = 0.0;
}

template <>
__device__ void set_point_from_xyzi<CloudPointTypeXYZIRCAEDT>(
  CloudPointTypeXYZIRCAEDT & point, const float * points_xyzi, uint32_t point_idx)
{
  const uint32_t base = point_idx * 4;
  point.x = points_xyzi[base + 0];
  point.y = points_xyzi[base + 1];
  point.z = points_xyzi[base + 2];
  point.intensity = static_cast<std::uint8_t>(points_xyzi[base + 3]);
  point.return_type = 0;
  point.channel = 0;
  point.azimuth = 0.0f;
  point.elevation = 0.0f;
  point.distance = 0.0f;
  point.time_stamp = 0u;
}

template <typename OutputPointT, typename InputPointT>
__device__ void set_point_from_input(OutputPointT & output_point, const InputPointT & input_point);

template <>
__device__ void set_point_from_input<CloudPointTypeXYZI, CloudPointTypeXYZI>(
  CloudPointTypeXYZI & output_point, const CloudPointTypeXYZI & input_point)
{
  output_point = input_point;
}

template <>
__device__ void set_point_from_input<CloudPointTypeXYZI, CloudPointTypeXYZIRC>(
  CloudPointTypeXYZI & output_point, const CloudPointTypeXYZIRC & input_point)
{
  output_point.x = input_point.x;
  output_point.y = input_point.y;
  output_point.z = input_point.z;
  output_point.intensity = static_cast<float>(input_point.intensity);
}

template <>
__device__ void set_point_from_input<CloudPointTypeXYZI, CloudPointTypeXYZIRT>(
  CloudPointTypeXYZI & output_point, const CloudPointTypeXYZIRT & input_point)
{
  output_point.x = input_point.x;
  output_point.y = input_point.y;
  output_point.z = input_point.z;
  output_point.intensity = input_point.intensity;
}

template <>
__device__ void set_point_from_input<CloudPointTypeXYZI, CloudPointTypeXYZIRADRT>(
  CloudPointTypeXYZI & output_point, const CloudPointTypeXYZIRADRT & input_point)
{
  output_point.x = input_point.x;
  output_point.y = input_point.y;
  output_point.z = input_point.z;
  output_point.intensity = input_point.intensity;
}

template <>
__device__ void set_point_from_input<CloudPointTypeXYZI, CloudPointTypeXYZIRCAEDT>(
  CloudPointTypeXYZI & output_point, const CloudPointTypeXYZIRCAEDT & input_point)
{
  output_point.x = input_point.x;
  output_point.y = input_point.y;
  output_point.z = input_point.z;
  output_point.intensity = static_cast<float>(input_point.intensity);
}

template <>
__device__ void set_point_from_input<CloudPointTypeXYZIRC, CloudPointTypeXYZIRC>(
  CloudPointTypeXYZIRC & output_point, const CloudPointTypeXYZIRC & input_point)
{
  output_point = input_point;
}

template <>
__device__ void set_point_from_input<CloudPointTypeXYZIRT, CloudPointTypeXYZIRT>(
  CloudPointTypeXYZIRT & output_point, const CloudPointTypeXYZIRT & input_point)
{
  output_point = input_point;
}

template <>
__device__ void set_point_from_input<CloudPointTypeXYZIRC, CloudPointTypeXYZIRCAEDT>(
  CloudPointTypeXYZIRC & output_point, const CloudPointTypeXYZIRCAEDT & input_point)
{
  output_point.x = input_point.x;
  output_point.y = input_point.y;
  output_point.z = input_point.z;
  output_point.intensity = input_point.intensity;
  output_point.return_type = input_point.return_type;
  output_point.channel = input_point.channel;
}

template <>
__device__ void set_point_from_input<CloudPointTypeXYZIRADRT, CloudPointTypeXYZIRADRT>(
  CloudPointTypeXYZIRADRT & output_point, const CloudPointTypeXYZIRADRT & input_point)
{
  output_point = input_point;
}

template <>
__device__ void set_point_from_input<CloudPointTypeXYZIRCAEDT, CloudPointTypeXYZIRCAEDT>(
  CloudPointTypeXYZIRCAEDT & output_point, const CloudPointTypeXYZIRCAEDT & input_point)
{
  output_point = input_point;
}

/**
 * @brief Kernel: for each point compute class from pred_probs; write seg/viz; if not in filter
 *        classes append to filtered cloud (from cloud_compact or set_point_from_xyzi).
 * @tparam InputPointT Input point type for raw compact points
 * @tparam OutputPointT Output point type for filtered cloud
 */
template <typename InputPointT, typename OutputPointT>
__global__ void fill_cloud_kernel(
  const float * points_xyzi, const InputPointT * cloud_compact, const uint32_t num_points_raw,
  const float * pred_probs, const uint32_t num_points, const bool active_comm_seg,
  const bool active_comm_viz, const bool active_comm_filtered,
  uint32_t * output_num_points_filtered, OutputSegmentationPointType * output_cloud_seg,
  OutputVisualizationPointType * output_cloud_viz, OutputPointT * output_cloud_filtered,
  PostprocessKernelParams params)
{
  uint32_t point_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (point_idx >= num_points) {
    return;
  }

  const float x = points_xyzi[point_idx * 4 + 0];
  const float y = points_xyzi[point_idx * 4 + 1];
  const float z = points_xyzi[point_idx * 4 + 2];
  const float * scores = pred_probs + static_cast<size_t>(point_idx) * params.num_classes;

  const bool apply_softmax = params.output_is_logits &&
    (active_comm_seg || (active_comm_filtered && params.num_filter_classes > 0));
  float best_score = -CUDART_INF_F;
  uint32_t class_id = params.num_classes - 1;

  if (active_comm_seg || active_comm_viz || apply_softmax) {
    for (uint32_t i = 0; i < params.num_classes; ++i) {
      if (scores[i] > best_score) {
        best_score = scores[i];
        class_id = i;
      }
    }
  }

  float inv_sum = 0.0f;
  if (apply_softmax && isfinite(best_score)) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < params.num_classes; ++i) {
      if (isfinite(scores[i])) {
        sum += expf(scores[i] - best_score);
      }
    }
    inv_sum = 1.0f / sum;
  }

  if (active_comm_seg) {
    output_cloud_seg[point_idx].x = x;
    output_cloud_seg[point_idx].y = y;
    output_cloud_seg[point_idx].z = z;
    output_cloud_seg[point_idx].class_id = static_cast<uint8_t>(class_id);
    output_cloud_seg[point_idx].probability = params.output_is_logits ? inv_sum : best_score;
  }

  if (active_comm_viz) {
    output_cloud_viz[point_idx].x = x;
    output_cloud_viz[point_idx].y = y;
    output_cloud_viz[point_idx].z = z;
    output_cloud_viz[point_idx].rgb = params.palette[class_id];
  }

  if (active_comm_filtered) {
    bool is_filtered = false;
    for (uint32_t i = 0; i < params.num_filter_classes; ++i) {
      const float score = scores[params.filter_classes[i]];
      const float prob = params.output_is_logits
        ? (inv_sum > 0.0f && isfinite(score) ? expf(score - best_score) * inv_sum : 0.0f)
        : score;
      if (prob >= params.threshold) {
        is_filtered = true;  // mark point as filtered
        break;
      }
    }
    if (!is_filtered) {  // if point is not filtered, add it to the output cloud
      const uint32_t append_idx = atomicAdd(output_num_points_filtered, 1);
      if (cloud_compact != nullptr && point_idx < num_points_raw) {
        set_point_from_input(output_cloud_filtered[append_idx], cloud_compact[point_idx]);
      } else {
        set_point_from_xyzi(output_cloud_filtered[append_idx], points_xyzi, point_idx);
      }
    }
  }
}

/**
 * @brief Launch fill_cloud_kernel for the given point type.
 */
template <typename InputPointT, typename OutputPointT>
cudaError_t PostprocessCuda::fillCloud_launch_impl(
  const float * points_xyzi, const InputPointT * cloud_compact, const uint32_t num_points_raw,
  const float * pred_probs, const uint32_t num_points, const utils::ActiveComm & active_comm,
  uint32_t * output_num_points_filtered, OutputSegmentationPointType * output_cloud_seg,
  OutputVisualizationPointType * output_cloud_viz, OutputPointT * output_cloud_filtered)
{
  dim3 block(utils::divup(num_points, utils::kernel_1d_size));
  dim3 threads(utils::kernel_1d_size);

  fill_cloud_kernel<<<block, threads, 0, stream_>>>(
    points_xyzi, cloud_compact, num_points_raw, pred_probs, num_points, active_comm.seg,
    active_comm.viz, active_comm.filtered, output_num_points_filtered, output_cloud_seg,
    output_cloud_viz, output_cloud_filtered, params_);

  return cudaGetLastError();
}

/* Explicit instantiations */
template cudaError_t PostprocessCuda::fillCloud_launch_impl<CloudPointTypeXYZI, CloudPointTypeXYZI>(
  const float *, const CloudPointTypeXYZI *, const uint32_t, const float *, const uint32_t,
  const utils::ActiveComm &, uint32_t *, OutputSegmentationPointType *,
  OutputVisualizationPointType *, CloudPointTypeXYZI *);
template cudaError_t
PostprocessCuda::fillCloud_launch_impl<CloudPointTypeXYZIRC, CloudPointTypeXYZI>(
  const float *, const CloudPointTypeXYZIRC *, const uint32_t, const float *, const uint32_t,
  const utils::ActiveComm &, uint32_t *, OutputSegmentationPointType *,
  OutputVisualizationPointType *, CloudPointTypeXYZI *);
template cudaError_t
PostprocessCuda::fillCloud_launch_impl<CloudPointTypeXYZIRC, CloudPointTypeXYZIRC>(
  const float *, const CloudPointTypeXYZIRC *, const uint32_t, const float *, const uint32_t,
  const utils::ActiveComm &, uint32_t *, OutputSegmentationPointType *,
  OutputVisualizationPointType *, CloudPointTypeXYZIRC *);
template cudaError_t
PostprocessCuda::fillCloud_launch_impl<CloudPointTypeXYZIRT, CloudPointTypeXYZI>(
  const float *, const CloudPointTypeXYZIRT *, const uint32_t, const float *, const uint32_t,
  const utils::ActiveComm &, uint32_t *, OutputSegmentationPointType *,
  OutputVisualizationPointType *, CloudPointTypeXYZI *);
template cudaError_t
PostprocessCuda::fillCloud_launch_impl<CloudPointTypeXYZIRT, CloudPointTypeXYZIRT>(
  const float *, const CloudPointTypeXYZIRT *, const uint32_t, const float *, const uint32_t,
  const utils::ActiveComm &, uint32_t *, OutputSegmentationPointType *,
  OutputVisualizationPointType *, CloudPointTypeXYZIRT *);
template cudaError_t
PostprocessCuda::fillCloud_launch_impl<CloudPointTypeXYZIRADRT, CloudPointTypeXYZI>(
  const float *, const CloudPointTypeXYZIRADRT *, const uint32_t, const float *, const uint32_t,
  const utils::ActiveComm &, uint32_t *, OutputSegmentationPointType *,
  OutputVisualizationPointType *, CloudPointTypeXYZI *);
template cudaError_t
PostprocessCuda::fillCloud_launch_impl<CloudPointTypeXYZIRADRT, CloudPointTypeXYZIRADRT>(
  const float *, const CloudPointTypeXYZIRADRT *, const uint32_t, const float *, const uint32_t,
  const utils::ActiveComm &, uint32_t *, OutputSegmentationPointType *,
  OutputVisualizationPointType *, CloudPointTypeXYZIRADRT *);
template cudaError_t
PostprocessCuda::fillCloud_launch_impl<CloudPointTypeXYZIRCAEDT, CloudPointTypeXYZI>(
  const float *, const CloudPointTypeXYZIRCAEDT *, const uint32_t, const float *, const uint32_t,
  const utils::ActiveComm &, uint32_t *, OutputSegmentationPointType *,
  OutputVisualizationPointType *, CloudPointTypeXYZI *);
template cudaError_t
PostprocessCuda::fillCloud_launch_impl<CloudPointTypeXYZIRCAEDT, CloudPointTypeXYZIRC>(
  const float *, const CloudPointTypeXYZIRCAEDT *, const uint32_t, const float *, const uint32_t,
  const utils::ActiveComm &, uint32_t *, OutputSegmentationPointType *,
  OutputVisualizationPointType *, CloudPointTypeXYZIRC *);
template cudaError_t
PostprocessCuda::fillCloud_launch_impl<CloudPointTypeXYZIRCAEDT, CloudPointTypeXYZIRCAEDT>(
  const float *, const CloudPointTypeXYZIRCAEDT *, const uint32_t, const float *, const uint32_t,
  const utils::ActiveComm &, uint32_t *, OutputSegmentationPointType *,
  OutputVisualizationPointType *, CloudPointTypeXYZIRCAEDT *);

/**
 * @brief Dispatch fillCloud by input format to templated implementation.
 */
cudaError_t PostprocessCuda::fillCloud_launch(
  const float * points_xyzi, const void * cloud_compact, const uint32_t num_points_raw,
  const float * pred_probs, const uint32_t num_points, CloudFormat input_format,
  CloudFormat output_format, const utils::ActiveComm & active_comm,
  uint32_t * output_num_points_filtered, OutputSegmentationPointType * output_cloud_seg,
  OutputVisualizationPointType * output_cloud_viz, void * output_cloud_filtered)
{
  switch (input_format) {
    case CloudFormat::XYZIRCAEDT:
      switch (output_format) {
        case CloudFormat::XYZIRCAEDT:
          return fillCloud_launch_impl(
            points_xyzi, static_cast<const CloudPointTypeXYZIRCAEDT *>(cloud_compact),
            num_points_raw, pred_probs, num_points, active_comm, output_num_points_filtered,
            output_cloud_seg, output_cloud_viz,
            static_cast<CloudPointTypeXYZIRCAEDT *>(output_cloud_filtered));
        case CloudFormat::XYZIRC:
          return fillCloud_launch_impl(
            points_xyzi, static_cast<const CloudPointTypeXYZIRCAEDT *>(cloud_compact),
            num_points_raw, pred_probs, num_points, active_comm, output_num_points_filtered,
            output_cloud_seg, output_cloud_viz,
            static_cast<CloudPointTypeXYZIRC *>(output_cloud_filtered));
        case CloudFormat::XYZI:
          return fillCloud_launch_impl(
            points_xyzi, static_cast<const CloudPointTypeXYZIRCAEDT *>(cloud_compact),
            num_points_raw, pred_probs, num_points, active_comm, output_num_points_filtered,
            output_cloud_seg, output_cloud_viz,
            static_cast<CloudPointTypeXYZI *>(output_cloud_filtered));
        default:
          return cudaErrorInvalidValue;
      }
    case CloudFormat::XYZIRADRT:
      switch (output_format) {
        case CloudFormat::XYZIRADRT:
          return fillCloud_launch_impl(
            points_xyzi, static_cast<const CloudPointTypeXYZIRADRT *>(cloud_compact),
            num_points_raw, pred_probs, num_points, active_comm, output_num_points_filtered,
            output_cloud_seg, output_cloud_viz,
            static_cast<CloudPointTypeXYZIRADRT *>(output_cloud_filtered));
        case CloudFormat::XYZI:
          return fillCloud_launch_impl(
            points_xyzi, static_cast<const CloudPointTypeXYZIRADRT *>(cloud_compact),
            num_points_raw, pred_probs, num_points, active_comm, output_num_points_filtered,
            output_cloud_seg, output_cloud_viz,
            static_cast<CloudPointTypeXYZI *>(output_cloud_filtered));
        default:
          return cudaErrorInvalidValue;
      }
    case CloudFormat::XYZIRC:
      switch (output_format) {
        case CloudFormat::XYZIRC:
          return fillCloud_launch_impl(
            points_xyzi, static_cast<const CloudPointTypeXYZIRC *>(cloud_compact), num_points_raw,
            pred_probs, num_points, active_comm, output_num_points_filtered, output_cloud_seg,
            output_cloud_viz, static_cast<CloudPointTypeXYZIRC *>(output_cloud_filtered));
        case CloudFormat::XYZI:
          return fillCloud_launch_impl(
            points_xyzi, static_cast<const CloudPointTypeXYZIRC *>(cloud_compact), num_points_raw,
            pred_probs, num_points, active_comm, output_num_points_filtered, output_cloud_seg,
            output_cloud_viz, static_cast<CloudPointTypeXYZI *>(output_cloud_filtered));
        default:
          return cudaErrorInvalidValue;
      }
    case CloudFormat::XYZIRT:
      switch (output_format) {
        case CloudFormat::XYZIRT:
          return fillCloud_launch_impl(
            points_xyzi, static_cast<const CloudPointTypeXYZIRT *>(cloud_compact), num_points_raw,
            pred_probs, num_points, active_comm, output_num_points_filtered, output_cloud_seg,
            output_cloud_viz, static_cast<CloudPointTypeXYZIRT *>(output_cloud_filtered));
        case CloudFormat::XYZI:
          return fillCloud_launch_impl(
            points_xyzi, static_cast<const CloudPointTypeXYZIRT *>(cloud_compact), num_points_raw,
            pred_probs, num_points, active_comm, output_num_points_filtered, output_cloud_seg,
            output_cloud_viz, static_cast<CloudPointTypeXYZI *>(output_cloud_filtered));
        default:
          return cudaErrorInvalidValue;
      }
    case CloudFormat::XYZI:
      if (output_format == CloudFormat::XYZI) {
        return fillCloud_launch_impl(
          points_xyzi, static_cast<const CloudPointTypeXYZI *>(cloud_compact), num_points_raw,
          pred_probs, num_points, active_comm, output_num_points_filtered, output_cloud_seg,
          output_cloud_viz, static_cast<CloudPointTypeXYZI *>(output_cloud_filtered));
      }
      return cudaErrorInvalidValue;
    default:
      return cudaErrorInvalidValue;
  }
}

}  // namespace lidar_frnet_ros1
