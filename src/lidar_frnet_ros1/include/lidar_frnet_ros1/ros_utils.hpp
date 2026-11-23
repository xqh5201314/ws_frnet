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

#ifndef LIDAR_FRNET_ROS1__ROS_UTILS_HPP_
#define LIDAR_FRNET_ROS1__ROS_UTILS_HPP_

#include "lidar_frnet_ros1/point_type.hpp"

#include <geometry_msgs/Point.h>
#include <geometry_msgs/Point32.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <visualization_msgs/Marker.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lidar_frnet_ros1::ros_utils
{

/**
 * @brief Layout descriptor for a point cloud (fields and point step size).
 */
struct PointCloudLayout
{
  /**
   * @brief Construct layout from field list and point step.
   * @param layout_fields Point field list (e.g. from sensor_msgs PointCloud2)
   * @param layout_point_step Byte stride per point
   */
  PointCloudLayout(
    std::vector<sensor_msgs::PointField> layout_fields, size_t layout_point_step)
  : fields(std::move(layout_fields)), point_step(layout_point_step)
  {
  }
  std::vector<sensor_msgs::PointField> fields;
  size_t point_step;
};

inline void resizeOutputCloud(
  const sensor_msgs::PointCloud2 & input, const PointCloudLayout & layout,
  const uint32_t num_points, sensor_msgs::PointCloud2 & output)
{
  output.fields = layout.fields;
  output.point_step = layout.point_step;
  output.header = input.header;
  output.height = 1;
  output.width = num_points;
  output.row_step = layout.point_step * num_points;
  output.is_bigendian = false;
  output.is_dense = input.is_dense;
  output.data.resize(output.row_step);
}

/**
 * @brief Build point cloud layout for segmentation output (x, y, z, class_id, probability).
 * @return Layout with FLOAT32 x,y,z, FLOAT32 probability and UINT8 class_id
 */
inline PointCloudLayout generateSegmentationPointCloudLayout()
{
  sensor_msgs::PointCloud2 msg;
  sensor_msgs::PointCloud2Modifier modifier(msg);
  modifier.setPointCloud2Fields(
    5, "x", 1, sensor_msgs::PointField::FLOAT32, "y", 1, sensor_msgs::PointField::FLOAT32,
    "z", 1, sensor_msgs::PointField::FLOAT32, "class_id", 1,
    sensor_msgs::PointField::UINT8, "probability", 1, sensor_msgs::PointField::FLOAT32);
  PointCloudLayout layout(msg.fields, msg.point_step);
  return layout;
}

/**
 * @brief Build point cloud layout for visualization output (x, y, z, rgb).
 * @return Layout with FLOAT32 x, y, z, rgb
 */
inline PointCloudLayout generateVisualizationPointCloudLayout()
{
  sensor_msgs::PointCloud2 msg;
  sensor_msgs::PointCloud2Modifier modifier(msg);
  modifier.setPointCloud2Fields(
    4, "x", 1, sensor_msgs::PointField::FLOAT32, "y", 1, sensor_msgs::PointField::FLOAT32,
    "z", 1, sensor_msgs::PointField::FLOAT32, "rgb", 1, sensor_msgs::PointField::FLOAT32);
  PointCloudLayout layout(msg.fields, msg.point_step);
  return layout;
}

/**
 * @brief Generate filtered point cloud layout for a supported cloud format.
 */
inline PointCloudLayout generateFilteredPointCloudLayout(CloudFormat format)
{
  sensor_msgs::PointCloud2 msg;
  sensor_msgs::PointCloud2Modifier modifier(msg);
  switch (format) {
    case CloudFormat::XYZIRCAEDT:
      modifier.setPointCloud2Fields(
        10, "x", 1, sensor_msgs::PointField::FLOAT32, "y", 1,
        sensor_msgs::PointField::FLOAT32, "z", 1, sensor_msgs::PointField::FLOAT32,
        "intensity", 1, sensor_msgs::PointField::UINT8, "return_type", 1,
        sensor_msgs::PointField::UINT8, "channel", 1, sensor_msgs::PointField::UINT16,
        "azimuth", 1, sensor_msgs::PointField::FLOAT32, "elevation", 1,
        sensor_msgs::PointField::FLOAT32, "distance", 1, sensor_msgs::PointField::FLOAT32,
        "time_stamp", 1, sensor_msgs::PointField::UINT32);
      break;
    case CloudFormat::XYZIRADRT:
      modifier.setPointCloud2Fields(
        9, "x", 1, sensor_msgs::PointField::FLOAT32, "y", 1,
        sensor_msgs::PointField::FLOAT32, "z", 1, sensor_msgs::PointField::FLOAT32,
        "intensity", 1, sensor_msgs::PointField::FLOAT32, "ring", 1,
        sensor_msgs::PointField::UINT16, "azimuth", 1, sensor_msgs::PointField::FLOAT32,
        "distance", 1, sensor_msgs::PointField::FLOAT32, "return_type", 1,
        sensor_msgs::PointField::UINT8, "time_stamp", 1,
        sensor_msgs::PointField::FLOAT64);
      break;
    case CloudFormat::XYZIRT:
      modifier.setPointCloud2Fields(
        6, "x", 1, sensor_msgs::PointField::FLOAT32, "y", 1,
        sensor_msgs::PointField::FLOAT32, "z", 1, sensor_msgs::PointField::FLOAT32,
        "intensity", 1, sensor_msgs::PointField::FLOAT32, "ring", 1,
        sensor_msgs::PointField::UINT16, "timestamp", 1, sensor_msgs::PointField::FLOAT64);
      break;
    case CloudFormat::XYZIRC:
      modifier.setPointCloud2Fields(
        6, "x", 1, sensor_msgs::PointField::FLOAT32, "y", 1,
        sensor_msgs::PointField::FLOAT32, "z", 1, sensor_msgs::PointField::FLOAT32,
        "intensity", 1, sensor_msgs::PointField::UINT8, "return_type", 1,
        sensor_msgs::PointField::UINT8, "channel", 1, sensor_msgs::PointField::UINT16);
      break;
    case CloudFormat::XYZI:
      modifier.setPointCloud2Fields(
        4, "x", 1, sensor_msgs::PointField::FLOAT32, "y", 1,
        sensor_msgs::PointField::FLOAT32, "z", 1, sensor_msgs::PointField::FLOAT32,
        "intensity", 1, sensor_msgs::PointField::FLOAT32);
      break;
    default:
      throw std::runtime_error("Unsupported filtered point cloud format.");
  }
  PointCloudLayout layout(msg.fields, msg.point_step);
  return layout;
}

inline const PointCloudLayout & filteredPointCloudLayout(CloudFormat format)
{
  static const std::array<PointCloudLayout, 5> layouts{
    generateFilteredPointCloudLayout(CloudFormat::XYZIRCAEDT),
    generateFilteredPointCloudLayout(CloudFormat::XYZIRADRT),
    generateFilteredPointCloudLayout(CloudFormat::XYZIRT),
    generateFilteredPointCloudLayout(CloudFormat::XYZIRC),
    generateFilteredPointCloudLayout(CloudFormat::XYZI)};
  return layouts.at(static_cast<size_t>(format));
}

inline CloudFormat detectCloudFormat(const sensor_msgs::PointCloud2 & cloud)
{
  if (cloud.is_bigendian) {
    return CloudFormat::UNKNOWN;
  }
  for (const auto format : {
         CloudFormat::XYZIRCAEDT, CloudFormat::XYZIRADRT, CloudFormat::XYZIRT,
         CloudFormat::XYZIRC, CloudFormat::XYZI}) {
    const auto & layout = filteredPointCloudLayout(format);
    if (cloud.point_step == layout.point_step && std::is_permutation(
          cloud.fields.begin(), cloud.fields.end(), layout.fields.begin(), layout.fields.end(),
          [](const sensor_msgs::PointField & a, const sensor_msgs::PointField & b) {
            return a.name == b.name && a.offset == b.offset && a.datatype == b.datatype &&
                   a.count == b.count;
          })) {
      return format;
    }
  }
  return CloudFormat::UNKNOWN;
}

inline bool hasValidCloudData(const sensor_msgs::PointCloud2 & cloud)
{
  return cloud.width > 0 && cloud.height > 0 && cloud.point_step > 0 &&
         cloud.row_step >= static_cast<uint64_t>(cloud.width) * cloud.point_step &&
         cloud.data.size() == static_cast<uint64_t>(cloud.row_step) * cloud.height;
}

/** Line scale for ego crop box wireframe (Marker scale.x). */
constexpr double EGO_CROP_BOX_LINE_SCALE = 0.05;
/** Red component for ego crop box color. */
constexpr float EGO_CROP_BOX_COLOR_R = 0.0f;
/** Green component for ego crop box color. */
constexpr float EGO_CROP_BOX_COLOR_G = 1.0f;
/** Blue component for ego crop box color. */
constexpr float EGO_CROP_BOX_COLOR_B = 0.0f;
/** Alpha component for ego crop box color. */
constexpr float EGO_CROP_BOX_COLOR_A = 1.0f;

/**
 * @brief Compute 8 corners of an AABB from bounds.
 *
 * Bounds order: [min_x, min_y, min_z, max_x, max_y, max_z].
 * Corner order: bottom face 0-3, top face 4-7.
 *
 * @param b Axis-aligned bounds (6 floats)
 * @param corners Output array of 8 geometry_msgs::Point corners
 */
inline void getEgoCropBoxCorners(
  const std::array<float, 6> & b, std::array<geometry_msgs::Point, 8> & corners)
{
  const double min_x = b[0];
  const double min_y = b[1];
  const double min_z = b[2];
  const double max_x = b[3];
  const double max_y = b[4];
  const double max_z = b[5];
  corners[0].x = max_x;
  corners[0].y = max_y;
  corners[0].z = min_z;
  corners[1].x = min_x;
  corners[1].y = max_y;
  corners[1].z = min_z;
  corners[2].x = min_x;
  corners[2].y = min_y;
  corners[2].z = min_z;
  corners[3].x = max_x;
  corners[3].y = min_y;
  corners[3].z = min_z;
  corners[4].x = max_x;
  corners[4].y = max_y;
  corners[4].z = max_z;
  corners[5].x = min_x;
  corners[5].y = max_y;
  corners[5].z = max_z;
  corners[6].x = min_x;
  corners[6].y = min_y;
  corners[6].z = max_z;
  corners[7].x = max_x;
  corners[7].y = min_y;
  corners[7].z = max_z;
}

/**
 * @brief Convert geometry_msgs::Point to Point32 (float coordinates).
 * @param p Input point (double x, y, z)
 * @return Point32 with same coordinates as float
 */
inline geometry_msgs::Point32 toPoint32(const geometry_msgs::Point & p)
{
  geometry_msgs::Point32 q;
  q.x = static_cast<float>(p.x);
  q.y = static_cast<float>(p.y);
  q.z = static_cast<float>(p.z);
  return q;
}

/**
 * @brief Fill marker message for ego crop box as LINE_LIST.
 *
 * Bounds order: [min_x, min_y, min_z, max_x, max_y, max_z].
 * Sets ns, id, type, action, pose, scale, color and points.
 *
 * @param bounds Axis-aligned box bounds (6 floats)
 * @param frame_id Frame ID for the message header
 * @param msg Output marker message (header, ns, type, scale, color, points are set)
 */
inline void setMarkerMsg(
  const std::array<float, 6> & bounds, const std::string & frame_id,
  visualization_msgs::Marker & msg)
{
  std::array<geometry_msgs::Point, 8> corners;
  getEgoCropBoxCorners(bounds, corners);
  msg.header.frame_id = frame_id;
  msg.ns = "ego_crop_box";
  msg.id = 0;
  msg.type = visualization_msgs::Marker::LINE_LIST;
  msg.action = visualization_msgs::Marker::ADD;
  msg.pose.orientation.w = 1.0;
  msg.scale.x = EGO_CROP_BOX_LINE_SCALE;
  msg.color.a = EGO_CROP_BOX_COLOR_A;
  msg.color.r = EGO_CROP_BOX_COLOR_R;
  msg.color.g = EGO_CROP_BOX_COLOR_G;
  msg.color.b = EGO_CROP_BOX_COLOR_B;
  msg.points.clear();
  msg.points.reserve(24);
  msg.points.push_back(corners[0]);
  msg.points.push_back(corners[1]);
  msg.points.push_back(corners[1]);
  msg.points.push_back(corners[2]);
  msg.points.push_back(corners[2]);
  msg.points.push_back(corners[3]);
  msg.points.push_back(corners[3]);
  msg.points.push_back(corners[0]);
  msg.points.push_back(corners[4]);
  msg.points.push_back(corners[5]);
  msg.points.push_back(corners[5]);
  msg.points.push_back(corners[6]);
  msg.points.push_back(corners[6]);
  msg.points.push_back(corners[7]);
  msg.points.push_back(corners[7]);
  msg.points.push_back(corners[4]);
  msg.points.push_back(corners[0]);
  msg.points.push_back(corners[4]);
  msg.points.push_back(corners[1]);
  msg.points.push_back(corners[5]);
  msg.points.push_back(corners[2]);
  msg.points.push_back(corners[6]);
  msg.points.push_back(corners[3]);
  msg.points.push_back(corners[7]);
}

}  // namespace lidar_frnet_ros1::ros_utils

#endif  // LIDAR_FRNET_ROS1__ROS_UTILS_HPP_
