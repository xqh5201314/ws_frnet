// Copyright 2026
//
// Licensed under the Apache License, Version 2.0.

#ifndef LIDAR_FRNET_ROS1__LIDAR_FRNET_NODE_HPP_
#define LIDAR_FRNET_ROS1__LIDAR_FRNET_NODE_HPP_

#include "lidar_frnet_ros1/lidar_frnet.hpp"

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>

#include <array>
#include <memory>
#include <optional>
#include <string>

namespace tf2_ros
{
class Buffer;
class TransformListener;
}

namespace lidar_frnet_ros1
{

class LidarFRNetNode
{
public:
  LidarFRNetNode();

private:
  void cloudCallback(const sensor_msgs::PointCloud2ConstPtr & msg);
  bool setStaticCropBoxTransform(const std::string & sensor_frame_id);
  void publishEgoCropBoxDebug(const ros::Time & stamp);

  ros::NodeHandle pnh_;
  ros::Subscriber cloud_sub_;
  ros::Publisher cloud_seg_pub_;
  ros::Publisher cloud_viz_pub_;
  ros::Publisher cloud_filtered_pub_;
  ros::Publisher ego_crop_box_marker_pub_;

  std::unique_ptr<LidarFRNet> frnet_;
  CloudFormat filtered_output_format_{CloudFormat::UNKNOWN};

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::optional<std::array<float, 12>> crop_sensor_to_ref_;
  std::string crop_sensor_frame_;
  std::string crop_reference_frame_;
  std::array<float, 6> crop_box_bounds_{};
  bool crop_box_enabled_{false};
  std::optional<visualization_msgs::Marker> ego_crop_box_marker_msg_;
};

}  // namespace lidar_frnet_ros1

#endif  // LIDAR_FRNET_ROS1__LIDAR_FRNET_NODE_HPP_
