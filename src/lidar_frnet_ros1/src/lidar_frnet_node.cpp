// Copyright 2026
//
// Licensed under the Apache License, Version 2.0.

#include "lidar_frnet_ros1/lidar_frnet_node.hpp"
#include "lidar_frnet_ros1/ros_utils.hpp"

#include <boost/version.hpp>
#include <geometry_msgs/TransformStamped.h>
#include <limits.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

// Older Boost.Thread headers evaluate PTHREAD_STACK_MIN in a preprocessor condition.
#pragma push_macro("PTHREAD_STACK_MIN")
#if BOOST_VERSION < 107800 && defined(__USE_DYNAMIC_STACK_SIZE)
#undef PTHREAD_STACK_MIN
#endif
#include <tf2_ros/transform_listener.h>
#pragma pop_macro("PTHREAD_STACK_MIN")

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace lidar_frnet_ros1
{

LidarFRNetNode::LidarFRNetNode() : pnh_("~")
{
  const auto class_names = pnh_.param<std::vector<std::string>>("class_names", {});
  const auto trt_config = TrtCommonConfig(
    pnh_.param<std::string>("onnx_path", ""), pnh_.param<std::string>("trt_precision", "fp16"),
    pnh_.param("use_cuda_graph", false));

  crop_reference_frame_ = pnh_.param<std::string>("filter/ego_crop_box/reference_frame", "base_link");
  crop_box_bounds_ = {
    static_cast<float>(pnh_.param<double>("filter/ego_crop_box/min_x", 0.0)),
    static_cast<float>(pnh_.param<double>("filter/ego_crop_box/min_y", 0.0)),
    static_cast<float>(pnh_.param<double>("filter/ego_crop_box/min_z", 0.0)),
    static_cast<float>(pnh_.param<double>("filter/ego_crop_box/max_x", 0.0)),
    static_cast<float>(pnh_.param<double>("filter/ego_crop_box/max_y", 0.0)),
    static_cast<float>(pnh_.param<double>("filter/ego_crop_box/max_z", 0.0))};
  crop_box_enabled_ = std::any_of(
    crop_box_bounds_.begin(), crop_box_bounds_.end(), [](float value) { return value != 0.0f; });
  if (crop_box_enabled_) {
    for (size_t axis = 0; axis < 3; ++axis) {
      if (!std::isfinite(crop_box_bounds_[axis]) || !std::isfinite(crop_box_bounds_[axis + 3]) ||
        crop_box_bounds_[axis] >= crop_box_bounds_[axis + 3] || crop_reference_frame_.empty()) {
        throw std::runtime_error("Ego crop box requires a reference frame and finite min < max bounds.");
      }
    }
  }

  const auto requested_format = pnh_.param<std::string>("filter/output_format", "");
  filtered_output_format_ = parse_cloud_format_string(requested_format);
  if (!requested_format.empty() && filtered_output_format_ == CloudFormat::UNKNOWN) {
    throw std::runtime_error("Unknown filter/output_format: " + requested_format);
  }
  const auto palette_param = pnh_.param<std::vector<int>>("palette", {});
  const std::vector<int64_t> palette(palette_param.begin(), palette_param.end());
  const auto postprocessing_params = utils::PostprocessingParams(
    pnh_.param<double>("filter/class_probability_threshold", 0.05),
    pnh_.param<std::vector<std::string>>("filter/classes", {}), requested_format,
    crop_box_bounds_, class_names, palette);

  const auto num_points_profile = pnh_.param<std::vector<int>>("num_points", {5000, 60000, 120000});
  const std::vector<int64_t> num_points(num_points_profile.begin(), num_points_profile.end());
  const auto unique_profile = pnh_.param<std::vector<int>>("num_unique_coors", {3000, 30000, 60000});
  const std::vector<int64_t> num_unique_coors(unique_profile.begin(), unique_profile.end());

  const auto network_params = utils::NetworkParams(
    class_names, num_points, num_unique_coors, pnh_.param("fov_up_deg", 52.6),
    pnh_.param("fov_down_deg", -52.6), pnh_.param("frustum_width", 256),
    pnh_.param("frustum_height", 128), pnh_.param("interpolation_width", 1024),
    pnh_.param("interpolation_height", 128), crop_box_enabled_, crop_box_bounds_,
    pnh_.param("interpolation_stride", 1));

  const bool build_only = pnh_.param("build_only", false);
  frnet_ = std::make_unique<LidarFRNet>(trt_config, network_params, postprocessing_params, build_only);
  if (build_only) {
    ROS_INFO("TensorRT engine is ready. Shutting down the node.");
    ros::shutdown();
    return;
  }

  if (crop_box_enabled_) {
    ros_utils::setMarkerMsg(crop_box_bounds_, crop_reference_frame_, ego_crop_box_marker_msg_.emplace());
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>();
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
  }

  cloud_sub_ = pnh_.subscribe("input/pointcloud", 1, &LidarFRNetNode::cloudCallback, this);
  cloud_seg_pub_ =
    pnh_.advertise<sensor_msgs::PointCloud2>("output/pointcloud/segmentation", 1);
  cloud_viz_pub_ =
    pnh_.advertise<sensor_msgs::PointCloud2>("output/pointcloud/visualization", 1);
  cloud_filtered_pub_ =
    pnh_.advertise<sensor_msgs::PointCloud2>("output/pointcloud/filtered", 1);
  ego_crop_box_marker_pub_ =
    pnh_.advertise<visualization_msgs::Marker>("debug/ego_crop_box_marker", 1);

  ROS_INFO("lidar_frnet is ready. Processing valid clouds on ~/input/pointcloud.");
}

bool LidarFRNetNode::setStaticCropBoxTransform(const std::string & sensor_frame_id)
{
  if (sensor_frame_id == crop_reference_frame_) {
    crop_sensor_to_ref_ = std::array<float, 12>{1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0};
    crop_sensor_frame_ = sensor_frame_id;
    return true;
  }
  geometry_msgs::TransformStamped tf;
  try {
    tf = tf_buffer_->lookupTransform(crop_reference_frame_, sensor_frame_id, ros::Time(0));
  } catch (const tf2::TransformException & ex) {
    ROS_WARN_THROTTLE(1.0, "Ego crop box TF lookup failed: %s", ex.what());
    return false;
  }

  tf2::Quaternion q;
  tf2::fromMsg(tf.transform.rotation, q);
  tf2::Matrix3x3 rotation(q);
  std::array<float, 12> transform{};
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      transform[r * 3 + c] = static_cast<float>(rotation[r][c]);
    }
  }
  transform[9] = static_cast<float>(tf.transform.translation.x);
  transform[10] = static_cast<float>(tf.transform.translation.y);
  transform[11] = static_cast<float>(tf.transform.translation.z);
  crop_sensor_to_ref_.emplace(transform);
  crop_sensor_frame_ = sensor_frame_id;
  ROS_INFO(
    "Cached ego crop box static transform %s -> %s", sensor_frame_id.c_str(),
    crop_reference_frame_.c_str());
  return true;
}

void LidarFRNetNode::cloudCallback(const sensor_msgs::PointCloud2ConstPtr & msg)
{
  const auto active_comm = utils::ActiveComm(
    true, cloud_viz_pub_.getNumSubscribers() > 0,
    cloud_filtered_pub_.getNumSubscribers() > 0);

  if (crop_box_enabled_ && (!crop_sensor_to_ref_ || crop_sensor_frame_ != msg->header.frame_id) &&
    !setStaticCropBoxTransform(msg->header.frame_id)) {
    return;
  }

  sensor_msgs::PointCloud2 cloud_seg_msg;
  sensor_msgs::PointCloud2 cloud_viz_msg;
  sensor_msgs::PointCloud2 cloud_filtered_msg;
  std::unordered_map<std::string, double> proc_timing;
  const std::array<float, 12> * crop_transform = crop_sensor_to_ref_ ? &*crop_sensor_to_ref_ : nullptr;
  const auto start = std::chrono::steady_clock::now();
  if (!frnet_->process(
        msg, cloud_seg_msg, cloud_viz_msg, cloud_filtered_msg, filtered_output_format_,
        active_comm, proc_timing, crop_transform)) {
    return;
  }
  const auto total_ms =
    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  ROS_INFO(
    "FRNet frame %u.%09u: preprocess %.3f ms, inference %.3f ms, postprocess %.3f ms, "
    "total %.3f ms",
    msg->header.stamp.sec, msg->header.stamp.nsec,
    proc_timing.at("debug/processing_time/preprocess_ms"),
    proc_timing.at("debug/processing_time/inference_ms"),
    proc_timing.at("debug/processing_time/postprocess_ms"), total_ms);

  cloud_seg_pub_.publish(cloud_seg_msg);
  if (active_comm.viz) {
    cloud_viz_pub_.publish(cloud_viz_msg);
  }
  if (active_comm.filtered) {
    cloud_filtered_pub_.publish(cloud_filtered_msg);
  }
  publishEgoCropBoxDebug(msg->header.stamp);
}

void LidarFRNetNode::publishEgoCropBoxDebug(const ros::Time & stamp)
{
  if (!crop_box_enabled_ || ego_crop_box_marker_pub_.getNumSubscribers() == 0 || !ego_crop_box_marker_msg_) {
    return;
  }
  auto marker = *ego_crop_box_marker_msg_;
  marker.header.stamp = stamp;
  ego_crop_box_marker_pub_.publish(marker);
}

}  // namespace lidar_frnet_ros1

int main(int argc, char ** argv)
{
  ros::init(argc, argv, "lidar_frnet");
  try {
    lidar_frnet_ros1::LidarFRNetNode node;
    ros::spin();
    return 0;
  } catch (const std::exception & e) {
    ROS_FATAL("lidar_frnet failed: %s", e.what());
  } catch (...) {
    ROS_FATAL("lidar_frnet failed with an unknown exception.");
  }
  return 1;
}
