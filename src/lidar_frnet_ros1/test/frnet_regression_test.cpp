// Licensed under the Apache License, Version 2.0.

#include "lidar_frnet_ros1/postprocess_kernel.hpp"
#include "lidar_frnet_ros1/preprocess_kernel.hpp"
#include "lidar_frnet_ros1/ros_utils.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>

namespace lidar_frnet_ros1
{

TEST(PointCloudLayout, ValidatesPackedFieldsAndRowPadding)
{
  sensor_msgs::PointCloud2 msg;
  for (const auto format : {
         CloudFormat::XYZIRCAEDT, CloudFormat::XYZIRADRT, CloudFormat::XYZIRT,
         CloudFormat::XYZIRC, CloudFormat::XYZI}) {
    const auto & layout = ros_utils::filteredPointCloudLayout(format);
    msg.fields = layout.fields;
    msg.point_step = layout.point_step;
    EXPECT_EQ(msg.point_step, get_point_step(format));
    EXPECT_LE(msg.point_step, max_point_step);
    std::reverse(msg.fields.begin(), msg.fields.end());
    EXPECT_EQ(ros_utils::detectCloudFormat(msg), format);
    ++msg.fields.front().offset;
    EXPECT_EQ(ros_utils::detectCloudFormat(msg), CloudFormat::UNKNOWN);
    --msg.fields.front().offset;
    msg.is_bigendian = true;
    EXPECT_EQ(ros_utils::detectCloudFormat(msg), CloudFormat::UNKNOWN);
    msg.is_bigendian = false;
  }

  msg.width = 4;
  msg.height = 2;
  msg.row_step = msg.width * msg.point_step + 8;
  msg.data.resize(msg.row_step * msg.height);
  EXPECT_TRUE(ros_utils::hasValidCloudData(msg));
  msg.data.pop_back();
  EXPECT_FALSE(ros_utils::hasValidCloudData(msg));
  --msg.row_step;
  msg.width = 100;
  msg.data.resize(msg.row_step * msg.height);
  EXPECT_FALSE(ros_utils::hasValidCloudData(msg));
}

TEST(ProcessingProfile, ChecksSizeAndOrder)
{
  EXPECT_THROW(utils::Profile({1, 2}), std::runtime_error);
  EXPECT_THROW(utils::Profile({0, 2, 3}), std::runtime_error);
  EXPECT_THROW(utils::Profile({2, 1, 3}), std::runtime_error);
  EXPECT_NO_THROW(utils::Profile({1, 2, 3}));
  EXPECT_THROW(
    utils::NetworkParams({"object"}, {1, 4, 8}, {1, 4, 8}, 45, -45, 8, 1, 8, 1, false, {}, -1),
    std::runtime_error);
}

TEST(PreprocessCuda, BoundsInterpolationAndVoxelWrites)
{
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    GTEST_SKIP() << "A CUDA device is required.";
  }
  auto stream = cuda_utils::make_stream();
  const auto params = utils::NetworkParams({"object"}, {1, 3, 3}, {1, 2, 2}, 45, -45, 8, 1, 8, 1);
  PreprocessCuda preprocess(params, stream.get());

  std::array<CloudPointTypeXYZI, 3> input{};
  for (size_t i = 0; i < input.size(); ++i) {
    const double angle = M_PI - 2.0 * M_PI * (1.5 + 2.0 * i) / 8.0;
    input[i] = {static_cast<float>(std::cos(angle)), static_cast<float>(std::sin(angle)), 0, 1};
  }
  const auto cloud = cuda_utils::make_unique<CloudPointTypeXYZI>(input.size());
  const auto points = cuda_utils::make_unique<float>(5 * 4);
  const auto coors = cuda_utils::make_unique<int64_t>(5 * 3);
  const auto keys = cuda_utils::make_unique<int64_t>(5);
  const auto count = cuda_utils::make_unique<uint32_t>(1);
  const auto occupied = cuda_utils::make_unique<uint32_t>(8);
  const auto projected = cuda_utils::make_unique<uint64_t>(8);
  CHECK_CUDA_ERROR(cudaMemcpy(cloud.get(), input.data(), sizeof(input), cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemsetAsync(keys.get(), 255, 5 * sizeof(int64_t), stream.get()));
  cuda_utils::clear_async(count.get(), 1, stream.get());
  cuda_utils::clear_async(occupied.get(), 8, stream.get());
  cuda_utils::clear_async(projected.get(), 8, stream.get());

  CHECK_CUDA_ERROR(preprocess.projectPoints_launch(
    cloud.get(), input.size(), CloudFormat::XYZI, count.get(), points.get(), coors.get(),
    keys.get(), occupied.get(), projected.get()));
  CHECK_CUDA_ERROR(preprocess.interpolatePoints_launch(
    occupied.get(), projected.get(), count.get(), points.get(), coors.get(), keys.get()));
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream.get()));

  uint32_t total = 0;
  std::array<int64_t, 2> unused_keys{};
  CHECK_CUDA_ERROR(cudaMemcpy(&total, count.get(), sizeof(total), cudaMemcpyDeviceToHost));
  CHECK_CUDA_ERROR(cudaMemcpy(
    unused_keys.data(), keys.get() + 3, sizeof(unused_keys), cudaMemcpyDeviceToHost));
  EXPECT_EQ(total, 5u);
  EXPECT_EQ(unused_keys, (std::array<int64_t, 2>{-1, -1}));

  const auto voxels = cuda_utils::make_unique<int64_t>(3 * 3);
  const auto inverse = cuda_utils::make_unique<int64_t>(3);
  CHECK_CUDA_ERROR(cudaMemsetAsync(voxels.get(), 255, 9 * sizeof(int64_t), stream.get()));
  EXPECT_EQ(preprocess.generateUniqueCoors(3, coors.get(), keys.get(), voxels.get(), inverse.get()), 3u);
  std::array<int64_t, 9> output{};
  CHECK_CUDA_ERROR(cudaMemcpy(output.data(), voxels.get(), sizeof(output), cudaMemcpyDeviceToHost));
  EXPECT_EQ(output, (std::array<int64_t, 9>{0, 0, 1, 0, 0, 3, -1, -1, -1}));

  const std::array<int64_t, 3> duplicate_keys{3, 1, 3};
  const std::array<int64_t, 9> duplicate_coors{0, 0, 3, 0, 0, 1, 0, 0, 3};
  CHECK_CUDA_ERROR(cudaMemcpy(
    keys.get(), duplicate_keys.data(), sizeof(duplicate_keys), cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(
    coors.get(), duplicate_coors.data(), sizeof(duplicate_coors), cudaMemcpyHostToDevice));
  EXPECT_EQ(preprocess.generateUniqueCoors(3, coors.get(), keys.get(), voxels.get(), inverse.get()), 2u);
  std::array<int64_t, 3> inverse_map{};
  CHECK_CUDA_ERROR(cudaMemcpy(
    inverse_map.data(), inverse.get(), sizeof(inverse_map), cudaMemcpyDeviceToHost));
  EXPECT_EQ(inverse_map, (std::array<int64_t, 3>{1, 0, 1}));
}

TEST(PreprocessCuda, SamplesInterpolationWithoutDroppingInputPoints)
{
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    GTEST_SKIP() << "A CUDA device is required.";
  }
  const auto stream = cuda_utils::make_stream();
  std::array<CloudPointTypeXYZI, 4> input{};
  for (size_t i = 0; i < input.size(); ++i) {
    const double angle = M_PI - 2.0 * M_PI * (0.5 + 2.0 * i) / 8.0;
    input[i] = {static_cast<float>(std::cos(angle)), static_cast<float>(std::sin(angle)), 0, 1};
  }
  const auto cloud = cuda_utils::make_unique<CloudPointTypeXYZI>(input.size());
  const auto points = cuda_utils::make_unique<float>(8 * 4);
  const auto coors = cuda_utils::make_unique<int64_t>(8 * 3);
  const auto keys = cuda_utils::make_unique<int64_t>(8);
  const auto count = cuda_utils::make_unique<uint32_t>(1);
  const auto occupied = cuda_utils::make_unique<uint32_t>(8);
  const auto projected = cuda_utils::make_unique<uint64_t>(8);
  CHECK_CUDA_ERROR(cudaMemcpy(cloud.get(), input.data(), sizeof(input), cudaMemcpyHostToDevice));

  const std::pair<int, uint32_t> cases[]{{0, 4}, {1, 7}, {3, 5}};
  for (const auto & [stride, expected] : cases) {
    SCOPED_TRACE(stride);
    const auto params = utils::NetworkParams(
      {"object"}, {1, 4, 8}, {1, 4, 8}, 45, -45, 8, 1, 8, 1, false, {}, stride);
    PreprocessCuda preprocess(params, stream.get());
    cuda_utils::clear_async(count.get(), 1, stream.get());
    cuda_utils::clear_async(occupied.get(), 8, stream.get());
    cuda_utils::clear_async(projected.get(), 8, stream.get());
    auto * flags = stride != 0 ? occupied.get() : nullptr;
    auto * image = stride != 0 ? projected.get() : nullptr;
    CHECK_CUDA_ERROR(preprocess.projectPoints_launch(
      cloud.get(), input.size(), CloudFormat::XYZI, count.get(), points.get(), coors.get(),
      keys.get(), flags, image));
    CHECK_CUDA_ERROR(preprocess.interpolatePoints_launch(
      flags, image, count.get(), points.get(), coors.get(), keys.get()));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream.get()));

    uint32_t total = 0;
    CHECK_CUDA_ERROR(cudaMemcpy(&total, count.get(), sizeof(total), cudaMemcpyDeviceToHost));
    EXPECT_EQ(total, expected);
    std::array<int64_t, 4> input_keys{};
    CHECK_CUDA_ERROR(cudaMemcpy(
      input_keys.data(), keys.get(), sizeof(input_keys), cudaMemcpyDeviceToHost));
    std::sort(input_keys.begin(), input_keys.end());
    EXPECT_EQ(input_keys, (std::array<int64_t, 4>{0, 2, 4, 6}));
  }
}

TEST(PostprocessCuda, UsesSoftmaxForLegacyLogits)
{
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    GTEST_SKIP() << "A CUDA device is required.";
  }
  auto stream = cuda_utils::make_stream();
  const utils::PostprocessingParams params(0.5, {"ground"}, "", {}, {"ground", "object"},
    {255, 0, 0, 0, 255, 0});
  PostprocessCuda postprocess(params, stream.get(), true);
  const std::array<float, 8> points{1, 0, 0, 1, 2, 0, 0, 1};
  const std::array<float, 4> logits{1000, 1001, 1003, 1000};
  const auto points_d = cuda_utils::make_unique<float>(points.size());
  const auto logits_d = cuda_utils::make_unique<float>(logits.size());
  const auto seg_d = cuda_utils::make_unique<OutputSegmentationPointType>(2);
  const auto filtered_d = cuda_utils::make_unique<CloudPointTypeXYZI>(2);
  const auto count = cuda_utils::make_unique<uint32_t>(1);
  CHECK_CUDA_ERROR(cudaMemcpy(points_d.get(), points.data(), sizeof(points), cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(logits_d.get(), logits.data(), sizeof(logits), cudaMemcpyHostToDevice));
  cuda_utils::clear_async(count.get(), 1, stream.get());
  CHECK_CUDA_ERROR(postprocess.fillCloud_launch(
    points_d.get(), nullptr, 0, logits_d.get(), 2, CloudFormat::XYZI, CloudFormat::XYZI,
    utils::ActiveComm(true, false, true), count.get(), seg_d.get(), nullptr, filtered_d.get()));
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream.get()));
  std::array<OutputSegmentationPointType, 2> output{};
  uint32_t filtered = 0;
  CHECK_CUDA_ERROR(cudaMemcpy(output.data(), seg_d.get(), sizeof(output), cudaMemcpyDeviceToHost));
  CHECK_CUDA_ERROR(cudaMemcpy(&filtered, count.get(), sizeof(filtered), cudaMemcpyDeviceToHost));
  EXPECT_EQ(output[0].class_id, 1);
  EXPECT_NEAR(output[0].probability, 0.7310586f, 1e-6f);
  EXPECT_EQ(output[1].class_id, 0);
  EXPECT_NEAR(output[1].probability, 0.9525741f, 1e-6f);
  EXPECT_EQ(filtered, 1u);
}

}  // namespace lidar_frnet_ros1

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
