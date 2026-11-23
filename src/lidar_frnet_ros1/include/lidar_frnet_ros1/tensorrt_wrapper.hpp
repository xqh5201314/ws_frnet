// Copyright 2026
//
// Licensed under the Apache License, Version 2.0.

#ifndef LIDAR_FRNET_ROS1__TENSORRT_WRAPPER_HPP_
#define LIDAR_FRNET_ROS1__TENSORRT_WRAPPER_HPP_

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#if NV_TENSORRT_MAJOR < 10
#error "FRNet requires TensorRT 10 or newer for its INT64 coordinate tensors."
#endif

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace lidar_frnet_ros1
{

struct TrtCommonConfig
{
  TrtCommonConfig(std::string onnx_path, std::string precision, bool use_cuda_graph = false);

  std::string onnx_path;
  std::string precision;
  std::string engine_path;
  std::size_t workspace_size{1ULL << 32U};
  bool use_cuda_graph;
};

struct NetworkIO
{
  std::string tensor_name;
  nvinfer1::Dims dims;
  nvinfer1::DataType data_type;
  nvinfer1::TensorIOMode mode;
};

struct ProfileDims
{
  std::string tensor_name;
  nvinfer1::Dims min;
  nvinfer1::Dims opt;
  nvinfer1::Dims max;
};

class TrtCommon
{
public:
  explicit TrtCommon(const TrtCommonConfig & config);
  ~TrtCommon();

  bool setup(
    const std::vector<ProfileDims> & profile_dims, const std::vector<NetworkIO> & network_io);
  bool setInputShape(const std::string & name, nvinfer1::Dims dims);
  bool setTensorAddress(const std::string & name, void * data);
  bool enqueueV3(cudaStream_t stream);
  const std::string & outputTensorName() const { return output_tensor_name_; }
  nvinfer1::Dims outputShape() const;

private:
  struct Logger : public nvinfer1::ILogger
  {
    void log(Severity severity, const char * msg) noexcept override;
  };

  std::unique_ptr<nvinfer1::IHostMemory> buildEngine(
    const std::vector<ProfileDims> & profile_dims, const std::vector<NetworkIO> & network_io);
  bool loadEngine();
  bool deserializeEngine(const void * data, size_t size);
  bool validateEngine(
    const std::vector<ProfileDims> & profile_dims, const std::vector<NetworkIO> & network_io);
  bool saveEngine(const nvinfer1::IHostMemory & plan) const;
  void resetCudaGraph();
  bool captureCudaGraph(cudaStream_t stream);

  TrtCommonConfig config_;
  Logger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  std::string output_tensor_name_;
  cudaGraphExec_t cuda_graph_{nullptr};
  cudaStream_t graph_stream_{nullptr};
  bool graph_warmed_up_{false};
  bool graph_enabled_;
};

}  // namespace lidar_frnet_ros1

#endif  // LIDAR_FRNET_ROS1__TENSORRT_WRAPPER_HPP_
