// Copyright 2026
//
// Licensed under the Apache License, Version 2.0.

#include "lidar_frnet_ros1/tensorrt_wrapper.hpp"

#include <NvInferPlugin.h>
#include <NvOnnxParser.h>

#include <ros/console.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace lidar_frnet_ros1
{

namespace
{
bool file_exists(const std::string & path)
{
  std::ifstream stream(path, std::ios::binary);
  return stream.good();
}

bool same_dims(const nvinfer1::Dims & a, const nvinfer1::Dims & b)
{
  return a.nbDims >= 0 && a.nbDims == b.nbDims && std::equal(a.d, a.d + a.nbDims, b.d);
}

bool validate_tensor(
  const NetworkIO & expected, const char * name, const nvinfer1::Dims & shape,
  nvinfer1::DataType data_type)
{
  if (data_type != expected.data_type) {
    ROS_WARN("TensorRT tensor '%s' has an unexpected data type.", name);
    return false;
  }
  if (!same_dims(shape, expected.dims)) {
    if (expected.mode == nvinfer1::TensorIOMode::kOUTPUT && shape.nbDims == 2) {
      ROS_WARN_STREAM(
        "TensorRT output '" << name << "' has " << shape.d[1] << " classes, but class_names has "
        << expected.dims.d[1] << ". Use the model's class configuration.");
    } else {
      ROS_WARN("TensorRT tensor '%s' has an unexpected shape.", name);
    }
    return false;
  }
  return true;
}

bool matches_name(const NetworkIO & expected, const char * name)
{
  return expected.tensor_name == name ||
         (expected.tensor_name == "pred_probs" && std::string_view(name) == "seg_logit");
}
}  // namespace

TrtCommonConfig::TrtCommonConfig(
  std::string onnx_path_arg, std::string precision_arg, bool use_cuda_graph_arg)
: onnx_path(std::move(onnx_path_arg)), precision(std::move(precision_arg)),
  use_cuda_graph(use_cuda_graph_arg)
{
  if (onnx_path.empty()) {
    throw std::runtime_error("Set onnx_path to the FRNet ONNX file for the configured sensor.");
  }
  if (precision != "fp16" && precision != "FP16" && precision != "fp32" && precision != "FP32") {
    throw std::runtime_error("trt_precision must be fp16 or fp32.");
  }
  engine_path = onnx_path + "." + precision + ".engine";
}

void TrtCommon::Logger::log(Severity severity, const char * msg) noexcept
{
  if (severity <= Severity::kWARNING) {
    ROS_WARN_STREAM("TensorRT: " << msg);
  }
}

TrtCommon::TrtCommon(const TrtCommonConfig & config)
: config_(config), graph_enabled_(config.use_cuda_graph)
{
}

TrtCommon::~TrtCommon()
{
  resetCudaGraph();
}

bool TrtCommon::setup(
  const std::vector<ProfileDims> & profile_dims, const std::vector<NetworkIO> & network_io)
{
  resetCudaGraph();
  graph_enabled_ = config_.use_cuda_graph;
  if (!initLibNvInferPlugins(&logger_, "")) {
    ROS_ERROR("Failed to initialize TensorRT standard plugins.");
    return false;
  }

  if (file_exists(config_.engine_path)) {
    std::error_code error;
    const auto model_time = std::filesystem::last_write_time(config_.onnx_path, error);
    const bool model_updated = !error &&
      std::filesystem::last_write_time(config_.engine_path, error) < model_time && !error;
    if (!model_updated && loadEngine() && validateEngine(profile_dims, network_io)) {
      context_.reset(engine_->createExecutionContext());
      if (context_) {
        ROS_INFO_STREAM("Using TensorRT engine: " << config_.engine_path);
        return true;
      }
    }
    ROS_WARN_STREAM("TensorRT engine needs rebuilding: " << config_.engine_path);
    context_.reset();
    engine_.reset();
    runtime_.reset();
  }
  if (!file_exists(config_.onnx_path)) {
    ROS_ERROR_STREAM("Cannot build TensorRT engine. ONNX file not found: " << config_.onnx_path);
    return false;
  }

  ROS_INFO_STREAM("Building TensorRT engine from: " << config_.onnx_path);
  const auto plan = buildEngine(profile_dims, network_io);
  if (!plan || !deserializeEngine(plan->data(), plan->size()) ||
    !validateEngine(profile_dims, network_io)) {
    return false;
  }
  context_.reset(engine_->createExecutionContext());
  return context_ && saveEngine(*plan);
}

std::unique_ptr<nvinfer1::IHostMemory> TrtCommon::buildEngine(
  const std::vector<ProfileDims> & profile_dims, const std::vector<NetworkIO> & network_io)
{
  std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger_));
  if (!builder) {
    ROS_ERROR("Failed to create the TensorRT builder.");
    return nullptr;
  }

  std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));
  std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
  if (!network || !config) {
    return nullptr;
  }
  std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger_));
  if (!parser || !parser->parseFromFile(
        config_.onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
    ROS_ERROR_STREAM("Failed to parse ONNX: " << config_.onnx_path);
    return nullptr;
  }

  const int32_t num_inputs = network->getNbInputs();
  const int32_t num_outputs = network->getNbOutputs();
  if (num_inputs != static_cast<int32_t>(profile_dims.size()) || num_outputs != 1) {
    ROS_ERROR("FRNet ONNX has an unexpected number of inputs or outputs.");
    return nullptr;
  }
  for (const auto & expected : network_io) {
    const bool is_input = expected.mode == nvinfer1::TensorIOMode::kINPUT;
    const nvinfer1::ITensor * tensor = nullptr;
    for (int32_t i = 0; i < (is_input ? num_inputs : num_outputs); ++i) {
      const auto * candidate = is_input ? network->getInput(i) : network->getOutput(i);
      if (matches_name(expected, candidate->getName())) {
        tensor = candidate;
        break;
      }
    }
    if (!tensor) {
      ROS_ERROR_STREAM("Missing FRNet ONNX tensor: " << expected.tensor_name);
      return nullptr;
    }
    if (!validate_tensor(expected, tensor->getName(), tensor->getDimensions(), tensor->getType())) {
      return nullptr;
    }
  }

  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, config_.workspace_size);
  if (config_.precision == "fp16" || config_.precision == "FP16") {
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
  }

  auto profile = builder->createOptimizationProfile();
  if (profile == nullptr) {
    return nullptr;
  }
  for (const auto & dims : profile_dims) {
    const char * name = dims.tensor_name.c_str();
    using nvinfer1::OptProfileSelector;
    if (!profile->setDimensions(name, OptProfileSelector::kMIN, dims.min) ||
      !profile->setDimensions(name, OptProfileSelector::kOPT, dims.opt) ||
      !profile->setDimensions(name, OptProfileSelector::kMAX, dims.max)) {
      ROS_ERROR("Invalid TensorRT dimensions for tensor '%s'.", name);
      return nullptr;
    }
  }
  if (config->addOptimizationProfile(profile) < 0) {
    ROS_ERROR("Failed to register the TensorRT optimization profile.");
    return nullptr;
  }

  std::unique_ptr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
  if (!plan) {
    ROS_ERROR("Failed to build TensorRT engine.");
  }
  return plan;
}

bool TrtCommon::saveEngine(const nvinfer1::IHostMemory & plan) const
{
  std::string temporary_path = config_.engine_path + ".XXXXXX";
  const int fd = mkstemp(temporary_path.data());
  if (fd < 0) {
    ROS_ERROR_STREAM("Failed to create TensorRT engine file beside: " << config_.engine_path);
    return false;
  }
  auto * file = fdopen(fd, "wb");
  bool written = false;
  if (file) {
    written = std::fwrite(plan.data(), 1, plan.size(), file) == plan.size();
    written = std::fclose(file) == 0 && written;
  } else {
    close(fd);
  }
  if (written && std::rename(temporary_path.c_str(), config_.engine_path.c_str()) == 0) {
    return true;
  }
  std::remove(temporary_path.c_str());
  ROS_ERROR_STREAM("Failed to save TensorRT engine: " << config_.engine_path);
  return false;
}

bool TrtCommon::loadEngine()
{
  std::ifstream stream(config_.engine_path, std::ios::binary);
  if (!stream) {
    ROS_ERROR_STREAM("Failed to open TensorRT engine: " << config_.engine_path);
    return false;
  }
  std::vector<char> data((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
  return deserializeEngine(data.data(), data.size());
}

bool TrtCommon::deserializeEngine(const void * data, size_t size)
{
  context_.reset();
  engine_.reset();
  runtime_.reset(nvinfer1::createInferRuntime(logger_));
  if (runtime_ && size > 0) {
    engine_.reset(runtime_->deserializeCudaEngine(data, size));
  }
  return engine_ != nullptr;
}

bool TrtCommon::validateEngine(
  const std::vector<ProfileDims> & profile_dims, const std::vector<NetworkIO> & network_io)
{
  if (engine_->getNbIOTensors() != static_cast<int32_t>(network_io.size()) ||
    engine_->getNbOptimizationProfiles() < 1) {
    ROS_WARN("TensorRT engine has an unexpected number of tensors or no optimization profile.");
    return false;
  }
  for (const auto & expected : network_io) {
    const char * name = nullptr;
    for (int32_t i = 0; i < engine_->getNbIOTensors(); ++i) {
      const auto * candidate = engine_->getIOTensorName(i);
      if (candidate && matches_name(expected, candidate)) {
        name = candidate;
        break;
      }
    }
    if (!name) {
      ROS_WARN_STREAM("Missing FRNet TensorRT tensor: " << expected.tensor_name);
      return false;
    }
    if (engine_->getTensorIOMode(name) != expected.mode ||
      engine_->getTensorLocation(name) != nvinfer1::TensorLocation::kDEVICE ||
      engine_->getTensorFormat(name) != nvinfer1::TensorFormat::kLINEAR) {
      ROS_WARN("TensorRT engine has an incompatible tensor '%s'.", name);
      return false;
    }
    if (!validate_tensor(expected, name, engine_->getTensorShape(name), engine_->getTensorDataType(name))) {
      return false;
    }
    if (expected.mode == nvinfer1::TensorIOMode::kOUTPUT) {
      output_tensor_name_ = name;
    }
  }
  for (const auto & expected : profile_dims) {
    const char * name = expected.tensor_name.c_str();
    using nvinfer1::OptProfileSelector;
    if (!same_dims(engine_->getProfileShape(name, 0, OptProfileSelector::kMIN), expected.min) ||
      !same_dims(engine_->getProfileShape(name, 0, OptProfileSelector::kOPT), expected.opt) ||
      !same_dims(engine_->getProfileShape(name, 0, OptProfileSelector::kMAX), expected.max)) {
      ROS_WARN("Cached TensorRT profile for '%s' differs from the configured min/opt/max.", name);
      return false;
    }
  }
  return true;
}

nvinfer1::Dims TrtCommon::outputShape() const
{
  return context_->getTensorShape(output_tensor_name_.c_str());
}

bool TrtCommon::setInputShape(const std::string & name, nvinfer1::Dims dims)
{
  if (!context_) return false;
  if (graph_enabled_) {
    if (same_dims(context_->getTensorShape(name.c_str()), dims)) return true;
    resetCudaGraph();
  }
  return context_->setInputShape(name.c_str(), dims);
}

bool TrtCommon::setTensorAddress(const std::string & name, void * data)
{
  if (!context_) return false;
  if (graph_enabled_) {
    if (context_->getTensorAddress(name.c_str()) == data && data) return true;
    resetCudaGraph();
  }
  return context_->setTensorAddress(name.c_str(), data);
}

bool TrtCommon::enqueueV3(cudaStream_t stream)
{
  if (!context_) return false;
  if (!graph_enabled_) return context_->enqueueV3(stream);

  if (graph_stream_ != stream) {
    resetCudaGraph();
    graph_stream_ = stream;
  }
  if (!graph_warmed_up_) {
    // TensorRT updates its internal state on the first enqueue after a shape change.
    graph_warmed_up_ = context_->enqueueV3(stream);
    return graph_warmed_up_;
  }
  if ((cuda_graph_ || captureCudaGraph(stream)) && cudaGraphLaunch(cuda_graph_, stream) == cudaSuccess) {
    return true;
  }
  resetCudaGraph();
  cudaGetLastError();  // Clear the handled Graph error before enqueuing normally.
  graph_enabled_ = false;
  ROS_WARN("CUDA Graph is unavailable. Using TensorRT enqueueV3 for this engine.");
  return context_->enqueueV3(stream);
}

void TrtCommon::resetCudaGraph()
{
  if (cuda_graph_) {
    cudaGraphExecDestroy(cuda_graph_);
    cuda_graph_ = nullptr;
  }
  graph_warmed_up_ = false;
}

bool TrtCommon::captureCudaGraph(cudaStream_t stream)
{
  if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
    return false;
  }
  const bool enqueued = context_->enqueueV3(stream);
  cudaGraph_t graph = nullptr;
  const auto status = cudaStreamEndCapture(stream, &graph);
  if (status == cudaSuccess && enqueued && graph &&
      cudaGraphInstantiate(&cuda_graph_, graph, 0) == cudaSuccess) {
    cudaGraphDestroy(graph);
    ROS_INFO_ONCE("Captured FRNet CUDA Graph for repeated input shapes.");
    return true;
  }
  if (graph) cudaGraphDestroy(graph);
  return false;
}

}  // namespace lidar_frnet_ros1
