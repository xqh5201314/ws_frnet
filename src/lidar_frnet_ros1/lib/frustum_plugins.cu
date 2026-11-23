// Licensed under the Apache License, Version 2.0.

#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <new>

namespace lidar_frnet_ros1
{
using namespace nvinfer1;
constexpr int threads_per_block = 128;
constexpr PluginFieldCollection no_fields{0, nullptr};

__device__ void accumulateMax(float * target, float value)
{
  if (isnan(value)) return;
  const int bits = __float_as_int(value);
  // Float bits are ordered as signed integers for positives. For negatives unsigned
  // order is reversed, with every positive bit pattern below the negative range.
  if (bits >= 0) {
    atomicMax(reinterpret_cast<int *>(target), bits);
  } else {
    atomicMin(reinterpret_cast<unsigned *>(target), static_cast<unsigned>(bits));
  }
}

__device__ void accumulateMax(__half2 * target, __half2 value)
{
  auto * bits = reinterpret_cast<unsigned *>(target);
  unsigned previous = atomicCAS(bits, 0U, 0U);
  for (;;) {
    const __half2_raw current{static_cast<uint16_t>(previous), static_cast<uint16_t>(previous >> 16U)};
    const __half2_raw maximum = __hmax2(__half2(current), value);
    const unsigned result = static_cast<unsigned>(maximum.x) | (static_cast<unsigned>(maximum.y) << 16U);
    if (result == previous) return;
    const auto observed = atomicCAS(bits, previous, result);
    if (observed == previous) return;
    previous = observed;
  }
}

template <typename T>
__global__ void poolFrustumMax(
  const int64_t * inverse_map, const T * updates, int64_t values, int64_t voxels,
  int64_t units_per_point, T * output)
{
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= values) return;
  const auto voxel = inverse_map[index / units_per_point];
  if (voxel < 0 || voxel >= voxels) return;
  accumulateMax(output + voxel * units_per_point + index % units_per_point, updates[index]);
}

class FrustumMaxPlugin final
  : public IPluginV3, public IPluginV3OneCore, public IPluginV3OneBuild, public IPluginV3OneRuntime
{
public:
  IPluginCapability * getCapabilityInterface(PluginCapabilityType type) noexcept override
  {
    switch (type) {
      case PluginCapabilityType::kCORE: return static_cast<IPluginV3OneCore *>(this);
      case PluginCapabilityType::kBUILD: return static_cast<IPluginV3OneBuild *>(this);
      case PluginCapabilityType::kRUNTIME: return static_cast<IPluginV3OneRuntime *>(this);
    }
    return nullptr;
  }
  IPluginV3 * clone() noexcept override { return new (std::nothrow) FrustumMaxPlugin(); }
  const char * getPluginName() const noexcept override { return "FrnetFrustumAtomicMax"; }
  const char * getPluginVersion() const noexcept override { return "1"; }
  const char * getPluginNamespace() const noexcept override { return ""; }
  int32_t getNbOutputs() const noexcept override { return 1; }

  int32_t getOutputDataTypes(
    DataType * output, int32_t nb_outputs, const DataType * input,
    int32_t nb_inputs) const noexcept override
  {
    if (nb_inputs != 3 || nb_outputs != 1) return -1;
    output[0] = input[0];
    return 0;
  }

  int32_t getOutputShapes(
    const DimsExprs * in, int32_t nb_inputs, const DimsExprs *, int32_t nb_shape_inputs,
    DimsExprs * out, int32_t nb_outputs, IExprBuilder &) noexcept override
  {
    if (nb_inputs != 3 || nb_outputs != 1 || nb_shape_inputs != 0 ||
        in[0].nbDims != 2 || in[1].nbDims != 1 || in[2].nbDims != 2) return -1;
    out[0] = in[0];
    return 0;
  }

  bool supportsFormatCombination(
    int32_t pos, const DynamicPluginTensorDesc * tensors, int32_t nb_inputs,
    int32_t nb_outputs) noexcept override
  {
    if (nb_inputs != 3 || nb_outputs != 1 || pos < 0 || pos >= 4 ||
        tensors[pos].desc.format != TensorFormat::kLINEAR) return false;
    const auto type = tensors[pos].desc.type;
    if (pos == 0) return type == DataType::kFLOAT || type == DataType::kHALF;
    return type == (pos == 1 ? DataType::kINT64 : tensors[0].desc.type);
  }

  int32_t configurePlugin(
    const DynamicPluginTensorDesc *, int32_t, const DynamicPluginTensorDesc *,
    int32_t) noexcept override { return 0; }

  int32_t onShapeChange(
    const PluginTensorDesc * in, int32_t nb_inputs, const PluginTensorDesc *,
    int32_t nb_outputs) noexcept override
  {
    if (nb_inputs != 3 || nb_outputs != 1) return -1;
    const auto voxels = in[0].dims.d[0];
    const auto channels = in[0].dims.d[1];
    const auto points = in[1].dims.d[0];
    constexpr int64_t max_blocks = std::numeric_limits<int32_t>::max();
    if (voxels <= 0 || points <= 0 || channels <= 0 ||
        points > max_blocks * threads_per_block / channels ||
        in[2].dims.d[0] != points || in[2].dims.d[1] != channels ||
        (in[0].type == DataType::kHALF && channels % 2 != 0)) return -1;
    return 0;
  }

  int32_t enqueue(
    const PluginTensorDesc * in, const PluginTensorDesc *, const void * const * input,
    void * const * output, void *, cudaStream_t stream) noexcept override
  {
    const auto voxels = in[0].dims.d[0];
    const auto channels = in[0].dims.d[1];
    const auto points = in[1].dims.d[0];
    const bool half = in[0].type == DataType::kHALF;
    const auto bytes = static_cast<size_t>(voxels) * channels * (half ? sizeof(__half) : sizeof(float));
    if (cudaMemcpyAsync(output[0], input[0], bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
      return -1;
    }
    const auto * inverse_map = static_cast<const int64_t *>(input[1]);
    const auto units_per_point = half ? channels / 2 : channels;
    const auto values = points * units_per_point;
    const auto blocks = (values - 1) / threads_per_block + 1;
    if (half) {
      poolFrustumMax<<<blocks, threads_per_block, 0, stream>>>(
        inverse_map, static_cast<const __half2 *>(input[2]), values, voxels, units_per_point,
        static_cast<__half2 *>(output[0]));
    } else {
      poolFrustumMax<<<blocks, threads_per_block, 0, stream>>>(
        inverse_map, static_cast<const float *>(input[2]), values, voxels, units_per_point,
        static_cast<float *>(output[0]));
    }
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
  }

  IPluginV3 * attachToContext(IPluginResourceContext *) noexcept override { return clone(); }
  const PluginFieldCollection * getFieldsToSerialize() noexcept override { return &no_fields; }
};

class FrustumMaxCreator final : public IPluginCreatorV3One
{
public:
  const char * getPluginName() const noexcept override { return "FrnetFrustumAtomicMax"; }
  const char * getPluginVersion() const noexcept override { return "1"; }
  const char * getPluginNamespace() const noexcept override { return ""; }
  const PluginFieldCollection * getFieldNames() noexcept override { return &no_fields; }
  IPluginV3 * createPlugin(
    const char *, const PluginFieldCollection *, TensorRTPhase) noexcept override
  {
    return new (std::nothrow) FrustumMaxPlugin();
  }
};

REGISTER_TENSORRT_PLUGIN(FrustumMaxCreator);
}  // namespace lidar_frnet_ros1
