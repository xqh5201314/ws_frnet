# FRNet 模型优化设计与验证记录

更新于 2026 年 11 月 23 日。适用于 `lidar_frnet_ros1` 中的 17 类 FRNet 模型和 nuScenes `scene-0916` 回放配置。

当前样例使用完整 CUDA 补点，启用 ONNX 形状改写和六处 CUDA 最大值归约。与原 FP16 引擎相比，同输入离线推理 P50 降低 **24.2%**，样例 mIoU 下降 **0.0132 个百分点**。节点重启后的 ROS 实跑总耗时 P50 降低 **17.9%**。CUDA Graph 已接入，样例默认使用普通提交。[样例启动配置](../samples/nuscenes_scene_0916/README.md) · [原始实验记录](../bug_log/nuscenes_scene_0916/BUG_ANALYSIS_nuscenes_scene_0916.md)

## 测试基线

| 项目 | 本次条件 |
| --- | --- |
| 硬件与软件 | RTX 4060 Laptop GPU，ROS Noetic，CUDA 12.8，TensorRT 10.10，`/home/xqh/env/cuda12_trt10.sh` |
| 模型产物 | 原版 `frnet.onnx`，算子版 `frnet_frustum_max.onnx`，各自构建 TensorRT FP16 引擎 |
| 输入数据 | 399 帧 XYZI 点云，`/nuscenes/lidar_top`，坐标系 `LIDAR_TOP`，体素滤波关闭 |
| 投影与补点 | 视锥网格 `480 × 32`，补点投影 `1920 × 32`，垂直视场 `−30°～10°` |
| 动态形状 | 点数 `N`，占用网格数 `U`。完整补点时，每帧平均由 34,723 个原始点生成 53,856 个网络输入点 |
| 精度评估 | 41 帧人工标注。按时间戳和 XYZ 匹配 1,074,426 个已知类原始点，取现场有标注的 12 类 IoU 平均值 |

ONNX 接口为 `points[N,4]`、`coors[N,3]`、`voxel_coors[U,3]`、`inverse_map[N]` 和输出 `seg_logit[N,17]`。TensorRT profile 的 `N` 为 `5000 / 45000 / 120000`，`U` 为 `1000 / 8000 / 15360`，依次为最小值、优化值和最大值。算子版的四路输入、输出签名和 186 个原有初始化张量逐项通过比对。[样例 YAML](../samples/nuscenes_scene_0916/frnet_nuscenes.yaml) · [模型绑定](../src/lidar_frnet_ros1/lib/lidar_frnet.cpp)

离线同输入对照每帧生成一次预处理张量供两个引擎共用，并逐帧交换执行顺序。推理完成耗时从 `enqueueV3` 开始计到 CUDA stream 同步结束。`trtexec` 固定帧另外记录 CPU 提交和 GPU 完成耗时。ROS 总耗时覆盖节点开始处理至发布前。P50 为中位数，P95 为第 95 百分位。各步提速以该步对应的对照组为分母，mIoU 变化以百分点表示。[节点计时](../src/lidar_frnet_ros1/src/lidar_frnet_node.cpp) · [阶段计时](../src/lidar_frnet_ros1/lib/lidar_frnet.cpp)

## 优化步骤

### 1 建立回放与分段计时

节点按输入点云时间戳记录预处理、推理、后处理和总耗时。每个成功处理的输入都会产生分割输出，RGB 可视化和过滤点云随各自订阅生成。Launch 自动启动预设 RViz，预测和标注使用同一类别配色。离线测试用已对齐的标注统计 IoU，并用 `trtexec` 对真实帧做逐层计时。这一步建立了速度和精度的对照方式。[节点实现](../src/lidar_frnet_ros1/src/lidar_frnet_node.cpp) · [RViz 和回放使用说明](../samples/nuscenes_scene_0916/README.md)

FRNet 在点与视锥特征之间反复融合，先从视锥取回上下文给逐点网络，再把点特征按格聚合。在原引擎的三个抽样帧中，显式 `ScatterElements` 占逐层计时的 14%～17%，索引、重排、拼接和逐元素融合占 37%～42%。编码器的最大值归约是其中一个耗时较高的算子。[FRNet 论文 III-B](https://arxiv.org/html/2312.04484#S3.SS2) · [逐层实测](../bug_log/nuscenes_scene_0916/BUG_ANALYSIS_nuscenes_scene_0916.md#tensorrt-算子实测与优化)

### 2 从输入维度取得占用网格数

原 ONNX 用 `ReduceMax(inverse_map) + 1` 计算 `U`。推理时 TensorRT 从 GPU 回读这个形状值，在层信息中表现为 `[DevicetoShapeHostCopy]`。CUDA 预处理用排序和前缀和生成连续编号 `0…U−1`，所以转换脚本用 `Shape(voxel_coors)` 和 `Gather(0)` 直接读取第一维。改写后对应的回读层消失，固定形状也可进行 Graph 捕获。[CUDA 网格编号](../src/lidar_frnet_ros1/lib/preprocess/preprocess_kernel.cu) · [ONNX 转换](../src/lidar_frnet_ros1/scripts/optimize_frnet_onnx.py) · [TensorRT 形状计算与同步](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/dynamic-shapes-advanced.html#execution-tensors-vs-shape-tensors)

单独改形状的实验中，第 200 帧 CPU 提交 P50 由 **25.55–29.75 ms** 降为 **1.26–1.43 ms**。整包推理完成 P50 为 **27.47 → 27.54 ms**，P95 为 **45.43 → 44.84 ms**。全包预测类别一致率 **99.9652%**，共 7,480 个类别变化。该轮精度记录为逐点一致率，带标注的 mIoU 见后续引擎对照。形状改写已纳入交付版。[独立形状改写实验](../bug_log/nuscenes_scene_0916/BUG_ANALYSIS_nuscenes_scene_0916.md#tensorrt-算子实测与优化)

### 3 评估 CUDA 补点步长

加入 `interpolation_stride`。CUDA 保留通过输入校验的原始点，对左右均有投影点的空像素取两侧 XYZI 均值生成补点。步长 `s > 1` 时通过 `(x + y) % s == 0` 筛选参与补点的列，让相邻行的筛选起点错开。`1` 执行完整补点，`0` 关闭补点。四组使用同一原 FP16 引擎，在同一进程中预热后交替执行 399 帧。样例曾采用步长 3，算子版样例已恢复到步长 1。[CUDA 补点实现](../src/lidar_frnet_ros1/lib/preprocess/preprocess_kernel.cu) · [步长实测和各类 IoU](../bug_log/nuscenes_scene_0916/BUG_ANALYSIS_nuscenes_scene_0916.md#补点步长实测)

| 补点步长 | 平均推理点数 | 推理 P50 / P95 | P50 降幅 | 总耗时 P50 | 样例 mIoU | mIoU 变化 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 53,856 | 26.29 / 29.25 ms | 基线 | 28.67 ms | 66.9599% | 基线 |
| 2 | 44,289 | 22.03 / 24.73 ms | 16.2% | 24.17 ms | 66.8038% | −0.1560 |
| 3 | 41,099 | 20.49 / 22.77 ms | 22.1% | 22.65 ms | 66.8259% | −0.1340 |
| 0 | 34,723 | 18.06 / 19.94 ms | 31.3% | 20.15 ms | 66.7070% | −0.2528 |

### 4 验证按网格分组的归约原型

原型由 `FrnetFrustumGroups` 一次构建每个网格的点链表，六处 `FrnetFrustumMax` 共用链表并遍历各格的点。它使用完整补点并包含形状改写。与同轮原 FP16 引擎相比，整包推理 P50 **25.04 → 23.61 ms**，下降 **5.7%**。mIoU **66.9603% → 66.9503%**，下降 **0.0100 个百分点**，逐点类别一致率 **99.8786%**。[原型逐帧结果](/tmp/frnet_segmented.iouAou/ab/ab_frames.tsv) · [原型标注混淆矩阵](/tmp/frnet_segmented.iouAou/ab/confusion.tsv)

第 200 帧逐层中位耗时合计由 **5.37 ms 增至 9.53 ms**，优化侧合计含一次分组与六次归约。链表遍历在多个归约层耗时较高，交付版改用直接原子归约。[原引擎逐层计时](/tmp/frnet_segmented.iouAou/same_frame_A1_layers.json) · [链表原型逐层计时](/tmp/frnet_segmented.iouAou/same_frame_B1_layers.json)

### 5 用 CUDA 原子最大值归约替换六处 ScatterElements

原图把 `inverse_map[N]` 展开到 `[N,C]`，为每个点的每个通道提供相同的网格索引。转换脚本匹配编码器一处、主干五处 `ScatterElements(max)`，以 TensorRT `IPluginV3` 算子 `FrnetFrustumAtomicMax` 接收初始网格特征 `[U,C]`、点到网格的 `inverse_map[N]` 和点特征 `[N,C]`。CUDA 先复制初始网格特征，再用一维 `inverse_map` 直接定位目标行，逐通道保留初始值与组内点特征的最大值。[转换脚本](../src/lidar_frnet_ros1/scripts/optimize_frnet_onnx.py) · [CUDA 插件](../src/lidar_frnet_ros1/lib/frustum_plugins.cu) · [ONNX ScatterElements 归约语义](https://onnx.ai/onnx/operators/onnx__ScatterElements.html#scatterelements-18)

插件支持 FP32 和 FP16。FP32 按符号选择有符号 `atomicMax` 或无符号 `atomicMin`，维持浮点数的最大值顺序。FP16 将两个相邻通道打包，用 `__hmax2` 分别比较并用 `atomicCAS` 提交。插件编入节点加载的 CUDA 库，通过 TensorRT 注册表解析。[归约与数据类型实现](../src/lidar_frnet_ros1/lib/frustum_plugins.cu) · [包构建配置](../src/lidar_frnet_ros1/CMakeLists.txt)

下表衡量形状改写与直接原子归约合并后的交付引擎。逐层结果取第 200 帧同一份输入，整包结果覆盖 399 帧完整补点。

| 指标 | 原 FP16 引擎 | 交付 FP16 引擎 | 变化 |
| --- | ---: | ---: | ---: |
| 六处归约层中位耗时之和 | 5.15 ms | 1.76 ms | 降低 65.8% |
| 整包推理 P50 | 24.92 ms | 18.89 ms | 降低 24.2% |
| 整包推理 P95 | 42.02 ms | 34.82 ms | 降低 17.1% |
| 41 帧标注 mIoU | 66.9621% | 66.9489% | −0.0132 个百分点 |
| 标注点准确率 | 91.8647% | 91.8634% | −0.0013 个百分点 |

306 帧推理耗时下降。21,488,466 个预测点中有 12,578 个类别改变，一致率 **99.9415%**。有标注的类别中，摩托车 IoU 降幅最大，从 65.4955% 降到 65.4169%，下降约 **0.079 个百分点**。数值差异对应重建 FP16 引擎的实测输出。[交付版离线结果](../bug_log/nuscenes_scene_0916/BUG_ANALYSIS_nuscenes_scene_0916.md#完整补点下的算子与-cuda-graph-实测)

### 6 接入 CUDA Graph 提交

节点参数 `use_cuda_graph` 控制捕获。形状变更后的第一次 `enqueueV3` 更新 TensorRT 内部状态。下一帧沿用相同形状、绑定地址和 CUDA stream 时捕获计算图，后续满足相同条件的帧用 `cudaGraphLaunch` 执行。形状、绑定地址或 stream 变化时销毁旧图，捕获或提交失败时回退到普通 `enqueueV3`。交付引擎完成了形状切换和输出地址重绑验证，原引擎完成了捕获失败回退验证。[节点 Graph 实现](../src/lidar_frnet_ros1/src/tensorrt_wrapper.cpp) · [TensorRT Graph 捕获约束](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/optimization.html#inference-with-cuda-graphs)

| 第 200 帧固定形状 `N=52560`、`U=12374` | 交付引擎普通提交 | 同引擎 CUDA Graph |
| --- | ---: | ---: |
| CPU 提交 P50 | 1.31–1.36 ms | 0.019 ms |
| GPU 完成 P50 | 20.07–20.12 ms | 20.34 ms |

固定形状时，CPU 提交 P50 下降约 **98.6%**，GPU 完成 P50 实测增加约 **0.22–0.27 ms**。固定输入的重复测试中，第 200 帧普通提交最多 8 个点的类别变化，Graph 最多 11 个，均在 52,560 个点中比较。两种提交的最大 logit 差都是 0.125，第 0 帧 logits 逐位一致。两处 `ScatterElements(add)` 的浮点累加顺序可能带来重跑波动。整包 399 帧有 399 种 `(N,U)` 组合，当前样例采用普通提交。[CUDA Graph 测试数据](../bug_log/nuscenes_scene_0916/BUG_ANALYSIS_nuscenes_scene_0916.md#完整补点下的算子与-cuda-graph-实测)

## 重启后的 ROS 回放

11 月 23 日 11:45:55 重启后加载 `frnet_frustum_max.onnx.fp16.engine`。RViz 运行期间按约 0.5 倍速回放，每组取 3 轮完整回放的 1,197 帧，输入时间戳逐帧对齐。[启用算子版的实跑记录](../bug_log/nuscenes_scene_0916/BUG_ANALYSIS_nuscenes_scene_0916.md#算子版重启后的实跑对照)

| 阶段 P50 / P95 | 旧版完整补点 | 重启前步长 3 | 算子版完整补点 |
| --- | ---: | ---: | ---: |
| 预处理 | 0.430 / 0.971 ms | 0.447 / 1.016 ms | 0.433 / 1.195 ms |
| 推理 | 22.439 / 24.224 ms | 19.216 / 20.725 ms | 18.151 / 19.884 ms |
| 后处理 | 0.812 / 1.465 ms | 0.591 / 0.997 ms | 0.791 / 1.429 ms |
| 总处理 | 23.782 / 25.724 ms | 20.304 / 21.982 ms | 19.517 / 21.364 ms |

同为完整补点时，推理 P50 降低 **19.1%**，推理 P95 降低 **17.9%**，总处理 P95 降低 **16.9%**，预处理 P95 增加 **0.224 ms**。相较重启前的步长 3，分割点数增加约 31%，后处理 P50 增加 0.20 ms，总处理 P50 降低 **3.9%**。首帧总处理耗时为 70.928 ms，后续三轮在同一帧为 19.979–20.676 ms，三个完整轮次共 1,197 帧均成功处理。[实跑时间窗与首帧证据](../bug_log/nuscenes_scene_0916/BUG_ANALYSIS_nuscenes_scene_0916.md#算子版重启后的实跑对照)

## 产物和复现

样例目录已部署优化 ONNX 和验证用 FP16 引擎。验证引擎 SHA-256 为 `c5fd6a5ca1579b2ed4fb1c311de0ddd1c1858e4210d64f470d93a685b32249db`。包的 5 项回归测试通过。若重新导出同结构的 17 类模型，可用安装了 `onnx` 的 Python 运行转换脚本，TensorRT 节点会为新模型构建缓存。[优化 ONNX](../samples/nuscenes_scene_0916/frnet_frustum_max.onnx) · [FP16 引擎](../samples/nuscenes_scene_0916/frnet_frustum_max.onnx.fp16.engine) · [回归测试](../src/lidar_frnet_ros1/test/frnet_regression_test.cpp)

```bash
cd /home/xqh/perception/ws_frnet
source /home/xqh/env/cuda12_trt10.sh
source devel/setup.bash

sample="$PWD/samples/nuscenes_scene_0916"
roslaunch lidar_frnet_ros1 lidar_frnet.launch \
  onnx_path:="$sample/frnet_frustum_max.onnx" \
  config_file:="$sample/frnet_nuscenes.yaml" \
  input_pointcloud:=/nuscenes/lidar_top \
  use_voxel_grid:=false interpolation_stride:=1 use_cuda_graph:=false
```

另一终端载入相同环境后运行 `rosbag play --loop --rate 0.5 samples/nuscenes_scene_0916/nuscenes_scene_0916.bag`。对照时把 `onnx_path` 指向样例目录的 `frnet.onnx`，按点云时间戳匹配完整回放。精度按时间戳和 XYZ 对齐 41 帧标注计算 mIoU，并比较整包 399 帧同输入下的预测类别。[回放指令与标注说明](../samples/nuscenes_scene_0916/README.md)
