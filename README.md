# FRNet ROS1 激光雷达语义分割

这是基于 Autoware FRNet 的 ROS1 Noetic 工作区。节点接收 `sensor_msgs/PointCloud2`，在 CUDA 上完成角度投影、补点和网格分组，用 TensorRT 执行 FRNet 推理，再生成逐点语义标签、RGB 可视化和可选的过滤点云。节点使用 ROS1 标准消息，编译时仅需此工作区的包与下文列出的依赖。[Autoware FRNet](https://github.com/autowarefoundation/autoware_universe/tree/main/perception/autoware_lidar_frnet) · [FRNet 论文](https://arxiv.org/abs/2312.04484)

nuScenes 样例使用 17 类模型和完整 CUDA 补点。算子版将六处按网格取最大值的操作改为 CUDA 原子归约，399 帧同输入离线对照中推理 P50 从 24.92 ms 降到 18.89 ms，41 帧标注的 mIoU 下降 0.0132 个百分点。[模型优化设计与验证](docs/frnet_model_optimization.md)

## 项目构成

| 目录 | 内容 |
| --- | --- |
| [`src/lidar_frnet_ros1/`](src/lidar_frnet_ros1/) | ROS1 节点和 TensorRT 封装位于 `src/`，CUDA 预处理、后处理和归约插件位于 `lib/`，接口与点类型位于 `include/`。`launch/` 和 `config/` 提供启动参数、雷达配置与 RViz，`cpkt/` 保存原始 17 类模型，`scripts/` 提供 ONNX 转换，`test/` 提供回归测试 |
| [`samples/autoware_lidar/`](samples/autoware_lidar/) | Autoware 的 ROS2 样例压缩包、提取左侧雷达后的 60 帧 ROS1 bag 和 17 类链路测试配置 |
| [`samples/nuscenes_scene_0916/`](samples/nuscenes_scene_0916/) | 399 帧点云和 41 帧逐点标注、原始及优化模型、FP16 引擎和 17 类配置 |
| [`docs/`](docs/) | 优化步骤、算子和 CUDA Graph 设计、耗时与精度对照 |
| [`bug_log/`](bug_log/) | Autoware 样例输出核对、nuScenes 算子测试与 ROS 回放分析。报告中的本机日志路径用于定位对应测试记录 |

## 环境与依赖

已验证的环境为 Ubuntu 20.04.6、ROS Noetic、CUDA 12.8、TensorRT 10.10.0.31 和 NVIDIA RTX 4060 Laptop GPU。编译需要支持 C++17 的编译器、CMake、catkin、CUDA Toolkit 及 TensorRT 开发头文件和库。RViz 用于桌面查看，启用可选的 5 cm 体素滤波时还需 `nodelet` 和 `pcl_ros`。Python 的 `onnx` 包用于重新生成优化模型。[构建配置](src/lidar_frnet_ros1/CMakeLists.txt) · [ROS 包依赖](src/lidar_frnet_ros1/package.xml)

仓库中的两个较大样例由 Git LFS 管理。克隆后在仓库根目录执行以下命令，可获取原始 ROS2 样例压缩包和可直接播放的 nuScenes ROS1 bag。[Git LFS 安装与协作](https://docs.github.com/en/repositories/working-with-files/managing-large-files/collaboration-with-git-large-file-storage)

```bash
git lfs install
git lfs pull
```

原开发机在每个编译或运行终端加载已有脚本。

```bash
source /home/xqh/env/cuda12_trt10.sh
```

在其他机器上，先安装匹配的 ROS Noetic、CUDA 和 TensorRT，再按实际安装位置设置环境。

```bash
source /opt/ros/noetic/setup.bash
export CUDA_HOME=/usr/local/cuda-12.8
export TENSORRT_ROOT=/opt/TensorRT-10.10.0.31
export PATH="$TENSORRT_ROOT/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$TENSORRT_ROOT/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

## 编译

在仓库根目录执行。`rosdep` 使用本机已初始化的 Noetic rosdep 索引安装 ROS 包依赖。`src/CMakeLists.txt` 指向 Noetic 的 catkin 顶层 CMake 文件。

```bash
rosdep install --from-paths src --ignore-src --rosdistro noetic -r -y
catkin_make -DCMAKE_BUILD_TYPE=Release \
  -DPYTHON_EXECUTABLE=/usr/bin/python3 \
  -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" \
  -DTENSORRT_ROOT="$TENSORRT_ROOT"
source devel/setup.bash
```

## 回放 nuScenes 样例

在已加载环境且位于仓库根目录的终端启动节点。样例使用完整补点和优化 ONNX，关闭体素滤波。桌面会话设置了 `DISPLAY` 时，launch 自动打开 RViz 并使用 `LIDAR_TOP` 坐标系。[样例配置](samples/nuscenes_scene_0916/frnet_nuscenes.yaml) · [回放说明](samples/nuscenes_scene_0916/README.md)

```bash
source devel/setup.bash
sample="$PWD/samples/nuscenes_scene_0916"
roslaunch lidar_frnet_ros1 lidar_frnet.launch \
  onnx_path:="$sample/frnet_frustum_max.onnx" \
  config_file:="$sample/frnet_nuscenes.yaml" \
  input_pointcloud:=/nuscenes/lidar_top \
  use_voxel_grid:=false
```

另开终端，同样加载 CUDA、TensorRT 和工作区环境，然后循环播放点云。

```bash
source devel/setup.bash
rosbag play --loop --rate 0.5 samples/nuscenes_scene_0916/nuscenes_scene_0916.bag
```

在 RViz 中查看 `FRNet Segmentation`。在有人工标注的关键帧暂停，可切换 `nuScenes Ground Truth`，按同一组 17 类配色对照。终端按输入时间戳打印每帧的预处理、推理、后处理和总耗时。

## 回放 Autoware 左侧雷达样例

该 ROS1 bag 含 60 帧 `velodyne_left` 点云，适合验证输入、输出和 RViz 链路。`frnet_17_sample.yaml` 使用编号形式的占位类别名，语义精度应通过搭配标注的 nuScenes 样例评估。[Autoware 样例验证记录](bug_log/autoware_lidar/BUG_ANALYSIS_autoware_lidar.md)

```bash
sample="$PWD/samples/autoware_lidar"
roslaunch lidar_frnet_ros1 lidar_frnet.launch \
  onnx_path:="$sample/frnet.onnx" \
  config_file:="$sample/frnet_17_sample.yaml" \
  input_pointcloud:=/sensing/lidar/left/pointcloud_raw \
  rviz_fixed_frame:=velodyne_left \
  use_voxel_grid:=false
```

在另一终端载入相同环境，运行 `rosbag play --loop samples/autoware_lidar/autoware_velodyne_left.bag`。`sample-rosbag.zip` 是 Autoware 原始 ROS2 样例。

## 接入自己的雷达和模型

提供模型 ONNX、匹配的 YAML 和雷达话题。YAML 要与模型训练时的类别顺序、投影视场和网格尺寸相符，点数与占用网格数也须落在 TensorRT profile 范围内。包内 `ros1_frnet_ot128.yaml` 和 `ros1_frnet_qt128.yaml` 对应 27 类 Autoware 模型。使用这些配置时请提供对应 ONNX。点云应在传感器坐标系内，`rviz_fixed_frame` 可设为该帧。支持的小端布局包括 XYZI、XYZIRC、XYZIRT、XYZIRCAEDT 和 XYZIRADRT。[完整包说明](src/lidar_frnet_ros1/README.md)

| 启动项或参数 | 用途 |
| --- | --- |
| `onnx_path` | ONNX 路径，TensorRT 引擎缓存通常写入相邻的 `<文件>.fp16.engine` |
| `config_file` | 类别、颜色、投影、补点、TensorRT profile 与过滤配置 |
| `input_pointcloud` | 雷达 `PointCloud2` 输入话题 |
| `use_voxel_grid` | 默认启用 5 cm 体素滤波，复现实验使用 `false` |
| `interpolation_stride` | `1` 完整补点，`2` 及以上按步长筛选补点列，`0` 关闭补点 |
| `use_cuda_graph` | 默认关闭。连续帧输入形状、显存绑定和 CUDA stream 一致时可以捕获并复用推理图 |
| `rviz`、`rviz_fixed_frame` | 控制 RViz 是否启动及其固定坐标系，终端运行可传 `rviz:=false` |
| `build_only` | 预建或验证 TensorRT 引擎后退出，可先用 `build_only:=true` 准备缓存 |

首次运行、ONNX 更新或已有引擎与本机环境不兼容时，节点会按 YAML 构建并缓存引擎。模型目录须可写，构建日志会显示实际加载或构建的文件路径。要重新导出当前 17 类结构的算子版，可在安装了 `onnx` 的 Python 环境中运行转换脚本。[转换实现](src/lidar_frnet_ros1/scripts/optimize_frnet_onnx.py)

```bash
sample="$PWD/samples/nuscenes_scene_0916"
python3 src/lidar_frnet_ros1/scripts/optimize_frnet_onnx.py \
  "$sample/frnet.onnx" "$sample/frnet_frustum_max.onnx"
```

## 输出话题与验证

| 默认话题 | 内容 |
| --- | --- |
| `/lidar_frnet/output/pointcloud/segmentation` | 有效输入的逐点分割，含 XYZ、类别编号和置信度 |
| `/lidar_frnet/output/pointcloud/visualization` | 订阅时生成按类别着色的 RGB 点云 |
| `/lidar_frnet/output/pointcloud/filtered` | 订阅时按 YAML 配置生成过滤点云 |

可用 `rostopic hz /lidar_frnet/output/pointcloud/segmentation` 确认输出频率。检查耗时时用相同 bag、播放速率和补点步长，按日志中的输入时间戳对齐。使用 `frnet.onnx` 可运行完整补点基线，样例精度以 41 帧标注上按时间戳和 XYZ 匹配的原始点为准。[对照方法和逐步优化记录](docs/frnet_model_optimization.md)

```bash
catkin_make run_tests_lidar_frnet_ros1
catkin_test_results
```

## 许可与数据来源

ROS1 包源码使用 Apache-2.0，并保留 Autoware 及 TIER IV 的版权声明。模型和样例遵循对应的上游许可。Autoware ROS2 样例版权归 TIER IV, Inc.（2020），仓库中的 ROS1 样例由其左侧雷达数据转换而来。[源码许可证](src/lidar_frnet_ros1/LICENSE) · [Autoware 样例来源](https://github.com/autowarefoundation/autoware/blob/main/ansible/roles/demo_artifacts/tasks/main.yaml)

nuScenes 样例来自 Motional 的 v1.0-mini 和 nuScenes-lidarseg mini，修改包括选取 `scene-0916`、转换为 ROS1 bag 和重映射语义类别。该样例按 CC BY-NC-SA 4.0 和 nuScenes 数据条款提供。[nuScenes 数据条款](https://www.nuscenes.org/terms-of-use) · [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
