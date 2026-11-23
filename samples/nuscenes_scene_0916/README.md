# nuScenes scene-0916 回放

这个 ROS1 bag 包含 nuScenes `scene-0916` 的 399 帧车顶雷达点云和 41 帧同时间戳的逐点标注。场景时长 19.9 秒，文件大小约 194 MiB。

| 话题 | 帧数 | 内容 |
| --- | ---: | --- |
| `/nuscenes/lidar_top` | 399 | `LIDAR_TOP` 坐标系下的 XYZI 点云，使用原始强度数值 |
| `/nuscenes/lidarseg/ground_truth` | 41 | 同坐标系下的 `class_id` 和 `rgb` |

`frnet.onnx` 指向工作区已有的 17 类 ONNX。`frnet_frustum_max.onnx` 使用相同权重和四路输入，在 TensorRT 中执行 CUDA 按格最大值归约。`frnet_nuscenes.yaml` 使用旧版 Autoware FRNet 的类别顺序和投影参数。逐点标注映射到同一类别顺序，预测和标注使用相同配色。[nuScenes 类别映射](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/lidarseg/README.md)

## 启动节点

```bash
cd /home/xqh/perception/ws_frnet
source /home/xqh/env/cuda12_trt10.sh
source devel/setup.bash

sample="$PWD/samples/nuscenes_scene_0916"
roslaunch lidar_frnet_ros1 lidar_frnet.launch \
  onnx_path:="$sample/frnet_frustum_max.onnx" \
  config_file:="$sample/frnet_nuscenes.yaml" \
  input_pointcloud:=/nuscenes/lidar_top \
  use_voxel_grid:=false
```

另开终端循环播放。

```bash
cd /home/xqh/perception/ws_frnet
source /home/xqh/env/cuda12_trt10.sh
source devel/setup.bash
rosbag play --loop --rate 0.5 samples/nuscenes_scene_0916/nuscenes_scene_0916.bag
```

桌面终端运行 launch 时会自动打开 RViz，固定坐标系为 `LIDAR_TOP`，两层点云均已配置 `RGB8`。启动时显示 FRNet 预测，在标注关键帧暂停回放并切换 `FRNet Segmentation` 和 `nuScenes Ground Truth` 即可对照。终端会按输入点云时间戳打印每帧的预处理、推理、后处理和总耗时。

| 内容 | 话题 |
| --- | --- |
| FRNet 预测 | `/lidar_frnet/output/pointcloud/visualization` |
| 数据集标注 | `/nuscenes/lidarseg/ground_truth` |

## 完整补点和 CUDA Graph

样例使用 `interpolation_stride: 1` 完整补点。把 `onnx_path` 指向 `frnet.onnx` 即可用原始引擎做同点数对照。

`use_cuda_graph:=true` 可在连续帧的点数、网格数和显存地址一致时复用推理图。本场景 399 帧的点数和网格数组合各不相同，样例启动采用默认的普通提交路径。[同输入实测](../../bug_log/nuscenes_scene_0916/BUG_ANALYSIS_nuscenes_scene_0916.md#完整补点下的算子与-cuda-graph-实测)

更新原始 ONNX 后，可用安装了 `onnx` 的 Python 重新生成优化版。节点会按新模型构建 FP16 缓存。

```bash
python3 src/lidar_frnet_ros1/scripts/optimize_frnet_onnx.py \
  "$sample/frnet.onnx" "$sample/frnet_frustum_max.onnx"
```

## 逐点验证

完整补点基线使用工作区的 ONNX 和 TensorRT FP16。在 0.5 倍速回放中收到全部 399 帧分割输出，时间戳与输入一致。在 41 帧标注中，按时间戳和 XYZ 坐标匹配原始点，统计下表。`unknown` 标注和模型插值生成的点未计入。工作区权重的 SHA-256 与 Autoware v1 发布记录不同，结果对应当前文件。[Autoware v1 模型记录](https://github.com/autowarefoundation/autoware/blob/1.6.0/ansible/roles/artifacts/tasks/main.yaml)

| 标注类别 | 正确分类点数 / 标注点数 |
| --- | ---: |
| 自行车 | 0 / 45 |
| 公交车 | 8,516 / 8,815 |
| 汽车 | 74,659 / 77,538 |
| 摩托车 | 2,497 / 3,640 |
| 行人 | 5,680 / 6,636 |
| 卡车 | 16,237 / 17,166 |
| 可行驶路面 | 408,047 / 415,468 |
| 其他平面 | 0 / 4 |
| 人行道 | 23,605 / 61,120 |
| 地形 | 111,434 / 123,217 |
| 人造物 | 127,007 / 143,661 |
| 植被 | 209,335 / 217,116 |

点云和标注来自 Motional 的 nuScenes v1.0-mini 和 nuScenes-lidarseg mini。原始数据按 CC BY-NC-SA 4.0 及 nuScenes 数据条款提供。[官方数据教程](https://www.nuscenes.org/tutorials/nuscenes_lidarseg_panoptic_tutorial.html) · [nuScenes 论文](https://openaccess.thecvf.com/content_CVPR_2020/papers/Caesar_nuScenes_A_Multimodal_Dataset_for_Autonomous_Driving_CVPR_2020_paper.pdf)
