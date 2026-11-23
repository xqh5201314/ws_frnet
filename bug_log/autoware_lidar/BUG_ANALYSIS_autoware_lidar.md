# Autoware 样例 FRNet 输出核对

## 结论

推理与 RViz 配色链路通过。启动日志确认使用 `frnet_17_sample.yaml`、样例 TensorRT 引擎和左侧雷达话题。告警时 FRNet 已收到点云，三路输出订阅数均为零，因此相应回调在推理前返回。截图中 `PointCloud2` 显示 `Status: Ok`，并启用了 `RGB8`。近似单色与样例的预测分布一致。当前结果的置信范围为场景级。

## 证据

| 时刻或范围（北京时间） | 检查项 | 证据 |
|---|---|---|
| 2026-11-18 15:10:35.060 至 15:10:45.277 | 输出订阅 | 三次告警命中三路输出订阅数均为零的条件。[回调](/home/xqh/perception/ws_frnet/src/lidar_frnet_ros1/src/lidar_frnet_node.cpp:145) |
| bag 起始后的 0 至 5.90 秒，共 60 帧 | 预测与颜色 | 766,309 个输出点，`class_14` 占 81.02%，`class_15` 占 18.59%。对应 RGB 为 `#E83DF2` 和 `#F23DBD`。逐点配色不一致数及无效概率数均为零。[配色逻辑](/home/xqh/perception/ws_frnet/src/lidar_frnet_ros1/lib/postprocess/postprocess_kernel.cu:267) |
| 同一段 bag，离线诊断 | 强度输入敏感性 | 原始强度范围为 1 至 157。将强度除以 255 后，主类别变为 `class_09`，占 87.87%。该试验仅用于诊断输入尺度敏感性。 |

样例只有 `velodyne_left` 的点云话题。RViz 固定坐标系使用 `velodyne_left` 时，可直接显示同坐标系的输出。全局 `No TF data` 对应样例缺少 TF 话题。[样例 bag](/home/xqh/perception/ws_frnet/samples/autoware_lidar/autoware_velodyne_left.bag)

## 后续验证

样例配置中的 17 个类别用数字编号，过滤列表为空。语义精度验证需要核实该 ONNX 的训练标签顺序与强度预处理，并用配套的标注点云评估。[样例配置](/home/xqh/perception/ws_frnet/samples/autoware_lidar/frnet_17_sample.yaml)

截图没有可对齐的帧时间，颜色结论来自同一样例的 60 帧离线复核。
