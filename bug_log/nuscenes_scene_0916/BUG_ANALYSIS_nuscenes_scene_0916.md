# nuScenes scene-0916 的 FRNet 耗时

## 结论

这次 0.5 倍速回放有 3,290 条成功处理记录，覆盖 8 轮完整的 399 帧和第 9 轮的前 98 帧。按输入时间戳对齐 bag 后，完整轮次均无缺帧。节点使用已有的 TensorRT FP16 引擎，RViz 同时运行。[启动记录](/home/xqh/.ros/log/7e1d254e-e3c9-11f1-a2b7-391b1cc295c4/roslaunch-xqh-Predator-PHN14-51-144912.log:37) · [引擎记录](/home/xqh/.ros/log/7e1d254e-e3c9-11f1-a2b7-391b1cc295c4/rosout.log:77372)

首帧耗时 70.305 ms，其中推理占 68.031 ms。同一个输入帧在随后 8 轮的总耗时为 22.283–27.628 ms，额外开销集中在首次推理。去掉首帧后，总耗时的中位数为 23.737 ms，P95 为 25.731 ms，最慢一帧为 29.340 ms。推理平均占总耗时的 94.4%。[首帧日志](/home/xqh/.ros/log/7e1d254e-e3c9-11f1-a2b7-391b1cc295c4/rosout.log:77377) · [最慢的后续帧](/home/xqh/.ros/log/7e1d254e-e3c9-11f1-a2b7-391b1cc295c4/rosout.log:79752)

## 证据

日志窗口为 2026-11-20 16:55:31.707–17:00:59.776（北京时间）。输入点云每帧 34,560–34,784 点，源帧间隔平均 50.001 ms。表中各项单位为毫秒。[回放数据](/home/xqh/perception/ws_frnet/samples/nuscenes_scene_0916/nuscenes_scene_0916.bag) · [统计窗口末帧](/home/xqh/.ros/log/7e1d254e-e3c9-11f1-a2b7-391b1cc295c4/rosout.log:80666)

| 阶段 | 首帧 | 后续平均 | 后续 P50 | 后续 P95 | 后续 P99 | 后续最大 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 预处理 | 0.533 | 0.495 | 0.427 | 0.946 | 1.347 | 2.028 |
| TensorRT 推理 | 68.031 | 22.456 | 22.447 | 24.259 | 25.290 | 27.634 |
| 后处理 | 1.737 | 0.829 | 0.794 | 1.424 | 1.766 | 2.804 |
| 总耗时 | 70.305 | 23.782 | 23.737 | 25.731 | 27.004 | 29.340 |

0.5 倍速下，约 100 ms 的帧间隔覆盖了首帧和后续帧的处理时间。按原速约 50 ms 的帧间隔估算，后续帧 P95 留有约 24 ms 的处理余量，首帧则超过一个帧间隔。预处理和后处理合计平均 1.32 ms，当前耗时主要由推理阶段决定。

预处理计入输入校验、CPU 到 GPU 拷贝、投影和输入张量尺寸设置。推理计入 `enqueueV3` 及等待 CUDA stream 完成。后处理计入生成输出点云及 GPU 到 CPU 拷贝。总耗时覆盖 `LidarFRNet::process()`，计时到 ROS 发布之前结束。[分段计时](/home/xqh/perception/ws_frnet/src/lidar_frnet_ros1/lib/lidar_frnet.cpp:94) · [总耗时边界](/home/xqh/perception/ws_frnet/src/lidar_frnet_ros1/src/lidar_frnet_node.cpp:159)

原速余量来自帧间隔和处理时间的比较。确认 1 倍速下的实际到帧率需要原速回放记录，细分首次推理内部开销需要 TensorRT 或 CUDA 的分项测量。

## 论文与本地模型对照

FRNet 先逐点提取特征，再用二维卷积处理角度网格。各层通过按索引取回网格特征、逐点更新、池化回网格来反复融合，最后按点输出类别。当前 ONNX 图有 53 个 `Conv` 节点和点级 `MatMul`、`Gather`、`Scatter`，与论文的两路计算对应。[FRNet 论文 §III-B](https://arxiv.org/html/2312.04484#S3.SS2) · [本地 ONNX](/home/xqh/perception/ws_frnet/src/lidar_frnet_ros1/cpkt/frnet.onnx)

按 399 帧 bag 和 CUDA 插值占用规则在 CPU 上复算，原始点数平均约 34,723，插值补点约 19,133，输入网络约 53,856 点，增加约 55%。增加的是逐点处理规模。主干的视锥网格为 `480 × 32`。这份计算量证据可以解释推理包含多路工作，现有计时粒度止于整个 TensorRT 执行阶段。[插值实现](/home/xqh/perception/ws_frnet/src/lidar_frnet_ros1/lib/preprocess/preprocess_kernel.cu:327) · [样例参数](/home/xqh/perception/ws_frnet/samples/nuscenes_scene_0916/frnet_nuscenes.yaml:73)

论文在 RTX 2080 Ti 的 SemanticKITTI 验证集上报告 FRNet 为 29.1 FPS，换算约 34.4 ms/帧。当前 22.45 ms 是 nuScenes 回放里 `enqueueV3` 到 CUDA stream 同步完成的中位耗时，输入、设备和统计范围各自不同。[FRNet 论文 §IV 与表 VII](https://arxiv.org/html/2312.04484#S4)

## TensorRT 算子实测与优化

从第 0、200、380 帧用项目的 CUDA 预处理提取四路真实输入，使用现有 TensorRT 10.10 FP16 引擎做单流逐层计时。RTX 4060 Laptop GPU 同时运行着 ROS 节点。引擎的 inspector 只提供层名，下表是按层名归组后占各次逐层计时总和的比例。

| 层类型 | 三帧占比 |
| --- | ---: |
| 索引、重排、拼接及逐元素融合层 | 37–42% |
| 显式 `ScatterElements` | 14–17% |
| 卷积 | 16–20% |
| 显式 `MatMul` / `Gemm` | 12–14% |
| 形状拷贝与 `Reformatting CopyNode` | 8–10% |

`/voxel_encoder/ScatterElements_2` 是按格取最大值，中位耗时 1.44–1.87 ms。`/backbone/fuse_layer1/conv` 与激活的融合层中位耗时 1.39–2.18 ms。如果要保留逐点计算规模，优先尝试复用 CUDA 预处理已生成的排序索引和分段边界，做按格归约，并减少点与网格之间反复的展开、重排和中间张量写入。[坐标排序与分组](/home/xqh/perception/ws_frnet/src/lidar_frnet_ros1/lib/preprocess/preprocess_kernel.cu:425)

原图用 `ReduceMax(inverse_map) + 1` 求网格数，引擎在此插入了 `[DevicetoShapeHostCopy]`。临时图改为取 `voxel_coors.shape[0]` 后，该拷贝消失。第 200 帧两次对照的 CPU 提交中位耗时从 25.55–29.75 ms 降到 1.26–1.43 ms，CUDA Graph 也能成功捕获。399 帧共 399 种 `(N, U)` 输入形状，固定形状的 Graph 收益适用于可重复使用同一图的场景。[TensorRT 形状同步](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/dynamic-shapes-advanced.html#execution-tensors-vs-shape-tensors) · [CUDA Graph 约束](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/optimization.html)

用相同预处理张量交替执行旧引擎和临时引擎，399 帧推理完成 P50 为 27.47 → 27.54 ms，P95 为 45.43 → 44.84 ms。CPU 提交改善明确，单帧完成时间接近。FP16 重建后的逐点类别一致率为 99.965%，21,488,466 个预测点中有 7,480 个类别变化。

减小逐点输入规模的速度收益更直接。跳过插值的三帧诊断中，第 200 帧由 52,560 点减到 34,688 点，GPU 推理中位耗时从 24.90 ms 降到 17.40 ms。该实验量化的是速度，插值调整的精度影响需要用标注帧衡量。[补点实现](/home/xqh/perception/ws_frnet/src/lidar_frnet_ros1/lib/preprocess/preprocess_kernel.cu:327)

## 补点步长实测

2026-11-23 使用 CUDA 12.8、TensorRT 10.10 和 RTX 4060 Laptop GPU，在 `LidarFRNet::process()` 内交替执行四个步长。所有档位加载同一 FP16 引擎，每档预热 6 次后处理全部 399 帧。校验了分割与 RGB 点云的帧标识、时间戳和点数，以及类别和置信度的有效性。5 项回归测试通过。[CUDA 补点代码](/home/xqh/perception/ws_frnet/src/lidar_frnet_ros1/lib/preprocess/preprocess_kernel.cu:330) · [回归测试](/home/xqh/perception/ws_frnet/src/lidar_frnet_ros1/test/frnet_regression_test.cpp)

| 补点步长 | 平均推理点数 | 推理 P50 | 推理 P95 | 总耗时 P50 | 样例 mIoU |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1，完整补点 | 53,856 | 26.29 ms | 29.25 ms | 28.67 ms | 66.96% |
| 2 | 44,289 | 22.03 ms | 24.73 ms | 24.17 ms | 66.80% |
| 3 | 41,099 | 20.49 ms | 22.77 ms | 22.65 ms | 66.83% |
| 0，关闭补点 | 34,723 | 18.06 ms | 19.94 ms | 20.15 ms | 66.71% |

IoU 计算使用 41 帧中按时间戳与 XYZ 对齐的 1,074,426 个已知类原始点。mIoU 对场景中有标注的 12 类等权平均，样例结果反映当前场景的取舍。[样例参数](/home/xqh/perception/ws_frnet/samples/nuscenes_scene_0916/frnet_nuscenes.yaml:73)

| 类别 | 标注点 | 步长 1 | 步长 2 | 步长 3 | 步长 0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 自行车 | 45 | 0.00% | 0.00% | 0.00% | 0.00% |
| 公交车 | 8,815 | 95.83% | 95.18% | 95.28% | 95.22% |
| 汽车 | 77,538 | 93.94% | 93.79% | 93.84% | 93.68% |
| 摩托车 | 3,640 | 65.47% | 65.37% | 64.97% | 64.10% |
| 行人 | 6,636 | 75.32% | 75.88% | 76.37% | 76.40% |
| 卡车 | 17,166 | 93.28% | 92.95% | 93.06% | 92.32% |
| 可行驶路面 | 415,468 | 89.44% | 89.40% | 89.39% | 89.42% |
| 其他平面 | 4 | 0.00% | 0.00% | 0.00% | 0.00% |
| 人行道 | 61,120 | 35.10% | 34.87% | 34.95% | 35.02% |
| 地形 | 123,217 | 82.30% | 82.03% | 82.04% | 82.30% |
| 人造物 | 143,661 | 82.34% | 82.01% | 81.91% | 81.97% |
| 植被 | 217,116 | 90.49% | 90.16% | 90.11% | 90.05% |

当时的样例使用步长 3，推理 P50 下降 22.1%，mIoU 下降 0.13 个百分点。步长 0 的摩托车 IoU 下降 1.37 个百分点，卡车下降 0.97 个百分点。通用节点的默认步长为 1。[启动参数](/home/xqh/perception/ws_frnet/src/lidar_frnet_ros1/launch/lidar_frnet.launch:7)

## 重启后的 ROS 实跑对照

11 月 23 日 10:31:46 重启的节点加载了 nuScenes YAML，补点步长为 3，使用与完整补点基线相同的 FP16 引擎。两次实跑均开启 RViz，原始点云以约 0.5 倍速回放。按输入时间戳配对，每次取首轮之后的 5 个完整循环，共 1,995 帧。[本次启动记录](/home/xqh/.ros/log/7e1d254e-e3c9-11f1-a2b7-391b1cc295c4/roslaunch-xqh-Predator-PHN14-51-176129.log:5)

| 阶段 | 完整补点 P50 | 步长 3 P50 | 完整补点 P95 | 步长 3 P95 |
| --- | ---: | ---: | ---: | ---: |
| 预处理 | 0.43 ms | 0.42 ms | 0.94 ms | 2.00 ms |
| 推理 | 22.48 ms | 19.61 ms | 24.17 ms | 22.05 ms |
| 后处理 | 0.82 ms | 0.77 ms | 1.45 ms | 2.16 ms |
| 总耗时 | 23.79 ms | 21.28 ms | 25.69 ms | 24.07 ms |

推理 P50 降低 12.8%，总耗时 P50 降低 10.6%。推理在 95.3% 的配对帧上更快。预处理和后处理的 P95 分别增加 1.07 ms 和 0.71 ms，总耗时 P95 下降 1.62 ms。两次实跑分属不同日期，现场负载会影响单帧对照结果。[基线配对起点](/home/xqh/.ros/log/7e1d254e-e3c9-11f1-a2b7-391b1cc295c4/rosout.log:77776) · [重启后配对起点](/home/xqh/.ros/log/7e1d254e-e3c9-11f1-a2b7-391b1cc295c4/rosout.log:114986)

截至 10:35:58.844，共记录 2,500 个成功处理帧。首轮从 bag 的第 2 帧开始记录，之后 5 个完整循环各有 399 条处理记录。首个处理帧总计 70.73 ms，推理占 68.26 ms。参与对照的后续帧总耗时最大为 27.82 ms。[首次处理](/home/xqh/.ros/log/7e1d254e-e3c9-11f1-a2b7-391b1cc295c4/rosout.log:114588) · [对照最慢帧](/home/xqh/.ros/log/7e1d254e-e3c9-11f1-a2b7-391b1cc295c4/rosout.log:116217)

分割精度基于同一引擎、同一场景的 41 帧标注离线对照。步长 3 的 mIoU 从 66.96% 降至 66.83%。公交车、摩托车分别下降 0.55 和 0.50 个百分点，行人增加 1.05 个百分点。实时耗时日志用于验证处理速度，精度结论适用于这份标注样例。

### 预处理和后处理的 P95

在每组 1,995 帧中，预处理超过 2 ms 的帧从 0 帧增至 101 帧，后处理从 6 帧增至 133 帧。步长 3 的同一输入时间戳在不同轮次中，预处理出现过 0.323–3.604 ms，另一输入时间戳的后处理为 0.401–3.111 ms。两段同时超过 2 ms 的帧有 2 个。[预处理最慢帧](/home/xqh/.ros/log/7e1d254e-e3c9-11f1-a2b7-391b1cc295c4/rosout.log:115743) · [后处理最慢帧](/home/xqh/.ros/log/7e1d254e-e3c9-11f1-a2b7-391b1cc295c4/rosout.log:115098)

同进程交替执行的 399 帧 A/B 中，完整补点到步长 3 的预处理 P95 为 2.63 → 2.33 ms，后处理 P95 为 2.78 → 2.60 ms。跨日期 ROS 实跑的上涨集中在阶段耗时尾部，同进程对照中两段 P95 都下降。

预处理计入主机到 GPU 的输入拷贝、投影和排序，以及回读点数和格子数的同步。后处理计入输出点云缓冲区的 `resize`、GPU 到主机拷贝和 stream 同步。普通主机内存的异步拷贝也可能在调用期间等待。当前墙钟日志记录整个阶段，尚缺阶段内计时来定位是哪次等待拉长了长尾。[预处理同步](/home/xqh/perception/ws_frnet/src/lidar_frnet_ros1/lib/lidar_frnet.cpp:171) · [后处理](/home/xqh/perception/ws_frnet/src/lidar_frnet_ros1/lib/lidar_frnet.cpp:254) · [CUDA 12.8 拷贝行为](/usr/local/cuda-12.8/include/cuda_runtime_api.h:93)

## 完整补点下的算子与 CUDA Graph 实测

2026-11-23 使用 CUDA 12.8、TensorRT 10.10 和 RTX 4060 Laptop GPU。整包 399 帧设置 `interpolation_stride: 1`，两个 FP16 引擎在每帧共用同一组输入显存，执行顺序逐帧交替。共处理 21,488,466 个输入网络的点。时间覆盖 `enqueueV3` 调用至 CUDA stream 同步完成，同机 ROS 回放继续运行。[样例配置](/home/xqh/perception/ws_frnet/samples/nuscenes_scene_0916/frnet_nuscenes.yaml:79)

| 指标 | 原始 FP16 | 直接 CUDA 归约 FP16 |
| --- | ---: | ---: |
| 推理 P50 | 24.92 ms | 18.89 ms |
| 推理 P95 | 42.02 ms | 34.82 ms |
| 推理最大 | 47.00 ms | 38.38 ms |
| 41 帧标注 mIoU | 66.9621% | 66.9489% |
| 标注点准确率 | 91.8647% | 91.8634% |

306 帧推理变快，P50 和 P95 分别下降 24.2% 和 17.1%。两引擎有 12,578 个逐点类别差异，一致率为 99.9415%。mIoU 对场景中有标注的 12 类等权平均，使用按时间戳和 XYZ 匹配的 1,074,426 个已知类原始点。各类 IoU 最大下降为摩托车的 0.079 个百分点。[逐帧耗时](/tmp/frnet_segmented.iouAou/atomic/ab/ab_frames.tsv) · [混淆矩阵](/tmp/frnet_segmented.iouAou/atomic/ab/confusion.tsv)

六处 `ScatterElements(max)` 由 CUDA 插件直接用 `inverse_map` 定位网格并做原子最大值归约，省掉索引向通道维度的展开。网格数来自 `voxel_coors.shape[0]`，引擎消除了对应的 `[DevicetoShapeHostCopy]`。ONNX 检查了 186 个已有初始化张量和输入输出签名，均与原图一致。第 200 帧的六层逐层中位耗时之和由 5.15 ms 降至 1.76 ms。[CUDA 插件](/home/xqh/perception/ws_frnet/src/lidar_frnet_ros1/lib/frustum_plugins.cu:18) · [ONNX 转换脚本](/home/xqh/perception/ws_frnet/src/lidar_frnet_ros1/scripts/optimize_frnet_onnx.py) · [原逐层计时](/tmp/frnet_segmented.iouAou/atomic/same_frame_A1_layers.json) · [新逐层计时](/tmp/frnet_segmented.iouAou/atomic/same_frame_B1_layers.json)

第 200 帧的固定形状为 `N=52560`、`U=12374`。使用 `trtexec` 单 stream 和同一份四路张量，原始引擎的 CPU 提交 P50 为 23.33–25.83 ms，优化引擎普通提交为 1.31–1.36 ms，CUDA Graph 为 0.019 ms。优化引擎的 GPU 完成 P50 在普通提交为 20.07–20.12 ms，开启 Graph 为 20.34 ms。样例的 399 帧有 399 种形状组合，实际回放采用默认的普通提交。[Graph 测试](/tmp/frnet_segmented.iouAou/atomic/same_frame_B_graph.log:99) · [TensorRT CUDA Graph 约束](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/optimization.html)

用工作区编译的库验证了重复形状捕获、形状切换、输出地址重绑和原 FP16 引擎捕获失败后的自动回退。第 200 帧固定输入重跑时，普通提交最多有 8 个点的类别变化，Graph 最多有 11 个，两者的最大 logit 差都为 0.125。第 0 帧两种提交方式的 logits 逐位一致。优化图保留两处 `ScatterElements(add)`，浮点累加顺序可能造成重跑时的数值波动。5 项回归测试通过。[Graph 实现](/home/xqh/perception/ws_frnet/src/lidar_frnet_ros1/src/tensorrt_wrapper.cpp:336) · [启动说明](/home/xqh/perception/ws_frnet/samples/nuscenes_scene_0916/README.md)

## 算子版重启后的实跑对照

11 月 23 日 11:45:55 重启后加载 `frnet_frustum_max.onnx.fp16.engine`，使用完整补点、关闭体素滤波，并按默认路径提交 TensorRT。RViz 同时运行，相邻源帧的日志间隔中位数约 100 ms，对应 0.5 倍速。首轮从 bag 第 2 帧开始记录。取其后三轮完整回放，每轮 399 帧，时间戳与 bag 逐帧匹配。对照为旧版完整补点的首轮之后三圈，以及重启前步长 3 的最后三圈。各组 1,197 帧，对照区间内无缺帧。[本次启动](/home/xqh/.ros/log/7e1d254e-e3c9-11f1-a2b7-391b1cc295c4/roslaunch-xqh-Predator-PHN14-51-185963.log:5) · [加载引擎](/home/xqh/.ros/log/7e1d254e-e3c9-11f1-a2b7-391b1cc295c4/rosout.log:159073) · [默认 Graph 参数](/home/xqh/perception/ws_frnet/src/lidar_frnet_ros1/src/lidar_frnet_node.cpp:38)

| 阶段 | 旧版完整补点 P50 / P95 | 重启前步长 3 P50 / P95 | 优化版完整补点 P50 / P95 |
| --- | ---: | ---: | ---: |
| 预处理 | 0.430 / 0.971 ms | 0.447 / 1.016 ms | 0.433 / 1.195 ms |
| 推理 | 22.439 / 24.224 ms | 19.216 / 20.725 ms | 18.151 / 19.884 ms |
| 后处理 | 0.812 / 1.465 ms | 0.591 / 0.997 ms | 0.791 / 1.429 ms |
| 总处理 | 23.782 / 25.724 ms | 20.304 / 21.982 ms | 19.517 / 21.364 ms |

和旧版完整补点相比，推理 P50 下降 19.1%，总处理 P50 下降 17.9%，预处理 P95 增加 0.22 ms。和重启前步长 3 相比，平均推理和分割输出点数从 41,099 恢复到 53,856，增加约 31%。后处理的逐点计算和拷贝随点数增长，后处理 P50 增加 0.20 ms，总处理 P50 下降 3.9%。[后处理拷贝](/home/xqh/perception/ws_frnet/src/lidar_frnet_ros1/lib/lidar_frnet.cpp:254) · [旧版完整补点配对起点](/home/xqh/.ros/log/7e1d254e-e3c9-11f1-a2b7-391b1cc295c4/rosout.log:77776) · [步长 3 配对起点](/home/xqh/.ros/log/7e1d254e-e3c9-11f1-a2b7-391b1cc295c4/rosout.log:157679) · [优化版配对起点](/home/xqh/.ros/log/7e1d254e-e3c9-11f1-a2b7-391b1cc295c4/rosout.log:159476) · [优化版配对末帧](/home/xqh/.ros/log/7e1d254e-e3c9-11f1-a2b7-391b1cc295c4/rosout.log:160672)

首帧总处理为 70.928 ms，其中推理 68.760 ms。同一输入帧在三个配对轮次的总处理耗时为 19.979–20.676 ms。三个配对轮次中最慢一帧为 24.018 ms。总处理计时覆盖节点内处理至发布前，实跑结果对应该回放速率和现场负载。[首帧](/home/xqh/.ros/log/7e1d254e-e3c9-11f1-a2b7-391b1cc295c4/rosout.log:159078) · [同帧第二次处理](/home/xqh/.ros/log/7e1d254e-e3c9-11f1-a2b7-391b1cc295c4/rosout.log:159477) · [计时边界](/home/xqh/perception/ws_frnet/src/lidar_frnet_ros1/src/lidar_frnet_node.cpp:161)

分割精度采用同一引擎文件的 41 帧标注离线对照，当前引擎与该离线测试文件的 SHA-256 均以 `c5fd6a5ca157` 开头。完整补点下场景 mIoU 从 66.9621% 降至 66.9489%，下降 0.0132 个百分点。全包 21,488,466 个逐点预测的类别一致率为 99.9415%，有标注的类别中摩托车 IoU 降幅最大，下降 0.079 个百分点。该精度结论适用于当前标注场景。[标注混淆矩阵](/tmp/frnet_segmented.iouAou/atomic/ab/confusion.tsv)
