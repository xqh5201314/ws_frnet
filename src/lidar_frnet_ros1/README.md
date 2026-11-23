# lidar_frnet_ros1

ROS1 port of `autoware_lidar_frnet` that uses standard ROS messages and does not depend on
Autoware packages.

## Interfaces

- Subscribes: `~input/pointcloud` (`sensor_msgs/PointCloud2`)
- Publishes: `~output/pointcloud/segmentation` (`sensor_msgs/PointCloud2`)
- Publishes: `~output/pointcloud/visualization` (`sensor_msgs/PointCloud2`)
- Publishes: `~output/pointcloud/filtered` (`sensor_msgs/PointCloud2`)
- Publishes: `~debug/ego_crop_box_marker` (`visualization_msgs/Marker`)

The node runs inference and publishes segmentation for each valid input it processes. Visualization
and filtered clouds are generated when their topics have subscribers.

## Build

Source the ROS Noetic and CUDA/TensorRT environment before building. Use TensorRT 10 or later.
For a separate TensorRT installation, set `TENSORRT_ROOT` in the environment or pass it to CMake.

```bash
catkin_make -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3 \
  -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" -DTENSORRT_ROOT="$TENSORRT_ROOT"
source devel/setup.bash
```

## Run

Use the Autoware ONNX and matching ROS1 YAML for the sensor (`ot128` or `qt128`). Load the YAML as
private ROS1 parameters with nested keys such as `filter/output_format`. The model directory needs to
be writable when TensorRT builds or refreshes the engine cache. To prepare the cache ahead of time,
pass `build_only:=true` to the same launch command.

```bash
roslaunch lidar_frnet_ros1 lidar_frnet.launch \
  onnx_path:=/path/to/frnet_qt128.onnx \
  config_file:=$(rospack find lidar_frnet_ros1)/config/ros1_frnet_qt128.yaml \
  input_pointcloud:=/points_raw \
  rviz_fixed_frame:=rslidar
```

On a desktop with `DISPLAY` set, the launch file opens RViz with the FRNet RGB cloud selected.
The default fixed frame is `LIDAR_TOP` for the nuScenes sample. Set `rviz_fixed_frame` to match
another sensor frame or pass `rviz:=false` for command-line runs. The node logs the input timestamp,
preprocessing, inference, postprocessing, and total processing time in milliseconds for every
successfully processed frame.

Set `use_voxel_grid:=false` to feed the cloud directly into FRNet. The default `true` runs the
existing 5 cm voxel filter. Input clouds must be in the sensor frame. Supported little-endian
packed point layouts are XYZI (16 bytes), XYZIRC (16), XYZIRT (26), XYZIRCAEDT (32) and XYZIRADRT
(35). Organized clouds may have padding at the end of each row. An empty `filter/output_format`
preserves the input point format.

Set `interpolation_stride` in the YAML or pass it to the launch file to control how many points
are added before inference. `1` uses every eligible interpolation column. Larger values sample
columns with a different starting offset on each row. `0` bypasses the interpolation projection
and passes the original valid, cropped points to TensorRT. The default is `1`.

The workspace's `cpkt/frnet.onnx` is a separate model with a 480 × 32 frustum and a 17-class
`seg_logit` tensor. It requires its own sensor settings, 17 class names in training order, and matching
palette. The OT128 and QT128 YAML files describe 27-class Autoware models. The node validates the
model's tensor shapes and converts legacy logits to probabilities before threshold filtering.

The TensorRT optimization for the bundled 17-class model is available through
`scripts/optimize_frnet_onnx.py`. The script needs Python `onnx` at conversion time. The resulting
ONNX uses the CUDA max pooling plugin loaded by the node. Launch with `use_cuda_graph:=true` to
enable CUDA Graph for consecutive frames with the same input shapes and buffer addresses.
[nuScenes sample](../../samples/nuscenes_scene_0916/README.md#完整补点和-cuda-graph)
