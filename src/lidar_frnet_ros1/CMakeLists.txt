cmake_minimum_required(VERSION 3.10)
project(lidar_frnet_ros1)

add_compile_options(-std=c++17 -Wall -Wextra -Wpedantic -Wno-deprecated-declarations)

set(TENSORRT_ROOT "$ENV{TENSORRT_ROOT}" CACHE PATH "TensorRT installation root")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  sensor_msgs
  tf2
  tf2_geometry_msgs
  tf2_ros
  visualization_msgs
)
if(POLICY CMP0146)
  cmake_policy(SET CMP0146 OLD)
endif()
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()
find_package(CUDA REQUIRED)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS "${TENSORRT_ROOT}"
  PATH_SUFFIXES include targets/${CMAKE_LIBRARY_ARCHITECTURE}/include
)
find_library(NVINFER nvinfer
  HINTS "${TENSORRT_ROOT}"
  PATH_SUFFIXES lib targets/${CMAKE_LIBRARY_ARCHITECTURE}/lib
)
find_library(NVONNXPARSER nvonnxparser
  HINTS "${TENSORRT_ROOT}"
  PATH_SUFFIXES lib targets/${CMAKE_LIBRARY_ARCHITECTURE}/lib
)
find_library(NVINFER_PLUGIN nvinfer_plugin
  HINTS "${TENSORRT_ROOT}"
  PATH_SUFFIXES lib targets/${CMAKE_LIBRARY_ARCHITECTURE}/lib
)

if(NOT TENSORRT_INCLUDE_DIR OR NOT NVINFER OR NOT NVONNXPARSER OR NOT NVINFER_PLUGIN)
  message(FATAL_ERROR "TensorRT was not found. Set -DTENSORRT_ROOT=/path/to/TensorRT or provide TENSORRT_INCLUDE_DIR, NVINFER, NVONNXPARSER, and NVINFER_PLUGIN.")
endif()
set(TensorRT_INCLUDE_DIRS ${TENSORRT_INCLUDE_DIR})
set(TensorRT_LIBRARIES ${NVINFER} ${NVONNXPARSER} ${NVINFER_PLUGIN})

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}_lib
  CATKIN_DEPENDS geometry_msgs roscpp sensor_msgs tf2 tf2_geometry_msgs tf2_ros visualization_msgs
  DEPENDS CUDA TensorRT
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)
include_directories(SYSTEM
  ${CUDA_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
)

list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr -diag-suppress 1675 --extended-lambda")
list(APPEND CUDA_NVCC_FLAGS "-std=c++17")
list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_75,code=sm_75")
list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_86,code=sm_86")
list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_87,code=sm_87")
if(CUDA_VERSION VERSION_GREATER_EQUAL "11.8")
  list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_89,code=sm_89")
endif()
if(CUDA_VERSION VERSION_GREATER_EQUAL "12.8")
  if(CUDA_VERSION VERSION_LESS "13.0")
    list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_101,code=sm_101")
  else()
    list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_110,code=sm_110")
  endif()
  list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_120,code=sm_120")
  list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_120,code=compute_120")
elseif(CUDA_VERSION VERSION_GREATER_EQUAL "11.8")
  list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_89,code=compute_89")
else()
  list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_87,code=compute_87")
endif()

cuda_add_library(${PROJECT_NAME}_cuda_lib
  lib/frustum_plugins.cu
  lib/preprocess/preprocess_kernel.cu
  lib/postprocess/postprocess_kernel.cu
)
target_link_libraries(${PROJECT_NAME}_cuda_lib ${NVINFER})

add_library(${PROJECT_NAME}_lib
  lib/lidar_frnet.cpp
  src/tensorrt_wrapper.cpp
)
target_link_libraries(${PROJECT_NAME}_lib
  ${PROJECT_NAME}_cuda_lib
  ${catkin_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${TensorRT_LIBRARIES}
)

add_executable(${PROJECT_NAME}_node
  src/lidar_frnet_node.cpp
)
target_link_libraries(${PROJECT_NAME}_node
  ${PROJECT_NAME}_lib
  ${catkin_LIBRARIES}
)

if(CATKIN_ENABLE_TESTING)
  catkin_add_gtest(frnet_regression_test test/frnet_regression_test.cpp)
  if(TARGET frnet_regression_test)
    target_link_libraries(frnet_regression_test ${PROJECT_NAME}_cuda_lib ${catkin_LIBRARIES})
  endif()
endif()

install(TARGETS ${PROJECT_NAME}_cuda_lib ${PROJECT_NAME}_lib ${PROJECT_NAME}_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)
install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
