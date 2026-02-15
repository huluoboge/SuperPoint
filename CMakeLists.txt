cmake_minimum_required(VERSION 3.18)
project(SuperPointONNX LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# CUDA 11.8路径
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.8")
set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")

# 查找CUDA
find_package(CUDA 11.8 REQUIRED)
message(STATUS "CUDA Version: ${CUDA_VERSION}")
message(STATUS "CUDA Toolkit: ${CUDA_TOOLKIT_ROOT_DIR}")

# 查找OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV Version: ${OpenCV_VERSION}")

# ONNX Runtime路径
# 选项1: 使用预编译库
set(ONNXRUNTIME_ROOT "/opt/onnxruntime-gpu" CACHE PATH "ONNX Runtime安装路径")

# 选项2: 使用系统安装的包
if(NOT EXISTS ${ONNXRUNTIME_ROOT})
    # 尝试查找Python pip安装的版本
    execute_process(
        COMMAND python -c "import onnxruntime; print(onnxruntime.__path__[0])"
        OUTPUT_VARIABLE PYTHON_ONNXRUNTIME_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(PYTHON_ONNXRUNTIME_PATH)
        message(STATUS "找到Python ONNX Runtime: ${PYTHON_ONNXRUNTIME_PATH}")
        set(ONNXRUNTIME_INCLUDE_DIR "${PYTHON_ONNXRUNTIME_PATH}/capi/include")
        set(ONNXRUNTIME_LIB_DIR "${PYTHON_ONNXRUNTIME_PATH}/capi")
    endif()
endif()

# 如果找到预编译包
if(EXISTS ${ONNXRUNTIME_ROOT})
    set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
    set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/lib")
endif()

message(STATUS "ONNX Runtime Include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime Lib: ${ONNXRUNTIME_LIB_DIR}")

# 包含目录
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
)

# 库目录
link_directories(
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    ${ONNXRUNTIME_LIB_DIR}
)

# 源文件
set(SOURCES
    superpoint_cpp_api.cpp
)

# 可执行文件
add_executable(superpoint_inference ${SOURCES})

# 链接库
target_link_libraries(superpoint_inference
    ${OpenCV_LIBS}
    onnxruntime
    cudart
    cudnn
)

# 设置RPATH
set_target_properties(superpoint_inference PROPERTIES
    INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}:${CUDA_TOOLKIT_ROOT_DIR}/lib64"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# 编译选项
target_compile_options(superpoint_inference PRIVATE
    -O3
    -march=native
)

# 安装
install(TARGETS superpoint_inference DESTINATION bin)

# 打印配置摘要
message(STATUS "==========================================")
message(STATUS "SuperPoint ONNX C++ 配置摘要")
message(STATUS "==========================================")
message(STATUS "CUDA: ${CUDA_VERSION} (${CUDA_TOOLKIT_ROOT_DIR})")
message(STATUS "OpenCV: ${OpenCV_VERSION}")
message(STATUS "ONNX Runtime: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "==========================================")
