#!/bin/bash
# 编译SuperPoint C++ API

set -e

echo "========================================================================"
echo "编译SuperPoint C++ API"
echo "========================================================================"

# 1. 检查CUDA 11.8
echo ""
echo "[1/5] 检查CUDA 11.8..."
if [ ! -f /usr/local/cuda-11.8/bin/nvcc ]; then
    echo "  ✗ CUDA 11.8 未安装"
    echo "  请先运行: ./install_cuda11_cudnn8.sh"
    exit 1
fi

echo "  ✓ CUDA 11.8: $(/usr/local/cuda-11.8/bin/nvcc --version | grep release)"

# 2. 检查cuDNN 8
echo ""
echo "[2/5] 检查cuDNN 8..."
if ! ldconfig -p | grep -q libcudnn.so.8; then
    echo "  ✗ cuDNN 8 未安装"
    echo "  请先运行: ./install_cuda11_cudnn8.sh"
    exit 1
fi
echo "  ✓ cuDNN 8 已安装"
ldconfig -p | grep libcudnn.so.8 | head -1

# 3. 检查ONNX Runtime C++
echo ""
echo "[3/5] 检查ONNX Runtime C++..."
if [ ! -d "/opt/onnxruntime-gpu" ]; then
    echo "  ✗ ONNX Runtime C++ 未安装"
    echo "  正在下载..."
    ./download_onnxruntime_cpp.sh
else
    echo "  ✓ ONNX Runtime: /opt/onnxruntime-gpu"
fi

# 4. 激活CUDA 11.8环境
echo ""
echo "[4/5] 激活CUDA 11.8环境..."
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

echo "  ✓ CUDA_HOME: $CUDA_HOME"
echo "  ✓ nvcc: $(which nvcc)"
nvcc --version | grep release

# 5. 编译
echo ""
echo "[5/5] 编译项目..."

# 创建build目录
mkdir -p build
cd build

# CMake配置
echo ""
echo "CMake配置..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc \
    -DONNXRUNTIME_ROOT=/opt/onnxruntime-gpu

# 编译
echo ""
echo "编译中..."
make -j$(nproc)

cd ..

# 验证可执行文件
echo ""
echo "========================================================================"
echo "编译完成！"
echo "========================================================================"

if [ -f "build/superpoint_inference" ]; then
    echo ""
    echo "✓ 可执行文件: build/superpoint_inference"
    echo ""
    echo "运行示例:"
    echo "  # 使用GPU"
    echo "  ./build/superpoint_inference IMG_0926.JPG gpu"
    echo ""
    echo "  # 使用CPU"
    echo "  ./build/superpoint_inference IMG_0926.JPG cpu"
    echo ""
    echo "注意: 运行前需要激活CUDA 11.8环境:"
    echo "  source ~/.cuda11_env"
    echo "  或"
    echo "  source switch_cuda.sh 11"
    echo ""
else
    echo ""
    echo "✗ 编译失败，未找到可执行文件"
    exit 1
fi

echo "========================================================================"
