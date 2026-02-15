#!/bin/bash
# SuperPoint C++ GPU 加速 - 一键安装脚本
# 为GTX 1060配置CUDA 11.8 + cuDNN 8 + ONNX Runtime C++

set -e

echo "========================================================================"
echo "SuperPoint C++ GPU加速 - 一键安装"
echo "========================================================================"
echo ""
echo "这个脚本将安装:"
echo "  1. CUDA 11.8 (与CUDA 12共存)"
echo "  2. cuDNN 8.9.7 for CUDA 11"
echo "  3. ONNX Runtime C++ 1.16.3 GPU"
echo "  4. 编译SuperPoint C++ API"
echo ""
echo "预计时间: 20-30分钟"
echo "所需空间: ~5GB"
echo ""
read -p "继续安装? (Y/n): " confirm
if [[ "$confirm" =~ ^[Nn]$ ]]; then
    echo "取消安装"
    exit 0
fi

# 步骤1: 安装CUDA 11.8 + cuDNN 8
echo ""
echo "========================================================================"
echo "步骤 1/3: 安装CUDA 11.8 + cuDNN 8"
echo "========================================================================"
if [ ! -f ./install_cuda11_cudnn8.sh ]; then
    echo "✗ 找不到 install_cuda11_cudnn8.sh"
    exit 1
fi

./install_cuda11_cudnn8.sh

# 步骤2: 下载ONNX Runtime C++
echo ""
echo "========================================================================"
echo "步骤 2/3: 下载ONNX Runtime C++"
echo "========================================================================"
if [ ! -f ./download_onnxruntime_cpp.sh ]; then
    echo "✗ 找不到 download_onnxruntime_cpp.sh"
    exit 1
fi

./download_onnxruntime_cpp.sh

# 步骤3: 编译C++ API
echo ""
echo "========================================================================"
echo "步骤 3/3: 编译SuperPoint C++ API"
echo "========================================================================"
if [ ! -f ./build_cpp_api.sh ]; then
    echo "✗ 找不到 build_cpp_api.sh"
    exit 1
fi

./build_cpp_api.sh

# 完成
echo ""
echo "========================================================================"
echo "🎉 安装完成！"
echo "========================================================================"
echo ""
echo "快速测试:"
echo "  # 1. 激活CUDA 11环境"
echo "  source ~/.cuda11_env"
echo ""
echo "  # 2. 运行GPU推理"
echo "  ./build/superpoint_inference IMG_0926.JPG gpu"
echo ""
echo "预期输出:"
echo "  ✓ 启用GPU推理 (CUDA)"
echo "  推理时间: ~45ms"
echo "  检测到 1549 个关键点"
echo ""
echo "文档:"
echo "  详细使用说明: CPP_API_GUIDE.md"
echo ""
echo "CUDA版本切换:"
echo "  source switch_cuda.sh 11  # SuperPoint C++"
echo "  source switch_cuda.sh 12  # PyTorch等"
echo ""
echo "========================================================================"
