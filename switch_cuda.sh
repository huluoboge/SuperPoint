#!/bin/bash
# CUDA版本切换脚本

if [ "$1" == "11" ] || [ "$1" == "11.8" ]; then
    export CUDA_HOME=/usr/local/cuda-11.8
    export PATH=/usr/local/cuda-11.8/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
    echo "✓ 切换到 CUDA 11.8"
    nvcc --version | grep release
elif [ "$1" == "12" ] || [ "$1" == "12.8" ]; then
    export CUDA_HOME=/usr/local/cuda-12.8
    export PATH=/usr/local/cuda-12.8/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
    echo "✓ 切换到 CUDA 12.8"
    nvcc --version | grep release
else
    echo "用法: source switch_cuda.sh [11|12]"
    echo "当前CUDA版本:"
    nvcc --version | grep release || echo "未设置CUDA环境"
fi
