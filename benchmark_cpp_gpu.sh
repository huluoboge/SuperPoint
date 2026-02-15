#!/bin/bash
# SuperPoint C++ GPU性能测试脚本

echo "========================================================================"
echo "SuperPoint C++ GPU 性能测试"
echo "========================================================================"

# 激活CUDA 11环境
source ~/.cuda11_env

echo ""
echo "测试配置:"
echo "  图像: IMG_0926.JPG (640x480 缩放后)"
echo "  设备: GPU (GTX 1060 + CUDA 11.8)"
echo "  测试次数: 10次"
echo ""

# 预热
echo "预热GPU..."
./build/superpoint_inference IMG_0926.JPG gpu > /dev/null 2>&1
./build/superpoint_inference IMG_0926.JPG gpu > /dev/null 2>&1
./build/superpoint_inference IMG_0926.JPG gpu > /dev/null 2>&1

echo ""
echo "开始性能测试 (10次运行)..."
echo "----------------------------------------------------------------------"

total=0
for i in {1..10}; do
    # 运行并提取推理时间
    output=$(./build/superpoint_inference IMG_0926.JPG gpu 2>&1)
    time=$(echo "$output" | grep "推理时间:" | awk '{print $2}' | sed 's/ms//')
    
    if [ ! -z "$time" ]; then
        total=$(echo "$total + $time" | bc)
        printf "运行 %2d: %6.1f ms\n" $i $time
    else
        echo "运行 $i: 失败"
    fi
done

echo "----------------------------------------------------------------------"

# 计算平均值
avg=$(echo "scale=1; $total / 10" | bc)

echo ""
echo "统计结果:"
echo "  总时间: $total ms"
echo "  平均推理时间: $avg ms"
echo "  平均FPS: $(echo "scale=1; 1000 / $avg" | bc)"

echo ""
echo "========================================================================"
echo "性能对比"
echo "========================================================================"
echo ""
echo "| 方案              | 推理时间 | FPS  | 设备       |"
echo "|-------------------|----------|------|------------|"
echo "| Python CPU        | 180ms    | 5.6  | CPU        |"
echo "| **C++ GPU**       | **${avg}ms** | **$(echo "scale=1; 1000 / $avg" | bc)** | **GPU (CUDA 11)** |"
echo ""
echo "提升倍数: $(echo "scale=2; 180 / $avg" | bc)x"
echo ""
echo "========================================================================"
