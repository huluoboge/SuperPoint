#!/bin/bash
# Quick start script for converting SuperPoint to TensorRT
# Usage: bash quick_convert.sh

set -e  # Exit on error

echo "=========================================="
echo "SuperPoint TensorRT 转换脚本"
echo "=========================================="
echo ""

# Configuration
WEIGHTS="weights/superpoint_v6_from_tf.pth"
ONNX_OUTPUT="superpoint.onnx"
TRT_OUTPUT="superpoint.trt"
IMAGE_HEIGHT=480
IMAGE_WIDTH=640
EXPORT_TYPE="dense"  # or "keypoints"
USE_FP16=true

# Check if weights exist
if [ ! -f "$WEIGHTS" ]; then
    echo "错误: 找不到权重文件: $WEIGHTS"
    echo "请确保权重文件存在"
    exit 1
fi

# Step 1: Check dependencies
echo "步骤 1: 检查依赖..."
python3 -c "import torch; print(f'  ✓ PyTorch {torch.__version__}')" || {
    echo "  ✗ PyTorch 未安装"
    echo "  请运行: pip install torch"
    exit 1
}

python3 -c "import onnx; print(f'  ✓ ONNX {onnx.__version__}')" || {
    echo "  ✗ ONNX 未安装"
    echo "  请运行: pip install onnx"
    exit 1
}

python3 -c "import tensorrt as trt; print(f'  ✓ TensorRT {trt.__version__}')" || {
    echo "  ✗ TensorRT 未安装"
    echo "  请运行: pip install tensorrt"
    exit 1
}

python3 -c "import pycuda.driver; print('  ✓ PyCUDA')" || {
    echo "  ✗ PyCUDA 未安装"
    echo "  请运行: pip install pycuda"
    exit 1
}

echo ""

# Step 2: Convert to ONNX
echo "步骤 2: 转换 PyTorch → ONNX..."
echo "  权重文件: $WEIGHTS"
echo "  输出文件: $ONNX_OUTPUT"
echo "  图像尺寸: ${IMAGE_HEIGHT}x${IMAGE_WIDTH}"
echo "  导出类型: $EXPORT_TYPE"
echo ""

python3 convert_to_onnx.py \
    --weights "$WEIGHTS" \
    --output "$ONNX_OUTPUT" \
    --height $IMAGE_HEIGHT \
    --width $IMAGE_WIDTH \
    --type "$EXPORT_TYPE"

if [ $? -ne 0 ]; then
    echo "错误: ONNX 转换失败"
    exit 1
fi

echo ""

# Step 3: Convert to TensorRT
echo "步骤 3: 转换 ONNX → TensorRT..."
echo "  输入文件: $ONNX_OUTPUT"
echo "  输出引擎: $TRT_OUTPUT"
echo "  FP16 模式: $USE_FP16"
echo ""

if [ "$USE_FP16" = true ]; then
    python3 convert_to_tensorrt.py \
        --onnx "$ONNX_OUTPUT" \
        --engine "$TRT_OUTPUT" \
        --fp16 \
        --workspace 2.0 \
        --test
else
    python3 convert_to_tensorrt.py \
        --onnx "$ONNX_OUTPUT" \
        --engine "$TRT_OUTPUT" \
        --workspace 2.0 \
        --test
fi

if [ $? -ne 0 ]; then
    echo "错误: TensorRT 转换失败"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ 转换完成！"
echo "=========================================="
echo ""
echo "生成的文件:"
echo "  ONNX 模型: $ONNX_OUTPUT"
echo "  TensorRT 引擎: $TRT_OUTPUT"
echo ""
echo "测试推理:"
echo "  python3 tensorrt_inference.py \\"
echo "    --engine $TRT_OUTPUT \\"
echo "    --image your_image.jpg \\"
echo "    --type $EXPORT_TYPE \\"
echo "    --benchmark"
echo ""

# Optional: Show file sizes
if command -v du &> /dev/null; then
    echo "文件大小:"
    du -h "$WEIGHTS" "$ONNX_OUTPUT" "$TRT_OUTPUT" | awk '{print "  " $2 ": " $1}'
    echo ""
fi

echo "详细使用说明请查看: TENSORRT_CONVERSION_GUIDE.md"
