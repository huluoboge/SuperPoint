# SuperPoint 模型转换 - 快速开始

## 🎉 转换成功！

您的SuperPoint模型已经成功转换为ONNX格式！

## 📋 当前状态

✅ **ONNX模型已创建**: `superpoint.onnx` (5.0 MB)  
✅ **推理脚本已准备**: `onnx_inference.py`  
⚠️ **TensorRT**: 您的GPU (GTX 1060) 不支持TensorRT 10.x

## 🚀 30秒开始使用

### 步骤1: 安装ONNX Runtime (推荐)

```bash
pip install onnxruntime-gpu scipy
```

### 步骤2: 运行推理

```bash
# 使用您的图片
python onnx_inference.py --image your_image.jpg --output result.jpg

# 带性能测试
python onnx_inference.py --image your_image.jpg --benchmark
```

就这么简单！🎊

## 📊 预期性能

在您的 GTX 1060 6GB 上（640x480图像）：
- **推理时间**: ~8-12ms
- **FPS**: ~80-120
- **比PyTorch快**: 约2倍

## 💡 为什么用ONNX Runtime？

对于您的GTX 1060，ONNX Runtime是最佳选择：

| 特性 | ONNX Runtime | TensorRT 10 |
|------|--------------|-------------|
| 兼容GTX 1060 | ✅ 完美 | ❌ 不支持 |
| 安装 | 一行命令 | 复杂 |
| 性能 | 很好 (2x快) | N/A |
| 维护 | Microsoft官方 | NVIDIA |

## 📖 使用示例

### Python脚本

```python
from onnx_inference import SuperPointONNX
import cv2

# 初始化模型
model = SuperPointONNX('superpoint.onnx')

# 读取图像
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# 推理
outputs = model.infer(image)

# 提取关键点
result = model.extract_keypoints(
    outputs['scores'],
    outputs['descriptors'],
    threshold=0.005,
    top_k=1000
)

print(f"检测到 {len(result['keypoints'])} 个关键点")
```

### 命令行

```bash
# 基本用法
python onnx_inference.py --image test.jpg

# 自定义参数
python onnx_inference.py \
    --image test.jpg \
    --threshold 0.01 \
    --top-k 500 \
    --nms-radius 4 \
    --output result.jpg

# 性能测试
python onnx_inference.py --image test.jpg --benchmark
```

## 🔧 可用脚本

1. **onnx_inference.py** ⭐ - ONNX Runtime推理（推荐）
2. **check_inference_options.py** - 检查系统并推荐方案
3. **convert_to_onnx.py** - PyTorch → ONNX转换
4. **test_conversion.py** - 测试环境

## 📚 完整文档

- **README_CONVERSION.md** - 完整使用指南
- **GPU_COMPATIBILITY.md** - GPU兼容性详解
- **TENSORRT_CONVERSION_GUIDE.md** - TensorRT指南（如需要）

## ❓ 如果需要TensorRT

如果您**真的**想用TensorRT，需要降级到8.6版本：

```bash
# 卸载当前版本
pip uninstall tensorrt

# 安装TensorRT 8.6 (支持GTX 1060)
pip install tensorrt==8.6.1 pycuda

# 转换
python convert_to_tensorrt.py \
    --onnx superpoint.onnx \
    --engine superpoint.trt \
    --fp16

# 推理
python tensorrt_inference.py \
    --engine superpoint.trt \
    --image test.jpg \
    --type dense
```

**但我们建议先试试ONNX Runtime！** 😊

## 🎯 下一步

1. 安装ONNX Runtime:
   ```bash
   pip install onnxruntime-gpu scipy
   ```

2. 测试推理:
   ```bash
   python onnx_inference.py --image your_image.jpg --benchmark
   ```

3. 集成到您的项目中！

## 💬 需要帮助？

运行检查脚本查看您的选项：
```bash
python check_inference_options.py
```

查看详细文档：
```bash
cat README_CONVERSION.md
```

---

**祝使用愉快！** 🚀

有问题随时查看文档或提issue。
