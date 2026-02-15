# SuperPoint ONNX and TensorRT 转换指南

这个指南将帮助您将 SuperPoint PyTorch 模型转换为 ONNX 和 TensorRT 格式，以便在 NVIDIA GPU 上获得更快的推理速度。

## 📋 前置要求

### 必需的包
```bash
# PyTorch 和相关依赖
pip install torch torchvision

# ONNX
pip install onnx onnxruntime

# TensorRT (如果使用 apt 安装)
# 您已经使用 apt 安装了 TensorRT，还需要安装 Python 绑定：
pip install tensorrt

# PyCUDA (用于 TensorRT 推理)
pip install pycuda

# OpenCV (用于图像处理)
pip install opencv-python

# 其他
pip install numpy scipy
```

### 检查 TensorRT 安装
```bash
python3 -c "import tensorrt; print(tensorrt.__version__)"
```

## 🔄 转换流程

转换分为两个步骤：
1. **PyTorch → ONNX**
2. **ONNX → TensorRT**

### 步骤 1: PyTorch 转换为 ONNX

有两种导出模式：

#### 模式 1: Dense 输出（推荐用于 TensorRT）
导出密集的特征图，后处理在 CPU 上进行。

```bash
python convert_to_onnx.py \
    --weights weights/superpoint_v6_from_tf.pth \
    --output superpoint_dense.onnx \
    --type dense \
    --height 480 \
    --width 640
```

**输出:**
- `scores`: 检测分数图 [B, H, W]
- `descriptors`: 密集描述符图 [B, 256, H/8, W/8]

#### 模式 2: Keypoints 输出
导出固定数量的关键点（带填充）。

```bash
python convert_to_onnx.py \
    --weights weights/superpoint_v6_from_tf.pth \
    --output superpoint_keypoints.onnx \
    --type keypoints \
    --max-keypoints 1024 \
    --height 480 \
    --width 640
```

**输出:**
- `keypoints`: 关键点坐标 [B, N, 2]
- `scores`: 关键点置信度 [B, N]
- `descriptors`: 关键点描述符 [B, N, 256]

#### 参数说明
- `--weights`: PyTorch 权重文件路径
- `--output`: 输出 ONNX 文件路径
- `--type`: 导出类型 (`dense` 或 `keypoints`)
- `--height`: 输入图像高度
- `--width`: 输入图像宽度
- `--max-keypoints`: 最大关键点数量（仅用于 keypoints 模式）
- `--opset`: ONNX opset 版本（默认 11）

### 步骤 2: ONNX 转换为 TensorRT

#### 基本转换 (FP32)
```bash
python convert_to_tensorrt.py \
    --onnx superpoint_dense.onnx \
    --engine superpoint_fp32.trt \
    --workspace 2.0
```

#### FP16 精度（推荐，速度快 2-3x）
```bash
python convert_to_tensorrt.py \
    --onnx superpoint_dense.onnx \
    --engine superpoint_fp16.trt \
    --fp16 \
    --workspace 2.0
```

#### 动态输入尺寸
如果需要支持不同尺寸的输入图像：

```bash
python convert_to_tensorrt.py \
    --onnx superpoint_dense.onnx \
    --engine superpoint_dynamic.trt \
    --fp16 \
    --dynamic-shapes \
    --min-height 240 --min-width 320 \
    --opt-height 480 --opt-width 640 \
    --max-height 960 --max-width 1280 \
    --workspace 4.0
```

#### 测试引擎
添加 `--test` 标志来测试构建的引擎：

```bash
python convert_to_tensorrt.py \
    --onnx superpoint_dense.onnx \
    --engine superpoint_fp16.trt \
    --fp16 \
    --test
```

#### 参数说明
- `--onnx`: ONNX 模型路径
- `--engine`: 输出 TensorRT 引擎路径
- `--fp16`: 启用 FP16 精度（更快）
- `--int8`: 启用 INT8 精度（需要校准）
- `--workspace`: 最大工作空间大小（GB）
- `--dynamic-shapes`: 启用动态输入尺寸
- `--test`: 构建后测试引擎

## 🚀 使用 TensorRT 引擎进行推理

### 基本用法

```bash
python tensorrt_inference.py \
    --engine superpoint_fp16.trt \
    --image test_image.jpg \
    --output result.jpg \
    --type dense \
    --threshold 0.005 \
    --top-k 1000
```

### 性能基准测试

```bash
python tensorrt_inference.py \
    --engine superpoint_fp16.trt \
    --image test_image.jpg \
    --type dense \
    --benchmark
```

### Python 代码示例

```python
from tensorrt_inference import SuperPointTRT
import cv2

# 加载模型
model = SuperPointTRT('superpoint_fp16.trt', output_type='dense')

# 读取图像
image = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)

# 推理
outputs = model.infer(image)

# 提取关键点
results = model.extract_keypoints_from_dense(
    outputs['scores'],
    outputs['descriptors'],
    threshold=0.005,
    top_k=1000
)

keypoints = results[0]['keypoints']  # [N, 2]
scores = results[0]['scores']        # [N]
descriptors = results[0]['descriptors']  # [N, 256]

print(f"检测到 {len(keypoints)} 个关键点")
```

## ⚡ 性能对比

预期性能提升（在 NVIDIA GPU 上）：

| 精度 | 相对速度 | 准确度 |
|------|----------|--------|
| FP32 | 1.0x | 100% |
| FP16 | 2-3x | ~99.9% |
| INT8 | 3-4x | ~99% |

典型推理时间（640x480 图像）：

| 平台 | PyTorch | TensorRT FP32 | TensorRT FP16 |
|------|---------|---------------|---------------|
| RTX 3090 | ~15ms | ~8ms | ~4ms |
| RTX 4090 | ~10ms | ~5ms | ~2ms |
| Jetson Xavier NX | ~50ms | ~30ms | ~15ms |

## 🔧 故障排除

### 1. TensorRT 导入错误
```
ImportError: No module named 'tensorrt'
```

**解决方案:**
```bash
# 检查 TensorRT 是否正确安装
dpkg -l | grep tensorrt

# 安装 Python 绑定
pip install tensorrt

# 或者从 NVIDIA 网站下载对应版本的 wheel 文件
```

### 2. CUDA 错误
```
pycuda._driver.Error: cuInit failed: no CUDA-capable device is detected
```

**解决方案:**
- 确保安装了 NVIDIA 驱动
- 检查 CUDA 是否正确安装：`nvidia-smi`
- 确保 CUDA 版本与 TensorRT 兼容

### 3. ONNX 导出警告
```
Warning: Constant folding ...
```

**解决方案:**
这些警告通常可以忽略。如果转换失败，尝试：
- 降低 opset 版本：`--opset 11`
- 禁用常量折叠（在代码中设置 `do_constant_folding=False`）

### 4. 动态形状错误
```
Error: Input shape does not match ...
```

**解决方案:**
- 使用 `--dynamic-shapes` 转换引擎
- 确保输入尺寸在 min/max 范围内
- 在推理时正确设置输入形状

## 📊 验证转换正确性

比较 PyTorch 和 TensorRT 的输出：

```python
import torch
import cv2
import numpy as np
from superpoint_pytorch import SuperPoint
from tensorrt_inference import SuperPointTRT

# 加载图像
image = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)
image_tensor = torch.from_numpy(image).float() / 255.0
image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)

# PyTorch 推理
pytorch_model = SuperPoint()
pytorch_model.load_state_dict(torch.load('weights/superpoint_v6_from_tf.pth'))
pytorch_model.eval()

with torch.no_grad():
    pytorch_out = pytorch_model({'image': image_tensor})

# TensorRT 推理
trt_model = SuperPointTRT('superpoint_fp16.trt', 'dense')
trt_out = trt_model.infer(image)

# 比较结果
print("检查输出差异...")
# 添加您的比较逻辑
```

## 📝 注意事项

1. **输入格式**: TensorRT 引擎期望归一化的输入 [0, 1]，grayscale 格式
2. **批处理**: 如果需要批处理，在转换时设置 `--max-batch-size`
3. **内存**: FP16 使用更少内存，但某些 GPU 可能不支持
4. **精度**: FP16 对大多数应用来说精度损失可忽略
5. **可移植性**: TensorRT 引擎是特定于 GPU 架构的，不可跨平台使用

## 🔗 相关资源

- [TensorRT 文档](https://docs.nvidia.com/deeplearning/tensorrt/)
- [ONNX 文档](https://onnx.ai/)
- [SuperPoint 论文](https://arxiv.org/abs/1712.07629)

## ❓ 常见问题

**Q: 可以在不同 GPU 上使用同一个引擎吗？**  
A: 不可以。TensorRT 引擎是为特定 GPU 架构优化的。需要在目标 GPU 上重新构建。

**Q: 为什么 TensorRT 推理比 PyTorch 快这么多？**  
A: TensorRT 进行了层融合、kernel 自动调优、精度校准等优化。

**Q: INT8 精度需要校准吗？**  
A: 是的。INT8 量化需要校准数据集。本脚本目前不包含校准功能。

**Q: 可以在 Jetson 设备上使用吗？**  
A: 可以！实际上 TensorRT 在 Jetson 上的优势更明显。确保使用 Jetson 上的 TensorRT 版本重新构建引擎。

---

如有问题，请查看日志输出或提交 issue。
