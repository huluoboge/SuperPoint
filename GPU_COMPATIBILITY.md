# GPU兼容性问题和解决方案

## 问题

您的系统配置：
- **GPU**: NVIDIA GeForce GTX 1060 6GB
- **计算能力**: SM 6.1 (CUDA Compute Capability 6.1)
- **TensorRT版本**: 10.15.1.29

**错误信息**:
```
Target GPU SM 61 is not supported by this TensorRT release
```

## 原因

TensorRT 10.x 版本只支持以下GPU架构：
- **SM 7.0+** (Volta及更新架构)
- 例如: RTX 2060/2070/2080, RTX 3060/3070/3080/3090, RTX 4060/4070/4080/4090, Tesla V100, A100等

GTX 1060 (SM 6.1, Pascal架构) 不在支持列表中。

## 解决方案

### 方案 1: 使用TensorRT 8.x (推荐用于您的GPU)

TensorRT 8.6是最后支持Pascal架构(SM 6.x)的版本。

#### 1.1 卸载当前TensorRT
```bash
pip uninstall tensorrt
sudo apt remove tensorrt* --purge  # 如果通过apt安装的
```

#### 1.2 安装TensorRT 8.6

从NVIDIA官网下载TensorRT 8.6:
https://developer.nvidia.com/tensorrt

或使用pip安装:
```bash
# For CUDA 11.x
pip install tensorrt==8.6.1

# 还需要安装
pip install pycuda
```

#### 1.3 重新运行转换
```bash
python convert_to_tensorrt.py \
    --onnx superpoint.onnx \
    --engine superpoint.trt \
    --fp16 \
    --workspace 2.0 \
    --test
```

### 方案 2: 在支持的GPU上构建引擎

如果您有访问RTX 20系列或更新GPU的权限：

1. 在该机器上构建TensorRT引擎
2. 将`.trt`文件复制到您的GTX 1060系统
3. **注意**: 引擎是特定于GPU架构的，此方案不适用

### 方案 3: 使用ONNX Runtime替代TensorRT

ONNX Runtime支持更广泛的GPU，包括GTX 1060：

```bash
pip install onnxruntime-gpu
```

使用ONNX Runtime进行推理：

```python
import onnxruntime as ort
import numpy as np
import cv2

# 创建推理会话
session = ort.InferenceSession(
    "superpoint.onnx",
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)

# 加载图像
image = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)
image = image.astype(np.float32) / 255.0
image = image[np.newaxis, np.newaxis, :, :]

# 推理
outputs = session.run(None, {'image': image})
scores = outputs[0]
descriptors = outputs[1]
```

性能比较（估计）：
- PyTorch: ~15ms
- ONNX Runtime GPU: ~8-10ms
- TensorRT 8 (如果可用): ~5-7ms

### 方案 4: 仅使用ONNX模型

ONNX模型已经成功创建，您可以：

1. **使用ONNX Runtime** (推荐)
2. **使用OpenCV DNN模块**:
   ```python
   import cv2
   net = cv2.dnn.readNetFromONNX('superpoint.onnx')
   net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
   net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
   ```

3. **使用PyTorch加载ONNX** (较慢但简单)

## 推荐行动

**对于GTX 1060用户，我强烈推荐方案3 (ONNX Runtime)**:

### 快速开始 - ONNX Runtime

1. 安装:
```bash
pip install onnxruntime-gpu
```

2. 创建简单的推理脚本 `onnx_inference.py`:
```python
import onnxruntime as ort
import cv2
import numpy as np
from tensorrt_inference import visualize_keypoints

# 加载模型
session = ort.InferenceSession(
    "superpoint.onnx",
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)

print(f"使用提供者: {session.get_providers()}")

# 加载图像
image = cv2.imread('test.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_normalized = gray.astype(np.float32) / 255.0
input_tensor = gray_normalized[np.newaxis, np.newaxis, :, :]

# 推理
outputs = session.run(None, {'image': input_tensor})
scores_map = outputs[0][0]  # [H, W]
descriptors_map = outputs[1][0]  # [256, H/8, W/8]

# 提取关键点
from scipy.ndimage import maximum_filter

# NMS
nms_radius = 4
max_score = maximum_filter(scores_map, size=nms_radius*2+1, mode='constant')
nms_mask = (scores_map == max_score)
scores_map = scores_map * nms_mask

# 阈值
threshold = 0.005
mask = scores_map > threshold
yx = np.argwhere(mask)
keypoints = yx[:, ::-1].astype(np.float32)  # (x, y)
kp_scores = scores_map[yx[:, 0], yx[:, 1]]

# Top-k
top_k = 1000
if len(keypoints) > top_k:
    indices = np.argsort(kp_scores)[::-1][:top_k]
    keypoints = keypoints[indices]
    kp_scores = kp_scores[indices]

print(f"检测到 {len(keypoints)} 个关键点")

# 可视化
vis = visualize_keypoints(image, keypoints, kp_scores)
cv2.imwrite('output_onnxruntime.jpg', vis)
print("保存结果到 output_onnxruntime.jpg")
```

3. 运行:
```bash
python onnx_inference.py
```

## TensorRT版本与GPU支持对照表

| TensorRT版本 | 最低CUDA计算能力 | 支持的GPU示例 |
|-------------|-----------------|--------------|
| 10.x | SM 7.0+ | RTX 20/30/40系列, V100, A100 |
| 8.6 | SM 5.3+ | GTX 10系列, RTX 20/30系列 |
| 7.x | SM 5.3+ | GTX 10系列及以上 |

## 检查您的选择

运行这个脚本查看可用选项：
```bash
python3 << 'EOF'
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
cap = torch.cuda.get_device_capability(0)
print(f"计算能力: {cap[0]}.{cap[1]}")

sm = cap[0] * 10 + cap[1]
print(f"\nSM版本: {sm}")

if sm >= 70:
    print("✓ 支持 TensorRT 10.x")
    print("✓ 支持 TensorRT 8.x")
elif sm >= 53:
    print("✗ 不支持 TensorRT 10.x")
    print("✓ 支持 TensorRT 8.x (推荐)")
else:
    print("✗ TensorRT支持受限")

print("\n推荐: ONNX Runtime (适用于所有GPU)")
EOF
```

## 总结

由于您的GTX 1060不支持TensorRT 10.x:

1. **最佳选择**: 使用ONNX Runtime GPU（安装简单，性能好）
2. **备选**: 降级TensorRT到8.6版本
3. **简单**: 直接使用ONNX模型配合PyTorch或OpenCV

ONNX模型已成功创建并可以使用！🎉
