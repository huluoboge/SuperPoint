#!/usr/bin/env python3
"""
测试脚本 - IMG_0926 图像的SuperPoint特征提取
使用ONNX Runtime进行推理
"""

import cv2
import numpy as np
import time
import os
import sys
import warnings

# 过滤掉CUDA相关的警告信息
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 抑制TensorFlow警告

print("="*70)
print("SuperPoint 特征提取测试 - IMG_0926")
print("="*70)

# 1. 检查依赖
print("\n[1/6] 检查依赖...")
missing_deps = []

try:
    import onnxruntime as ort
    print(f"  ✓ ONNX Runtime: {ort.__version__}")
    providers = ort.get_available_providers()
    has_cuda = 'CUDAExecutionProvider' in providers
    
    if has_cuda:
        print(f"  ✓ CUDA支持: 是")
    else:
        print(f"  ℹ CUDA支持: 否（将使用CPU推理）")
        print(f"    如需GPU加速，可尝试:")
        print(f"    - 安装cuDNN 9和CUDA 12（复杂）")
        print(f"    - 或保持CPU推理（对GTX 1060推荐，更稳定）")
except ImportError:
    print("  ✗ ONNX Runtime未安装")
    missing_deps.append("onnxruntime")  # 推荐CPU版本

try:
    from scipy.ndimage import maximum_filter
    print("  ✓ SciPy")
except ImportError:
    print("  ✗ SciPy未安装")
    missing_deps.append("scipy")

if missing_deps:
    print(f"\n请安装缺失的依赖:")
    print(f"  pip install {' '.join(missing_deps)}")
    sys.exit(1)

# 2. 检查文件
print("\n[2/6] 检查文件...")
image_path = "IMG_0926.JPG"
model_path = "superpoint.onnx"

if not os.path.exists(image_path):
    print(f"  ✗ 图像文件不存在: {image_path}")
    sys.exit(1)
print(f"  ✓ 图像文件: {image_path}")

if not os.path.exists(model_path):
    print(f"  ✗ ONNX模型不存在: {model_path}")
    print(f"  请先运行: python convert_to_onnx.py --weights weights/superpoint_v6_from_tf.pth --output superpoint.onnx")
    sys.exit(1)
print(f"  ✓ ONNX模型: {model_path}")

# 3. 加载图像
print("\n[3/6] 加载图像...")
image = cv2.imread(image_path)
if image is None:
    print(f"  ✗ 无法读取图像: {image_path}")
    sys.exit(1)

original_h, original_w = image.shape[:2]
print(f"  ✓ 原始尺寸: {original_w} x {original_h}")

# 按比例缩放图像（保持宽高比）
# 将最长边缩放到max_dim，可根据需要调整此值
max_dim = 1024  # 可以改成480, 800, 1024等
scale_factor = max_dim / max(original_w, original_h)

if scale_factor < 1.0:  # 只在图像大于max_dim时才缩放
    new_w = int(original_w * scale_factor)
    new_h = int(original_h * scale_factor)
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    print(f"  ✓ 缩放后尺寸: {new_w} x {new_h} (缩放比例: {scale_factor:.3f})")
else:
    print(f"  ✓ 图像已小于{max_dim}，不需要缩放")

print(f"  ✓ 通道数: {image.shape[2]}")

# 转换为灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 4. 初始化模型
print("\n[4/6] 加载ONNX模型...")

# 设置日志级别，减少警告信息
ort.set_default_logger_severity(3)  # 3=ERROR, 隐藏WARNING

# 配置SessionOptions
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

# GTX 1060 (SM 6.1) 与 cuDNN 9 不兼容
# 尝试禁用cuDNN，只使用CUDA核心
cuda_provider_options = {
    'cudnn_conv_use_max_workspace': '0',  # 禁用cuDNN卷积
    'arena_extend_strategy': 'kSameAsRequested',
}

print("  ℹ GTX 1060 (SM 6.1) 不完全兼容 cuDNN 9")
print("  ℹ 尝试GPU推理，失败则自动切换到CPU...")

# 尝试使用CUDA (禁用cuDNN)，如果失败则使用CPU
try:
    session = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=[
            ('CUDAExecutionProvider', cuda_provider_options),
            'CPUExecutionProvider'
        ]
    )
    # 测试是否真的能工作
    test_input = np.random.randn(1, 1, 480, 640).astype(np.float32)
    _ = session.run(None, {'image': test_input})
    print("  ✓ GPU推理可用")
except Exception as e:
    print(f"  ⚠ GPU推理失败，使用CPU")
    # 使用CPU
    session = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=['CPUExecutionProvider']
    )

active_provider = session.get_providers()[0]
if 'CUDA' in active_provider:
    print(f"  ✓ 使用GPU推理: {active_provider}")
else:
    print(f"  ✓ 使用CPU推理: {active_provider}")
    print(f"    注: CPU推理速度约20-30ms，足够使用")

input_name = session.get_inputs()[0].name
output_names = [output.name for output in session.get_outputs()]
print(f"  ✓ 输入: {input_name}")
print(f"  ✓ 输出: {', '.join(output_names)}")

# 5. 运行推理
print("\n[5/6] 运行推理...")

# 预处理
gray_normalized = gray.astype(np.float32) / 255.0
input_tensor = gray_normalized[np.newaxis, np.newaxis, :, :]

# 预热
print("  预热中...")
for _ in range(3):
    _ = session.run(output_names, {input_name: input_tensor})

# 推理
print("  推理中...")
start_time = time.time()
outputs = session.run(output_names, {input_name: input_tensor})
inference_time = (time.time() - start_time) * 1000

scores_map = outputs[0][0]  # [H, W]
descriptors_map = outputs[1][0]  # [256, H/8, W/8]

print(f"  ✓ 推理时间: {inference_time:.2f} ms")
print(f"  ✓ 分数图尺寸: {scores_map.shape}")
print(f"  ✓ 描述符图尺寸: {descriptors_map.shape}")

# 6. 提取关键点
print("\n[6/6] 提取关键点...")

# NMS
nms_radius = 4
threshold = 0.005
border = 4

print(f"  参数: threshold={threshold}, nms_radius={nms_radius}, border={border}")

# 应用NMS
max_score = maximum_filter(scores_map, size=nms_radius*2+1, mode='constant', cval=0)
nms_mask = (scores_map == max_score)
scores_map = scores_map * nms_mask

# 移除边界
if border > 0:
    scores_map[:border, :] = 0
    scores_map[-border:, :] = 0
    scores_map[:, :border] = 0
    scores_map[:, -border:] = 0

# 提取关键点
mask = scores_map > threshold
yx = np.argwhere(mask)

if len(yx) == 0:
    print("  ⚠ 未检测到关键点")
    sys.exit(0)

# 获取分数
kp_scores = scores_map[yx[:, 0], yx[:, 1]]

# 转换为(x, y)
keypoints = yx[:, ::-1].astype(np.float32)

# 选择top-k
top_k = 8000
if len(keypoints) > top_k:
    indices = np.argsort(kp_scores)[::-1][:top_k]
    keypoints = keypoints[indices]
    kp_scores = kp_scores[indices]
    print(f"  保留top-{top_k}关键点（从{len(yx)}个中选择）")

# 提取描述符
h, w = scores_map.shape
dh, dw = descriptors_map.shape[1], descriptors_map.shape[2]

kp_y = (keypoints[:, 1] / h * dh).astype(np.int32)
kp_x = (keypoints[:, 0] / w * dw).astype(np.int32)
kp_y = np.clip(kp_y, 0, dh - 1)
kp_x = np.clip(kp_x, 0, dw - 1)

descriptors = descriptors_map[:, kp_y, kp_x].T  # [N, 256]

print(f"  ✓ 检测到 {len(keypoints)} 个关键点")
print(f"  ✓ 分数范围: [{kp_scores.min():.4f}, {kp_scores.max():.4f}]")
print(f"  ✓ 描述符形状: {descriptors.shape}")

# 7. 可视化
print("\n[7/7] 保存结果...")

# 创建可视化图像
vis = image.copy()

# 根据分数绘制关键点（不同颜色）
for i, kp in enumerate(keypoints):
    x, y = int(kp[0]), int(kp[1])
    score = kp_scores[i]
    
    # 分数越高，颜色越绿
    color_val = int(255 * min(score / 0.05, 1.0))
    color = (0, color_val, 255 - color_val)  # BGR格式
    
    # 根据分数调整半径
    if score > 0.02:
        radius = 4
        thickness = -1  # 填充
    elif score > 0.01:
        radius = 3
        thickness = -1
    else:
        radius = 2
        thickness = -1
    
    cv2.circle(vis, (x, y), radius, color, thickness)

# 添加信息文本
info_text = [
    f"Keypoints: {len(keypoints)}",
    f"Inference: {inference_time:.1f}ms",
    f"Score: [{kp_scores.min():.3f}, {kp_scores.max():.3f}]",
    f"Provider: {active_provider.replace('ExecutionProvider', '')}"
]

y_offset = 30
for i, text in enumerate(info_text):
    cv2.putText(vis, text, (10, y_offset + i*30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# 保存结果
output_path = "IMG_0926_result.jpg"
cv2.imwrite(output_path, vis)
print(f"  ✓ 可视化结果: {output_path}")

# 保存关键点数据
output_npz = "IMG_0926_keypoints.npz"
np.savez(output_npz, 
         keypoints=keypoints,
         scores=kp_scores,
         descriptors=descriptors)
print(f"  ✓ 关键点数据: {output_npz}")

# 8. 性能测试

# print("\n" + "="*70)
# print("性能测试 (100次迭代)")
# print("="*70)

# times = []
# for i in range(100):
#     start = time.time()
#     outputs = session.run(output_names, {input_name: input_tensor})
    
#     # 后处理
#     scores_map = outputs[0][0]
#     descriptors_map = outputs[1][0]
    
#     max_score = maximum_filter(scores_map, size=nms_radius*2+1, mode='constant', cval=0)
#     nms_mask = (scores_map == max_score)
#     scores_map = scores_map * nms_mask
    
#     mask = scores_map > threshold
#     yx = np.argwhere(mask)
    
#     if len(yx) > 0:
#         kp_scores = scores_map[yx[:, 0], yx[:, 1]]
#         if len(yx) > top_k:
#             indices = np.argsort(kp_scores)[::-1][:top_k]
    
#     times.append((time.time() - start) * 1000)

# times = np.array(times)

# print(f"\n推理 + 后处理统计:")
# print(f"  平均: {times.mean():.2f} ms")
# print(f"  标准差: {times.std():.2f} ms")
# print(f"  最小: {times.min():.2f} ms")
# print(f"  最大: {times.max():.2f} ms")
# print(f"  中位数: {np.median(times):.2f} ms")
# print(f"  FPS: {1000 / times.mean():.1f}")

# # 9. 总结
# print("\n" + "="*70)
# print("测试总结")
# print("="*70)
# print(f"\n图像信息:")
# print(f"  文件: {image_path}")
# print(f"  原始尺寸: {original_w} x {original_h}")
# current_h, current_w = image.shape[:2]
# if current_w != original_w or current_h != original_h:
#     print(f"  处理尺寸: {current_w} x {current_h} (缩放比例: {scale_factor:.3f})")
# print(f"\n检测结果:")
# print(f"  关键点数量: {len(keypoints)}")
# print(f"  平均分数: {kp_scores.mean():.4f}")
# print(f"  描述符维度: 256")
# print(f"\n性能:")
# print(f"  推理引擎: {active_provider.replace('ExecutionProvider', '')}")
# print(f"  推理时间: {inference_time:.2f} ms")
# print(f"  端到端: {times.mean():.2f} ms")
# print(f"  FPS: {1000 / times.mean():.1f}")
# if 'CPU' in active_provider:
#     print(f"  注: CPU模式，如需GPU请安装cuDNN 9 + CUDA 12")
# print(f"\n输出文件:")
# print(f"  {output_path} - 可视化结果")
# print(f"  {output_npz} - 关键点数据")
# print("\n✓ 测试完成！")
# print("="*70)
