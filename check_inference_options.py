#!/usr/bin/env python3
"""
检查您的系统并推荐最佳SuperPoint推理方案
"""
import sys

print("="*60)
print("SuperPoint 推理方案检查")
print("="*60)

# 1. 检查GPU
print("\n1. GPU信息:")
try:
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        sm = cap[0] * 10 + cap[1]
        
        print(f"   GPU: {gpu_name}")
        print(f"   计算能力: SM {cap[0]}.{cap[1]}")
        
        # 判断TensorRT兼容性
        supports_trt10 = sm >= 70
        supports_trt8 = sm >= 53
        
        print(f"\n   TensorRT 10.x支持: {'✅ 是' if supports_trt10 else '❌ 否'}")
        print(f"   TensorRT 8.6支持: {'✅ 是' if supports_trt8 else '❌ 否'}")
    else:
        print("   ❌ CUDA不可用，将使用CPU")
        supports_trt10 = False
        supports_trt8 = False
        sm = 0
except Exception as e:
    print(f"   错误: {e}")
    supports_trt10 = False
    supports_trt8 = False
    sm = 0

# 2. 检查已安装的包
print("\n2. 已安装的推理引擎:")
packages = {}

# PyTorch
try:
    import torch
    packages['PyTorch'] = torch.__version__
    print(f"   ✅ PyTorch: {torch.__version__}")
except:
    print(f"   ❌ PyTorch: 未安装")

# ONNX
try:
    import onnx
    packages['ONNX'] = onnx.__version__
    print(f"   ✅ ONNX: {onnx.__version__}")
except:
    print(f"   ❌ ONNX: 未安装")

# ONNX Runtime
try:
    import onnxruntime as ort
    packages['ONNX Runtime'] = ort.__version__
    providers = ort.get_available_providers()
    has_cuda_provider = 'CUDAExecutionProvider' in providers
    print(f"   ✅ ONNX Runtime: {ort.__version__}")
    if has_cuda_provider:
        print(f"      GPU加速: ✅ 可用")
    else:
        print(f"      GPU加速: ❌ 不可用 (仅CPU)")
except:
    print(f"   ❌ ONNX Runtime: 未安装")
    has_cuda_provider = False

# TensorRT
try:
    import tensorrt as trt
    packages['TensorRT'] = trt.__version__
    major_version = int(trt.__version__.split('.')[0])
    print(f"   ✅ TensorRT: {trt.__version__}")
    
    if major_version >= 10 and not supports_trt10:
        print(f"      ⚠️  版本{major_version}.x不支持您的GPU (SM {sm})")
        print(f"      建议: 降级到 TensorRT 8.6")
    elif major_version < 10 and supports_trt8:
        print(f"      ✅ 兼容您的GPU")
except:
    print(f"   ❌ TensorRT: 未安装")
    major_version = None

# OpenCV
try:
    import cv2
    packages['OpenCV'] = cv2.__version__
    print(f"   ✅ OpenCV: {cv2.__version__}")
except:
    print(f"   ❌ OpenCV: 未安装")

# 3. 检查模型文件
print("\n3. 模型文件:")
import os

models = {
    'superpoint.onnx': 'ONNX模型',
    'superpoint.trt': 'TensorRT引擎',
    'weights/superpoint_v6_from_tf.pth': 'PyTorch权重'
}

for path, desc in models.items():
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024**2)
        print(f"   ✅ {desc}: {path} ({size_mb:.1f} MB)")
    else:
        print(f"   ❌ {desc}: {path} (不存在)")

# 4. 推荐方案
print("\n" + "="*60)
print("推荐的推理方案:")
print("="*60)

recommendations = []

# 评估各个方案
if 'ONNX Runtime' in packages and has_cuda_provider and os.path.exists('superpoint.onnx'):
    recommendations.append({
        'rank': 1,
        'name': 'ONNX Runtime (GPU)',
        'performance': '⭐⭐⭐⭐ (约8-12ms)',
        'ease': '⭐⭐⭐⭐⭐ 非常简单',
        'command': 'python onnx_inference.py --image test.jpg',
        'install': '已安装并就绪',
        'notes': '最佳选择！性能优秀且易用'
    })
elif os.path.exists('superpoint.onnx'):
    recommendations.append({
        'rank': 1,
        'name': 'ONNX Runtime (GPU)',
        'performance': '⭐⭐⭐⭐ (约8-12ms)',
        'ease': '⭐⭐⭐⭐⭐ 非常简单',
        'command': 'python onnx_inference.py --image test.jpg',
        'install': 'pip install onnxruntime-gpu',
        'notes': '推荐！一行命令安装'
    })

if major_version and major_version >= 10 and supports_trt10 and os.path.exists('superpoint.trt'):
    recommendations.append({
        'rank': 2,
        'name': 'TensorRT 10.x',
        'performance': '⭐⭐⭐⭐⭐ (约4-6ms)',
        'ease': '⭐⭐⭐ 中等',
        'command': 'python tensorrt_inference.py --engine superpoint.trt --image test.jpg',
        'install': '已安装并就绪',
        'notes': '最快，但仅限支持的GPU'
    })
elif major_version and major_version < 10 and supports_trt8 and os.path.exists('superpoint.trt'):
    recommendations.append({
        'rank': 2,
        'name': 'TensorRT 8.x',
        'performance': '⭐⭐⭐⭐⭐ (约6-8ms)',
        'ease': '⭐⭐⭐ 中等',
        'command': 'python tensorrt_inference.py --engine superpoint.trt --image test.jpg',
        'install': '已安装',
        'notes': '非常快，适合您的GPU'
    })
elif supports_trt8 and os.path.exists('superpoint.onnx'):
    recommendations.append({
        'rank': 3,
        'name': 'TensorRT 8.x (需安装)',
        'performance': '⭐⭐⭐⭐⭐ (约6-8ms)',
        'ease': '⭐⭐ 复杂',
        'command': 'python convert_to_tensorrt.py --onnx superpoint.onnx --engine superpoint.trt --fp16',
        'install': 'pip install tensorrt==8.6.1 pycuda',
        'notes': '需要降级TensorRT，较复杂'
    })

if 'PyTorch' in packages and os.path.exists('weights/superpoint_v6_from_tf.pth'):
    recommendations.append({
        'rank': 4,
        'name': 'PyTorch (原始)',
        'performance': '⭐⭐⭐ (约15-20ms)',
        'ease': '⭐⭐⭐⭐ 简单',
        'command': '使用原始SuperPoint代码',
        'install': '已安装',
        'notes': '最简单但较慢'
    })

# 排序并显示
recommendations.sort(key=lambda x: x['rank'])

for i, rec in enumerate(recommendations[:3], 1):  # 只显示前3个
    print(f"\n方案 {i}: {rec['name']}")
    print(f"   性能:   {rec['performance']}")
    print(f"   易用性: {rec['ease']}")
    print(f"   安装:   {rec['install']}")
    print(f"   运行:   {rec['command']}")
    print(f"   备注:   {rec['notes']}")

# 5. 下一步操作
print("\n" + "="*60)
print("建议的下一步:")
print("="*60)

if recommendations:
    best = recommendations[0]
    print(f"\n推荐使用: {best['name']}\n")
    
    if 'pip install' in best['install']:
        print(f"1. 安装依赖:")
        print(f"   {best['install']}")
        print()
    
    print(f"2. 运行推理:")
    print(f"   {best['command']}")
    print()
    
    if 'ONNX Runtime' in best['name']:
        print("3. 查看文档:")
        print("   cat README_CONVERSION.md")
else:
    print("\n请先安装必要的包和模型文件。")
    print("\n快速开始:")
    print("1. pip install onnxruntime-gpu")
    print("2. python convert_to_onnx.py --weights weights/superpoint_v6_from_tf.pth --output superpoint.onnx")
    print("3. python onnx_inference.py --image test.jpg")

print("\n" + "="*60)
print("详细文档: README_CONVERSION.md, GPU_COMPATIBILITY.md")
print("="*60)
