#!/usr/bin/env python3
"""
CUDA环境检查脚本
检查CUDA、cuDNN和ONNX Runtime的配置
"""

import sys
import os
import subprocess

print("="*70)
print("CUDA环境检查")
print("="*70)

# 1. 检查NVIDIA驱动
print("\n1. NVIDIA驱动:")
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        lines = result.stdout.split('\n')
        for line in lines[:10]:  # 只显示前几行
            if 'NVIDIA' in line or 'Driver' in line or 'CUDA' in line:
                print(f"   {line.strip()}")
        print("   ✓ NVIDIA驱动正常")
    else:
        print("   ✗ nvidia-smi失败")
except FileNotFoundError:
    print("   ✗ 未找到nvidia-smi命令")
    print("   请安装NVIDIA驱动")

# 2. 检查CUDA
print("\n2. CUDA Toolkit:")
try:
    result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
    if result.returncode == 0:
        version_line = [l for l in result.stdout.split('\n') if 'release' in l.lower()]
        if version_line:
            print(f"   {version_line[0].strip()}")
        print("   ✓ CUDA Toolkit已安装")
        
        # 检查版本
        if 'release 12' in result.stdout:
            print("   ✓ CUDA 12.x (兼容cuDNN 9)")
        elif 'release 11' in result.stdout:
            print("   ⚠ CUDA 11.x (需要cuDNN 8)")
    else:
        print("   ✗ nvcc命令失败")
except FileNotFoundError:
    print("   ✗ 未找到nvcc命令")
    print("   CUDA可能未安装或不在PATH中")

# 3. 检查cuDNN
print("\n3. cuDNN:")
try:
    result = subprocess.run(['ldconfig', '-p'], capture_output=True, text=True)
    cudnn_lines = [l for l in result.stdout.split('\n') if 'cudnn' in l.lower()]
    
    if cudnn_lines:
        print("   已安装的cuDNN库:")
        cudnn_versions = set()
        for line in cudnn_lines:
            if 'libcudnn.so' in line:
                # 提取版本号
                if 'libcudnn.so.9' in line:
                    cudnn_versions.add('9')
                    path = line.split('=>')[-1].strip()
                    print(f"   ✓ cuDNN 9: {path}")
                elif 'libcudnn.so.8' in line:
                    cudnn_versions.add('8')
                    path = line.split('=>')[-1].strip()
                    print(f"   ✓ cuDNN 8: {path}")
        
        if '9' in cudnn_versions:
            print("   ✓ cuDNN 9 已安装 (CUDA 12兼容)")
        elif '8' in cudnn_versions:
            print("   ⚠ cuDNN 8 已安装 (适用于CUDA 11)")
            print("   建议安装cuDNN 9用于CUDA 12")
        
        # 检查dpkg
        try:
            dpkg_result = subprocess.run(['dpkg', '-l'], capture_output=True, text=True)
            cudnn_packages = [l for l in dpkg_result.stdout.split('\n') if 'cudnn' in l.lower()]
            if cudnn_packages:
                print("\n   已安装的包:")
                for pkg in cudnn_packages[:5]:
                    parts = pkg.split()
                    if len(parts) >= 3:
                        print(f"   - {parts[1]} ({parts[2]})")
        except:
            pass
            
    else:
        print("   ✗ 未找到cuDNN库")
        print("   请安装cuDNN (参考 INSTALL_CUDNN.md)")
except Exception as e:
    print(f"   ✗ 检查失败: {e}")

# 4. 检查Python包
print("\n4. Python环境:")
print(f"   Python版本: {sys.version.split()[0]}")

# PyTorch (可选)
try:
    import torch
    print(f"   ✓ PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"   ✓ PyTorch CUDA可用")
        try:
            print(f"     GPU: {torch.cuda.get_device_name(0)}")
        except:
            pass
    else:
        print(f"   ⚠ PyTorch CUDA不可用")
except ImportError:
    print("   - PyTorch未安装")

# ONNX Runtime
try:
    import onnxruntime as ort
    print(f"   ✓ ONNX Runtime: {ort.__version__}")
    
    providers = ort.get_available_providers()
    print(f"\n   可用的Execution Providers:")
    for provider in providers:
        if provider == 'CUDAExecutionProvider':
            print(f"   ✓ {provider} (GPU)")
        else:
            print(f"   - {provider}")
    
    if 'CUDAExecutionProvider' in providers:
        print("\n   ✓ GPU推理已启用！")
        
        # 尝试创建CUDA会话
        try:
            test_session = ort.InferenceSession(
                "superpoint.onnx" if os.path.exists("superpoint.onnx") else None,
                providers=['CUDAExecutionProvider']
            ) if os.path.exists("superpoint.onnx") else None
            
            if test_session:
                print("   ✓ CUDA会话测试成功")
        except Exception as e:
            print(f"   ⚠ CUDA会话测试失败: {str(e)[:60]}...")
            print("   可能原因:")
            print("     - cuDNN未正确安装")
            print("     - CUDA版本不匹配")
            print("     - 需要重启Python环境")
    else:
        print("\n   ✗ GPU推理未启用")
        print("   请安装cuDNN 9 (参考 INSTALL_CUDNN.md)")
        
except ImportError:
    print("   ✗ ONNX Runtime未安装")
    print("   安装: pip install onnxruntime-gpu")

# 5. 环境变量
print("\n5. 环境变量:")
cuda_vars = ['CUDA_HOME', 'CUDA_PATH', 'LD_LIBRARY_PATH']
found_any = False
for var in cuda_vars:
    value = os.environ.get(var)
    if value:
        print(f"   {var}: {value[:60]}...")
        found_any = True

if not found_any:
    print("   (未设置CUDA相关环境变量，通常是正常的)")

# 6. 总结和建议
print("\n" + "="*70)
print("总结和建议")
print("="*70)

issues = []
recommendations = []

# 检查CUDA
if subprocess.run(['which', 'nvcc'], capture_output=True).returncode != 0:
    issues.append("CUDA Toolkit未安装或不在PATH中")
    recommendations.append("安装CUDA 12: https://developer.nvidia.com/cuda-downloads")

# 检查cuDNN
result = subprocess.run(['ldconfig', '-p'], capture_output=True, text=True)
if 'libcudnn.so.9' not in result.stdout:
    issues.append("cuDNN 9未安装")
    recommendations.append("运行安装脚本: ./install_cudnn9.sh")
    recommendations.append("或参考文档: INSTALL_CUDNN.md")

# 检查ONNX Runtime
try:
    import onnxruntime as ort
    if 'CUDAExecutionProvider' not in ort.get_available_providers():
        issues.append("ONNX Runtime GPU未启用")
        recommendations.append("如果cuDNN已安装，尝试:")
        recommendations.append("  pip uninstall onnxruntime onnxruntime-gpu")
        recommendations.append("  pip install onnxruntime-gpu")
        recommendations.append("  然后重启Python")
except ImportError:
    issues.append("ONNX Runtime未安装")
    recommendations.append("安装: pip install onnxruntime-gpu")

if issues:
    print("\n发现的问题:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    
    print("\n建议操作:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
else:
    print("\n✓ 环境配置正常！")
    print("\n可以运行:")
    print("  python test_img_0926.py")
    print("\n预期应该看到GPU推理")

print("\n" + "="*70)
