#!/usr/bin/env python3
"""
Simple test script to verify the conversion works.
"""
import torch
import warnings
warnings.filterwarnings('ignore')

print("="*50)
print("Testing SuperPoint Conversion Environment")
print("="*50)

# Check PyTorch
print(f"\n1. PyTorch version: {torch.__version__}")

# Check CUDA
print(f"2. CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    try:
        device_name = torch.cuda.get_device_name(0)
        compute_capability = torch.cuda.get_device_capability(0)
        print(f"   GPU: {device_name}")
        print(f"   Compute Capability: {compute_capability[0]}.{compute_capability[1]}")
        
        # Test if we can actually use CUDA
        try:
            test_tensor = torch.zeros(1).cuda()
            print(f"   ✓ CUDA is usable")
            use_cuda = True
        except Exception as e:
            print(f"   ✗ CUDA not compatible with this PyTorch version")
            print(f"   Note: Conversion will run on CPU (this is fine!)")
            use_cuda = False
    except Exception as e:
        print(f"   Error checking CUDA: {e}")
        use_cuda = False
else:
    print("   Using CPU for conversion")
    use_cuda = False

# Check weights file
import os
weights_path = 'weights/superpoint_v6_from_tf.pth'
print(f"\n3. Weights file: {weights_path}")
if os.path.exists(weights_path):
    size_mb = os.path.getsize(weights_path) / (1024**2)
    print(f"   ✓ Found ({size_mb:.2f} MB)")
else:
    print(f"   ✗ Not found!")
    print(f"   Please ensure weights file exists")

# Check packages
print(f"\n4. Required packages:")
packages = {
    'torch': 'PyTorch',
    'onnx': 'ONNX',
    'cv2': 'OpenCV (opencv-python)',
    'numpy': 'NumPy'
}

all_ok = True
for pkg, name in packages.items():
    try:
        if pkg == 'cv2':
            import cv2
            print(f"   ✓ {name}: {cv2.__version__}")
        else:
            mod = __import__(pkg)
            print(f"   ✓ {name}: {mod.__version__}")
    except ImportError:
        print(f"   ✗ {name}: Not installed")
        print(f"      Install with: pip install {pkg if pkg != 'cv2' else 'opencv-python'}")
        all_ok = False

# TensorRT (optional for ONNX export)
try:
    import tensorrt as trt
    print(f"   ✓ TensorRT (optional): {trt.__version__}")
    has_trt = True
except ImportError:
    print(f"   - TensorRT: Not installed (only needed for TensorRT conversion)")
    has_trt = False

print("\n" + "="*50)
print("Summary:")
print("="*50)

if all_ok:
    print("✓ All required packages installed!")
    print("\nYou can now run:")
    print("  python convert_to_onnx.py --weights weights/superpoint_v6_from_tf.pth --output superpoint.onnx --type dense")
    
    if has_trt:
        print("\nThen convert to TensorRT:")
        print("  python convert_to_tensorrt.py --onnx superpoint.onnx --engine superpoint.trt --fp16")
    else:
        print("\nTo convert to TensorRT, install it first:")
        print("  pip install tensorrt pycuda")
else:
    print("✗ Some packages are missing. Please install them first.")

if not use_cuda:
    print("\nNote: Conversion will run on CPU. This is normal and works fine!")
    print("      (Only actual inference benefits from GPU)")

print("")
