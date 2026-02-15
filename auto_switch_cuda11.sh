#!/bin/bash

echo "=========================================="
echo "统一切换到 CUDA 11.8 环境（自动执行）"
echo "=========================================="
echo ""

# 步骤 1: 卸载当前 PyTorch
echo "[1/7] 卸载 CUDA 12.8 版本的 PyTorch..."
pip uninstall -y torch torchvision torchaudio 2>&1 | grep -E "(Successfully|WARNING|Found)" || echo "  PyTorch 未安装"
echo "✓ 完成"
echo ""

# 步骤 2: 安装 CUDA 11.8 版本的 PyTorch
echo "[2/7] 安装 CUDA 11.8 版本的 PyTorch 2.5.1..."
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
echo "✓ 完成"
echo ""

# 步骤 3: 备份并更新 .bashrc
echo "[3/7] 配置 ~/.bashrc..."
cp ~/.bashrc ~/.bashrc.backup_$(date +%Y%m%d_%H%M%S)

# 移除旧的 CUDA 配置
sed -i '/export PATH=.*cuda.*bin/d' ~/.bashrc
sed -i '/export LD_LIBRARY_PATH=.*cuda.*lib/d' ~/.bashrc
sed -i '/export CUDA_HOME/d' ~/.bashrc

# 添加新的 CUDA 11.8 配置
cat >> ~/.bashrc << 'EOF'

# ===== CUDA 11.8 环境配置 (默认) =====
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
EOF

echo "✓ 完成（备份: ~/.bashrc.backup_*）"
echo ""

# 步骤 4: 更新系统符号链接
echo "[4/7] 更新 /usr/local/cuda 符号链接..."
sudo rm -f /usr/local/cuda
sudo ln -s /usr/local/cuda-11.8 /usr/local/cuda
echo "✓ 完成（/usr/local/cuda -> cuda-11.8）"
echo ""

# 步骤 5: 立即加载环境
echo "[5/7] 加载新环境变量..."
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
echo "✓ 完成"
echo ""

# 步骤 6: 验证安装
echo "[6/7] 验证环境..."
echo ""
echo "CUDA 版本:"
nvcc --version | grep "release"
echo ""

echo "PyTorch 验证:"
python << 'PYCODE'
import torch
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA 版本: {torch.version.cuda}")
    print(f"  cuDNN 版本: {torch.backends.cudnn.version()}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
PYCODE
echo ""

# 步骤 7: GPU 测试
echo "[7/7] GPU 计算测试..."
python << 'PYCODE'
import torch
if torch.cuda.is_available():
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.mm(x, y)
    print(f"  ✓ GPU 矩阵乘法成功")
    print(f"  设备: {z.device}")
else:
    print("  ✗ CUDA 不可用")
PYCODE
echo ""

echo "=========================================="
echo "✓ 切换完成！"
echo "=========================================="
echo ""
echo "当前配置:"
echo "  - CUDA: 11.8.89 + cuDNN 8"
echo "  - PyTorch: CUDA 11.8 版本"
echo "  - 系统默认: CUDA 11.8"
echo ""
echo "使用方式:"
echo "  1. 当前终端: 已生效"
echo "  2. 新终端: 自动加载 CUDA 11.8"
echo "  3. Python/PyTorch: 统一使用 CUDA 11.8"
echo "  4. C++ SuperPoint: 继续正常工作"
echo ""
echo "环境变量:"
echo "  CUDA_HOME: $CUDA_HOME"
echo "  nvcc: $(which nvcc)"
echo ""
