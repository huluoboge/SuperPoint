#!/bin/bash

echo "=========================================="
echo "统一切换到 CUDA 11.8 环境"
echo "=========================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 步骤 1: 检查当前环境
echo -e "${YELLOW}步骤 1: 检查当前环境${NC}"
echo "当前 PyTorch 版本:"
python -c "import torch; print('  版本:', torch.__version__); print('  CUDA:', torch.version.cuda)" 2>/dev/null || echo "  PyTorch 未安装"
echo ""
echo "当前 CUDA 路径:"
echo "  nvcc: $(which nvcc 2>/dev/null || echo '未找到')"
echo ""

# 步骤 2: 卸载 CUDA 12.8 版本的 PyTorch
echo -e "${YELLOW}步骤 2: 卸载 CUDA 12.8 版本的 PyTorch${NC}"
read -p "确认卸载当前 PyTorch 2.8.0+cu128？(y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip uninstall -y torch torchvision torchaudio
    echo -e "${GREEN}✓ PyTorch 已卸载${NC}"
else
    echo -e "${RED}✗ 取消操作${NC}"
    exit 1
fi
echo ""

# 步骤 3: 安装 CUDA 11.8 版本的 PyTorch
echo -e "${YELLOW}步骤 3: 安装 CUDA 11.8 版本的 PyTorch${NC}"
echo "安装 PyTorch 2.5.1 (对应 CUDA 11.8)..."
echo ""
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo ""

# 步骤 4: 配置系统默认使用 CUDA 11.8
echo -e "${YELLOW}步骤 4: 配置系统默认使用 CUDA 11.8${NC}"

# 备份 .bashrc
cp ~/.bashrc ~/.bashrc.backup_$(date +%Y%m%d_%H%M%S)
echo -e "${GREEN}✓ 已备份 ~/.bashrc${NC}"

# 移除旧的 CUDA 配置（如果存在）
sed -i '/export PATH=\/usr\/local\/cuda/d' ~/.bashrc
sed -i '/export LD_LIBRARY_PATH=\/usr\/local\/cuda/d' ~/.bashrc
sed -i '/export CUDA_HOME/d' ~/.bashrc

# 添加 CUDA 11.8 配置
cat >> ~/.bashrc << 'EOF'

# CUDA 11.8 环境配置 (默认)
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
EOF

echo -e "${GREEN}✓ 已添加 CUDA 11.8 配置到 ~/.bashrc${NC}"
echo ""

# 步骤 5: 更新系统符号链接
echo -e "${YELLOW}步骤 5: 更新系统符号链接${NC}"
echo "需要 root 权限来更新 /usr/local/cuda 符号链接..."
sudo rm -f /usr/local/cuda
sudo ln -s /usr/local/cuda-11.8 /usr/local/cuda
echo -e "${GREEN}✓ /usr/local/cuda -> /usr/local/cuda-11.8${NC}"
echo ""

# 步骤 6: 立即加载新环境
echo -e "${YELLOW}步骤 6: 加载新环境${NC}"
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
echo -e "${GREEN}✓ 环境变量已更新${NC}"
echo ""

# 步骤 7: 验证新环境
echo -e "${YELLOW}步骤 7: 验证新环境${NC}"
echo "----------------------------------------"
echo "系统 CUDA 版本:"
nvcc --version | grep "release"
echo ""

echo "PyTorch 配置:"
python -c "import torch; print('  PyTorch 版本:', torch.__version__); print('  CUDA 可用:', torch.cuda.is_available()); print('  CUDA 版本:', torch.version.cuda); print('  cuDNN 版本:', torch.backends.cudnn.version()); print('  GPU 设备:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
echo ""

echo "环境变量:"
echo "  CUDA_HOME: $CUDA_HOME"
echo "  nvcc 路径: $(which nvcc)"
echo ""

# 步骤 8: 测试 GPU
echo -e "${YELLOW}步骤 8: 测试 PyTorch GPU${NC}"
python -c "
import torch
print('创建测试张量...')
if torch.cuda.is_available():
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.mm(x, y)
    print('✓ GPU 计算成功！')
    print(f'  结果形状: {z.shape}')
    print(f'  设备: {z.device}')
else:
    print('✗ CUDA 不可用')
"
echo ""

echo "=========================================="
echo -e "${GREEN}✓ 切换完成！${NC}"
echo "=========================================="
echo ""
echo "重要提示:"
echo "1. 已将 CUDA 11.8 设置为系统默认"
echo "2. PyTorch 已切换到 CUDA 11.8 版本"
echo "3. ~/.bashrc 已更新（备份文件: ~/.bashrc.backup_*）"
echo "4. 当前终端已生效"
echo "5. 新终端会自动使用 CUDA 11.8"
echo ""
echo "如需回退到 CUDA 12.8，可以:"
echo "  sudo ln -sf /usr/local/cuda-12.8 /usr/local/cuda"
echo "  pip install torch --index-url https://download.pytorch.org/whl/cu128"
echo ""
