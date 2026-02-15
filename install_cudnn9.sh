#!/bin/bash
# cuDNN 9 安装脚本 for CUDA 12
# 适用于 Ubuntu/Debian 系统

set -e  # 遇到错误立即退出

echo "=========================================="
echo "cuDNN 9 安装脚本 (CUDA 12)"
echo "=========================================="
echo ""

# 检查CUDA版本
echo "1. 检查CUDA版本..."
if command -v nvcc &> /dev/null; then
    nvcc_version=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    echo "   ✓ 找到 CUDA: $nvcc_version"
else
    echo "   ✗ 未找到 nvcc 命令"
    echo "   请确保CUDA 12已安装并在PATH中"
    exit 1
fi

# 检查是否已安装cuDNN
echo ""
echo "2. 检查现有cuDNN..."
if ldconfig -p | grep -q libcudnn.so.9; then
    echo "   ✓ cuDNN 9 已经安装"
    cudnn_path=$(ldconfig -p | grep libcudnn.so.9 | awk '{print $4}' | head -1)
    echo "   路径: $cudnn_path"
    
    read -p "   是否要重新安装? (y/N): " reinstall
    if [[ ! "$reinstall" =~ ^[Yy]$ ]]; then
        echo "   跳过安装"
        exit 0
    fi
else
    echo "   未找到cuDNN 9"
fi

# 方法1: 使用apt安装 (推荐)
echo ""
echo "=========================================="
echo "方法1: 使用APT安装 (推荐)"
echo "=========================================="
echo ""
echo "这是最简单的方法，适用于Ubuntu 20.04/22.04/24.04"
echo ""

read -p "是否使用APT安装? (Y/n): " use_apt
if [[ ! "$use_apt" =~ ^[Nn]$ ]]; then
    echo ""
    echo "开始APT安装..."
    
    # 添加NVIDIA仓库
    echo ""
    echo "3. 添加NVIDIA仓库..."
    
    # 检测Ubuntu版本
    ubuntu_version=$(lsb_release -rs | cut -d'.' -f1)
    echo "   检测到 Ubuntu $ubuntu_version"
    
    # 下载CUDA keyring
    if [ ! -f /usr/share/keyrings/cuda-archive-keyring.gpg ]; then
        echo "   下载CUDA GPG密钥..."
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(lsb_release -rs | tr -d '.')/x86_64/cuda-keyring_1.1-1_all.deb
        sudo dpkg -i cuda-keyring_1.1-1_all.deb
        rm cuda-keyring_1.1-1_all.deb
    else
        echo "   ✓ CUDA密钥已存在"
    fi
    
    # 更新包列表
    echo ""
    echo "4. 更新包列表..."
    sudo apt update
    
    # 安装cuDNN
    echo ""
    echo "5. 安装cuDNN 9..."
    echo "   这可能需要几分钟..."
    
    # 注: libcudnn9-samples-cuda-12 包不存在，只安装运行库
    sudo apt install -y \
        libcudnn9-cuda-12
    
    echo ""
    echo "✓ cuDNN 9 安装完成！"
    
else
    # 方法2: 手动下载安装
    echo ""
    echo "=========================================="
    echo "方法2: 手动下载安装"
    echo "=========================================="
    echo ""
    echo "请按以下步骤操作："
    echo ""
    echo "1. 访问 NVIDIA cuDNN 下载页面:"
    echo "   https://developer.nvidia.com/cudnn-downloads"
    echo ""
    echo "2. 登录并选择:"
    echo "   - cuDNN 9.x"
    echo "   - CUDA 12.x"
    echo "   - Linux x86_64"
    echo "   - 选择 'Local Installer for Ubuntu' (.deb)"
    echo ""
    echo "3. 下载.deb文件后，运行:"
    echo "   sudo dpkg -i cudnn-local-repo-*.deb"
    echo "   sudo cp /var/cudnn-*/cudnn-*-keyring.gpg /usr/share/keyrings/"
    echo "   sudo apt update"
    echo "   sudo apt install -y libcudnn9-cuda-12"
    echo ""
    exit 0
fi

# 验证安装
echo ""
echo "=========================================="
echo "验证安装"
echo "=========================================="
echo ""

echo "6. 验证cuDNN安装..."

# 检查库文件
if ldconfig -p | grep -q libcudnn.so.9; then
    cudnn_path=$(ldconfig -p | grep libcudnn.so.9 | awk '{print $4}' | head -1)
    echo "   ✓ libcudnn.so.9: $cudnn_path"
    
    # 获取版本
    if command -v dpkg &> /dev/null; then
        cudnn_version=$(dpkg -l | grep libcudnn9 | head -1 | awk '{print $3}')
        echo "   ✓ cuDNN版本: $cudnn_version"
    fi
else
    echo "   ✗ 未找到 libcudnn.so.9"
    echo "   可能需要重启或运行: sudo ldconfig"
fi

# 更新库缓存
echo ""
echo "7. 更新动态链接库缓存..."
sudo ldconfig
echo "   ✓ 完成"

# 检查ONNX Runtime
echo ""
echo "8. 检查ONNX Runtime..."
if python3 -c "import onnxruntime" 2>/dev/null; then
    echo "   ✓ ONNX Runtime已安装"
    
    # 测试CUDA Provider
    echo ""
    echo "9. 测试CUDA Provider..."
    python3 << 'EOF'
import warnings
warnings.filterwarnings('ignore')

try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    
    if 'CUDAExecutionProvider' in providers:
        print("   ✓ CUDAExecutionProvider 可用")
        
        # 尝试创建会话
        try:
            session = ort.InferenceSession(
                "superpoint.onnx" if __import__('os').path.exists("superpoint.onnx") else None,
                providers=['CUDAExecutionProvider']
            ) if __import__('os').path.exists("superpoint.onnx") else None
            
            if session:
                print("   ✓ CUDA推理会话创建成功")
            else:
                print("   ℹ 需要ONNX模型文件来完整测试")
        except Exception as e:
            print(f"   ⚠ CUDA会话创建失败: {str(e)[:80]}")
    else:
        print("   ✗ CUDAExecutionProvider 不可用")
        print("   可用的providers:", providers)
except Exception as e:
    print(f"   ✗ 错误: {e}")
EOF
else
    echo "   ✗ ONNX Runtime未安装"
    echo ""
    echo "请安装ONNX Runtime GPU版本:"
    echo "   pip install onnxruntime-gpu"
fi

echo ""
echo "=========================================="
echo "安装完成！"
echo "=========================================="
echo ""
echo "下一步:"
echo "1. 确保已安装 onnxruntime-gpu:"
echo "   pip install onnxruntime-gpu"
echo ""
echo "2. 测试SuperPoint推理:"
echo "   python test_img_0926.py"
echo ""
echo "3. 如果仍然使用CPU，尝试:"
echo "   - 重启终端"
echo "   - 运行: sudo ldconfig"
echo "   - 检查: python -c 'import onnxruntime; print(onnxruntime.get_available_providers())'"
echo ""
