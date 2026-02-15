# cuDNN 9 安装指南 (CUDA 12)

## 快速安装 (推荐)

### 方法1: 使用脚本自动安装

```bash
chmod +x install_cudnn9.sh
./install_cudnn9.sh
```

### 方法2: 手动安装 (APT)

最简单的方法，适用于Ubuntu 20.04/22.04/24.04：

```bash
# 1. 添加NVIDIA仓库 (如果还没有)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# 2. 安装cuDNN 9
sudo apt install -y libcudnn9-cuda-12

# 3. 更新库缓存
sudo ldconfig

# 4. 验证
ldconfig -p | grep cudnn
```

输出应该显示：
```
libcudnn.so.9 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libcudnn.so.9
```

### 方法3: 从NVIDIA官网下载

1. **访问**: https://developer.nvidia.com/cudnn-downloads

2. **选择**:
   - cuDNN 9.x for CUDA 12.x
   - Linux x86_64
   - Ubuntu (你的版本)
   - Local Installer (.deb)

3. **安装** (假设下载文件为 `cudnn-local-repo-*.deb`):
```bash
sudo dpkg -i cudnn-local-repo-ubuntu2204-9.*.deb
sudo cp /var/cudnn-local-repo-*/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt install -y libcudnn9-cuda-12
sudo ldconfig
```

## 验证安装

### 1. 检查cuDNN库

```bash
ldconfig -p | grep cudnn
```

应该看到 `libcudnn.so.9`

### 2. 检查版本

```bash
dpkg -l | grep cudnn
```

### 3. 测试ONNX Runtime CUDA

```python
python3 << EOF
import onnxruntime as ort
print("可用的Providers:", ort.get_available_providers())

if 'CUDAExecutionProvider' in ort.get_available_providers():
    print("✓ CUDA Provider可用！")
else:
    print("✗ CUDA Provider不可用")
EOF
```

### 4. 运行SuperPoint测试

```bash
python test_img_0926.py
```

应该看到：
```
✓ 使用GPU推理: CUDAExecutionProvider
```

而不是：
```
✓ 使用CPU推理: CPUExecutionProvider
```

## 常见问题

### 问题1: "libcudnn.so.9: cannot open shared object file"

**解决方案**:
```bash
# 更新动态链接库缓存
sudo ldconfig

# 检查是否安装成功
ldconfig -p | grep cudnn
```

### 问题2: 安装后仍使用CPU

**可能原因**:
1. ONNX Runtime版本不对
2. 需要重启Python环境
3. 环境变量问题

**解决方案**:
```bash
# 1. 确保安装GPU版本
pip uninstall onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu

# 2. 检查
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"

# 3. 应该看到 'CUDAExecutionProvider' 在列表中
```

### 问题3: CUDA版本不匹配

您的系统: CUDA 12
需要: cuDNN 9 for CUDA 12

检查CUDA版本:
```bash
nvcc --version
nvidia-smi  # 查看驱动支持的最高CUDA版本
```

### 问题4: Ubuntu版本问题

不同Ubuntu版本使用不同的仓库URL：
- Ubuntu 20.04: `ubuntu2004`
- Ubuntu 22.04: `ubuntu2204`
- Ubuntu 24.04: `ubuntu2404`

修改上面命令中的版本号。

## 安装后设置

### 环境变量 (通常不需要)

如果手动安装，可能需要：

```bash
# 添加到 ~/.bashrc
export LD_LIBRARY_PATH=/usr/local/cuda-12/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12

# 生效
source ~/.bashrc
```

### 验证完整环境

运行检查脚本：
```bash
python check_cuda_environment.py
```

## 性能提升预期

安装cuDNN 9后，GPU推理性能：

| 操作 | CPU | GPU (GTX 1060) | 提升 |
|------|-----|----------------|------|
| 推理 | ~20-30ms | ~8-12ms | 2-3x |
| FPS | ~35-50 | ~80-120 | 2-3x |

**注意**: GTX 1060虽然较老，但仍能获得显著加速！

## 卸载cuDNN (如需要)

```bash
sudo apt remove --purge libcudnn9-cuda-12
sudo apt autoremove
sudo ldconfig
```

## 参考链接

- **cuDNN下载**: https://developer.nvidia.com/cudnn-downloads
- **ONNX Runtime GPU**: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
- **CUDA Toolkit**: https://developer.nvidia.com/cuda-downloads

## 快速命令总结

```bash
# 一键安装 (Ubuntu 22.04为例)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y libcudnn9-cuda-12
sudo ldconfig

# 安装ONNX Runtime GPU
pip install onnxruntime-gpu

# 测试
python test_img_0926.py
```

完成！🎉
