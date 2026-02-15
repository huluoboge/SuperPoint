# TensorRT版本与GTX 1060 (SM 6.1)兼容性

## 官方支持矩阵

根据NVIDIA官方文档和社区反馈：

### TensorRT 10.x (2024+)
- **最低要求**: SM 7.0 (Volta架构)
- **GTX 1060 (SM 6.1)**: ❌ **不支持**
- 状态: 当前系统已安装 TensorRT 10.13.3

### TensorRT 8.6.x (2023年3月)
- **最低要求**: 存在争议
- **官方文档**: 部分文档显示要求SM 7.0+
- **实际情况**: 很多用户报告SM 6.x已不支持
- **GTX 1060 (SM 6.1)**: ⚠️ **可能不支持** (需验证)

### TensorRT 8.5.x (2022年)
- **最低要求**: SM 5.3+
- **GTX 1060 (SM 6.1)**: ✓ **支持**
- 建议版本: **8.5.3.1** (最稳定)

### TensorRT 8.4.x (2022年)
- **最低要求**: SM 5.3+
- **GTX 1060 (SM 6.1)**: ✓ **支持**
- 建议版本: 8.4.3.1

### TensorRT 8.2.x (2021年)
- **最低要求**: SM 5.3+
- **GTX 1060 (SM 6.1)**: ✓ **支持**
- 建议版本: 8.2.5.1

### TensorRT 7.x (2020-2021)
- **最低要求**: SM 5.3+
- **GTX 1060 (SM 6.1)**: ✓ **支持**
- 建议版本: 7.2.3.4 (最后的7.x版本)

## GPU架构时间线

| GPU系列 | 架构 | CUDA Capability | 发布年份 | TensorRT 8.6+ | TensorRT 8.5 | TensorRT 7.x |
|---------|------|-----------------|----------|---------------|--------------|--------------|
| GTX 10系列 | Pascal | SM 6.0-6.1 | 2016 | ❌ | ✓ | ✓ |
| GTX 16系列 | Turing | SM 7.5 | 2019 | ✓ | ✓ | ✓ |
| RTX 20系列 | Turing | SM 7.5 | 2018 | ✓ | ✓ | ✓ |
| RTX 30系列 | Ampere | SM 8.6 | 2020 | ✓ | ✓ | ✓ |

## 推荐安装方案

### 方案1: TensorRT 8.5.3.1 (推荐) ✓✓✓

**优点**:
- ✓ 确认支持SM 6.1
- ✓ 较新版本，性能好
- ✓ 支持CUDA 11.8
- ✓ 功能完整

**CUDA版本要求**: CUDA 11.x

**安装步骤**:

1. **卸载TensorRT 10.x**:
```bash
sudo apt remove --purge 'tensorrt*' 'libnvinfer*' 'nv-tensorrt*'
sudo apt autoremove
```

2. **降级到CUDA 11.8** (如果当前是CUDA 12):
```bash
# 下载CUDA 11.8
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run

# 安装（不覆盖驱动）
sudo sh cuda_11.8.0_520.61.05_linux.run --toolkit --silent --override

# 设置环境变量
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
```

3. **安装cuDNN 8** (for CUDA 11):
```bash
# 添加NVIDIA仓库
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# 卸载cuDNN 9
sudo apt remove --purge libcudnn9-cuda-12

# 安装cuDNN 8
sudo apt install -y libcudnn8=8.9.7.*-1+cuda11.8 libcudnn8-dev=8.9.7.*-1+cuda11.8
sudo apt-mark hold libcudnn8 libcudnn8-dev  # 防止自动更新
sudo ldconfig
```

4. **下载TensorRT 8.5.3.1**:

访问: https://developer.nvidia.com/nvidia-tensorrt-8x-download

选择:
- TensorRT 8.5 GA Update 3
- CUDA 11.x
- Ubuntu 22.04
- Tar File (推荐) 或 Debian

**Tar安装**:
```bash
# 解压
tar -xzvf TensorRT-8.5.3.1.Linux.x86_64-gnu.cuda-11.8.cudnn8.6.tar.gz
mv TensorRT-8.5.3.1 /opt/tensorrt

# 设置环境变量
export LD_LIBRARY_PATH=/opt/tensorrt/lib:$LD_LIBRARY_PATH
export PATH=/opt/tensorrt/bin:$PATH

# 添加到~/.bashrc
echo 'export LD_LIBRARY_PATH=/opt/tensorrt/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export PATH=/opt/tensorrt/bin:$PATH' >> ~/.bashrc
```

**Debian安装**:
```bash
# 如果下载的是deb包
sudo dpkg -i nv-tensorrt-repo-ubuntu2204-cuda11.8-trt8.5.3.1-ga-*_amd64.deb
sudo apt-key add /var/nv-tensorrt-repo-*/7fa2af80.pub
sudo apt update
sudo apt install tensorrt
```

5. **安装Python包**:
```bash
cd /opt/tensorrt/python
pip install tensorrt-8.5.3.1-cp39-none-linux_x86_64.whl
```

6. **验证安装**:
```bash
python3 -c "import tensorrt as trt; print(trt.__version__)"
# 应输出: 8.5.3.1
```

### 方案2: TensorRT 8.4.x (保守选择) ✓✓

如果8.5有问题，可以尝试8.4:
- 确认支持SM 6.1
- CUDA 11.x
- 安装步骤同上，版本号改为8.4.3.1

### 方案3: TensorRT 7.2.x (最保守) ✓

如果需要最大兼容性:
- 绝对支持SM 6.1
- CUDA 10.2/11.x
- 版本: 7.2.3.4

## 重要注意事项

### CUDA版本要求

| TensorRT | CUDA 版本 | cuDNN |
|----------|-----------|-------|
| 8.5.3.1 | 11.8 | 8.6+ |
| 8.4.3.1 | 11.6 | 8.4+ |
| 7.2.3.4 | 10.2/11.x | 8.0+ |

**当前系统**: CUDA 12.8 + cuDNN 9
**需要降级到**: CUDA 11.8 + cuDNN 8

### 性能预期

GTX 1060使用TensorRT 8.5:
- 推理时间: ~30-50ms (vs CPU 180ms)
- 加速比: 3-6x
- FPS: ~20-30

### 是否值得降级？

**考虑因素**:
1. **当前CPU性能**: 180ms (5.6 FPS)
2. **TensorRT预期**: 30-50ms (20-30 FPS)
3. **降级成本**: 需要重装CUDA 11 + cuDNN 8 + TensorRT 8.5
4. **影响范围**: 可能影响系统其他CUDA应用

**建议**:
- 如果5.6 FPS够用 → **保持CPU模式** ✓
- 如果需要20+ FPS → **降级到TensorRT 8.5** ✓
- 如果需要60+ FPS → **升级GPU到RTX 3060+** ✓✓✓

## 验证步骤

安装后运行:

```bash
# 1. 检查TensorRT版本
python -c "import tensorrt as trt; print(f'TensorRT: {trt.__version__}')"

# 2. 检查CUDA版本
nvcc --version

# 3. 检查cuDNN版本
dpkg -l | grep cudnn

# 4. 测试转换
python convert_to_tensorrt.py

# 5. 测试推理
python tensorrt_inference.py
```

## 快速决策树

```
需要TensorRT吗？
├─ 不需要（CPU够用）→ 保持当前配置 ✓
└─ 需要
   ├─ 愿意降级CUDA? 
   │  ├─ 是 → 安装TensorRT 8.5 + CUDA 11 ✓
   │  └─ 否 → 考虑升级GPU ✓✓
   └─ 需要高性能（60+ FPS）→ 升级到RTX 3060 ✓✓✓
```

## 参考资料

- TensorRT支持矩阵: https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/
- TensorRT下载: https://developer.nvidia.com/tensorrt
- CUDA Toolkit归档: https://developer.nvidia.com/cuda-toolkit-archive
- cuDNN归档: https://developer.nvidia.com/cudnn-archive

## 结论

**对于GTX 1060 (SM 6.1)**:
- TensorRT 8.6+ ❌ **不推荐/不支持**
- TensorRT 8.5.3.1 ✓ **推荐** (需要CUDA 11.8)
- TensorRT 8.4/8.2 ✓ 兼容
- TensorRT 7.2.x ✓ 最保守

**下一步**:
1. 确认是否真的需要TensorRT（当前CPU模式5.6 FPS可能够用）
2. 如需要，准备降级到CUDA 11.8环境
3. 安装TensorRT 8.5.3.1
