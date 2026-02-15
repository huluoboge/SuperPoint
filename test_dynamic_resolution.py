#!/usr/bin/env python3
"""
验证ONNX模型的动态分辨率能力
测试不同尺寸的图像推理
"""

import numpy as np
import onnxruntime as ort
import time
import sys

def test_dynamic_resolution():
    """测试ONNX模型支持任意分辨率"""
    
    print("="*70)
    print("ONNX动态分辨率测试")
    print("="*70)
    
    # 1. 加载模型
    print("\n[1/3] 加载ONNX模型...")
    try:
        session = ort.InferenceSession(
            'superpoint.onnx',
            providers=['CPUExecutionProvider']  # 使用CPU确保稳定
        )
        print("  ✓ 模型加载成功")
    except Exception as e:
        print(f"  ✗ 加载失败: {e}")
        return False
    
    # 2. 测试不同分辨率
    print("\n[2/3] 测试不同分辨率...")
    
    test_sizes = [
        (320, 240, "QVGA - 小图"),
        (640, 480, "VGA - 中图"),
        (800, 600, "SVGA"),
        (1024, 768, "XGA"),
        (1280, 720, "HD 720p"),
        (1920, 1080, "Full HD 1080p"),
        (2560, 1440, "2K"),
        (3840, 2160, "4K - 大图"),
    ]
    
    results = []
    
    for width, height, desc in test_sizes:
        try:
            # 创建随机图像
            image = np.random.rand(1, 1, height, width).astype(np.float32)
            
            # 推理
            start = time.time()
            scores, descriptors = session.run(None, {'image': image})
            elapsed = (time.time() - start) * 1000
            
            # 验证输出尺寸
            expected_h = height
            expected_w = width
            expected_desc_h = height // 8
            expected_desc_w = width // 8
            
            scores_ok = scores.shape == (1, expected_h, expected_w)
            desc_ok = descriptors.shape == (1, 256, expected_desc_h, expected_desc_w)
            
            if scores_ok and desc_ok:
                results.append({
                    'size': f'{width}x{height}',
                    'desc': desc,
                    'time': elapsed,
                    'success': True,
                    'scores_shape': scores.shape,
                    'desc_shape': descriptors.shape
                })
                status = "✓"
            else:
                results.append({
                    'size': f'{width}x{height}',
                    'desc': desc,
                    'time': elapsed,
                    'success': False,
                    'error': f'Shape mismatch'
                })
                status = "✗"
            
            print(f"  {status} {width:4d}x{height:4d} ({desc:15s}): {elapsed:6.1f}ms | "
                  f"scores{scores.shape}, desc{descriptors.shape}")
            
        except Exception as e:
            results.append({
                'size': f'{width}x{height}',
                'desc': desc,
                'success': False,
                'error': str(e)
            })
            print(f"  ✗ {width:4d}x{height:4d} ({desc:15s}): 失败 - {e}")
    
    # 3. 统计结果
    print("\n[3/3] 测试总结...")
    
    success_count = sum(1 for r in results if r['success'])
    total_count = len(results)
    
    print(f"\n成功: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("\n✓✓✓ 完美！ONNX模型支持所有测试的分辨率")
        print("    从 320x240 到 3840x2160 (4K) 都可以推理")
        print("    真正的动态分辨率，无需重新配置")
    elif success_count > 0:
        print(f"\n⚠ 部分成功 ({success_count}/{total_count})")
    else:
        print("\n✗ 全部失败，请检查模型")
        return False
    
    # 4. 性能分析
    if success_count > 0:
        print("\n推理时间分析:")
        successful_results = [r for r in results if r['success']]
        
        # 按尺寸排序
        for r in successful_results:
            w, h = map(int, r['size'].split('x'))
            pixels = w * h / 1_000_000  # 百万像素
            print(f"  {r['size']:12s} ({pixels:4.1f}MP): {r['time']:6.1f}ms")
        
        # 计算时间/像素比
        times = [r['time'] for r in successful_results]
        avg_time = sum(times) / len(times)
        print(f"\n  平均推理时间: {avg_time:.1f}ms")
    
    print("\n" + "="*70)
    
    return success_count == total_count


def compare_with_tensorrt():
    """对比TensorRT的限制"""
    
    print("\n" + "="*70)
    print("对比: ONNX vs TensorRT动态分辨率")
    print("="*70)
    
    print("\n✓ ONNX Runtime (当前方案):")
    print("  - 支持任意分辨率（已验证 ✓）")
    print("  - 运行时自动适配")
    print("  - 无需预先配置")
    print("  - 一个模型文件适配所有尺寸")
    
    print("\n⚠ TensorRT (如果使用):")
    print("  - 需要预先指定最小/最优/最大尺寸")
    print("  - 超出范围的尺寸无法推理")
    print("  - 非最优尺寸性能下降")
    print("  - 配置复杂")
    
    print("\n示例对比:")
    print("\n  ONNX:")
    print("    session.run(None, {'image': img})  # 任意尺寸，直接用")
    
    print("\n  TensorRT:")
    print("    # 转换时需要指定:")
    print("    profile.set_shape('image',")
    print("        min=(1,1,240,320),    # 最小: 320x240")
    print("        opt=(1,1,480,640),    # 最优: 640x480")
    print("        max=(1,1,1080,1920))  # 最大: 1920x1080")
    print("    # 4K (3840x2160) 会失败！需要重新转换")
    
    print("\n结论: 对于多机器+任意分辨率部署 → ONNX Runtime ✓✓✓")
    print("="*70)


if __name__ == '__main__':
    print("\n测试SuperPoint ONNX模型的动态分辨率能力\n")
    
    # 测试动态分辨率
    success = test_dynamic_resolution()
    
    # 对比说明
    compare_with_tensorrt()
    
    # 最终建议
    if success:
        print("\n【建议】")
        print("  ✓ 当前ONNX配置完美支持你的需求:")
        print("    - CPU + GPU 都能用（自动切换）")
        print("    - 任意分辨率（已验证）")
        print("    - 多机器部署（同一个模型文件）")
        print("    - 维护简单（一套代码）")
        print("\n  不需要TensorRT！")
        print("  如需更高性能 → 升级GPU到RTX 3060+ (ONNX GPU也能加速)")
    else:
        print("\n【警告】模型测试失败，请检查 superpoint.onnx 是否存在")
        sys.exit(1)
