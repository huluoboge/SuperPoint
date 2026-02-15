#!/usr/bin/env python3
"""
Script to convert ONNX model to TensorRT engine.
Requires TensorRT to be installed (tensorrt, pycuda).

Usage:
    python convert_to_tensorrt.py --onnx superpoint.onnx --engine superpoint.trt --fp16
"""

import argparse
import os
import numpy as np


def convert_onnx_to_tensorrt(onnx_path, engine_path, fp16=False, int8=False, 
                             max_batch_size=1, max_workspace_size=1,
                             min_shape=None, opt_shape=None, max_shape=None):
    """
    Convert ONNX model to TensorRT engine.
    
    Args:
        onnx_path: Path to ONNX model
        engine_path: Output path for TensorRT engine
        fp16: Enable FP16 precision
        int8: Enable INT8 precision
        max_batch_size: Maximum batch size
        max_workspace_size: Maximum workspace size in GB
        min_shape: Minimum input shape (height, width)
        opt_shape: Optimal input shape (height, width)
        max_shape: Maximum input shape (height, width)
    """
    try:
        import tensorrt as trt
    except ImportError:
        print("Error: TensorRT is not installed!")
        print("Please install TensorRT:")
        print("  pip install tensorrt")
        print("Or follow: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html")
        return None
    
    # Create TensorRT logger
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    
    print(f"TensorRT version: {trt.__version__}")
    print(f"Loading ONNX model: {onnx_path}")
    
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX file not found: {onnx_path}")
        return None
    
    # Create builder and network
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX model
    print("Parsing ONNX model...")
    with open(onnx_path, 'rb') as model_file:
        if not parser.parse(model_file.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    print(f"✓ ONNX model parsed successfully")
    print(f"  Network inputs: {network.num_inputs}")
    print(f"  Network outputs: {network.num_outputs}")
    
    # Print network info
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        print(f"  Input {i}: {input_tensor.name}, shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
    
    for i in range(network.num_outputs):
        output_tensor = network.get_output(i)
        print(f"  Output {i}: {output_tensor.name}, shape: {output_tensor.shape}, dtype: {output_tensor.dtype}")
    
    # Create builder config
    config = builder.create_builder_config()
    
    # Set workspace size
    workspace_size = int(max_workspace_size * (1 << 30))  # Convert GB to bytes
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
    print(f"Max workspace size: {max_workspace_size} GB")
    
    # Set precision
    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("✓ FP16 mode enabled")
        else:
            print("Warning: FP16 is not supported on this platform")
    
    if int8:
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("✓ INT8 mode enabled")
            print("Note: INT8 calibration is not implemented in this script")
        else:
            print("Warning: INT8 is not supported on this platform")
    
    # Configure dynamic shapes if specified
    if min_shape and opt_shape and max_shape:
        print("Configuring dynamic input shapes...")
        profile = builder.create_optimization_profile()
        
        # Assuming first input is the image
        input_name = network.get_input(0).name
        
        # Shapes should be (batch, channels, height, width)
        min_shape_full = (1, 1, min_shape[0], min_shape[1])
        opt_shape_full = (max_batch_size, 1, opt_shape[0], opt_shape[1])
        max_shape_full = (max_batch_size, 1, max_shape[0], max_shape[1])
        
        profile.set_shape(input_name, min_shape_full, opt_shape_full, max_shape_full)
        config.add_optimization_profile(profile)
        
        print(f"  Min shape: {min_shape_full}")
        print(f"  Opt shape: {opt_shape_full}")
        print(f"  Max shape: {max_shape_full}")
    
    # Build engine
    print("\nBuilding TensorRT engine... (this may take a few minutes)")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("ERROR: Failed to build TensorRT engine")
        return None
    
    # Save engine
    print(f"Saving TensorRT engine to: {engine_path}")
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    
    print(f"✓ TensorRT engine saved successfully!")
    print(f"  Engine file size: {os.path.getsize(engine_path) / (1024**2):.2f} MB")
    
    return engine_path


def test_tensorrt_engine(engine_path, input_shape=(1, 1, 480, 640)):
    """
    Test the TensorRT engine with random input.
    
    Args:
        engine_path: Path to TensorRT engine file
        input_shape: Input tensor shape (B, C, H, W)
    """
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
    except ImportError as e:
        print(f"Error: Required package not installed: {e}")
        print("Install with: pip install tensorrt pycuda")
        return
    
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    print(f"\nTesting TensorRT engine: {engine_path}")
    
    # Load engine
    with open(engine_path, 'rb') as f:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())
    
    if engine is None:
        print("ERROR: Failed to load engine")
        return
    
    # Create execution context
    context = engine.create_execution_context()
    
    # Set input shape if using dynamic shapes
    if engine.num_optimization_profiles > 0:
        context.set_input_shape(engine.get_tensor_name(0), input_shape)
    
    # Allocate buffers
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    
    print("Engine bindings:")
    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
        shape = context.get_tensor_shape(tensor_name)
        size = trt.volume(shape)
        
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        
        bindings.append(int(device_mem))
        
        if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
            inputs.append({'host': host_mem, 'device': device_mem})
            print(f"  Input: {tensor_name}, shape: {shape}, dtype: {dtype}")
            # Fill with random data
            host_mem[:] = np.random.randn(*shape).astype(dtype).flatten()
        else:
            outputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})
            print(f"  Output: {tensor_name}, shape: {shape}, dtype: {dtype}")
    
    # Run inference
    print("\nRunning inference...")
    
    # Transfer input data to device
    for inp in inputs:
        cuda.memcpy_htod_async(inp['device'], inp['host'], stream)
    
    # Execute
    context.execute_async_v3(stream_handle=stream.handle)
    
    # Transfer predictions back
    for out in outputs:
        cuda.memcpy_dtoh_async(out['host'], out['device'], stream)
    
    # Synchronize
    stream.synchronize()
    
    print("✓ Inference successful!")
    
    # Print output statistics
    for i, out in enumerate(outputs):
        data = out['host'].reshape(out['shape'])
        print(f"\nOutput {i}:")
        print(f"  Shape: {data.shape}")
        print(f"  Min: {data.min():.6f}, Max: {data.max():.6f}, Mean: {data.mean():.6f}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Convert ONNX to TensorRT')
    parser.add_argument('--onnx', type=str, required=True,
                        help='Path to ONNX model')
    parser.add_argument('--engine', type=str, default='superpoint.trt',
                        help='Output TensorRT engine file path')
    parser.add_argument('--fp16', action='store_true',
                        help='Enable FP16 precision')
    parser.add_argument('--int8', action='store_true',
                        help='Enable INT8 precision')
    parser.add_argument('--max-batch-size', type=int, default=1,
                        help='Maximum batch size')
    parser.add_argument('--workspace', type=float, default=1.0,
                        help='Maximum workspace size in GB')
    parser.add_argument('--dynamic-shapes', action='store_true',
                        help='Enable dynamic input shapes')
    parser.add_argument('--min-height', type=int, default=240,
                        help='Minimum input height (for dynamic shapes)')
    parser.add_argument('--min-width', type=int, default=320,
                        help='Minimum input width (for dynamic shapes)')
    parser.add_argument('--opt-height', type=int, default=480,
                        help='Optimal input height (for dynamic shapes)')
    parser.add_argument('--opt-width', type=int, default=640,
                        help='Optimal input width (for dynamic shapes)')
    parser.add_argument('--max-height', type=int, default=960,
                        help='Maximum input height (for dynamic shapes)')
    parser.add_argument('--max-width', type=int, default=1280,
                        help='Maximum input width (for dynamic shapes)')
    parser.add_argument('--test', action='store_true',
                        help='Test the engine after building')
    
    args = parser.parse_args()
    
    # Set up dynamic shapes if requested
    min_shape = None
    opt_shape = None
    max_shape = None
    
    if args.dynamic_shapes:
        min_shape = (args.min_height, args.min_width)
        opt_shape = (args.opt_height, args.opt_width)
        max_shape = (args.max_height, args.max_width)
    
    # Convert to TensorRT
    engine_path = convert_onnx_to_tensorrt(
        args.onnx,
        args.engine,
        args.fp16,
        args.int8,
        args.max_batch_size,
        args.workspace,
        min_shape,
        opt_shape,
        max_shape
    )
    
    # Test the engine
    if engine_path and args.test:
        test_shape = (args.max_batch_size, 1, args.opt_height, args.opt_width)
        test_tensorrt_engine(engine_path, test_shape)


if __name__ == '__main__':
    main()
