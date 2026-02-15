#!/usr/bin/env python3
"""
Example script for running inference with TensorRT SuperPoint model.

Usage:
    python tensorrt_inference.py --engine superpoint.trt --image path/to/image.jpg
"""

import argparse
import numpy as np
import cv2
import time


class SuperPointTRT:
    """SuperPoint TensorRT inference wrapper"""
    
    def __init__(self, engine_path, output_type='dense'):
        """
        Initialize TensorRT engine.
        
        Args:
            engine_path: Path to TensorRT engine file
            output_type: 'dense' or 'keypoints'
        """
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
        except ImportError as e:
            raise ImportError(f"Required package not installed: {e}\n"
                            "Install with: pip install tensorrt pycuda")
        
        self.cuda = cuda
        self.output_type = output_type
        
        # Load engine
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        # Allocate buffers
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.output_shapes = []
        
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.input_name = tensor_name
                self.input_dtype = dtype
            else:
                self.output_shapes.append(tensor_name)
        
        print(f"✓ TensorRT engine loaded: {engine_path}")
    
    def _allocate_buffers(self, input_shape):
        """Allocate buffers for the given input shape"""
        # Clear previous allocations
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        # Set input shape
        self.context.set_input_shape(self.input_name, input_shape)
        
        # Allocate buffers
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            shape = self.context.get_tensor_shape(tensor_name)
            size = trt.volume(shape)
            
            host_mem = self.cuda.pagelocked_empty(size, dtype)
            device_mem = self.cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({
                    'host': host_mem, 
                    'device': device_mem, 
                    'shape': shape,
                    'name': tensor_name
                })
    
    def infer(self, image):
        """
        Run inference on an image.
        
        Args:
            image: Input image (numpy array, grayscale or RGB)
        
        Returns:
            Dictionary with outputs (depends on output_type)
        """
        # Preprocess image
        if len(image.shape) == 2:
            # Grayscale
            image = image[np.newaxis, np.newaxis, :, :]
        elif len(image.shape) == 3:
            # RGB to grayscale or add batch dimension
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = image[np.newaxis, np.newaxis, :, :]
        
        # Normalize to [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        input_shape = image.shape
        
        # Allocate buffers for this input shape
        self._allocate_buffers(input_shape)
        
        # Copy input data
        self.inputs[0]['host'][:] = image.flatten()
        
        # Transfer input to device
        self.cuda.memcpy_htod_async(self.inputs[0]['device'], 
                                     self.inputs[0]['host'], 
                                     self.stream)
        
        # Run inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        # Transfer outputs back
        for out in self.outputs:
            self.cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        
        self.stream.synchronize()
        
        # Parse outputs
        results = {}
        for out in self.outputs:
            data = out['host'].reshape(out['shape'])
            name = out['name']
            
            if 'scores' in name.lower() or name == 'scores':
                results['scores'] = data
            elif 'descriptor' in name.lower():
                results['descriptors'] = data
            elif 'keypoint' in name.lower():
                results['keypoints'] = data
            else:
                results[name] = data
        
        return results
    
    def extract_keypoints_from_dense(self, scores, descriptors, threshold=0.005, 
                                     top_k=None, stride=8, nms_radius=4):
        """
        Extract keypoints from dense outputs.
        
        Args:
            scores: Dense score map [B, H, W]
            descriptors: Dense descriptor map [B, D, H/8, W/8]
            threshold: Detection threshold
            top_k: Keep top-k keypoints (None for all)
            stride: Stride of the descriptor map
            nms_radius: Radius for non-maximum suppression
        
        Returns:
            List of (keypoints, scores, descriptors) for each image in batch
        """
        from scipy.ndimage import maximum_filter
        
        batch_size = scores.shape[0]
        results = []
        
        for b in range(batch_size):
            score = scores[b]
            desc = descriptors[b]
            
            # Apply NMS
            if nms_radius > 0:
                max_score = maximum_filter(score, size=nms_radius*2+1, mode='constant')
                nms_mask = (score == max_score)
                score = score * nms_mask
            
            # Find keypoints above threshold
            mask = score > threshold
            yx = np.argwhere(mask)
            
            if len(yx) == 0:
                results.append({
                    'keypoints': np.zeros((0, 2)),
                    'scores': np.zeros(0),
                    'descriptors': np.zeros((0, desc.shape[0]))
                })
                continue
            
            # Get scores
            kp_scores = score[yx[:, 0], yx[:, 1]]
            
            # Convert (y, x) to (x, y)
            keypoints = yx[:, ::-1].astype(np.float32)
            
            # Sort by score and select top-k
            if top_k is not None and len(keypoints) > top_k:
                indices = np.argsort(kp_scores)[::-1][:top_k]
                keypoints = keypoints[indices]
                kp_scores = kp_scores[indices]
            
            # Sample descriptors at keypoint locations
            h, w = score.shape
            dh, dw = desc.shape[1], desc.shape[2]
            
            # Normalize keypoints to descriptor map coordinates
            kp_y = (keypoints[:, 1] / h * dh).astype(np.int32)
            kp_x = (keypoints[:, 0] / w * dw).astype(np.int32)
            
            # Clip to valid range
            kp_y = np.clip(kp_y, 0, dh - 1)
            kp_x = np.clip(kp_x, 0, dw - 1)
            
            # Extract descriptors
            kp_descriptors = desc[:, kp_y, kp_x].T
            
            results.append({
                'keypoints': keypoints,
                'scores': kp_scores,
                'descriptors': kp_descriptors
            })
        
        return results


def visualize_keypoints(image, keypoints, scores=None, radius=3):
    """Draw keypoints on image"""
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()
    
    for i, kp in enumerate(keypoints):
        x, y = int(kp[0]), int(kp[1])
        
        # Color by score if available
        if scores is not None:
            color_val = int(255 * min(scores[i] / 0.05, 1.0))
            color = (0, color_val, 255 - color_val)
        else:
            color = (0, 255, 0)
        
        cv2.circle(vis, (x, y), radius, color, -1)
    
    return vis


def main():
    parser = argparse.ArgumentParser(description='SuperPoint TensorRT Inference')
    parser.add_argument('--engine', type=str, required=True,
                        help='Path to TensorRT engine')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--output', type=str, default='output_keypoints.jpg',
                        help='Output visualization path')
    parser.add_argument('--type', type=str, default='dense',
                        choices=['dense', 'keypoints'],
                        help='Model output type')
    parser.add_argument('--threshold', type=float, default=0.005,
                        help='Detection threshold')
    parser.add_argument('--top-k', type=int, default=1000,
                        help='Maximum number of keypoints')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run benchmark')
    
    args = parser.parse_args()
    
    # Load image
    print(f"Loading image: {args.image}")
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image: {args.image}")
        return
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(f"Image shape: {gray.shape}")
    
    # Initialize TensorRT model
    print(f"Loading TensorRT engine...")
    model = SuperPointTRT(args.engine, args.type)
    
    # Warm up
    print("Warming up...")
    for _ in range(5):
        _ = model.infer(gray)
    
    # Run inference
    print("Running inference...")
    start_time = time.time()
    outputs = model.infer(gray)
    inference_time = (time.time() - start_time) * 1000
    
    print(f"✓ Inference time: {inference_time:.2f} ms")
    
    # Process outputs
    if args.type == 'dense':
        print("\nExtracting keypoints from dense outputs...")
        results = model.extract_keypoints_from_dense(
            outputs['scores'],
            outputs['descriptors'],
            threshold=args.threshold,
            top_k=args.top_k
        )
        keypoints = results[0]['keypoints']
        scores = results[0]['scores']
        descriptors = results[0]['descriptors']
    else:
        # Keypoints mode
        keypoints = outputs['keypoints'][0]
        scores = outputs['scores'][0]
        descriptors = outputs['descriptors'][0]
        
        # Filter out padding
        valid_mask = keypoints[:, 0] >= 0
        keypoints = keypoints[valid_mask]
        scores = scores[valid_mask]
        descriptors = descriptors[valid_mask]
    
    print(f"\nDetected {len(keypoints)} keypoints")
    if len(keypoints) > 0:
        print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        print(f"Descriptor shape: {descriptors.shape}")
    
    # Visualize
    print(f"\nSaving visualization to: {args.output}")
    vis = visualize_keypoints(image, keypoints, scores)
    cv2.imwrite(args.output, vis)
    
    # Benchmark
    if args.benchmark:
        print("\n" + "="*50)
        print("Running benchmark...")
        num_iterations = 100
        times = []
        
        for i in range(num_iterations):
            start = time.time()
            _ = model.infer(gray)
            times.append((time.time() - start) * 1000)
        
        times = np.array(times)
        print(f"Iterations: {num_iterations}")
        print(f"Mean: {times.mean():.2f} ms")
        print(f"Std: {times.std():.2f} ms")
        print(f"Min: {times.min():.2f} ms")
        print(f"Max: {times.max():.2f} ms")
        print(f"FPS: {1000 / times.mean():.1f}")
    
    print("\n✓ Done!")


if __name__ == '__main__':
    main()
