#!/usr/bin/env python3
"""
ONNX Runtime inference for SuperPoint.
Compatible with all GPUs including GTX 1060 (and newer).

Usage:
    python onnx_inference.py --model superpoint.onnx --image test.jpg
"""

import argparse
import numpy as np
import cv2
import time


class SuperPointONNX:
    """SuperPoint ONNX Runtime inference wrapper"""
    
    def __init__(self, model_path):
        """
        Initialize ONNX Runtime session.
        
        Args:
            model_path: Path to ONNX model file
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("ONNX Runtime not installed!\n"
                            "Install with: pip install onnxruntime-gpu")
        
        self.ort = ort
        
        # Create session with GPU support
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # Get model info
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        active_provider = self.session.get_providers()[0]
        print(f"✓ ONNX Runtime initialized")
        print(f"  Provider: {active_provider}")
        print(f"  Input: {self.input_name}")
        print(f"  Outputs: {self.output_names}")
    
    def infer(self, image):
        """
        Run inference on an image.
        
        Args:
            image: Input image (numpy array, grayscale or RGB)
        
        Returns:
            Dictionary with 'scores' and 'descriptors'
        """
        # Preprocess
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Normalize to [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        # Add batch and channel dimensions
        input_tensor = image[np.newaxis, np.newaxis, :, :]
        
        # Run inference
        outputs = self.session.run(
            self.output_names,
            {self.input_name: input_tensor}
        )
        
        return {
            'scores': outputs[0],
            'descriptors': outputs[1]
        }
    
    def extract_keypoints(self, scores, descriptors, threshold=0.005, 
                         nms_radius=4, top_k=1000, border=4):
        """
        Extract keypoints from dense outputs.
        
        Args:
            scores: Dense score map [B, H, W]
            descriptors: Dense descriptor map [B, D, H/8, W/8]
            threshold: Detection threshold
            nms_radius: NMS radius
            top_k: Maximum number of keypoints
            border: Border pixels to remove
        
        Returns:
            Dictionary with keypoints, scores, and descriptors
        """
        from scipy.ndimage import maximum_filter
        
        score = scores[0]  # Remove batch dimension
        desc = descriptors[0]  # [D, H/8, W/8]
        
        # Apply NMS
        if nms_radius > 0:
            max_score = maximum_filter(
                score, size=nms_radius*2+1, mode='constant', cval=0
            )
            nms_mask = (score == max_score)
            score = score * nms_mask
        
        # Remove borders
        if border > 0:
            score[:border, :] = 0
            score[-border:, :] = 0
            score[:, :border] = 0
            score[:, -border:] = 0
        
        # Find keypoints above threshold
        mask = score > threshold
        yx = np.argwhere(mask)
        
        if len(yx) == 0:
            return {
                'keypoints': np.zeros((0, 2), dtype=np.float32),
                'scores': np.zeros(0, dtype=np.float32),
                'descriptors': np.zeros((0, desc.shape[0]), dtype=np.float32)
            }
        
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
        kp_descriptors = desc[:, kp_y, kp_x].T  # [N, D]
        
        return {
            'keypoints': keypoints,
            'scores': kp_scores,
            'descriptors': kp_descriptors
        }


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
    parser = argparse.ArgumentParser(description='SuperPoint ONNX Runtime Inference')
    parser.add_argument('--model', type=str, default='superpoint.onnx',
                        help='Path to ONNX model')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--output', type=str, default='output_onnx.jpg',
                        help='Output visualization path')
    parser.add_argument('--threshold', type=float, default=0.005,
                        help='Detection threshold')
    parser.add_argument('--nms-radius', type=int, default=4,
                        help='NMS radius')
    parser.add_argument('--top-k', type=int, default=1000,
                        help='Maximum number of keypoints')
    parser.add_argument('--border', type=int, default=4,
                        help='Border pixels to remove')
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
    
    # Initialize model
    print(f"\nLoading ONNX model: {args.model}")
    model = SuperPointONNX(args.model)
    
    # Warm up
    print("Warming up...")
    for _ in range(5):
        _ = model.infer(gray)
    
    # Run inference
    print("\nRunning inference...")
    start_time = time.time()
    outputs = model.infer(gray)
    inference_time = (time.time() - start_time) * 1000
    
    print(f"✓ Inference time: {inference_time:.2f} ms")
    print(f"  Scores shape: {outputs['scores'].shape}")
    print(f"  Descriptors shape: {outputs['descriptors'].shape}")
    
    # Extract keypoints
    print("\nExtracting keypoints...")
    start_time = time.time()
    result = model.extract_keypoints(
        outputs['scores'],
        outputs['descriptors'],
        threshold=args.threshold,
        nms_radius=args.nms_radius,
        top_k=args.top_k,
        border=args.border
    )
    extraction_time = (time.time() - start_time) * 1000
    
    keypoints = result['keypoints']
    scores = result['scores']
    descriptors = result['descriptors']
    
    print(f"✓ Extraction time: {extraction_time:.2f} ms")
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
            outputs = model.infer(gray)
            result = model.extract_keypoints(
                outputs['scores'],
                outputs['descriptors'],
                threshold=args.threshold,
                nms_radius=args.nms_radius,
                top_k=args.top_k,
                border=args.border
            )
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
