#!/usr/bin/env python3
"""
Script to convert SuperPoint PyTorch model to ONNX format.
This creates an ONNX model suitable for TensorRT conversion.

Usage:
    python convert_to_onnx.py --weights weights/superpoint_v6_from_tf.pth --output superpoint.onnx
"""

import argparse
import torch
import torch.nn as nn
from superpoint_pytorch import SuperPoint, batched_nms, sample_descriptors
from types import SimpleNamespace


class SuperPointONNX(nn.Module):
    """
    Wrapper for SuperPoint model for ONNX export.
    Outputs dense feature maps instead of sparse keypoints for better ONNX compatibility.
    """
    
    def __init__(self, superpoint_model):
        super().__init__()
        self.backbone = superpoint_model.backbone
        self.detector = superpoint_model.detector
        self.descriptor = superpoint_model.descriptor
        self.conf = superpoint_model.conf
        self.stride = superpoint_model.stride
        
    def forward(self, image):
        """
        Forward pass that outputs dense feature maps.
        
        Args:
            image: Input image tensor [B, C, H, W], can be RGB or grayscale
        
        Returns:
            scores: Detection scores [B, H, W]
            descriptors: Dense descriptor map [B, 256, H/8, W/8]
        """
        # Convert RGB to grayscale if needed
        if image.shape[1] == 3:
            scale = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            image = (image * scale).sum(1, keepdim=True)
        
        # Extract features
        features = self.backbone(image)
        
        # Dense descriptors (normalized)
        descriptors_dense = torch.nn.functional.normalize(
            self.descriptor(features), p=2, dim=1
        )
        
        # Detection scores
        scores = self.detector(features)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, self.stride, self.stride)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(
            b, h * self.stride, w * self.stride
        )
        
        # Note: NMS is NOT applied here for better TensorRT compatibility
        # Apply NMS in post-processing using tensorrt_inference.py
        
        return scores, descriptors_dense


class SuperPointONNXWithKeypoints(nn.Module):
    """
    Wrapper that extracts top-k keypoints for ONNX export.
    This version outputs a fixed number of keypoints.
    """
    
    def __init__(self, superpoint_model, max_num_keypoints=1024):
        super().__init__()
        self.backbone = superpoint_model.backbone
        self.detector = superpoint_model.detector
        self.descriptor = superpoint_model.descriptor
        self.conf = superpoint_model.conf
        self.stride = superpoint_model.stride
        self.max_num_keypoints = max_num_keypoints
        
    def forward(self, image):
        """
        Forward pass that outputs fixed number of keypoints.
        
        Args:
            image: Input image tensor [B, C, H, W]
        
        Returns:
            keypoints: [B, max_num_keypoints, 2] (x, y coordinates, padded with -1)
            scores: [B, max_num_keypoints] (confidence scores, padded with 0)
            descriptors: [B, max_num_keypoints, 256] (descriptors, padded with 0)
        """
        b, _, h, w = image.shape
        
        # Convert RGB to grayscale if needed
        if image.shape[1] == 3:
            scale = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            image = (image * scale).sum(1, keepdim=True)
        
        # Extract features
        features = self.backbone(image)
        
        # Dense descriptors
        descriptors_dense = torch.nn.functional.normalize(
            self.descriptor(features), p=2, dim=1
        )
        
        # Detection scores
        scores_map = self.detector(features)
        scores_map = torch.nn.functional.softmax(scores_map, 1)[:, :-1]
        _, _, h_s, w_s = scores_map.shape
        scores_map = scores_map.permute(0, 2, 3, 1).reshape(b, h_s, w_s, self.stride, self.stride)
        scores_map = scores_map.permute(0, 1, 3, 2, 4).reshape(
            b, h_s * self.stride, w_s * self.stride
        )
        
        # Note: NMS is NOT applied here for better TensorRT compatibility
        # Post-processing will need to apply NMS if needed
        
        # Remove borders
        if self.conf.remove_borders:
            pad = self.conf.remove_borders
            scores_map[:, :pad] = -1
            scores_map[:, :, :pad] = -1
            scores_map[:, -pad:] = -1
            scores_map[:, :, -pad:] = -1
        
        # Extract keypoints for each batch
        all_keypoints = []
        all_scores = []
        all_descriptors = []
        
        for i in range(b):
            score = scores_map[i]
            
            # Find keypoints above threshold
            yx = torch.nonzero(score > self.conf.detection_threshold)
            scores_valid = score[yx[:, 0], yx[:, 1]]
            
            # Convert (y, x) to (x, y)
            keypoints = yx.flip(1).float()
            
            # Get top-k
            num_keypoints = min(len(keypoints), self.max_num_keypoints)
            if num_keypoints > 0:
                if num_keypoints < len(keypoints):
                    # Select top-k
                    scores_valid, indices = torch.topk(scores_valid, num_keypoints, dim=0)
                    keypoints = keypoints[indices]
                
                # Sample descriptors
                desc = sample_descriptors(
                    keypoints[None], descriptors_dense[i:i+1], self.stride
                )
                desc = desc.squeeze(0).transpose(0, 1)  # [N, 256]
            else:
                # No keypoints found
                desc = torch.zeros((0, self.conf.descriptor_dim), 
                                   device=image.device, dtype=image.dtype)
            
            # Pad to max_num_keypoints
            if num_keypoints < self.max_num_keypoints:
                pad_size = self.max_num_keypoints - num_keypoints
                keypoints = torch.cat([
                    keypoints,
                    -torch.ones((pad_size, 2), device=image.device, dtype=keypoints.dtype)
                ], dim=0)
                scores_valid = torch.cat([
                    scores_valid,
                    torch.zeros(pad_size, device=image.device, dtype=scores_valid.dtype)
                ], dim=0)
                desc = torch.cat([
                    desc,
                    torch.zeros((pad_size, self.conf.descriptor_dim), 
                               device=image.device, dtype=desc.dtype)
                ], dim=0)
            
            all_keypoints.append(keypoints)
            all_scores.append(scores_valid)
            all_descriptors.append(desc)
        
        keypoints_out = torch.stack(all_keypoints, dim=0)
        scores_out = torch.stack(all_scores, dim=0)
        descriptors_out = torch.stack(all_descriptors, dim=0)
        
        return keypoints_out, scores_out, descriptors_out


def convert_to_onnx(weights_path, output_path, image_height=480, image_width=640,
                    export_type='dense', max_num_keypoints=1024, opset_version=11):
    """
    Convert SuperPoint PyTorch model to ONNX format.
    
    Args:
        weights_path: Path to PyTorch weights file
        output_path: Output path for ONNX model
        image_height: Input image height
        image_width: Input image width
        export_type: 'dense' for dense feature maps or 'keypoints' for fixed keypoints
        max_num_keypoints: Maximum number of keypoints (only for 'keypoints' mode)
        opset_version: ONNX opset version
    """
    # Load the PyTorch model
    # Check CUDA availability and compatibility
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    
    use_cuda = False
    if torch.cuda.is_available():
        try:
            # Try to create a small tensor on CUDA to test compatibility
            _ = torch.zeros(1).cuda()
            use_cuda = True
            print(f"Using device: cuda")
        except Exception as e:
            print(f"CUDA available but not compatible: {e}")
            print(f"Using device: cpu (ONNX/TensorRT conversion works fine on CPU)")
    else:
        print(f"Using device: cpu (ONNX/TensorRT conversion works fine on CPU)")
    
    device = torch.device('cuda' if use_cuda else 'cpu')
    
    # Load SuperPoint model
    superpoint = SuperPoint(
        nms_radius=4,
        max_num_keypoints=None,
        detection_threshold=0.005,
        remove_borders=4,
        descriptor_dim=256,
    )
    
    # Load weights
    print(f"Loading weights from {weights_path}")
    state_dict = torch.load(weights_path, map_location='cpu')
    superpoint.load_state_dict(state_dict)
    superpoint.eval()
    if use_cuda:
        superpoint.to(device)
    
    # For ONNX export, use CPU to avoid compatibility issues
    export_device = 'cpu'
    if use_cuda:
        print("Note: Using CPU for ONNX export (more stable)")
        superpoint.to('cpu')
    
    # Create ONNX wrapper model
    if export_type == 'dense':
        print("Creating ONNX model with dense outputs...")
        onnx_model = SuperPointONNX(superpoint)
        dummy_input = torch.randn(1, 1, image_height, image_width)
        input_names = ['image']
        output_names = ['scores', 'descriptors']
        dynamic_axes = {
            'image': {0: 'batch', 2: 'height', 3: 'width'},
            'scores': {0: 'batch', 1: 'height', 2: 'width'},
            'descriptors': {0: 'batch', 2: 'height', 3: 'width'}
        }
    elif export_type == 'keypoints':
        print(f"Creating ONNX model with top-{max_num_keypoints} keypoints...")
        onnx_model = SuperPointONNXWithKeypoints(superpoint, max_num_keypoints)
        dummy_input = torch.randn(1, 1, image_height, image_width)
        input_names = ['image']
        output_names = ['keypoints', 'scores', 'descriptors']
        dynamic_axes = {
            'image': {0: 'batch', 2: 'height', 3: 'width'},
            'keypoints': {0: 'batch'},
            'scores': {0: 'batch'},
            'descriptors': {0: 'batch'}
        }
    else:
        raise ValueError(f"Unknown export_type: {export_type}")
    
    onnx_model.eval()
    # Keep model on CPU for export
    
    # Test forward pass
    print("Testing forward pass...")
    with torch.no_grad():
        outputs = onnx_model(dummy_input)
        if export_type == 'dense':
            print(f"  Scores shape: {outputs[0].shape}")
            print(f"  Descriptors shape: {outputs[1].shape}")
        else:
            print(f"  Keypoints shape: {outputs[0].shape}")
            print(f"  Scores shape: {outputs[1].shape}")
            print(f"  Descriptors shape: {outputs[2].shape}")
    
    # Export to ONNX
    print(f"\nExporting to ONNX: {output_path}")
    torch.onnx.export(
        onnx_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )
    
    print("✓ ONNX export successful!")
    
    # Verify the ONNX model
    print("\nVerifying ONNX model...")
    try:
        import onnx
        onnx_model_check = onnx.load(output_path)
        onnx.checker.check_model(onnx_model_check)
        print("✓ ONNX model is valid!")
    except ImportError:
        print("Note: Install 'onnx' package to verify the exported model")
    except Exception as e:
        print(f"Warning: ONNX verification failed: {e}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Convert SuperPoint to ONNX')
    parser.add_argument('--weights', type=str, 
                        default='weights/superpoint_v6_from_tf.pth',
                        help='Path to PyTorch weights file')
    parser.add_argument('--output', type=str, 
                        default='superpoint.onnx',
                        help='Output ONNX file path')
    parser.add_argument('--height', type=int, default=480,
                        help='Input image height')
    parser.add_argument('--width', type=int, default=640,
                        help='Input image width')
    parser.add_argument('--type', type=str, default='dense',
                        choices=['dense', 'keypoints'],
                        help='Export type: dense (feature maps) or keypoints (fixed number)')
    parser.add_argument('--max-keypoints', type=int, default=1024,
                        help='Maximum number of keypoints (for keypoints mode)')
    parser.add_argument('--opset', type=int, default=11,
                        help='ONNX opset version')
    
    args = parser.parse_args()
    
    convert_to_onnx(
        args.weights,
        args.output,
        args.height,
        args.width,
        args.type,
        args.max_keypoints,
        args.opset
    )


if __name__ == '__main__':
    main()
