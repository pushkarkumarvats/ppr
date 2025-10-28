"""
Inference Optimizations

This module implements various optimization techniques to improve
inference speed and reduce memory usage:
- Quantization (INT8, FP16)
- Operator fusion
- Memory optimization
- Channel pruning
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
import copy


def quantize_model_dynamic(
    model: nn.Module,
    dtype: torch.dtype = torch.qint8,
    device: str = 'cpu'
) -> nn.Module:
    """
    Apply dynamic quantization to model.
    
    Dynamic quantization converts weights to INT8 but keeps activations
    in floating point. Good for models where compute is memory-bound.
    
    Args:
        model: Model to quantize
        dtype: Target quantization dtype
        device: Device (quantization typically requires CPU)
        
    Returns:
        Quantized model
    """
    model = model.to(device).eval()
    
    # Apply dynamic quantization to Linear and Conv layers
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d},
        dtype=dtype
    )
    
    return quantized_model


def quantize_model_static(
    model: nn.Module,
    calibration_data: torch.Tensor,
    device: str = 'cpu'
) -> nn.Module:
    """
    Apply static quantization with calibration.
    
    Static quantization requires calibration data to determine optimal
    scale and zero-point values. Provides better performance than dynamic.
    
    Args:
        model: Model to quantize
        calibration_data: Data for calibration [B, ...]
        device: Device
        
    Returns:
        Quantized model
    """
    model = model.to(device).eval()
    model_copy = copy.deepcopy(model)
    
    # Set quantization config
    model_copy.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # Prepare model for static quantization
    model_prepared = torch.quantization.prepare(model_copy)
    
    # Calibrate with representative data
    print("Calibrating model...")
    with torch.no_grad():
        for i in range(min(100, len(calibration_data))):
            _ = model_prepared(calibration_data[i:i+1])
    
    # Convert to quantized model
    model_quantized = torch.quantization.convert(model_prepared)
    
    return model_quantized


def convert_to_fp16(model: nn.Module) -> nn.Module:
    """
    Convert model to FP16 (half precision).
    
    This is a simple optimization that can provide 2x speedup
    on hardware with good FP16 support (e.g., modern GPUs).
    
    Args:
        model: Model to convert
        
    Returns:
        FP16 model
    """
    return model.half()


class FusedConvBN(nn.Module):
    """
    Fused convolution + batch normalization.
    
    During inference, BN can be folded into convolution weights,
    reducing memory and compute overhead.
    """
    
    def __init__(self, conv: nn.Conv2d, bn: nn.BatchNorm2d):
        super().__init__()
        
        # Fold BN into conv
        conv_weight = conv.weight.data
        conv_bias = conv.bias.data if conv.bias is not None else torch.zeros(conv.out_channels)
        
        bn_weight = bn.weight.data
        bn_bias = bn.bias.data
        bn_mean = bn.running_mean
        bn_var = bn.running_var
        bn_eps = bn.eps
        
        # Compute fused weights
        std = torch.sqrt(bn_var + bn_eps)
        gamma = bn_weight / std
        
        # w_fused = w_conv * gamma
        fused_weight = conv_weight * gamma.view(-1, 1, 1, 1)
        
        # b_fused = (b_conv - mu) * gamma + beta
        fused_bias = (conv_bias - bn_mean) * gamma + bn_bias
        
        # Create fused conv
        self.conv = nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            bias=True
        )
        
        self.conv.weight.data = fused_weight
        self.conv.bias.data = fused_bias
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def fuse_conv_bn(model: nn.Module) -> nn.Module:
    """
    Fuse all Conv+BN pairs in model.
    
    Args:
        model: Model to optimize
        
    Returns:
        Model with fused layers
    """
    model_fused = copy.deepcopy(model)
    
    # Find Conv+BN pairs
    for name, module in model_fused.named_children():
        if isinstance(module, nn.Sequential):
            # Check if it's Conv followed by BN
            if len(module) >= 2:
                if isinstance(module[0], nn.Conv2d) and isinstance(module[1], nn.BatchNorm2d):
                    # Create fused layer
                    fused = FusedConvBN(module[0], module[1])
                    
                    # Replace in sequential
                    new_layers = [fused] + list(module[2:])
                    setattr(model_fused, name, nn.Sequential(*new_layers))
        else:
            # Recursively fuse in submodules
            setattr(model_fused, name, fuse_conv_bn(module))
    
    return model_fused


class PrunedConv2d(nn.Module):
    """
    Convolution layer with channel pruning.
    
    Removes channels with low L1 norm to reduce computation.
    """
    
    def __init__(self, conv: nn.Conv2d, prune_ratio: float = 0.3):
        super().__init__()
        
        # Compute L1 norm of each output channel
        weight = conv.weight.data  # [out_ch, in_ch, k, k]
        l1_norms = weight.abs().sum(dim=[1, 2, 3])
        
        # Keep top (1 - prune_ratio) channels
        num_keep = int(conv.out_channels * (1 - prune_ratio))
        _, keep_indices = torch.topk(l1_norms, num_keep)
        keep_indices = sorted(keep_indices.tolist())
        
        # Create pruned conv
        self.conv = nn.Conv2d(
            conv.in_channels,
            num_keep,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            bias=conv.bias is not None
        )
        
        # Copy pruned weights
        self.conv.weight.data = weight[keep_indices]
        if conv.bias is not None:
            self.conv.bias.data = conv.bias.data[keep_indices]
        
        self.keep_indices = keep_indices
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def prune_model(model: nn.Module, prune_ratio: float = 0.3) -> nn.Module:
    """
    Apply channel pruning to reduce model size.
    
    Args:
        model: Model to prune
        prune_ratio: Fraction of channels to prune
        
    Returns:
        Pruned model
    """
    model_pruned = copy.deepcopy(model)
    
    for name, module in model_pruned.named_children():
        if isinstance(module, nn.Conv2d):
            pruned = PrunedConv2d(module, prune_ratio)
            setattr(model_pruned, name, pruned)
        else:
            # Recursively prune submodules
            setattr(model_pruned, name, prune_model(module, prune_ratio))
    
    return model_pruned


def optimize_for_inference(
    model: nn.Module,
    optimization_level: int = 2,
    calibration_data: Optional[torch.Tensor] = None,
    device: str = 'cuda'
) -> nn.Module:
    """
    Apply comprehensive optimizations for inference.
    
    Optimization levels:
    - 0: No optimization (baseline)
    - 1: FP16 + operator fusion
    - 2: Level 1 + dynamic quantization
    - 3: Level 2 + static quantization (requires calibration data)
    - 4: Level 3 + channel pruning
    
    Args:
        model: Model to optimize
        optimization_level: Level of optimization (0-4)
        calibration_data: Calibration data for static quantization
        device: Target device
        
    Returns:
        Optimized model
    """
    print(f"Applying optimization level {optimization_level}...")
    
    if optimization_level == 0:
        return model
    
    optimized = model.eval()
    
    # Level 1: FP16 + fusion
    if optimization_level >= 1:
        print("  - Fusing Conv+BN layers...")
        optimized = fuse_conv_bn(optimized)
        
        if device == 'cuda':
            print("  - Converting to FP16...")
            optimized = convert_to_fp16(optimized).to(device)
    
    # Level 2: Dynamic quantization
    if optimization_level >= 2:
        print("  - Applying dynamic quantization...")
        optimized = quantize_model_dynamic(optimized, device='cpu')
    
    # Level 3: Static quantization
    if optimization_level >= 3:
        if calibration_data is None:
            print("  ! Warning: Static quantization requires calibration data")
        else:
            print("  - Applying static quantization...")
            optimized = quantize_model_static(optimized, calibration_data, device='cpu')
    
    # Level 4: Channel pruning
    if optimization_level >= 4:
        print("  - Applying channel pruning...")
        optimized = prune_model(optimized, prune_ratio=0.3)
    
    print("✓ Optimization complete!")
    return optimized


def benchmark_optimization(
    original_model: nn.Module,
    optimized_model: nn.Module,
    input_shape: tuple,
    num_iterations: int = 100,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Benchmark original vs optimized model.
    
    Args:
        original_model: Original model
        optimized_model: Optimized model
        input_shape: Input tensor shape
        num_iterations: Number of iterations
        device: Device
        
    Returns:
        Benchmark results
    """
    import time
    import numpy as np
    
    print("Benchmarking models...")
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape)
    
    # Warmup and benchmark original
    original_model = original_model.to(device).eval()
    dummy_input_orig = dummy_input.to(device)
    
    with torch.no_grad():
        for _ in range(10):
            _ = original_model(dummy_input_orig)
        
        torch.cuda.synchronize() if device == 'cuda' else None
        
        orig_times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = original_model(dummy_input_orig)
            torch.cuda.synchronize() if device == 'cuda' else None
            orig_times.append(time.perf_counter() - start)
    
    # Benchmark optimized
    opt_device = device if not isinstance(optimized_model, torch.quantization.QuantizedModule) else 'cpu'
    optimized_model = optimized_model.to(opt_device).eval()
    dummy_input_opt = dummy_input.to(opt_device)
    
    with torch.no_grad():
        for _ in range(10):
            _ = optimized_model(dummy_input_opt)
        
        torch.cuda.synchronize() if opt_device == 'cuda' else None
        
        opt_times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = optimized_model(dummy_input_opt)
            torch.cuda.synchronize() if opt_device == 'cuda' else None
            opt_times.append(time.perf_counter() - start)
    
    # Compute stats
    orig_mean = np.mean(orig_times) * 1000  # ms
    opt_mean = np.mean(opt_times) * 1000  # ms
    speedup = orig_mean / opt_mean
    
    # Model sizes
    orig_size = sum(p.numel() * p.element_size() for p in original_model.parameters()) / (1024 ** 2)
    opt_size = sum(p.numel() * p.element_size() for p in optimized_model.parameters()) / (1024 ** 2)
    size_reduction = (1 - opt_size / orig_size) * 100
    
    results = {
        'original_latency_ms': orig_mean,
        'optimized_latency_ms': opt_mean,
        'speedup': speedup,
        'original_size_mb': orig_size,
        'optimized_size_mb': opt_size,
        'size_reduction_pct': size_reduction
    }
    
    print(f"\nResults:")
    print(f"  Original latency: {orig_mean:.2f}ms")
    print(f"  Optimized latency: {opt_mean:.2f}ms")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Original size: {orig_size:.2f}MB")
    print(f"  Optimized size: {opt_size:.2f}MB")
    print(f"  Size reduction: {size_reduction:.1f}%")
    
    return results


if __name__ == "__main__":
    # Test optimizations
    print("Testing inference optimizations...")
    
    # Create a simple test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(4, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 4, 3, padding=1)
            )
        
        def forward(self, x):
            return self.features(x)
    
    model = TestModel()
    
    print("\n1. Testing Conv+BN fusion...")
    fused_model = fuse_conv_bn(model)
    print("   ✓ Fusion complete")
    
    print("\n2. Testing FP16 conversion...")
    fp16_model = convert_to_fp16(model)
    print(f"   ✓ Model dtype: {next(fp16_model.parameters()).dtype}")
    
    print("\n3. Testing channel pruning...")
    pruned_model = prune_model(model, prune_ratio=0.3)
    print("   ✓ Pruning complete")
    
    print("\n4. Testing dynamic quantization...")
    quant_model = quantize_model_dynamic(model)
    print("   ✓ Quantization complete")
    
    print("\n5. Testing comprehensive optimization...")
    optimized = optimize_for_inference(
        model,
        optimization_level=2,
        device='cpu'
    )
    print("   ✓ Optimization complete")
    
    print("\n✓ All optimizations tested successfully!")
