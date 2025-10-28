"""Real-Time Inference Pipeline for RAW Image Enhancement"""

import torch
import torch.nn as nn
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict
from pathlib import Path

from models.raw_diffusion_unet import RAWVAE, RAWDiffusionUNet, DDPMScheduler
from models.consistency_distillation import ConsistencyModel, AdaptiveStepSelector
from models.optical_flow import AlignmentModule
from models.lens_aberration_module import AberrationCorrectionModule
from data.preprocessing import raw_to_srgb_simple


class LatencyTracker:
    """Track inference latency for each pipeline stage."""
    
    def __init__(self):
        self.timings = defaultdict(list)
        self.current_stage = None
        self.start_time = None
    
    def start(self, stage_name: str):
        """Start timing a stage."""
        self.current_stage = stage_name
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self.start_time = time.perf_counter()
    
    def end(self):
        """End timing the current stage."""
        if self.current_stage is None:
            return
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = (time.perf_counter() - self.start_time) * 1000  # Convert to ms
        self.timings[self.current_stage].append(elapsed)
        
        self.current_stage = None
        self.start_time = None
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get timing statistics."""
        stats = {}
        for stage, times in self.timings.items():
            times_array = np.array(times)
            stats[stage] = {
                'mean': float(np.mean(times_array)),
                'std': float(np.std(times_array)),
                'min': float(np.min(times_array)),
                'max': float(np.max(times_array)),
                'p50': float(np.percentile(times_array, 50)),
                'p95': float(np.percentile(times_array, 95)),
                'p99': float(np.percentile(times_array, 99)),
            }
        return stats
    
    def reset(self):
        """Reset all timings."""
        self.timings.clear()


class RealTimePipeline:
    """Real-time inference pipeline with optimizations."""
    
    def __init__(
        self,
        vae: RAWVAE,
        consistency_model: ConsistencyModel,
        alignment: AlignmentModule,
        aberration: Optional[AberrationCorrectionModule] = None,
        device: str = 'cuda',
        num_inference_steps: int = 2,
        use_adaptive_steps: bool = True,
        enable_profiling: bool = False
    ):
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.use_adaptive_steps = use_adaptive_steps
        self.enable_profiling = enable_profiling
        
        # Models
        self.vae = vae.to(device).eval()
        self.consistency_model = consistency_model.to(device).eval()
        self.alignment = alignment.to(device).eval()
        self.aberration = aberration.to(device).eval() if aberration else None
        
        # Disable gradients
        self.vae.requires_grad_(False)
        self.consistency_model.requires_grad_(False)
        self.alignment.requires_grad_(False)
        if self.aberration is not None:
            self.aberration.requires_grad_(False)
        
        # Adaptive step selector
        if use_adaptive_steps:
            self.step_selector = AdaptiveStepSelector(
                min_steps=2,
                max_steps=4,
                complexity_threshold=0.3
            )
        else:
            self.step_selector = None
        
        # Latency tracker
        self.tracker = LatencyTracker() if enable_profiling else None
        
        # Warmup
        self._warmup()
    
    def _warmup(self, num_iterations: int = 5):
        """Warmup the pipeline to ensure stable timing."""
        print("Warming up pipeline...")
        dummy_burst = torch.randn(1, 8, 4, 512, 512, device=self.device)
        
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = self.forward(dummy_burst)
        
        print("Warmup complete!")
    
    @torch.no_grad()
    def forward(
        self,
        burst: torch.Tensor,
        metadata: Optional[Dict] = None,
        return_intermediate: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Run inference on burst.
        
        Args:
            burst: Input burst [B, T, 4, H, W]
            metadata: Optional lens metadata
            return_intermediate: If True, return intermediate results
            
        Returns:
            Dictionary with outputs and timing info
        """
        B, T = burst.shape[:2]
        results = {}
        
        # Stage 1: Burst alignment
        if self.tracker:
            self.tracker.start('alignment')
        
        aligned = self.alignment(burst, reference_idx=T // 2)
        
        if self.tracker:
            self.tracker.end()
        
        # Stage 2: Merge aligned frames
        if self.tracker:
            self.tracker.start('merge')
        
        merged = aligned.mean(dim=1)  # [B, 4, H, W]
        
        if self.tracker:
            self.tracker.end()
        
        # Stage 3: Aberration correction
        if self.aberration is not None and metadata is not None:
            if self.tracker:
                self.tracker.start('aberration')
            
            merged = self.aberration(merged, metadata)
            
            if self.tracker:
                self.tracker.end()
        
        # Stage 4: VAE encoding
        if self.tracker:
            self.tracker.start('vae_encode')
        
        condition_latent = self.vae.encode(merged).sample()
        
        if self.tracker:
            self.tracker.end()
        
        # Stage 5: Determine number of steps
        if self.step_selector is not None:
            if self.tracker:
                self.tracker.start('step_selection')
            
            num_steps = self.step_selector.select_steps(condition_latent)
            
            if self.tracker:
                self.tracker.end()
        else:
            num_steps = self.num_inference_steps
        
        # Stage 6: Consistency model inference
        if self.tracker:
            self.tracker.start('consistency')
        
        enhanced_latent = self.consistency_model.generate(
            condition=condition_latent,
            num_steps=num_steps
        )
        
        if self.tracker:
            self.tracker.end()
        
        # Stage 7: VAE decoding
        if self.tracker:
            self.tracker.start('vae_decode')
        
        enhanced = self.vae.decode(enhanced_latent)
        
        if self.tracker:
            self.tracker.end()
        
        # Prepare results
        results['enhanced'] = enhanced
        results['num_steps'] = num_steps
        
        if return_intermediate:
            results['aligned'] = aligned
            results['merged'] = merged
            results['condition_latent'] = condition_latent
            results['enhanced_latent'] = enhanced_latent
        
        return results
    
    def benchmark(
        self,
        burst_sizes: List[Tuple[int, int]] = [(512, 512), (1024, 1024), (2048, 2048)],
        burst_lengths: List[int] = [4, 8, 16],
        num_iterations: int = 100,
        warmup_iterations: int = 10
    ) -> Dict:
        """
        Comprehensive benchmarking.
        
        Args:
            burst_sizes: List of (H, W) sizes to test
            burst_lengths: List of burst lengths to test
            num_iterations: Number of iterations per configuration
            warmup_iterations: Number of warmup iterations
            
        Returns:
            Benchmark results
        """
        print("Running comprehensive benchmark...")
        results = {}
        
        for H, W in burst_sizes:
            for T in burst_lengths:
                config_name = f"{H}x{W}_T{T}"
                print(f"\nBenchmarking {config_name}...")
                
                # Create dummy data
                dummy_burst = torch.randn(1, T, 4, H, W, device=self.device)
                
                # Warmup
                for _ in range(warmup_iterations):
                    _ = self.forward(dummy_burst)
                
                # Benchmark
                self.tracker.reset()
                latencies = []
                
                for _ in range(num_iterations):
                    start = time.perf_counter()
                    _ = self.forward(dummy_burst)
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    elapsed = (time.perf_counter() - start) * 1000
                    latencies.append(elapsed)
                
                # Collect results
                latencies = np.array(latencies)
                results[config_name] = {
                    'total_latency': {
                        'mean': float(np.mean(latencies)),
                        'std': float(np.std(latencies)),
                        'min': float(np.min(latencies)),
                        'max': float(np.max(latencies)),
                        'p50': float(np.percentile(latencies, 50)),
                        'p95': float(np.percentile(latencies, 95)),
                        'p99': float(np.percentile(latencies, 99)),
                    },
                    'stage_timings': self.tracker.get_stats(),
                    'meets_target': float(np.percentile(latencies, 95)) < 30.0
                }
                
                print(f"  Mean latency: {np.mean(latencies):.2f}ms")
                print(f"  P95 latency: {np.percentile(latencies, 95):.2f}ms")
                print(f"  Meets <30ms target: {results[config_name]['meets_target']}")
        
        return results
    
    def export_onnx(
        self,
        save_path: str,
        input_size: Tuple[int, int] = (512, 512),
        burst_length: int = 8,
        opset_version: int = 14
    ):
        """
        Export pipeline to ONNX format.
        
        Args:
            save_path: Path to save ONNX model
            input_size: Input image size (H, W)
            burst_length: Burst length
            opset_version: ONNX opset version
        """
        print(f"Exporting to ONNX: {save_path}")
        
        H, W = input_size
        dummy_input = torch.randn(1, burst_length, 4, H, W, device=self.device)
        
        # Create wrapper for ONNX export
        class ONNXWrapper(nn.Module):
            def __init__(self, pipeline):
                super().__init__()
                self.pipeline = pipeline
            
            def forward(self, burst):
                return self.pipeline.forward(burst)['enhanced']
        
        wrapper = ONNXWrapper(self)
        
        torch.onnx.export(
            wrapper,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['burst'],
            output_names=['enhanced'],
            dynamic_axes={
                'burst': {0: 'batch', 1: 'time'},
                'enhanced': {0: 'batch'}
            }
        )
        
        print(f"✓ Exported to {save_path}")
    
    def get_model_size(self) -> Dict[str, float]:
        """Get model size in MB for each component."""
        sizes = {}
        
        def count_parameters(model, name):
            num_params = sum(p.numel() for p in model.parameters())
            size_mb = num_params * 4 / (1024 ** 2)  # Assuming float32
            sizes[name] = {
                'parameters': num_params,
                'size_mb': size_mb
            }
        
        count_parameters(self.vae, 'vae')
        count_parameters(self.consistency_model, 'consistency')
        count_parameters(self.alignment, 'alignment')
        if self.aberration is not None:
            count_parameters(self.aberration, 'aberration')
        
        total_params = sum(s['parameters'] for s in sizes.values())
        total_size = sum(s['size_mb'] for s in sizes.values())
        
        sizes['total'] = {
            'parameters': total_params,
            'size_mb': total_size
        }
        
        return sizes
    
    def memory_profile(
        self,
        burst_size: Tuple[int, int] = (1024, 1024),
        burst_length: int = 8
    ) -> Dict[str, float]:
        """Profile memory usage."""
        if not torch.cuda.is_available():
            return {'error': 'CUDA not available'}
        
        H, W = burst_size
        dummy_burst = torch.randn(1, burst_length, 4, H, W, device=self.device)
        
        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        # Measure memory
        memory_before = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        
        _ = self.forward(dummy_burst)
        
        memory_after = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        memory_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        
        return {
            'before_mb': memory_before,
            'after_mb': memory_after,
            'peak_mb': memory_peak,
            'allocated_mb': memory_after - memory_before
        }


if __name__ == "__main__":
    # Test real-time pipeline
    print("Testing real-time inference pipeline...")
    
    from models.raw_diffusion_unet import RAWVAE
    from models.consistency_distillation import ConsistencyModel
    from models.optical_flow import RAWOpticalFlow, AlignmentModule
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize models (small configs for testing)
    vae = RAWVAE(
        in_channels=4,
        latent_channels=16,
        channels=64,
        num_res_blocks=2
    )
    
    flow_net = RAWOpticalFlow(
        in_channels=4,
        feature_dim=128,
        num_levels=4
    )
    
    alignment = AlignmentModule(flow_net)
    
    consistency = ConsistencyModel(
        in_channels=16,
        model_channels=128,
        num_res_blocks=2,
        attention_resolutions=[8, 16]
    )
    
    # Create pipeline
    pipeline = RealTimePipeline(
        vae=vae,
        consistency_model=consistency,
        alignment=alignment,
        device=device,
        num_inference_steps=2,
        use_adaptive_steps=True,
        enable_profiling=True
    )
    
    print("\n1. Testing forward pass...")
    burst = torch.randn(1, 8, 4, 512, 512, device=device)
    results = pipeline.forward(burst, return_intermediate=True)
    print(f"   Enhanced shape: {results['enhanced'].shape}")
    print(f"   Number of steps: {results['num_steps']}")
    
    print("\n2. Getting model size...")
    sizes = pipeline.get_model_size()
    print(f"   Model sizes:")
    for name, info in sizes.items():
        print(f"   - {name}: {info['size_mb']:.2f} MB ({info['parameters']:,} params)")
    
    print("\n3. Memory profiling...")
    memory_stats = pipeline.memory_profile(burst_size=(512, 512), burst_length=8)
    print(f"   Memory stats:")
    for k, v in memory_stats.items():
        print(f"   - {k}: {v:.2f} MB")
    
    print("\n4. Running mini benchmark...")
    bench_results = pipeline.benchmark(
        burst_sizes=[(512, 512)],
        burst_lengths=[8],
        num_iterations=20,
        warmup_iterations=5
    )
    
    for config, stats in bench_results.items():
        print(f"\n   Config: {config}")
        print(f"   Mean latency: {stats['total_latency']['mean']:.2f}ms")
        print(f"   P95 latency: {stats['total_latency']['p95']:.2f}ms")
        print(f"   Meets target: {stats['meets_target']}")
    
    print("\n✓ Real-time pipeline tested successfully!")
