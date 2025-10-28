"""
Final Integration and Optimization

This script performs final integration of all components and
applies comprehensive optimizations for production deployment.
"""

import torch
import torch.nn as nn
from pathlib import Path
import yaml
import json
from typing import Dict, Optional

from models.raw_diffusion_unet import RAWVAE
from models.consistency_distillation import ConsistencyModel
from models.optical_flow import RAWOpticalFlow, AlignmentModule
from models.lens_aberration_module import AberrationCorrectionModule
from inference.realtime_pipeline import RealTimePipeline
from inference.optimization import optimize_for_inference
from deployment.coreml_converter import PipelineConverter


class ProductionPipeline:
    """
    Production-ready integrated pipeline with all optimizations.
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        config_path: str,
        device: str = 'cuda',
        optimization_level: int = 2
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        self.optimization_level = optimization_level
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize models
        self.vae = None
        self.consistency_model = None
        self.flow_net = None
        self.alignment = None
        self.aberration = None
        self.pipeline = None
        
        print("Initializing production pipeline...")
        self._load_models()
        self._optimize_models()
        self._create_pipeline()
        print("✓ Production pipeline ready!")
    
    def _load_models(self):
        """Load all model checkpoints."""
        print("Loading model checkpoints...")
        
        # VAE
        vae_path = self.checkpoint_dir / 'vae_final.pt'
        if vae_path.exists():
            self.vae = RAWVAE(**self.config['model']['vae'])
            checkpoint = torch.load(vae_path, map_location='cpu')
            self.vae.load_state_dict(checkpoint['model_state_dict'])
            print("  ✓ VAE loaded")
        
        # Optical Flow
        flow_path = self.checkpoint_dir / 'flow_final.pt'
        if flow_path.exists():
            self.flow_net = RAWOpticalFlow(**self.config['model']['optical_flow'])
            checkpoint = torch.load(flow_path, map_location='cpu')
            self.flow_net.load_state_dict(checkpoint['model_state_dict'])
            self.alignment = AlignmentModule(self.flow_net)
            print("  ✓ Optical flow loaded")
        
        # Consistency Model
        consistency_path = self.checkpoint_dir / 'consistency_final.pt'
        if consistency_path.exists():
            self.consistency_model = ConsistencyModel(**self.config['model']['consistency'])
            checkpoint = torch.load(consistency_path, map_location='cpu')
            self.consistency_model.load_state_dict(checkpoint['model_state_dict'])
            print("  ✓ Consistency model loaded")
        
        # Aberration Correction (optional)
        aberration_path = self.checkpoint_dir / 'aberration_final.pt'
        if aberration_path.exists():
            self.aberration = AberrationCorrectionModule(**self.config['model']['aberration'])
            checkpoint = torch.load(aberration_path, map_location='cpu')
            self.aberration.load_state_dict(checkpoint['model_state_dict'])
            print("  ✓ Aberration correction loaded")
    
    def _optimize_models(self):
        """Apply optimizations to all models."""
        print(f"\nApplying optimization level {self.optimization_level}...")
        
        if self.optimization_level > 0:
            if self.vae:
                self.vae = optimize_for_inference(
                    self.vae,
                    optimization_level=self.optimization_level,
                    device=self.device
                )
            
            if self.flow_net:
                self.flow_net = optimize_for_inference(
                    self.flow_net,
                    optimization_level=self.optimization_level,
                    device=self.device
                )
            
            if self.consistency_model:
                self.consistency_model = optimize_for_inference(
                    self.consistency_model,
                    optimization_level=self.optimization_level,
                    device=self.device
                )
            
            if self.aberration:
                self.aberration = optimize_for_inference(
                    self.aberration,
                    optimization_level=self.optimization_level,
                    device=self.device
                )
            
            print("  ✓ Models optimized")
    
    def _create_pipeline(self):
        """Create integrated inference pipeline."""
        print("\nCreating inference pipeline...")
        
        self.pipeline = RealTimePipeline(
            vae=self.vae,
            consistency_model=self.consistency_model,
            alignment=self.alignment,
            aberration=self.aberration,
            device=self.device,
            num_inference_steps=2,
            use_adaptive_steps=True,
            enable_profiling=True
        )
        
        print("  ✓ Pipeline created")
    
    def benchmark(self, save_path: Optional[str] = None) -> Dict:
        """Run comprehensive benchmark."""
        print("\nRunning benchmark...")
        
        results = self.pipeline.benchmark(
            burst_sizes=[(512, 512), (1024, 1024), (2048, 2048)],
            burst_lengths=[4, 8, 16],
            num_iterations=100,
            warmup_iterations=20
        )
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"  ✓ Benchmark saved to {save_path}")
        
        return results
    
    def export_for_mobile(self, output_dir: str):
        """Export models for mobile deployment."""
        print(f"\nExporting for mobile deployment to {output_dir}...")
        
        converter = PipelineConverter(output_dir)
        
        # Convert each component
        if self.alignment:
            converter.convert_component(
                'alignment',
                self.alignment,
                input_shape=(8, 4, 1024, 1024),
                compute_units='CPU_AND_NE'
            )
        
        # VAE encoder
        if self.vae:
            converter.convert_component(
                'vae_encoder',
                self.vae.encoder,
                input_shape=(4, 1024, 1024),
                compute_units='CPU_AND_NE'
            )
        
        # Consistency model
        if self.consistency_model:
            converter.convert_component(
                'consistency',
                self.consistency_model,
                input_shape=(16, 128, 128),
                compute_units='CPU_AND_NE'
            )
        
        # VAE decoder
        if self.vae:
            converter.convert_component(
                'vae_decoder',
                self.vae.decoder,
                input_shape=(16, 128, 128),
                compute_units='CPU_AND_NE'
            )
        
        # Aberration (optional)
        if self.aberration:
            converter.convert_component(
                'aberration',
                self.aberration,
                input_shape=(4, 1024, 1024),
                compute_units='CPU_AND_NE'
            )
        
        # Generate Swift wrapper and docs
        converter.export_pipeline_config()
        converter.generate_swift_wrapper()
        converter.create_ios_integration_guide()
        
        print("  ✓ Mobile export complete")
    
    def validate(self) -> Dict[str, bool]:
        """Validate pipeline functionality."""
        print("\nValidating pipeline...")
        
        validation_results = {}
        
        # Test forward pass
        try:
            test_burst = torch.randn(1, 8, 4, 512, 512, device=self.device)
            results = self.pipeline.forward(test_burst)
            validation_results['forward_pass'] = True
            print("  ✓ Forward pass")
        except Exception as e:
            print(f"  ✗ Forward pass failed: {e}")
            validation_results['forward_pass'] = False
        
        # Test latency target
        try:
            test_burst = torch.randn(1, 8, 4, 1024, 1024, device=self.device)
            import time
            start = time.perf_counter()
            _ = self.pipeline.forward(test_burst)
            latency = (time.perf_counter() - start) * 1000
            meets_target = latency < 30.0
            validation_results['latency_target'] = meets_target
            print(f"  {'✓' if meets_target else '✗'} Latency: {latency:.2f}ms (target: <30ms)")
        except Exception as e:
            print(f"  ✗ Latency test failed: {e}")
            validation_results['latency_target'] = False
        
        # Test model sizes
        try:
            sizes = self.pipeline.get_model_size()
            total_size = sizes['total']['size_mb']
            meets_size = total_size < 1000  # <1GB
            validation_results['model_size'] = meets_size
            print(f"  {'✓' if meets_size else '✗'} Model size: {total_size:.2f}MB")
        except Exception as e:
            print(f"  ✗ Size check failed: {e}")
            validation_results['model_size'] = False
        
        # Test memory usage
        try:
            memory_stats = self.pipeline.memory_profile()
            peak_memory = memory_stats.get('peak_mb', 0)
            meets_memory = peak_memory < 2000  # <2GB
            validation_results['memory_usage'] = meets_memory
            print(f"  {'✓' if meets_memory else '✗'} Peak memory: {peak_memory:.2f}MB")
        except Exception as e:
            print(f"  ✗ Memory test failed: {e}")
            validation_results['memory_usage'] = False
        
        return validation_results
    
    def generate_deployment_package(self, output_dir: str):
        """Generate complete deployment package."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nGenerating deployment package in {output_dir}...")
        
        # 1. Export mobile models
        self.export_for_mobile(str(output_path / 'mobile'))
        
        # 2. Save optimized PyTorch models
        torch_dir = output_path / 'pytorch'
        torch_dir.mkdir(exist_ok=True)
        
        if self.vae:
            torch.save(self.vae.state_dict(), torch_dir / 'vae.pt')
        if self.consistency_model:
            torch.save(self.consistency_model.state_dict(), torch_dir / 'consistency.pt')
        if self.flow_net:
            torch.save(self.flow_net.state_dict(), torch_dir / 'flow.pt')
        
        print("  ✓ PyTorch models saved")
        
        # 3. Run and save benchmark
        benchmark_results = self.benchmark(str(output_path / 'benchmark.json'))
        
        # 4. Save validation results
        validation_results = self.validate()
        with open(output_path / 'validation.json', 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        # 5. Generate deployment README
        readme_content = self._generate_deployment_readme(validation_results, benchmark_results)
        with open(output_path / 'DEPLOYMENT.md', 'w') as f:
            f.write(readme_content)
        
        print(f"\n✓ Deployment package complete: {output_dir}")
    
    def _generate_deployment_readme(self, validation: Dict, benchmark: Dict) -> str:
        """Generate deployment README."""
        return f"""# Deployment Package

Generated: {Path(__file__).parent}

## Validation Results

{self._format_validation(validation)}

## Performance Benchmarks

{self._format_benchmarks(benchmark)}

## Package Contents

- `pytorch/`: Optimized PyTorch models
- `mobile/`: CoreML models for iOS deployment
- `benchmark.json`: Detailed performance metrics
- `validation.json`: Validation test results

## Deployment Instructions

### PyTorch Deployment

```python
import torch
from inference.realtime_pipeline import RealTimePipeline

# Load models
vae = torch.load('pytorch/vae.pt')
consistency = torch.load('pytorch/consistency.pt')

# Create pipeline
pipeline = RealTimePipeline(vae, consistency, ...)
```

### iOS Deployment

See `mobile/iOS_Integration_Guide.md` for complete instructions.

## Requirements

- PyTorch 2.1.0+
- iOS 16.0+ (for mobile deployment)
- CUDA 12.1+ (for GPU inference)

## Support

For issues or questions, please contact: your.email@example.com
"""
    
    def _format_validation(self, validation: Dict) -> str:
        """Format validation results."""
        lines = []
        for test, passed in validation.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            lines.append(f"- {test}: {status}")
        return "\n".join(lines)
    
    def _format_benchmarks(self, benchmark: Dict) -> str:
        """Format benchmark results."""
        lines = []
        for config, stats in benchmark.items():
            latency = stats['total_latency']['p95']
            meets_target = stats['meets_target']
            status = "✓" if meets_target else "✗"
            lines.append(f"- {config}: {latency:.2f}ms (P95) {status}")
        return "\n".join(lines)


def main():
    """Main integration script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Final Integration and Optimization')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Directory containing model checkpoints')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='./deployment',
                       help='Output directory for deployment package')
    parser.add_argument('--optimization_level', type=int, default=2,
                       choices=[0, 1, 2, 3, 4],
                       help='Optimization level (0=none, 4=max)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Create production pipeline
    pipeline = ProductionPipeline(
        checkpoint_dir=args.checkpoint_dir,
        config_path=args.config,
        device=args.device,
        optimization_level=args.optimization_level
    )
    
    # Generate deployment package
    pipeline.generate_deployment_package(args.output_dir)
    
    print("\n" + "="*70)
    print("FINAL INTEGRATION COMPLETE!")
    print("="*70)
    print(f"\nDeployment package: {args.output_dir}")
    print("\nNext steps:")
    print("1. Review validation results in deployment/validation.json")
    print("2. Test mobile deployment using deployment/mobile/")
    print("3. Deploy to production")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
