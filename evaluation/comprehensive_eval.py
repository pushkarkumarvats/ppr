"""Evaluation & Benchmarking Suite"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
import pandas as pd


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    model_name: str
    image_size: Tuple[int, int]
    batch_size: int
    num_steps: int
    avg_latency_ms: float
    std_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_fps: float
    memory_mb: float
    model_size_mb: float
    psnr: Optional[float] = None
    ssim: Optional[float] = None
    lpips: Optional[float] = None


class ComprehensiveEvaluator:
    """
    Comprehensive evaluation including quality metrics, speed, and memory usage.
    """
    
    def __init__(
        self,
        device: str = 'cuda',
        save_dir: str = './evaluation_results'
    ):
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = []
    
    def evaluate_model(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        model_name: str,
        num_inference_steps: int = 2,
        num_samples: int = 100
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation of a model.
        
        Args:
            model: Model to evaluate
            dataloader: Test data loader
            model_name: Name of the model
            num_inference_steps: Number of inference steps
            num_samples: Number of samples to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        model.eval()
        
        metrics = {
            'psnr': [],
            'ssim': [],
            'lpips': [],
            'latency': [],
            'memory': []
        }
        
        print(f"\nEvaluating {model_name}...")
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
                if i >= num_samples:
                    break
                
                burst = batch['burst'].to(self.device)
                target = batch['target'].to(self.device) if 'target' in batch else None
                
                # Measure latency
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                
                # Run inference
                if hasattr(model, 'forward'):
                    output = model.forward(burst, num_inference_steps=num_inference_steps)
                    if isinstance(output, dict):
                        enhanced = output['enhanced']
                    else:
                        enhanced = output
                else:
                    enhanced = model(burst)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                latency = (time.time() - start_time) * 1000  # ms
                
                metrics['latency'].append(latency)
                
                # Measure memory
                if torch.cuda.is_available():
                    memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
                    metrics['memory'].append(memory)
                    torch.cuda.reset_peak_memory_stats()
                
                # Calculate quality metrics if target available
                if target is not None:
                    psnr = self._calculate_psnr(enhanced, target)
                    ssim = self._calculate_ssim(enhanced, target)
                    
                    metrics['psnr'].append(psnr)
                    metrics['ssim'].append(ssim)
        
        # Aggregate results
        results = {
            'model_name': model_name,
            'num_inference_steps': num_inference_steps,
            'avg_latency_ms': np.mean(metrics['latency']),
            'std_latency_ms': np.std(metrics['latency']),
            'p50_latency_ms': np.percentile(metrics['latency'], 50),
            'p95_latency_ms': np.percentile(metrics['latency'], 95),
            'p99_latency_ms': np.percentile(metrics['latency'], 99),
            'throughput_fps': 1000.0 / np.mean(metrics['latency']),
            'avg_memory_mb': np.mean(metrics['memory']) if metrics['memory'] else 0,
        }
        
        if metrics['psnr']:
            results['avg_psnr'] = np.mean(metrics['psnr'])
            results['std_psnr'] = np.std(metrics['psnr'])
        
        if metrics['ssim']:
            results['avg_ssim'] = np.mean(metrics['ssim'])
            results['std_ssim'] = np.std(metrics['ssim'])
        
        self.results.append(results)
        
        return results
    
    def _calculate_psnr(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Calculate PSNR between two images."""
        mse = F.mse_loss(img1, img2).item()
        if mse == 0:
            return float('inf')
        return 20 * np.log10(1.0 / np.sqrt(mse))
    
    def _calculate_ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Calculate SSIM between two images (simplified)."""
        # Simplified SSIM calculation
        mu1 = img1.mean()
        mu2 = img2.mean()
        sigma1 = img1.std()
        sigma2 = img2.std()
        sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 ** 2 + sigma2 ** 2 + c2))
        
        return ssim.item()
    
    def compare_models(
        self,
        models: Dict[str, nn.Module],
        dataloader: torch.utils.data.DataLoader,
        num_inference_steps_list: List[int] = [2, 4, 8],
        num_samples: int = 100
    ):
        """
        Compare multiple models across different configurations.
        
        Args:
            models: Dictionary mapping model names to model instances
            dataloader: Test data loader
            num_inference_steps_list: List of inference steps to test
            num_samples: Number of samples to evaluate
        """
        print("="*70)
        print("MODEL COMPARISON")
        print("="*70)
        
        for model_name, model in models.items():
            for num_steps in num_inference_steps_list:
                self.evaluate_model(
                    model=model,
                    dataloader=dataloader,
                    model_name=f"{model_name}_{num_steps}steps",
                    num_inference_steps=num_steps,
                    num_samples=num_samples
                )
        
        self.save_results()
        self.plot_comparison()
    
    def save_results(self):
        """Save evaluation results to JSON and CSV."""
        # Save as JSON
        json_path = self.save_dir / 'evaluation_results.json'
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save as CSV
        csv_path = self.save_dir / 'evaluation_results.csv'
        df = pd.DataFrame(self.results)
        df.to_csv(csv_path, index=False)
        
        print(f"\n✓ Results saved to:")
        print(f"  JSON: {json_path}")
        print(f"  CSV: {csv_path}")
    
    def plot_comparison(self):
        """Create visualization comparing models."""
        if not self.results:
            return
        
        df = pd.DataFrame(self.results)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Latency comparison
        ax = axes[0, 0]
        df.plot(x='model_name', y='avg_latency_ms', kind='bar', ax=ax, color='skyblue')
        ax.set_title('Average Latency', fontsize=14, fontweight='bold')
        ax.set_ylabel('Latency (ms)')
        ax.set_xlabel('')
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot 2: Throughput comparison
        ax = axes[0, 1]
        df.plot(x='model_name', y='throughput_fps', kind='bar', ax=ax, color='lightgreen')
        ax.set_title('Throughput', fontsize=14, fontweight='bold')
        ax.set_ylabel('FPS')
        ax.set_xlabel('')
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot 3: PSNR comparison (if available)
        ax = axes[1, 0]
        if 'avg_psnr' in df.columns:
            df.plot(x='model_name', y='avg_psnr', kind='bar', ax=ax, color='coral')
            ax.set_title('Image Quality (PSNR)', fontsize=14, fontweight='bold')
            ax.set_ylabel('PSNR (dB)')
            ax.set_xlabel('')
            ax.grid(axis='y', alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            ax.text(0.5, 0.5, 'PSNR data not available', 
                   ha='center', va='center', fontsize=12)
        
        # Plot 4: Memory usage
        ax = axes[1, 1]
        df.plot(x='model_name', y='avg_memory_mb', kind='bar', ax=ax, color='plum')
        ax.set_title('Memory Usage', fontsize=14, fontweight='bold')
        ax.set_ylabel('Memory (MB)')
        ax.set_xlabel('')
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plot_path = self.save_dir / 'model_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  Plot: {plot_path}")
        plt.close()
    
    def generate_report(self) -> str:
        """Generate a comprehensive markdown report."""
        report = ["# Model Evaluation Report\n"]
        report.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"**Device**: {self.device}\n")
        report.append(f"**Number of Models**: {len(self.results)}\n\n")
        
        report.append("## Summary Table\n")
        report.append("| Model | Steps | Latency (ms) | Throughput (FPS) | PSNR (dB) | Memory (MB) |")
        report.append("|-------|-------|--------------|------------------|-----------|-------------|")
        
        for result in self.results:
            psnr_str = f"{result.get('avg_psnr', 0):.2f}" if 'avg_psnr' in result else "N/A"
            report.append(
                f"| {result['model_name']} | "
                f"{result['num_inference_steps']} | "
                f"{result['avg_latency_ms']:.2f} | "
                f"{result['throughput_fps']:.2f} | "
                f"{psnr_str} | "
                f"{result['avg_memory_mb']:.2f} |"
            )
        
        report.append("\n## Detailed Results\n")
        for result in self.results:
            report.append(f"\n### {result['model_name']}\n")
            report.append(f"- **Inference Steps**: {result['num_inference_steps']}")
            report.append(f"- **Avg Latency**: {result['avg_latency_ms']:.2f} ms (±{result['std_latency_ms']:.2f})")
            report.append(f"- **p95 Latency**: {result['p95_latency_ms']:.2f} ms")
            report.append(f"- **p99 Latency**: {result['p99_latency_ms']:.2f} ms")
            report.append(f"- **Throughput**: {result['throughput_fps']:.2f} FPS")
            report.append(f"- **Memory**: {result['avg_memory_mb']:.2f} MB")
            
            if 'avg_psnr' in result:
                report.append(f"- **PSNR**: {result['avg_psnr']:.2f} dB (±{result['std_psnr']:.2f})")
            
            if 'avg_ssim' in result:
                report.append(f"- **SSIM**: {result['avg_ssim']:.4f} (±{result['std_ssim']:.4f})")
        
        report_text = "\n".join(report)
        
        # Save report
        report_path = self.save_dir / 'evaluation_report.md'
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(f"  Report: {report_path}")
        
        return report_text


class ABTestingFramework:
    """
    A/B testing framework for comparing different model versions.
    """
    
    def __init__(self, save_dir: str = './ab_testing'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.tests = []
    
    def run_ab_test(
        self,
        model_a: nn.Module,
        model_b: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        name_a: str = "Model A",
        name_b: str = "Model B",
        num_samples: int = 100,
        device: str = 'cuda'
    ) -> Dict:
        """
        Run A/B test between two models.
        
        Returns statistical comparison results.
        """
        print(f"\nRunning A/B test: {name_a} vs {name_b}")
        
        evaluator = ComprehensiveEvaluator(device=device, save_dir=self.save_dir)
        
        # Evaluate both models
        results_a = evaluator.evaluate_model(model_a, dataloader, name_a, num_samples=num_samples)
        results_b = evaluator.evaluate_model(model_b, dataloader, name_b, num_samples=num_samples)
        
        # Statistical comparison
        comparison = {
            'model_a': name_a,
            'model_b': name_b,
            'latency_improvement': (results_a['avg_latency_ms'] - results_b['avg_latency_ms']) / results_a['avg_latency_ms'] * 100,
            'throughput_improvement': (results_b['throughput_fps'] - results_a['throughput_fps']) / results_a['throughput_fps'] * 100,
        }
        
        if 'avg_psnr' in results_a and 'avg_psnr' in results_b:
            comparison['psnr_difference'] = results_b['avg_psnr'] - results_a['avg_psnr']
        
        print(f"\n✓ A/B Test Results:")
        print(f"  Latency: {comparison['latency_improvement']:.2f}% {'improvement' if comparison['latency_improvement'] > 0 else 'degradation'}")
        print(f"  Throughput: {comparison['throughput_improvement']:.2f}% {'improvement' if comparison['throughput_improvement'] > 0 else 'degradation'}")
        
        if 'psnr_difference' in comparison:
            print(f"  PSNR: {comparison['psnr_difference']:.2f} dB difference")
        
        self.tests.append(comparison)
        
        return comparison


def main():
    """Example usage of evaluation suite."""
    print("="*70)
    print("Comprehensive Evaluation & Benchmarking Suite")
    print("="*70)
    
    # This is a placeholder - integrate with your actual models
    print("\nTo use this evaluation suite:")
    print("1. Load your trained models")
    print("2. Create a test dataloader")
    print("3. Run evaluation:")
    print()
    print("Example:")
    print("  evaluator = ComprehensiveEvaluator(device='cuda')")
    print("  results = evaluator.evaluate_model(")
    print("      model=your_model,")
    print("      dataloader=test_loader,")
    print("      model_name='consistency_2step'")
    print("  )")
    print()
    print("For A/B testing:")
    print("  framework = ABTestingFramework()")
    print("  comparison = framework.run_ab_test(")
    print("      model_a=baseline_model,")
    print("      model_b=optimized_model,")
    print("      dataloader=test_loader")
    print("  )")


if __name__ == "__main__":
    main()
