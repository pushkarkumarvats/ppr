"""
Performance Monitoring and Profiling Tools

This module provides comprehensive monitoring and debugging utilities
for the RAW enhancement pipeline.
"""

import torch
import time
import psutil
import json
from typing import Dict, List, Optional, Any
from collections import defaultdict
from pathlib import Path
from datetime import datetime
import numpy as np


class PerformanceProfiler:
    """
    Profile inference pipeline performance with detailed breakdowns.
    """
    
    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.timings = defaultdict(list)
        self.memory_usage = []
        self.active_stage = None
        self.start_time = None
        
        self.session_start = datetime.now()
        self.session_id = self.session_start.strftime("%Y%m%d_%H%M%S")
    
    def start_stage(self, stage_name: str):
        """Start profiling a stage."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        self.active_stage = stage_name
        self.start_time = time.perf_counter()
    
    def end_stage(self):
        """End profiling the active stage."""
        if self.active_stage is None:
            return
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed = (time.perf_counter() - self.start_time) * 1000  # ms
        self.timings[self.active_stage].append(elapsed)
        
        # Record memory usage
        if torch.cuda.is_available():
            memory_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            self.memory_usage.append({
                'stage': self.active_stage,
                'memory_mb': memory_mb,
                'timestamp': time.time()
            })
        
        self.active_stage = None
        self.start_time = None
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
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
                'count': len(times)
            }
        
        return stats
    
    def save_report(self, filename: Optional[str] = None):
        """Save profiling report."""
        if filename is None:
            filename = f"profile_{self.session_id}.json"
        
        report = {
            'session_id': self.session_id,
            'session_start': self.session_start.isoformat(),
            'duration_seconds': (datetime.now() - self.session_start).total_seconds(),
            'statistics': self.get_statistics(),
            'memory_usage': self.memory_usage,
            'system_info': self.get_system_info()
        }
        
        filepath = self.log_dir / filename
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Profile saved to {filepath}")
        return filepath
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        info = {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024 ** 3),
            'python_version': __import__('sys').version
        }
        
        if torch.cuda.is_available():
            info['cuda_available'] = True
            info['cuda_version'] = torch.version.cuda
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        else:
            info['cuda_available'] = False
        
        return info
    
    def print_summary(self):
        """Print profiling summary."""
        stats = self.get_statistics()
        
        print("\n" + "="*70)
        print("PERFORMANCE PROFILING SUMMARY")
        print("="*70)
        
        total_time = sum(s['mean'] for s in stats.values())
        
        print(f"\nTotal Pipeline Time: {total_time:.2f}ms (mean)")
        print(f"\nStage Breakdown:")
        print("-"*70)
        
        for stage, stage_stats in sorted(stats.items(), key=lambda x: x[1]['mean'], reverse=True):
            percentage = (stage_stats['mean'] / total_time) * 100
            print(f"\n{stage}:")
            print(f"  Mean: {stage_stats['mean']:.2f}ms ({percentage:.1f}%)")
            print(f"  Std:  {stage_stats['std']:.2f}ms")
            print(f"  Min:  {stage_stats['min']:.2f}ms")
            print(f"  Max:  {stage_stats['max']:.2f}ms")
            print(f"  P95:  {stage_stats['p95']:.2f}ms")
            print(f"  Runs: {stage_stats['count']}")
        
        if self.memory_usage:
            print("\n" + "-"*70)
            print("Memory Usage:")
            max_memory = max(m['memory_mb'] for m in self.memory_usage)
            print(f"  Peak: {max_memory:.2f}MB")
        
        print("="*70 + "\n")


class ModelDebugger:
    """
    Debug model issues with activation inspection and gradient flow.
    """
    
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.activations = {}
        self.gradients = {}
        self.hooks = []
    
    def register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.activations[name] = {
                        'mean': output.mean().item(),
                        'std': output.std().item(),
                        'min': output.min().item(),
                        'max': output.max().item(),
                        'shape': tuple(output.shape),
                        'has_nan': torch.isnan(output).any().item(),
                        'has_inf': torch.isinf(output).any().item()
                    }
            return hook
        
        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    grad = grad_output[0]
                    self.gradients[name] = {
                        'mean': grad.mean().item(),
                        'std': grad.std().item(),
                        'min': grad.min().item(),
                        'max': grad.max().item(),
                        'norm': grad.norm().item(),
                        'has_nan': torch.isnan(grad).any().item(),
                        'has_inf': torch.isinf(grad).any().item()
                    }
            return hook
        
        # Register hooks for all modules
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                self.hooks.append(module.register_forward_hook(forward_hook(name)))
                self.hooks.append(module.register_full_backward_hook(backward_hook(name)))
    
    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def check_activations(self) -> Dict[str, List[str]]:
        """Check for problematic activations."""
        issues = defaultdict(list)
        
        for name, stats in self.activations.items():
            if stats['has_nan']:
                issues['nan_activations'].append(name)
            if stats['has_inf']:
                issues['inf_activations'].append(name)
            if abs(stats['mean']) > 100:
                issues['large_activations'].append(f"{name} (mean={stats['mean']:.2f})")
            if stats['std'] < 1e-6:
                issues['dead_activations'].append(f"{name} (std={stats['std']:.2e})")
        
        return dict(issues)
    
    def check_gradients(self) -> Dict[str, List[str]]:
        """Check for gradient issues."""
        issues = defaultdict(list)
        
        for name, stats in self.gradients.items():
            if stats['has_nan']:
                issues['nan_gradients'].append(name)
            if stats['has_inf']:
                issues['inf_gradients'].append(name)
            if stats['norm'] < 1e-8:
                issues['vanishing_gradients'].append(f"{name} (norm={stats['norm']:.2e})")
            if stats['norm'] > 100:
                issues['exploding_gradients'].append(f"{name} (norm={stats['norm']:.2f})")
        
        return dict(issues)
    
    def print_debug_report(self):
        """Print comprehensive debug report."""
        print("\n" + "="*70)
        print("MODEL DEBUG REPORT")
        print("="*70)
        
        # Activation issues
        act_issues = self.check_activations()
        if act_issues:
            print("\nACTIVATION ISSUES:")
            for issue_type, modules in act_issues.items():
                print(f"\n  {issue_type}:")
                for module in modules:
                    print(f"    - {module}")
        else:
            print("\n✓ No activation issues detected")
        
        # Gradient issues
        grad_issues = self.check_gradients()
        if grad_issues:
            print("\nGRADIENT ISSUES:")
            for issue_type, modules in grad_issues.items():
                print(f"\n  {issue_type}:")
                for module in modules:
                    print(f"    - {module}")
        else:
            print("\n✓ No gradient issues detected")
        
        print("\n" + "="*70 + "\n")
    
    def save_debug_info(self, filepath: str):
        """Save debug information to file."""
        info = {
            'activations': self.activations,
            'gradients': self.gradients,
            'activation_issues': self.check_activations(),
            'gradient_issues': self.check_gradients()
        }
        
        with open(filepath, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"Debug info saved to {filepath}")


class InferenceLogger:
    """
    Log inference results for monitoring and analysis.
    """
    
    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.log_dir / f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.file_handle = open(self.log_file, 'a')
    
    def log_inference(
        self,
        input_shape: tuple,
        output_shape: tuple,
        latency_ms: float,
        metrics: Dict[str, float],
        metadata: Optional[Dict] = None
    ):
        """Log single inference result."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'input_shape': input_shape,
            'output_shape': output_shape,
            'latency_ms': latency_ms,
            'metrics': metrics,
            'metadata': metadata or {}
        }
        
        self.file_handle.write(json.dumps(log_entry) + '\n')
        self.file_handle.flush()
    
    def close(self):
        """Close log file."""
        if self.file_handle:
            self.file_handle.close()
    
    def __del__(self):
        self.close()


class ResourceMonitor:
    """
    Monitor system resources during execution.
    """
    
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.monitoring = False
        self.samples = []
    
    def start(self):
        """Start monitoring."""
        self.monitoring = True
        self.samples = []
        
        import threading
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.start()
    
    def stop(self):
        """Stop monitoring."""
        self.monitoring = False
        if hasattr(self, 'thread'):
            self.thread.join()
    
    def _monitor_loop(self):
        """Monitoring loop."""
        while self.monitoring:
            sample = {
                'timestamp': time.time(),
                'cpu_percent': psutil.cpu_percent(interval=None),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_used_gb': psutil.virtual_memory().used / (1024 ** 3)
            }
            
            if torch.cuda.is_available():
                sample['gpu_memory_used_mb'] = torch.cuda.memory_allocated() / (1024 ** 2)
                sample['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / (1024 ** 2)
            
            self.samples.append(sample)
            time.sleep(self.interval)
    
    def get_summary(self) -> Dict[str, float]:
        """Get resource usage summary."""
        if not self.samples:
            return {}
        
        cpu_values = [s['cpu_percent'] for s in self.samples]
        memory_values = [s['memory_percent'] for s in self.samples]
        
        summary = {
            'cpu_mean': np.mean(cpu_values),
            'cpu_max': np.max(cpu_values),
            'memory_mean': np.mean(memory_values),
            'memory_max': np.max(memory_values)
        }
        
        if torch.cuda.is_available():
            gpu_mem_values = [s['gpu_memory_used_mb'] for s in self.samples]
            summary['gpu_memory_mean_mb'] = np.mean(gpu_mem_values)
            summary['gpu_memory_max_mb'] = np.max(gpu_mem_values)
        
        return summary


if __name__ == "__main__":
    print("Testing monitoring tools...")
    
    # Test profiler
    print("\n1. Testing PerformanceProfiler...")
    profiler = PerformanceProfiler(log_dir="./test_logs")
    
    for i in range(5):
        profiler.start_stage('test_stage')
        time.sleep(0.01)
        profiler.end_stage()
    
    profiler.print_summary()
    profiler.save_report("test_profile.json")
    print("   ✓ Profiler tested")
    
    # Test debugger
    print("\n2. Testing ModelDebugger...")
    test_model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 5)
    )
    
    debugger = ModelDebugger(test_model)
    debugger.register_hooks()
    
    x = torch.randn(2, 10, requires_grad=True)
    y = test_model(x)
    loss = y.sum()
    loss.backward()
    
    debugger.print_debug_report()
    debugger.remove_hooks()
    print("   ✓ Debugger tested")
    
    # Test logger
    print("\n3. Testing InferenceLogger...")
    logger = InferenceLogger(log_dir="./test_logs")
    logger.log_inference(
        input_shape=(1, 4, 512, 512),
        output_shape=(1, 4, 512, 512),
        latency_ms=25.5,
        metrics={'psnr': 35.2, 'ssim': 0.95}
    )
    logger.close()
    print("   ✓ Logger tested")
    
    # Test resource monitor
    print("\n4. Testing ResourceMonitor...")
    monitor = ResourceMonitor(interval=0.1)
    monitor.start()
    time.sleep(0.5)
    monitor.stop()
    summary = monitor.get_summary()
    print(f"   Resource summary: {summary}")
    print("   ✓ Monitor tested")
    
    print("\n✓ All monitoring tools tested successfully!")
