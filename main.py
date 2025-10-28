"""
Main entry point for RAW Fusion Diffusion
"""

import argparse
import sys
from pathlib import Path

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Real-Time Multi-Frame RAW Fusion with Diffusion-Guided Detail Recovery"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Training command
    train_parser = subparsers.add_parser("train", help="Train models")
    train_parser.add_argument("--config", type=str, required=True, help="Training config file")
    train_parser.add_argument("--stage", type=str, choices=["vae", "flow", "diffusion", "consistency"], help="Training stage")
    
    # Inference command
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    infer_parser.add_argument("--input", type=str, required=True, help="Input burst directory")
    infer_parser.add_argument("--output", type=str, required=True, help="Output file")
    infer_parser.add_argument("--steps", type=int, default=4, help="Number of inference steps")
    
    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert to CoreML")
    convert_parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    convert_parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    convert_parser.add_argument("--optimize-ane", action="store_true", help="Optimize for Apple Neural Engine")
    
    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark performance")
    bench_parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    bench_parser.add_argument("--num-runs", type=int, default=100, help="Number of benchmark runs")
    
    args = parser.parse_args()
    
    if args.command == "train":
        from training.train_diffusion import train
        train(args.config, args.stage)
    elif args.command == "infer":
        from inference.realtime_pipeline import infer
        infer(args.checkpoint, args.input, args.output, args.steps)
    elif args.command == "convert":
        from deployment.coreml_converter import convert
        convert(args.checkpoint, args.output_dir, args.optimize_ane)
    elif args.command == "benchmark":
        from inference.realtime_pipeline import benchmark
        benchmark(args.checkpoint, args.num_runs)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
