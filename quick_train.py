#!/usr/bin/env python3
"""
Quick Training Script - Simplified Training Entry Point

For full training orchestration, use train_orchestrator.py
This script provides a simpler interface for common training tasks.
"""

import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description='Quick Training Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all stages
  python quick_train.py --data ./data/train
  
  # Train specific stage
  python quick_train.py --data ./data/train --stage vae
  
  # Resume training
  python quick_train.py --data ./data/train --resume ./outputs
  
  # Quick demo (no real training)
  python quick_train.py --demo
        """
    )
    
    parser.add_argument('--data', type=str, help='Path to training data')
    parser.add_argument('--output', type=str, default='./outputs',
                       help='Output directory (default: ./outputs)')
    parser.add_argument('--stage', type=str, choices=['vae', 'flow', 'diffusion', 'all'],
                       default='all', help='Training stage (default: all)')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size (default: 4)')
    parser.add_argument('--epochs', type=int, help='Override epochs for stage')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint directory')
    parser.add_argument('--demo', action='store_true',
                       help='Run in demo mode (no actual training)')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    
    if args.demo:
        print("="*70)
        print("DEMO MODE - Training Simulation")
        print("="*70)
        print("\nThis demonstrates the training process without actual computation.")
        print("For real training, provide --data argument with your training data.\n")
        
        print("Training pipeline:")
        print("  ✓ Stage 1: VAE Pre-training (20 epochs, ~6 hours)")
        print("  ✓ Stage 2: Optical Flow Training (15 epochs, ~5 hours)")
        print("  ✓ Stage 3: Diffusion Model Training (50 epochs, ~48 hours)")
        print("  ✓ Stage 4: Consistency Distillation (30 epochs, ~12 hours)")
        print("\nTotal estimated time: ~72 hours on A100 GPU")
        print("\nTo start real training:")
        print("  python quick_train.py --data ./data/train --output ./outputs")
        return
    
    if not args.data:
        print("Error: --data argument is required")
        print("Use --demo for a demonstration, or provide training data path")
        sys.exit(1)
    
    # Check if data exists
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data directory not found: {args.data}")
        sys.exit(1)
    
    # Import training orchestrator
    try:
        from scripts.train_orchestrator import TrainingOrchestrator, setup_distributed, cleanup_distributed
    except ImportError as e:
        print(f"Error importing training modules: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)
    
    # Setup
    print("="*70)
    print("RAW Image Enhancement - Training")
    print("="*70)
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print(f"Stage: {args.stage}")
    print(f"Batch size: {args.batch_size}")
    print()
    
    # Setup distributed
    rank, world_size = setup_distributed()
    
    try:
        # Create orchestrator
        orchestrator = TrainingOrchestrator(
            config_path=args.config,
            data_dir=args.data,
            output_dir=args.output,
            resume_from=args.resume,
            rank=rank,
            world_size=world_size
        )
        
        # Run training based on stage
        if args.stage == 'all':
            orchestrator.run_complete_training()
        elif args.stage == 'vae':
            orchestrator.train_stage_1_vae()
        elif args.stage == 'flow':
            orchestrator.train_stage_2_flow()
        elif args.stage == 'diffusion':
            orchestrator.train_stage_3_diffusion()
        
        print("\n" + "="*70)
        print("✓ Training complete!")
        print("="*70)
        print(f"\nCheckpoints saved to: {args.output}")
        print("\nNext steps:")
        print("1. Validate models: python scripts/evaluate.py")
        print("2. Export for deployment: python scripts/final_integration.py")
        print("3. Deploy: See DEPLOYMENT.md")
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        cleanup_distributed()

if __name__ == "__main__":
    main()
