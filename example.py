"""Example: RAW Enhancement Pipeline Usage"""

import torch
import numpy as np
from pathlib import Path
import argparse
from typing import List

# Import pipeline components
try:
    from inference.realtime_pipeline import RealTimePipeline
    from models.raw_diffusion_unet import RAWVAE
    from models.consistency_distillation import ConsistencyModel
    from models.optical_flow import RAWOpticalFlow, AlignmentModule
    from data.raw_loader import load_proraw_dng
    from data.preprocessing import raw_to_srgb_simple
except ImportError as e:
    print(f"Warning: Could not import required modules: {e}")
    print("This is a demonstration script. Install full dependencies to run.")


def load_burst_from_files(file_paths: List[str]) -> torch.Tensor:
    """
    Load a burst of RAW images from DNG files.
    
    Args:
        file_paths: List of paths to DNG files
        
    Returns:
        Burst tensor [1, T, 4, H, W]
    """
    burst_frames = []
    
    for path in file_paths:
        print(f"Loading {path}...")
        raw_data, metadata = load_proraw_dng(path)
        
        # Convert to tensor [4, H, W]
        raw_tensor = torch.from_numpy(raw_data).float()
        burst_frames.append(raw_tensor)
    
    # Stack into burst [T, 4, H, W]
    burst = torch.stack(burst_frames, dim=0)
    
    # Add batch dimension [1, T, 4, H, W]
    burst = burst.unsqueeze(0)
    
    return burst


def create_demo_burst(num_frames: int = 8, size: int = 512) -> torch.Tensor:
    """
    Create a synthetic demo burst for testing.
    
    Args:
        num_frames: Number of frames in burst
        size: Image size
        
    Returns:
        Demo burst tensor [1, T, 4, H, W]
    """
    print(f"Creating demo burst: {num_frames} frames of {size}x{size}")
    
    # Create base pattern
    base = torch.randn(1, 4, size, size) * 0.5 + 0.5
    
    # Add slight variations per frame
    burst_frames = []
    for i in range(num_frames):
        noise = torch.randn_like(base) * 0.02
        offset = torch.randn(1, 4, 1, 1) * 0.01
        frame = base + noise + offset
        burst_frames.append(frame.squeeze(0))
    
    burst = torch.stack(burst_frames, dim=0).unsqueeze(0)
    
    return burst


def load_models(checkpoint_dir: str, device: str = 'cuda') -> RealTimePipeline:
    """
    Load trained models and create pipeline.
    
    Args:
        checkpoint_dir: Directory containing model checkpoints
        device: Device to load models on
        
    Returns:
        Initialized pipeline
    """
    checkpoint_path = Path(checkpoint_dir)
    
    print(f"Loading models from {checkpoint_path}...")
    
    # Load VAE
    vae = RAWVAE(in_channels=4, latent_channels=16, channels=64, num_res_blocks=2)
    vae_ckpt = checkpoint_path / 'vae_final.pt'
    if vae_ckpt.exists():
        vae.load_state_dict(torch.load(vae_ckpt, map_location=device)['model_state_dict'])
        print("✓ VAE loaded")
    else:
        print("⚠ VAE checkpoint not found, using random weights")
    
    # Load Optical Flow
    flow_net = RAWOpticalFlow(in_channels=4, feature_dim=128, num_levels=4)
    flow_ckpt = checkpoint_path / 'flow_final.pt'
    if flow_ckpt.exists():
        flow_net.load_state_dict(torch.load(flow_ckpt, map_location=device)['model_state_dict'])
        print("✓ Optical Flow loaded")
    else:
        print("⚠ Flow checkpoint not found, using random weights")
    
    alignment = AlignmentModule(flow_net)
    
    # Load Consistency Model
    consistency = ConsistencyModel(in_channels=16, model_channels=128, num_res_blocks=2)
    consistency_ckpt = checkpoint_path / 'consistency_final.pt'
    if consistency_ckpt.exists():
        consistency.load_state_dict(torch.load(consistency_ckpt, map_location=device)['model_state_dict'])
        print("✓ Consistency Model loaded")
    else:
        print("⚠ Consistency checkpoint not found, using random weights")
    
    # Create pipeline
    pipeline = RealTimePipeline(
        vae=vae,
        consistency_model=consistency,
        alignment=alignment,
        device=device,
        num_inference_steps=2,
        use_adaptive_steps=True
    )
    
    print("✓ Pipeline ready!")
    
    return pipeline


def enhance_burst(
    pipeline: RealTimePipeline,
    burst: torch.Tensor,
    num_steps: int = 2
) -> dict:
    """
    Run enhancement on a burst.
    
    Args:
        pipeline: Enhancement pipeline
        burst: Input burst [1, T, 4, H, W]
        num_steps: Number of inference steps
        
    Returns:
        Dictionary with enhanced image and metrics
    """
    import time
    
    print(f"\nEnhancing burst with {num_steps} steps...")
    print(f"Burst shape: {burst.shape}")
    
    device = next(pipeline.vae.parameters()).device
    burst = burst.to(device)
    
    start_time = time.time()
    
    with torch.no_grad():
        result = pipeline.forward(burst, num_inference_steps=num_steps)
    
    latency = (time.time() - start_time) * 1000  # ms
    
    print(f"✓ Enhancement complete in {latency:.2f}ms")
    print(f"  Actual steps used: {result.get('num_steps', num_steps)}")
    
    result['latency_ms'] = latency
    
    return result


def save_output(enhanced: torch.Tensor, output_path: str):
    """
    Save enhanced RAW image.
    
    Args:
        enhanced: Enhanced RAW tensor [1, 4, H, W]
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove batch dimension
    enhanced = enhanced.squeeze(0)
    
    # Convert to numpy
    enhanced_np = enhanced.cpu().numpy()
    
    # Save as numpy array (in practice, would save as DNG)
    np.save(output_path.with_suffix('.npy'), enhanced_np)
    
    print(f"✓ Enhanced RAW saved to {output_path.with_suffix('.npy')}")
    
    # Also save RGB preview
    try:
        rgb = raw_to_srgb_simple(enhanced_np)
        rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        
        from PIL import Image
        rgb_pil = Image.fromarray(rgb_uint8.transpose(1, 2, 0))
        rgb_pil.save(output_path.with_suffix('.png'))
        
        print(f"✓ RGB preview saved to {output_path.with_suffix('.png')}")
    except Exception as e:
        print(f"⚠ Could not save RGB preview: {e}")


def main():
    parser = argparse.ArgumentParser(description='RAW Enhancement Example')
    parser.add_argument('--checkpoints', type=str, default='./outputs',
                       help='Path to model checkpoints')
    parser.add_argument('--input', type=str, nargs='+',
                       help='Input DNG files (burst sequence)')
    parser.add_argument('--output', type=str, default='./output/enhanced',
                       help='Output file path')
    parser.add_argument('--steps', type=int, default=2,
                       help='Number of inference steps (2, 4, or 8)')
    parser.add_argument('--demo', action='store_true',
                       help='Run with demo/synthetic data')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("RAW Image Enhancement Pipeline - Example")
    print("="*70)
    
    # Load models
    try:
        pipeline = load_models(args.checkpoints, device=args.device)
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Running in demo mode with synthetic models...")
        args.demo = True
    
    # Load or create burst
    if args.demo or not args.input:
        burst = create_demo_burst(num_frames=8, size=512)
    else:
        try:
            burst = load_burst_from_files(args.input)
        except Exception as e:
            print(f"Error loading burst: {e}")
            print("Falling back to demo burst...")
            burst = create_demo_burst(num_frames=8, size=512)
    
    # Enhance
    try:
        result = enhance_burst(pipeline, burst, num_steps=args.steps)
        enhanced = result['enhanced']
        
        # Save output
        save_output(enhanced, args.output)
        
        # Print stats
        print("\n" + "="*70)
        print("Enhancement Statistics")
        print("="*70)
        print(f"Latency: {result['latency_ms']:.2f}ms")
        print(f"Steps: {result.get('num_steps', args.steps)}")
        print(f"Device: {args.device}")
        print(f"Output shape: {enhanced.shape}")
        print("="*70)
        
    except Exception as e:
        print(f"Error during enhancement: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
