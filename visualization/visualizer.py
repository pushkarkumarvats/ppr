"""Data Visualization Tools"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import json


class TrainingVisualizer:
    """
    Visualize training progress and metrics.
    """
    
    def __init__(self, log_dir: str = './outputs/logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
    def plot_training_curves(
        self,
        metrics_file: str,
        save_path: Optional[str] = None
    ):
        """
        Plot training loss and metrics over time.
        
        Args:
            metrics_file: Path to JSON file with training metrics
            save_path: Where to save the plot
        """
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Training Loss
        ax = axes[0, 0]
        if 'train_loss' in metrics:
            epochs = range(len(metrics['train_loss']))
            ax.plot(epochs, metrics['train_loss'], 'b-', label='Train Loss', linewidth=2)
            if 'val_loss' in metrics:
                ax.plot(epochs, metrics['val_loss'], 'r--', label='Val Loss', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training Loss', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 2: PSNR
        ax = axes[0, 1]
        if 'psnr' in metrics:
            epochs = range(len(metrics['psnr']))
            ax.plot(epochs, metrics['psnr'], 'g-', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('PSNR (dB)')
            ax.set_title('PSNR over Training', fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # Plot 3: SSIM
        ax = axes[1, 0]
        if 'ssim' in metrics:
            epochs = range(len(metrics['ssim']))
            ax.plot(epochs, metrics['ssim'], 'orange', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('SSIM')
            ax.set_title('SSIM over Training', fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Learning Rate
        ax = axes[1, 1]
        if 'lr' in metrics:
            iterations = range(len(metrics['lr']))
            ax.plot(iterations, metrics['lr'], 'purple', linewidth=2)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule', fontweight='bold')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved training curves to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_image_comparison(
        self,
        images: Dict[str, torch.Tensor],
        titles: Optional[Dict[str, str]] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot comparison of input, output, and target images.
        
        Args:
            images: Dict with keys like 'input', 'output', 'target'
            titles: Custom titles for each image
            save_path: Where to save the plot
        """
        num_images = len(images)
        fig, axes = plt.subplots(1, num_images, figsize=(5*num_images, 5))
        
        if num_images == 1:
            axes = [axes]
        
        for idx, (key, img) in enumerate(images.items()):
            ax = axes[idx]
            
            # Convert tensor to numpy
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu()
                if img.dim() == 4:
                    img = img[0]  # Remove batch dim
                if img.shape[0] in [1, 3, 4]:
                    img = img.permute(1, 2, 0)  # CHW -> HWC
                img = img.numpy()
            
            # Normalize to [0, 1]
            img = np.clip(img, 0, 1)
            
            # Handle different channel counts
            if img.shape[-1] == 1:
                img = img.squeeze(-1)
                ax.imshow(img, cmap='gray')
            elif img.shape[-1] == 4:
                # Show RGB channels only for Bayer
                rgb = np.stack([img[:,:,0], img[:,:,1], img[:,:,3]], axis=-1)
                ax.imshow(rgb)
            else:
                ax.imshow(img)
            
            title = titles.get(key, key.title()) if titles else key.title()
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved comparison to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_attention_maps(
        self,
        attention_weights: torch.Tensor,
        save_path: Optional[str] = None
    ):
        """
        Visualize attention weights from transformer layers.
        
        Args:
            attention_weights: Tensor of shape [heads, H, W]
            save_path: Where to save the plot
        """
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()
        
        num_heads = attention_weights.shape[0]
        cols = min(4, num_heads)
        rows = (num_heads + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        axes = axes.flatten() if num_heads > 1 else [axes]
        
        for idx in range(num_heads):
            ax = axes[idx]
            im = ax.imshow(attention_weights[idx], cmap='viridis', interpolation='nearest')
            ax.set_title(f'Head {idx+1}', fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Hide extra subplots
        for idx in range(num_heads, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Attention Maps', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved attention maps to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_optical_flow(
        self,
        flow: torch.Tensor,
        image: Optional[torch.Tensor] = None,
        save_path: Optional[str] = None
    ):
        """
        Visualize optical flow vectors.
        
        Args:
            flow: Flow tensor [2, H, W] or [B, 2, H, W]
            image: Optional background image
            save_path: Where to save the plot
        """
        if isinstance(flow, torch.Tensor):
            flow = flow.detach().cpu()
            if flow.dim() == 4:
                flow = flow[0]  # Remove batch
        
        flow_np = flow.numpy()
        
        # Create flow visualization
        hsv = np.zeros((flow_np.shape[1], flow_np.shape[2], 3), dtype=np.uint8)
        hsv[:, :, 1] = 255
        
        # Calculate magnitude and angle
        mag, ang = cv2.cartToPolar(flow_np[0], flow_np[1])
        hsv[:, :, 0] = ang * 180 / np.pi / 2
        hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot flow colormap
        axes[0].imshow(rgb)
        axes[0].set_title('Optical Flow (Color)', fontweight='bold')
        axes[0].axis('off')
        
        # Plot flow vectors
        step = 16
        y, x = np.mgrid[step//2:flow_np.shape[1]:step, step//2:flow_np.shape[2]:step]
        fx = flow_np[0, y, x]
        fy = flow_np[1, y, x]
        
        if image is not None:
            if isinstance(image, torch.Tensor):
                image = image.detach().cpu().numpy()
            axes[1].imshow(image, cmap='gray')
        else:
            axes[1].imshow(np.zeros_like(rgb[:,:,0]), cmap='gray')
        
        axes[1].quiver(x, y, fx, fy, color='red', scale=50, width=0.003)
        axes[1].set_title('Flow Vectors', fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved flow visualization to {save_path}")
        else:
            plt.show()
        
        plt.close()


# Try to import cv2 for flow visualization
try:
    import cv2
except ImportError:
    print("OpenCV not installed. Flow visualization will be limited.")
    print("Install with: pip install opencv-python")


def main():
    """Example usage."""
    print("="*70)
    print("Data Visualization Tools")
    print("="*70)
    
    print("\nAvailable visualizations:")
    print("1. Training curves (loss, PSNR, SSIM, LR)")
    print("2. Image comparisons (input vs output)")
    print("3. Attention maps")
    print("4. Optical flow vectors")
    
    print("\nUsage examples:")
    print()
    print("# Visualize training")
    print("visualizer = TrainingVisualizer()")
    print("visualizer.plot_training_curves('training_metrics.json')")
    print()
    print("# Compare images")
    print("visualizer.plot_image_comparison({")
    print("    'input': input_img,")
    print("    'output': output_img,")
    print("    'target': target_img")
    print("})")


if __name__ == "__main__":
    main()
