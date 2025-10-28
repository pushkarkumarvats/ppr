"""
Quality Metrics for RAW Image Enhancement

This module implements comprehensive quality metrics for evaluating
RAW image processing results, including traditional metrics (PSNR, SSIM)
and perceptual metrics (LPIPS, NIQE).

Key Components:
- PSNR: Peak Signal-to-Noise Ratio
- SSIM: Structural Similarity Index
- LPIPS: Learned Perceptual Image Patch Similarity
- NIQE: Natural Image Quality Evaluator
- HallucinationDetector: Detect unrealistic generated details
- TemporalStabilityMetric: Measure temporal flickering
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np
from math import exp


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """
    Compute Peak Signal-to-Noise Ratio.
    
    Args:
        pred: Predicted image [B, C, H, W]
        target: Ground truth image [B, C, H, W]
        max_val: Maximum pixel value
        
    Returns:
        PSNR in dB
    """
    mse = F.mse_loss(pred, target, reduction='mean')
    if mse == 0:
        return torch.tensor(float('inf'))
    return 20 * torch.log10(max_val / torch.sqrt(mse))


def gaussian_kernel(kernel_size: int, sigma: float, channels: int) -> torch.Tensor:
    """Create 2D Gaussian kernel."""
    x = torch.arange(kernel_size).float() - kernel_size // 2
    gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    kernel_1d = gauss / gauss.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel = kernel_2d.expand(channels, 1, kernel_size, kernel_size).contiguous()
    return kernel


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    max_val: float = 1.0,
    return_map: bool = False
) -> torch.Tensor:
    """
    Compute Structural Similarity Index (SSIM).
    
    Args:
        pred: Predicted image [B, C, H, W]
        target: Ground truth image [B, C, H, W]
        window_size: Size of Gaussian window
        sigma: Gaussian standard deviation
        max_val: Maximum pixel value
        return_map: If True, return SSIM map instead of average
        
    Returns:
        SSIM score or map
    """
    C = pred.shape[1]
    
    # Create Gaussian window
    window = gaussian_kernel(window_size, sigma, C)
    window = window.to(pred.device).type(pred.dtype)
    
    # Constants for stability
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    
    # Compute local means
    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=C)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # Compute local variances and covariance
    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=C) - mu1_mu2
    
    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if return_map:
        return ssim_map
    else:
        return ssim_map.mean()


class LPIPS(nn.Module):
    """
    Learned Perceptual Image Patch Similarity.
    
    This is a simplified version that works directly on RAW Bayer images.
    For RGB images, consider using the official LPIPS implementation.
    """
    
    def __init__(self, in_channels: int = 4):
        super().__init__()
        
        # Feature extraction network
        self.layers = nn.ModuleList([
            self._make_layer(in_channels, 64),
            self._make_layer(64, 128),
            self._make_layer(128, 256),
            self._make_layer(256, 512),
            self._make_layer(512, 512)
        ])
        
        # Linear layers for each scale
        self.lins = nn.ModuleList([
            nn.Conv2d(64, 1, 1, bias=False),
            nn.Conv2d(128, 1, 1, bias=False),
            nn.Conv2d(256, 1, 1, bias=False),
            nn.Conv2d(512, 1, 1, bias=False),
            nn.Conv2d(512, 1, 1, bias=False)
        ])
        
        # Initialize linear layers
        for lin in self.lins:
            nn.init.constant_(lin.weight, 1.0)
        
        # Freeze after initialization
        self.requires_grad_(False)
    
    def _make_layer(self, in_ch: int, out_ch: int) -> nn.Module:
        """Create a feature extraction layer."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute LPIPS distance.
        
        Args:
            pred: Predicted image [B, C, H, W]
            target: Ground truth image [B, C, H, W]
            
        Returns:
            LPIPS distance (lower is better)
        """
        # Extract features at multiple scales
        feats_pred = []
        feats_target = []
        
        x_pred = pred
        x_target = target
        
        for layer in self.layers:
            x_pred = layer(x_pred)
            x_target = layer(x_target)
            feats_pred.append(x_pred)
            feats_target.append(x_target)
            # Downsample for next scale
            x_pred = F.avg_pool2d(x_pred, 2)
            x_target = F.avg_pool2d(x_target, 2)
        
        # Compute normalized differences at each scale
        diffs = []
        for feat_pred, feat_target, lin in zip(feats_pred, feats_target, self.lins):
            # Normalize features
            feat_pred = feat_pred / (feat_pred.norm(dim=1, keepdim=True) + 1e-8)
            feat_target = feat_target / (feat_target.norm(dim=1, keepdim=True) + 1e-8)
            
            # Squared difference
            diff = (feat_pred - feat_target) ** 2
            
            # Apply linear weighting and spatial pooling
            diff = lin(diff)
            diffs.append(diff.mean())
        
        # Average across scales
        return sum(diffs) / len(diffs)


class NIQE(nn.Module):
    """
    Natural Image Quality Evaluator.
    
    No-reference quality metric that measures deviation from natural
    image statistics. Lower scores indicate better quality.
    
    This is a simplified implementation for RAW images.
    """
    
    def __init__(self, patch_size: int = 96):
        super().__init__()
        self.patch_size = patch_size
        
        # Mean and covariance of natural image features
        # These should be computed from a dataset of pristine images
        # For now, we use placeholder values
        self.register_buffer('mu_pristine', torch.zeros(36))
        self.register_buffer('cov_pristine', torch.eye(36))
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract NIQE features from image patches.
        
        Features include:
        - Local mean and variance
        - Gradient statistics
        - Log-Gabor filter responses
        """
        B, C, H, W = x.shape
        
        features = []
        
        # Reshape to patches
        patches = x.unfold(2, self.patch_size, self.patch_size // 2)
        patches = patches.unfold(3, self.patch_size, self.patch_size // 2)
        patches = patches.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        
        num_patches = patches.shape[2]
        
        for i in range(num_patches):
            patch = patches[:, :, i]  # [B, C, P, P]
            
            # Local mean and variance
            mu = patch.mean(dim=[2, 3])  # [B, C]
            var = patch.var(dim=[2, 3])  # [B, C]
            
            # Gradient magnitude
            grad_x = patch[:, :, :, 1:] - patch[:, :, :, :-1]
            grad_y = patch[:, :, 1:, :] - patch[:, :, :-1, :]
            grad_mag = torch.sqrt(grad_x.pow(2).mean(dim=[2, 3]) + 
                                 grad_y.pow(2).mean(dim=[2, 3]))  # [B, C]
            
            # Combine features
            patch_features = torch.cat([mu, var, grad_mag], dim=1)  # [B, C*3]
            features.append(patch_features)
        
        # Average across patches
        features = torch.stack(features, dim=1).mean(dim=1)  # [B, C*3]
        
        return features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute NIQE score.
        
        Args:
            x: Input image [B, C, H, W]
            
        Returns:
            NIQE score (lower is better)
        """
        features = self.extract_features(x)
        
        # Compute Mahalanobis distance to pristine distribution
        # distance = sqrt((f - mu)^T * Cov^-1 * (f - mu))
        
        mu = self.mu_pristine[:features.shape[1]]
        cov_inv = torch.inverse(self.cov_pristine[:features.shape[1], :features.shape[1]])
        
        diff = features - mu.unsqueeze(0)
        distance = torch.sqrt((diff @ cov_inv * diff).sum(dim=1))
        
        return distance.mean()


class HallucinationDetector(nn.Module):
    """
    Detects hallucinated details in enhanced images.
    
    Compares the enhanced image against the input burst to identify
    details that are not supported by the input data.
    """
    
    def __init__(self, threshold: float = 0.3):
        super().__init__()
        self.threshold = threshold
    
    def compute_detail_map(self, x: torch.Tensor) -> torch.Tensor:
        """Extract high-frequency detail map."""
        # Multi-scale Laplacian
        detail = 0
        for sigma in [0.5, 1.0, 2.0]:
            kernel_size = int(6 * sigma + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            # Gaussian blur
            kernel = gaussian_kernel(kernel_size, sigma, x.shape[1])
            kernel = kernel.to(x.device).type(x.dtype)
            blurred = F.conv2d(x, kernel, padding=kernel_size // 2, groups=x.shape[1])
            
            # Difference (high-freq)
            detail = detail + (x - blurred).abs()
        
        return detail / 3.0
    
    def forward(
        self,
        enhanced: torch.Tensor,
        burst: torch.Tensor,
        reference: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Detect hallucinated details.
        
        Args:
            enhanced: Enhanced output [B, C, H, W]
            burst: Input burst [B, T, C, H, W]
            reference: Optional ground truth [B, C, H, W]
            
        Returns:
            Dictionary with hallucination metrics
        """
        # Extract details
        enhanced_detail = self.compute_detail_map(enhanced)
        
        # Compute detail support from burst
        burst_details = []
        for t in range(burst.shape[1]):
            detail = self.compute_detail_map(burst[:, t])
            burst_details.append(detail)
        
        burst_details = torch.stack(burst_details, dim=1)  # [B, T, C, H, W]
        burst_mean = burst_details.mean(dim=1)  # [B, C, H, W]
        burst_std = burst_details.std(dim=1)  # [B, C, H, W]
        
        # Hallucination score: enhanced detail that exceeds burst statistics
        hallucination_map = F.relu(enhanced_detail - (burst_mean + 2 * burst_std))
        hallucination_score = hallucination_map.mean()
        
        # Percentage of pixels with hallucinations
        hallucination_ratio = (hallucination_map > self.threshold).float().mean()
        
        results = {
            'hallucination_score': hallucination_score,
            'hallucination_ratio': hallucination_ratio,
            'hallucination_map': hallucination_map
        }
        
        # If reference available, compute precision/recall
        if reference is not None:
            ref_detail = self.compute_detail_map(reference)
            
            # True positives: detail in both enhanced and reference
            tp = ((enhanced_detail > self.threshold) & 
                  (ref_detail > self.threshold)).float().mean()
            
            # False positives: detail in enhanced but not reference
            fp = ((enhanced_detail > self.threshold) & 
                  (ref_detail <= self.threshold)).float().mean()
            
            # False negatives: detail in reference but not enhanced
            fn = ((enhanced_detail <= self.threshold) & 
                  (ref_detail > self.threshold)).float().mean()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            results['precision'] = precision
            results['recall'] = recall
            results['f1'] = f1
        
        return results


class TemporalStabilityMetric(nn.Module):
    """
    Measures temporal stability (flickering) in video sequences.
    
    Computes frame-to-frame consistency with motion compensation.
    """
    
    def __init__(self):
        super().__init__()
    
    def warp_image(
        self,
        img: torch.Tensor,
        flow: torch.Tensor
    ) -> torch.Tensor:
        """Warp image using optical flow."""
        B, C, H, W = img.shape
        
        # Create sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=img.device),
            torch.arange(W, device=img.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).float()
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)
        
        # Add flow
        warped_grid = grid + flow
        
        # Normalize
        warped_grid[:, 0] = 2.0 * warped_grid[:, 0] / (W - 1) - 1.0
        warped_grid[:, 1] = 2.0 * warped_grid[:, 1] / (H - 1) - 1.0
        
        warped_grid = warped_grid.permute(0, 2, 3, 1)
        
        return F.grid_sample(img, warped_grid, align_corners=True, padding_mode='border')
    
    def forward(
        self,
        frames: torch.Tensor,
        flows: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute temporal stability metrics.
        
        Args:
            frames: Video frames [B, T, C, H, W]
            flows: Optical flows [B, T-1, 2, H, W]
            
        Returns:
            Dictionary with stability metrics
        """
        B, T = frames.shape[:2]
        
        frame_diffs = []
        warped_diffs = []
        
        for t in range(T - 1):
            frame_t = frames[:, t]
            frame_t1 = frames[:, t + 1]
            
            # Direct frame difference
            diff = (frame_t - frame_t1).abs().mean()
            frame_diffs.append(diff)
            
            # Motion-compensated difference
            if flows is not None:
                flow = flows[:, t]
                frame_t_warped = self.warp_image(frame_t, flow)
                warped_diff = (frame_t_warped - frame_t1).abs().mean()
                warped_diffs.append(warped_diff)
        
        results = {
            'temporal_variance': torch.stack(frame_diffs).var(),
            'average_frame_diff': torch.stack(frame_diffs).mean()
        }
        
        if flows is not None:
            results['warped_diff'] = torch.stack(warped_diffs).mean()
            results['flicker_score'] = torch.stack(warped_diffs).var()
        
        return results


class MetricCalculator:
    """Unified interface for computing all metrics."""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.lpips = LPIPS().to(device)
        self.niqe = NIQE().to(device)
        self.hallucination_detector = HallucinationDetector().to(device)
        self.temporal_stability = TemporalStabilityMetric().to(device)
    
    def compute_all_metrics(
        self,
        pred: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        burst: Optional[torch.Tensor] = None,
        flows: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute all available metrics.
        
        Args:
            pred: Predicted image(s) [B, C, H, W] or [B, T, C, H, W]
            target: Ground truth [B, C, H, W]
            burst: Input burst [B, T, C, H, W]
            flows: Optical flows [B, T-1, 2, H, W]
            
        Returns:
            Dictionary of metric values
        """
        metrics = {}
        
        # Handle video input
        if pred.ndim == 5:
            # Compute temporal metrics
            temporal_metrics = self.temporal_stability(pred, flows)
            metrics.update({f'temporal_{k}': v.item() 
                          for k, v in temporal_metrics.items()})
            
            # Use middle frame for other metrics
            pred = pred[:, pred.shape[1] // 2]
        
        # Reference-based metrics (if target available)
        if target is not None:
            metrics['psnr'] = psnr(pred, target).item()
            metrics['ssim'] = ssim(pred, target).item()
            metrics['lpips'] = self.lpips(pred, target).item()
        
        # No-reference metrics
        metrics['niqe'] = self.niqe(pred).item()
        
        # Hallucination detection (if burst available)
        if burst is not None:
            halluc_metrics = self.hallucination_detector(pred, burst, target)
            metrics.update({f'hallucination_{k}': v.item() if isinstance(v, torch.Tensor) and v.numel() == 1 else v
                          for k, v in halluc_metrics.items() 
                          if k != 'hallucination_map'})
        
        return metrics


if __name__ == "__main__":
    # Test metrics
    print("Testing quality metrics...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    B, C, H, W = 2, 4, 256, 256
    T = 8
    
    # Create dummy data
    pred = torch.randn(B, C, H, W, device=device)
    target = torch.randn(B, C, H, W, device=device)
    burst = torch.randn(B, T, C, H, W, device=device)
    video = torch.randn(B, T, C, H, W, device=device)
    flows = torch.randn(B, T-1, 2, H, W, device=device)
    
    print("\n1. Testing PSNR...")
    psnr_val = psnr(pred, target)
    print(f"   PSNR: {psnr_val:.2f} dB")
    
    print("\n2. Testing SSIM...")
    ssim_val = ssim(pred, target)
    print(f"   SSIM: {ssim_val:.4f}")
    
    print("\n3. Testing LPIPS...")
    lpips_metric = LPIPS().to(device)
    lpips_val = lpips_metric(pred, target)
    print(f"   LPIPS: {lpips_val:.4f}")
    
    print("\n4. Testing NIQE...")
    niqe_metric = NIQE().to(device)
    niqe_val = niqe_metric(pred)
    print(f"   NIQE: {niqe_val:.4f}")
    
    print("\n5. Testing HallucinationDetector...")
    halluc_detector = HallucinationDetector().to(device)
    halluc_metrics = halluc_detector(pred, burst, target)
    print(f"   Hallucination metrics:")
    for k, v in halluc_metrics.items():
        if k != 'hallucination_map':
            print(f"   - {k}: {v.item():.4f}")
    
    print("\n6. Testing TemporalStabilityMetric...")
    temporal_metric = TemporalStabilityMetric().to(device)
    temporal_metrics = temporal_metric(video, flows)
    print(f"   Temporal stability metrics:")
    for k, v in temporal_metrics.items():
        print(f"   - {k}: {v.item():.4f}")
    
    print("\n7. Testing MetricCalculator...")
    calculator = MetricCalculator(device)
    all_metrics = calculator.compute_all_metrics(pred, target, burst)
    print(f"   All metrics:")
    for k, v in all_metrics.items():
        if isinstance(v, (int, float)):
            print(f"   - {k}: {v:.4f}")
    
    print("\n✓ All metrics tested successfully!")
