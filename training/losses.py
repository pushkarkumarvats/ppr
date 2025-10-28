"""
Loss Functions for RAW Image Enhancement

This module implements specialized loss functions for training diffusion models
on RAW images, including perceptual losses, hallucination penalties, and 
temporal consistency constraints.

Key Components:
- RAWPerceptualLoss: Feature-based perceptual loss for RAW images
- HallucinationPenaltyLoss: Prevents unrealistic detail generation
- TemporalConsistencyLoss: Enforces temporal stability across bursts
- EdgePreservationLoss: Preserves high-frequency details
- ChromaticAberrationLoss: Penalizes color fringing artifacts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


class RAWPerceptualLoss(nn.Module):
    """
    Perceptual loss for RAW images using multi-scale feature matching.
    
    Unlike traditional perceptual losses that use ImageNet-pretrained networks,
    this computes features directly from RAW Bayer patterns to preserve
    color accuracy and fine details.
    """
    
    def __init__(
        self,
        in_channels: int = 4,  # Bayer RGGB
        feature_dims: Tuple[int, ...] = (32, 64, 128, 256),
        feature_weights: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0),
        normalize_features: bool = True
    ):
        super().__init__()
        self.feature_weights = feature_weights
        self.normalize_features = normalize_features
        
        # Build multi-scale feature extractor
        self.feature_extractors = nn.ModuleList()
        prev_dim = in_channels
        
        for dim in feature_dims:
            extractor = nn.Sequential(
                nn.Conv2d(prev_dim, dim, 3, padding=1),
                nn.GroupNorm(8, dim),
                nn.SiLU(),
                nn.Conv2d(dim, dim, 3, padding=1),
                nn.GroupNorm(8, dim),
                nn.SiLU()
            )
            self.feature_extractors.append(extractor)
            prev_dim = dim
        
        # Freeze feature extractor after initialization
        self.requires_grad_(False)
    
    def extract_features(self, x: torch.Tensor) -> list:
        """Extract multi-scale features from RAW image."""
        features = []
        current = x
        
        for extractor in self.feature_extractors:
            current = extractor(current)
            features.append(current)
            # Downsample for next scale
            current = F.avg_pool2d(current, 2)
        
        return features
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute perceptual loss between predicted and target RAW images.
        
        Args:
            pred: Predicted RAW image [B, 4, H, W]
            target: Ground truth RAW image [B, 4, H, W]
            mask: Optional spatial mask [B, 1, H, W]
            
        Returns:
            Perceptual loss scalar
        """
        pred_features = self.extract_features(pred)
        target_features = self.extract_features(target)
        
        loss = 0.0
        for weight, pred_feat, target_feat in zip(
            self.feature_weights, pred_features, target_features
        ):
            if self.normalize_features:
                pred_feat = F.normalize(pred_feat, dim=1)
                target_feat = F.normalize(target_feat, dim=1)
            
            feat_loss = F.l1_loss(pred_feat, target_feat, reduction='none')
            
            if mask is not None:
                # Downsample mask to match feature resolution
                h, w = pred_feat.shape[2:]
                feat_mask = F.interpolate(mask, size=(h, w), mode='nearest')
                feat_loss = feat_loss * feat_mask
                feat_loss = feat_loss.sum() / (feat_mask.sum() + 1e-8)
            else:
                feat_loss = feat_loss.mean()
            
            loss = loss + weight * feat_loss
        
        return loss / len(self.feature_weights)


class HallucinationPenaltyLoss(nn.Module):
    """
    Penalizes hallucinated details that are not supported by the input.
    
    This loss prevents the diffusion model from generating unrealistic
    high-frequency details by enforcing consistency with the input burst.
    """
    
    def __init__(
        self,
        frequency_threshold: float = 0.3,
        spatial_threshold: float = 0.5,
        penalty_weight: float = 1.0
    ):
        super().__init__()
        self.frequency_threshold = frequency_threshold
        self.spatial_threshold = spatial_threshold
        self.penalty_weight = penalty_weight
    
    def extract_high_freq(self, x: torch.Tensor) -> torch.Tensor:
        """Extract high-frequency components using Laplacian."""
        kernel = torch.tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        
        # Apply to each channel
        high_freq = []
        for c in range(x.shape[1]):
            hf = F.conv2d(x[:, c:c+1], kernel, padding=1)
            high_freq.append(hf)
        
        return torch.cat(high_freq, dim=1)
    
    def compute_support_map(
        self,
        burst: torch.Tensor,
        pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute spatial support map indicating where details are valid.
        
        Args:
            burst: Input burst frames [B, T, 4, H, W]
            pred: Predicted output [B, 4, H, W]
            
        Returns:
            Support map [B, 1, H, W] with values in [0, 1]
        """
        B, T = burst.shape[:2]
        
        # Average high-frequency content across burst
        burst_hf = []
        for t in range(T):
            hf = self.extract_high_freq(burst[:, t])
            burst_hf.append(hf.abs())
        
        burst_hf = torch.stack(burst_hf, dim=1)  # [B, T, 4, H, W]
        
        # Compute mean and variance
        hf_mean = burst_hf.mean(dim=1)  # [B, 4, H, W]
        hf_var = burst_hf.var(dim=1)  # [B, 4, H, W]
        
        # Support is high where burst has consistent high-freq content
        support = hf_mean / (hf_var.sqrt() + 1e-6)
        support = support.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Normalize to [0, 1]
        support = torch.sigmoid(support - self.spatial_threshold)
        
        return support
    
    def forward(
        self,
        pred: torch.Tensor,
        burst: torch.Tensor,
        reference: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute hallucination penalty.
        
        Args:
            pred: Predicted output [B, 4, H, W]
            burst: Input burst [B, T, 4, H, W]
            reference: Optional ground truth [B, 4, H, W]
            
        Returns:
            Hallucination penalty loss
        """
        # Extract high-frequency details from prediction
        pred_hf = self.extract_high_freq(pred)
        
        # Compute support map from burst
        support = self.compute_support_map(burst, pred)
        
        # Penalize high-frequency content where support is low
        unsupported_hf = pred_hf.abs() * (1 - support)
        
        # Higher penalty for very high frequencies
        penalty = unsupported_hf.pow(2).mean()
        
        # If reference available, also penalize deviation
        if reference is not None:
            ref_hf = self.extract_high_freq(reference)
            deviation = (pred_hf - ref_hf).abs() * (1 - support)
            penalty = penalty + deviation.mean()
        
        return self.penalty_weight * penalty


class TemporalConsistencyLoss(nn.Module):
    """
    Enforces temporal consistency across burst frames.
    
    This loss ensures that the output remains stable when processing
    slightly different frames from the same burst sequence.
    """
    
    def __init__(
        self,
        use_optical_flow: bool = True,
        flow_weight: float = 1.0,
        feature_weight: float = 1.0
    ):
        super().__init__()
        self.use_optical_flow = use_optical_flow
        self.flow_weight = flow_weight
        self.feature_weight = feature_weight
        
        # Simple feature extractor for temporal consistency
        self.feature_net = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1)
        )
    
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
        grid = torch.stack([grid_x, grid_y], dim=0).float()  # [2, H, W]
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, 2, H, W]
        
        # Add flow to grid
        warped_grid = grid + flow
        
        # Normalize to [-1, 1]
        warped_grid[:, 0] = 2.0 * warped_grid[:, 0] / (W - 1) - 1.0
        warped_grid[:, 1] = 2.0 * warped_grid[:, 1] / (H - 1) - 1.0
        
        # Permute to [B, H, W, 2]
        warped_grid = warped_grid.permute(0, 2, 3, 1)
        
        # Sample
        return F.grid_sample(img, warped_grid, align_corners=True, padding_mode='border')
    
    def forward(
        self,
        pred_t0: torch.Tensor,
        pred_t1: torch.Tensor,
        flow_01: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute temporal consistency loss between two predictions.
        
        Args:
            pred_t0: Prediction from frame t [B, 4, H, W]
            pred_t1: Prediction from frame t+1 [B, 4, H, W]
            flow_01: Optical flow from t0 to t1 [B, 2, H, W]
            
        Returns:
            Temporal consistency loss
        """
        loss = 0.0
        
        if self.use_optical_flow and flow_01 is not None:
            # Warp pred_t0 to t1 using flow
            pred_t0_warped = self.warp_image(pred_t0, flow_01)
            
            # Compute photometric loss
            photo_loss = F.l1_loss(pred_t0_warped, pred_t1)
            loss = loss + self.flow_weight * photo_loss
        
        # Feature-level consistency
        feat_t0 = self.feature_net(pred_t0)
        feat_t1 = self.feature_net(pred_t1)
        
        if self.use_optical_flow and flow_01 is not None:
            feat_t0 = self.warp_image(feat_t0, flow_01)
        
        feat_loss = F.l1_loss(feat_t0, feat_t1)
        loss = loss + self.feature_weight * feat_loss
        
        return loss


class EdgePreservationLoss(nn.Module):
    """
    Preserves edges and high-frequency details in the output.
    
    This loss is crucial for maintaining sharp details in RAW images
    while allowing smooth regions to be denoised.
    """
    
    def __init__(
        self,
        edge_weight: float = 1.0,
        gradient_weight: float = 0.5
    ):
        super().__init__()
        self.edge_weight = edge_weight
        self.gradient_weight = gradient_weight
        
        # Sobel kernels
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    
    def compute_gradients(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute image gradients using Sobel operators."""
        grad_x = []
        grad_y = []
        
        for c in range(x.shape[1]):
            gx = F.conv2d(x[:, c:c+1], self.sobel_x, padding=1)
            gy = F.conv2d(x[:, c:c+1], self.sobel_y, padding=1)
            grad_x.append(gx)
            grad_y.append(gy)
        
        return torch.cat(grad_x, dim=1), torch.cat(grad_y, dim=1)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        edge_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute edge preservation loss.
        
        Args:
            pred: Predicted image [B, 4, H, W]
            target: Ground truth image [B, 4, H, W]
            edge_mask: Optional edge importance mask [B, 1, H, W]
            
        Returns:
            Edge preservation loss
        """
        # Compute gradients
        pred_gx, pred_gy = self.compute_gradients(pred)
        target_gx, target_gy = self.compute_gradients(target)
        
        # Gradient magnitude
        pred_mag = torch.sqrt(pred_gx.pow(2) + pred_gy.pow(2) + 1e-8)
        target_mag = torch.sqrt(target_gx.pow(2) + target_gy.pow(2) + 1e-8)
        
        # Edge loss on gradient magnitude
        edge_loss = F.l1_loss(pred_mag, target_mag, reduction='none')
        
        if edge_mask is not None:
            edge_loss = edge_loss * edge_mask
            edge_loss = edge_loss.sum() / (edge_mask.sum() + 1e-8)
        else:
            edge_loss = edge_loss.mean()
        
        # Gradient direction loss
        grad_loss = (
            F.l1_loss(pred_gx, target_gx, reduction='mean') +
            F.l1_loss(pred_gy, target_gy, reduction='mean')
        )
        
        return self.edge_weight * edge_loss + self.gradient_weight * grad_loss


class ChromaticAberrationLoss(nn.Module):
    """
    Penalizes chromatic aberration (color fringing) artifacts.
    
    This loss helps the lens aberration correction module by providing
    a signal to reduce color misalignment.
    """
    
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
    
    def compute_color_misalignment(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute color channel misalignment in Bayer pattern.
        
        For RGGB Bayer: channels are [R, G1, G2, B]
        We check for misalignment between color channels.
        """
        r, g1, g2, b = x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4]
        
        # Average greens
        g = (g1 + g2) / 2.0
        
        # Compute gradient misalignment between R-G and B-G
        rg_diff = r - g
        bg_diff = b - g
        
        # Look for high-frequency misalignment (edges)
        rg_grad = F.avg_pool2d(rg_diff.abs(), 3, stride=1, padding=1)
        bg_grad = F.avg_pool2d(bg_diff.abs(), 3, stride=1, padding=1)
        
        # Chromatic aberration appears as correlated color gradients
        misalignment = rg_grad * bg_grad
        
        return misalignment
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute chromatic aberration loss.
        
        Args:
            pred: Predicted RAW image [B, 4, H, W]
            target: Ground truth RAW image [B, 4, H, W]
            
        Returns:
            Chromatic aberration loss
        """
        pred_ca = self.compute_color_misalignment(pred)
        target_ca = self.compute_color_misalignment(target)
        
        # Penalize excess chromatic aberration in prediction
        loss = F.l1_loss(pred_ca, target_ca)
        
        return self.weight * loss


class CombinedLoss(nn.Module):
    """
    Combined loss function for training RAW diffusion models.
    
    This combines all loss components with configurable weights.
    """
    
    def __init__(
        self,
        l1_weight: float = 1.0,
        perceptual_weight: float = 1.0,
        hallucination_weight: float = 0.5,
        temporal_weight: float = 0.3,
        edge_weight: float = 0.5,
        chroma_weight: float = 0.2,
        use_temporal: bool = True
    ):
        super().__init__()
        
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.hallucination_weight = hallucination_weight
        self.temporal_weight = temporal_weight
        self.edge_weight = edge_weight
        self.chroma_weight = chroma_weight
        self.use_temporal = use_temporal
        
        # Initialize loss modules
        self.perceptual_loss = RAWPerceptualLoss()
        self.hallucination_loss = HallucinationPenaltyLoss()
        self.temporal_loss = TemporalConsistencyLoss() if use_temporal else None
        self.edge_loss = EdgePreservationLoss()
        self.chroma_loss = ChromaticAberrationLoss()
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        burst: Optional[torch.Tensor] = None,
        pred_t1: Optional[torch.Tensor] = None,
        flow_01: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            pred: Predicted image [B, 4, H, W]
            target: Ground truth image [B, 4, H, W]
            burst: Input burst [B, T, 4, H, W] (for hallucination penalty)
            pred_t1: Prediction at t+1 (for temporal consistency)
            flow_01: Optical flow (for temporal consistency)
            mask: Optional spatial mask [B, 1, H, W]
            
        Returns:
            Dictionary of loss components and total loss
        """
        losses = {}
        
        # L1 loss
        l1_loss = F.l1_loss(pred, target, reduction='none')
        if mask is not None:
            l1_loss = (l1_loss * mask).sum() / (mask.sum() + 1e-8)
        else:
            l1_loss = l1_loss.mean()
        losses['l1'] = self.l1_weight * l1_loss
        
        # Perceptual loss
        if self.perceptual_weight > 0:
            perceptual = self.perceptual_loss(pred, target, mask)
            losses['perceptual'] = self.perceptual_weight * perceptual
        
        # Hallucination penalty
        if self.hallucination_weight > 0 and burst is not None:
            hallucination = self.hallucination_loss(pred, burst, target)
            losses['hallucination'] = self.hallucination_weight * hallucination
        
        # Temporal consistency
        if self.use_temporal and self.temporal_weight > 0 and pred_t1 is not None:
            temporal = self.temporal_loss(pred, pred_t1, flow_01)
            losses['temporal'] = self.temporal_weight * temporal
        
        # Edge preservation
        if self.edge_weight > 0:
            edge_mask = mask if mask is not None else None
            edge = self.edge_loss(pred, target, edge_mask)
            losses['edge'] = self.edge_weight * edge
        
        # Chromatic aberration
        if self.chroma_weight > 0:
            chroma = self.chroma_loss(pred, target)
            losses['chroma'] = self.chroma_weight * chroma
        
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses


if __name__ == "__main__":
    # Test loss functions
    print("Testing RAW loss functions...")
    
    B, C, H, W = 2, 4, 256, 256
    T = 8
    
    # Create dummy data
    pred = torch.randn(B, C, H, W)
    target = torch.randn(B, C, H, W)
    burst = torch.randn(B, T, C, H, W)
    pred_t1 = torch.randn(B, C, H, W)
    flow = torch.randn(B, 2, H, W)
    mask = torch.ones(B, 1, H, W)
    
    # Test individual losses
    print("\n1. Testing RAWPerceptualLoss...")
    perceptual_loss = RAWPerceptualLoss()
    loss = perceptual_loss(pred, target, mask)
    print(f"   Perceptual loss: {loss.item():.4f}")
    
    print("\n2. Testing HallucinationPenaltyLoss...")
    hallucination_loss = HallucinationPenaltyLoss()
    loss = hallucination_loss(pred, burst, target)
    print(f"   Hallucination loss: {loss.item():.4f}")
    
    print("\n3. Testing TemporalConsistencyLoss...")
    temporal_loss = TemporalConsistencyLoss()
    loss = temporal_loss(pred, pred_t1, flow)
    print(f"   Temporal loss: {loss.item():.4f}")
    
    print("\n4. Testing EdgePreservationLoss...")
    edge_loss = EdgePreservationLoss()
    loss = edge_loss(pred, target, mask)
    print(f"   Edge loss: {loss.item():.4f}")
    
    print("\n5. Testing ChromaticAberrationLoss...")
    chroma_loss = ChromaticAberrationLoss()
    loss = chroma_loss(pred, target)
    print(f"   Chromatic aberration loss: {loss.item():.4f}")
    
    print("\n6. Testing CombinedLoss...")
    combined_loss = CombinedLoss()
    losses = combined_loss(pred, target, burst, pred_t1, flow, mask)
    print(f"   Loss components:")
    for name, value in losses.items():
        print(f"   - {name}: {value.item():.4f}")
    
    print("\n✓ All loss functions tested successfully!")
