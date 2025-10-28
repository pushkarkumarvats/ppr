"""
Optical Flow Alignment Module
Modified RAFT (Recurrent All-Pairs Field Transforms) for RAW Bayer pattern input
Provides sub-pixel accurate alignment for burst sequences
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RAWOpticalFlow(nn.Module):
    """
    Modified RAFT optical flow for RAW Bayer input
    Estimates dense optical flow with sub-pixel accuracy
    """
    
    def __init__(
        self,
        feature_dim: int = 128,
        num_scales: int = 4,
        num_iterations: int = 12,
        corr_radius: int = 4,
        corr_levels: int = 4,
        dropout: float = 0.0,
    ):
        """
        Args:
            feature_dim: Feature dimension for correlation
            num_scales: Number of pyramid scales
            num_iterations: Number of GRU refinement iterations
            corr_radius: Correlation search radius
            corr_levels: Number of correlation pyramid levels
            dropout: Dropout rate
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_scales = num_scales
        self.num_iterations = num_iterations
        self.corr_radius = corr_radius
        self.corr_levels = corr_levels
        
        # Feature encoder for RAW Bayer (4 channels)
        self.feature_encoder = FeatureEncoder(
            in_channels=4,
            feature_dim=feature_dim,
            num_scales=num_scales
        )
        
        # Context encoder for initial flow and hidden state
        self.context_encoder = ContextEncoder(
            in_channels=4,
            hidden_dim=128,
            output_dim=256
        )
        
        # GRU-based flow refinement
        self.update_block = UpdateBlock(
            corr_levels=corr_levels,
            corr_radius=corr_radius,
            hidden_dim=128,
            input_dim=128 + 256,  # correlation features + context
        )
        
        # Flow head to predict residual flow
        self.flow_head = FlowHead(hidden_dim=128, output_dim=2)
        
        # Confidence/occlusion head
        self.confidence_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        reference_raw: torch.Tensor,
        target_raw: torch.Tensor,
        iters: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate optical flow from reference to target
        
        Args:
            reference_raw: Reference RAW image [B, 4, H, W]
            target_raw: Target RAW image [B, 4, H, W]
            iters: Number of refinement iterations (default: self.num_iterations)
            
        Returns:
            flow: Optical flow [B, 2, H, W]
            confidence: Confidence map [B, 1, H, W]
        """
        if iters is None:
            iters = self.num_iterations
        
        # Extract features at multiple scales
        ref_features = self.feature_encoder(reference_raw)
        target_features = self.feature_encoder(target_raw)
        
        # Extract context features for refinement
        context_features = self.context_encoder(reference_raw)
        
        # Initialize flow and hidden state
        B, _, H, W = reference_raw.shape
        flow = torch.zeros(B, 2, H // 8, W // 8, device=reference_raw.device)
        hidden = torch.tanh(context_features[:, :128])
        
        # Build correlation pyramid
        corr_pyramid = self._build_correlation_pyramid(
            ref_features[-1],  # Use finest scale
            target_features[-1]
        )
        
        # Iterative refinement
        flow_predictions = []
        
        for _ in range(iters):
            # Index correlation pyramid
            corr_features = self._index_correlation(corr_pyramid, flow)
            
            # Concatenate with context
            flow_context = torch.cat([corr_features, context_features], dim=1)
            
            # Update flow and hidden state
            hidden, delta_flow = self.update_block(hidden, flow_context, flow)
            
            # Predict residual flow
            flow = flow + delta_flow
            
            flow_predictions.append(flow)
        
        # Upsample flow to original resolution
        flow_up = self._upsample_flow(flow, scale_factor=8)
        
        # Predict confidence/occlusion mask
        confidence = self.confidence_head(hidden)
        confidence_up = F.interpolate(
            confidence,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )
        
        return flow_up, confidence_up
    
    def _build_correlation_pyramid(
        self,
        fmap1: torch.Tensor,
        fmap2: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Build correlation pyramid for efficient search
        
        Args:
            fmap1: Features from reference [B, C, H, W]
            fmap2: Features from target [B, C, H, W]
            
        Returns:
            List of correlation volumes at different scales
        """
        B, C, H, W = fmap1.shape
        
        # Normalize features
        fmap1 = fmap1 / (torch.norm(fmap1, dim=1, keepdim=True) + 1e-8)
        fmap2 = fmap2 / (torch.norm(fmap2, dim=1, keepdim=True) + 1e-8)
        
        # Compute all-pairs correlation
        corr = torch.einsum('bchw,bcHW->bhwHW', fmap1, fmap2)
        corr = corr.view(B, H, W, H * W)
        
        # Build pyramid by average pooling
        corr_pyramid = [corr]
        
        for _ in range(self.corr_levels - 1):
            # Pool target dimension
            corr = corr.view(B, H, W, H, W)
            corr = F.avg_pool2d(corr.permute(0, 3, 4, 1, 2), 2, stride=2)
            corr = corr.permute(0, 3, 4, 1, 2)
            H, W = H // 2, W // 2
            corr = corr.reshape(B, H, W, -1)
            corr_pyramid.append(corr)
        
        return corr_pyramid
    
    def _index_correlation(
        self,
        corr_pyramid: List[torch.Tensor],
        flow: torch.Tensor
    ) -> torch.Tensor:
        """
        Index correlation pyramid using current flow estimate
        
        Args:
            corr_pyramid: List of correlation volumes
            flow: Current flow estimate [B, 2, H, W]
            
        Returns:
            Correlation features [B, corr_levels * (2*radius+1)^2, H, W]
        """
        B, _, H, W = flow.shape
        radius = self.corr_radius
        
        # Generate sampling grid around flow
        dx = torch.linspace(-radius, radius, 2 * radius + 1, device=flow.device)
        dy = torch.linspace(-radius, radius, 2 * radius + 1, device=flow.device)
        delta = torch.stack(torch.meshgrid(dy, dx, indexing='ij'), dim=-1)
        delta = delta.view(1, 2 * radius + 1, 2 * radius + 1, 2)
        
        # Sample from each pyramid level
        corr_samples = []
        
        for i, corr in enumerate(corr_pyramid):
            # Scale flow to current pyramid level
            scale = 2 ** i
            flow_scaled = flow / scale
            
            # Create sampling coordinates
            centroid = flow_scaled.permute(0, 2, 3, 1).unsqueeze(1).unsqueeze(1)
            coords = centroid + delta  # [B, 2r+1, 2r+1, H, W, 2]
            coords = coords.view(B, -1, H, W, 2)
            
            # Bilinear sampling
            corr_level = corr.view(B, H, W, -1).permute(0, 3, 1, 2)
            sampled = self._bilinear_sample(corr_level, coords)
            
            corr_samples.append(sampled)
        
        # Concatenate all levels
        corr_features = torch.cat(corr_samples, dim=1)
        
        return corr_features
    
    @staticmethod
    def _bilinear_sample(
        volume: torch.Tensor,
        coords: torch.Tensor
    ) -> torch.Tensor:
        """Bilinear sampling from correlation volume"""
        B, C, H, W = volume.shape
        
        # Normalize coordinates to [-1, 1]
        coords = coords.clone()
        coords[..., 0] = 2 * coords[..., 0] / (W - 1) - 1
        coords[..., 1] = 2 * coords[..., 1] / (H - 1) - 1
        
        # Reshape for grid_sample
        coords = coords.view(B, -1, 1, 2)
        
        # Sample
        sampled = F.grid_sample(
            volume,
            coords,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )
        
        # Reshape back
        sampled = sampled.view(B, C, -1)
        
        return sampled
    
    @staticmethod
    def _upsample_flow(flow: torch.Tensor, scale_factor: int = 8) -> torch.Tensor:
        """
        Upsample flow to full resolution
        
        Args:
            flow: Coarse flow [B, 2, H, W]
            scale_factor: Upsampling factor
            
        Returns:
            Upsampled flow [B, 2, H*scale, W*scale]
        """
        # Bilinear upsampling
        flow_up = F.interpolate(
            flow,
            scale_factor=scale_factor,
            mode='bilinear',
            align_corners=False
        )
        
        # Scale flow values
        flow_up = flow_up * scale_factor
        
        return flow_up


class FeatureEncoder(nn.Module):
    """Multi-scale feature encoder for RAW images"""
    
    def __init__(
        self,
        in_channels: int = 4,
        feature_dim: int = 128,
        num_scales: int = 4
    ):
        super().__init__()
        
        self.num_scales = num_scales
        
        # Initial convolution for Bayer pattern
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=1, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks for feature extraction
        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 96, stride=2)
        self.layer3 = self._make_layer(96, 128, stride=2)
        self.layer4 = self._make_layer(128, feature_dim, stride=2)
        
        # Feature projection
        self.proj = nn.Conv2d(feature_dim, feature_dim, 1)
    
    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1
    ) -> nn.Module:
        """Create residual layer"""
        layers = []
        
        # Residual block with bottleneck
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        layers.append(ResidualBlock(out_channels, out_channels, 1))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale features
        
        Returns:
            List of features at different scales [1x, 1/2x, 1/4x, 1/8x]
        """
        features = []
        
        x = self.conv1(x)
        features.append(x)  # 1x
        
        x = self.layer1(x)
        features.append(x)  # 1x
        
        x = self.layer2(x)
        features.append(x)  # 1/2x
        
        x = self.layer3(x)
        features.append(x)  # 1/4x
        
        x = self.layer4(x)
        x = self.proj(x)
        features.append(x)  # 1/8x
        
        return features


class ResidualBlock(nn.Module):
    """Residual block with bottleneck"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, 1)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels // 2, 3, stride=stride, padding=1)
        self.conv3 = nn.Conv2d(out_channels // 2, out_channels, 1)
        
        self.norm1 = nn.InstanceNorm2d(out_channels // 2)
        self.norm2 = nn.InstanceNorm2d(out_channels // 2)
        self.norm3 = nn.InstanceNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.InstanceNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.norm3(out)
        
        out = out + residual
        out = self.relu(out)
        
        return out


class ContextEncoder(nn.Module):
    """Context encoder for GRU initialization"""
    
    def __init__(
        self,
        in_channels: int = 4,
        hidden_dim: int = 128,
        output_dim: int = 256
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 64, 7, padding=3)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 96, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(96, 128, 3, stride=2, padding=1)
        
        self.conv_out = nn.Conv2d(128, output_dim, 1)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv_out(x)
        
        return x


class UpdateBlock(nn.Module):
    """GRU-based flow update block"""
    
    def __init__(
        self,
        corr_levels: int = 4,
        corr_radius: int = 4,
        hidden_dim: int = 128,
        input_dim: int = 384
    ):
        super().__init__()
        
        corr_dim = corr_levels * (2 * corr_radius + 1) ** 2
        
        self.encoder = nn.Sequential(
            nn.Conv2d(corr_dim + 2, 128, 3, padding=1),  # +2 for current flow
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # GRU cell
        self.gru = ConvGRU(hidden_dim, input_dim + 64)
        
        # Flow predictor
        self.flow_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 3, padding=1)
        )
    
    def forward(
        self,
        hidden: torch.Tensor,
        context: torch.Tensor,
        flow: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden: Hidden state [B, hidden_dim, H, W]
            context: Context features [B, input_dim, H, W]
            flow: Current flow [B, 2, H, W]
            
        Returns:
            new_hidden: Updated hidden state
            delta_flow: Flow update
        """
        # Encode correlation + flow
        corr_flow = torch.cat([context, flow], dim=1)
        motion_features = self.encoder(corr_flow)
        
        # Concatenate with context
        inp = torch.cat([context, motion_features], dim=1)
        
        # GRU update
        hidden = self.gru(hidden, inp)
        
        # Predict flow residual
        delta_flow = self.flow_head(hidden)
        
        return hidden, delta_flow


class ConvGRU(nn.Module):
    """Convolutional GRU cell"""
    
    def __init__(self, hidden_dim: int, input_dim: int):
        super().__init__()
        
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
    
    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        hx = torch.cat([h, x], dim=1)
        
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        
        h = (1 - z) * h + z * q
        
        return h


class FlowHead(nn.Module):
    """Flow prediction head"""
    
    def __init__(self, hidden_dim: int = 128, output_dim: int = 2):
        super().__init__()
        
        self.conv1 = nn.Conv2d(hidden_dim, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.relu(self.conv1(x)))


class AlignmentModule(nn.Module):
    """
    Multi-frame burst alignment module
    Aligns all frames in burst to reference frame
    """
    
    def __init__(self, num_frames: int = 8):
        super().__init__()
        
        self.num_frames = num_frames
        self.optical_flow = RAWOpticalFlow()
    
    def forward(
        self,
        raw_burst: torch.Tensor,
        reference_idx: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align burst sequence to reference frame
        
        Args:
            raw_burst: Burst frames [B, N, 4, H, W]
            reference_idx: Index of reference frame
            
        Returns:
            aligned_burst: Aligned frames [B, N, 4, H, W]
            confidence_maps: Alignment confidence [B, N, 1, H, W]
        """
        B, N, C, H, W = raw_burst.shape
        
        reference_frame = raw_burst[:, reference_idx]
        
        aligned_frames = []
        confidence_maps = []
        
        for i in range(N):
            if i == reference_idx:
                # Reference frame doesn't need alignment
                aligned_frames.append(reference_frame)
                confidence_maps.append(torch.ones(B, 1, H, W, device=raw_burst.device))
            else:
                target_frame = raw_burst[:, i]
                
                # Estimate optical flow
                flow, confidence = self.optical_flow(reference_frame, target_frame)
                
                # Warp target to reference
                aligned = self.backward_warp(target_frame, flow)
                
                aligned_frames.append(aligned)
                confidence_maps.append(confidence)
        
        # Stack frames
        aligned_burst = torch.stack(aligned_frames, dim=1)
        confidence_stack = torch.stack(confidence_maps, dim=1)
        
        return aligned_burst, confidence_stack
    
    @staticmethod
    def backward_warp(frame: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """
        Backward warp frame using optical flow
        
        Args:
            frame: Frame to warp [B, C, H, W]
            flow: Optical flow [B, 2, H, W]
            
        Returns:
            Warped frame [B, C, H, W]
        """
        B, C, H, W = frame.shape
        
        # Create sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=frame.device),
            torch.arange(W, dtype=torch.float32, device=frame.device),
            indexing='ij'
        )
        
        grid = torch.stack([grid_x, grid_y], dim=0)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)
        
        # Add flow
        grid = grid + flow
        
        # Normalize to [-1, 1]
        grid[:, 0] = 2 * grid[:, 0] / (W - 1) - 1
        grid[:, 1] = 2 * grid[:, 1] / (H - 1) - 1
        
        # Transpose for grid_sample
        grid = grid.permute(0, 2, 3, 1)
        
        # Warp
        warped = F.grid_sample(
            frame,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )
        
        return warped
    
    def compute_alignment_error(
        self,
        reference: torch.Tensor,
        aligned: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute alignment error map
        
        Args:
            reference: Reference frame [B, C, H, W]
            aligned: Aligned frame [B, C, H, W]
            
        Returns:
            Error map [B, 1, H, W]
        """
        # L1 distance
        error = torch.abs(reference - aligned).mean(dim=1, keepdim=True)
        
        return error


if __name__ == "__main__":
    print("Testing Optical Flow module...")
    
    # Test with dummy RAW images
    B, C, H, W = 2, 4, 256, 256
    
    ref_raw = torch.randn(B, C, H, W)
    target_raw = ref_raw + torch.randn(B, C, H, W) * 0.1
    
    # Test optical flow
    print("\n1. Testing RAWOpticalFlow...")
    flow_model = RAWOpticalFlow()
    flow_model.eval()
    
    with torch.no_grad():
        flow, confidence = flow_model(ref_raw, target_raw, iters=6)
    
    print(f"   Flow shape: {flow.shape}")
    print(f"   Confidence shape: {confidence.shape}")
    print(f"   Flow range: [{flow.min():.2f}, {flow.max():.2f}]")
    print(f"   Confidence range: [{confidence.min():.2f}, {confidence.max():.2f}]")
    
    # Test alignment module
    print("\n2. Testing AlignmentModule...")
    burst = torch.randn(B, 8, C, H, W)
    
    align_module = AlignmentModule(num_frames=8)
    align_module.eval()
    
    with torch.no_grad():
        aligned_burst, conf_maps = align_module(burst)
    
    print(f"   Aligned burst shape: {aligned_burst.shape}")
    print(f"   Confidence maps shape: {conf_maps.shape}")
    
    # Test backward warp
    print("\n3. Testing backward_warp...")
    test_flow = torch.randn(B, 2, H, W) * 5
    warped = AlignmentModule.backward_warp(ref_raw, test_flow)
    print(f"   Warped shape: {warped.shape}")
    
    print("\n✓ All optical flow tests passed!")
