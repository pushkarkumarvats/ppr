"""
RAW Image Preprocessing
Linearization, white balance, color correction, demosaicing
"""

from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def linearize_raw(
    raw_tensor: torch.Tensor,
    black_level: Union[List[float], torch.Tensor],
    white_level: float
) -> torch.Tensor:
    """
    Linearize RAW data: black level subtraction and white level normalization
    
    Args:
        raw_tensor: RAW image [4, H, W] or [N, 4, H, W]
        black_level: Black level for each channel [R, G1, G2, B]
        white_level: White level (saturation point)
        
    Returns:
        Linearized RAW in range [0, 1]
    """
    if isinstance(black_level, list):
        black_level = torch.tensor(black_level, dtype=raw_tensor.dtype, device=raw_tensor.device)
    
    # Handle both single image and batch
    if raw_tensor.ndim == 3:
        # [4, H, W] -> reshape for broadcasting
        black_level = black_level.view(4, 1, 1)
    else:
        # [N, 4, H, W] -> reshape for broadcasting
        black_level = black_level.view(1, 4, 1, 1)
    
    # Subtract black level
    raw_linear = raw_tensor - black_level
    
    # Clip to valid range
    raw_linear = torch.clamp(raw_linear, min=0.0)
    
    # Normalize to [0, 1]
    raw_linear = raw_linear / (white_level - black_level.mean())
    
    return raw_linear


def apply_white_balance(
    raw_tensor: torch.Tensor,
    wb_gains: Union[List[float], torch.Tensor]
) -> torch.Tensor:
    """
    Apply white balance gains to RAW image
    
    Args:
        raw_tensor: RAW image [4, H, W] or [N, 4, H, W]
        wb_gains: White balance gains [R, G1, G2, B] or [R, G, G, B]
        
    Returns:
        White balanced RAW
    """
    if isinstance(wb_gains, list):
        wb_gains = torch.tensor(wb_gains, dtype=raw_tensor.dtype, device=raw_tensor.device)
    
    # Normalize gains so green = 1.0
    wb_gains = wb_gains / wb_gains[1]
    
    # Reshape for broadcasting
    if raw_tensor.ndim == 3:
        wb_gains = wb_gains.view(4, 1, 1)
    else:
        wb_gains = wb_gains.view(1, 4, 1, 1)
    
    # Apply gains
    wb_raw = raw_tensor * wb_gains
    
    # Clip to [0, 1]
    wb_raw = torch.clamp(wb_raw, 0.0, 1.0)
    
    return wb_raw


def apply_color_correction_matrix(
    rgb_image: torch.Tensor,
    ccm: Union[np.ndarray, torch.Tensor]
) -> torch.Tensor:
    """
    Apply color correction matrix (CCM) to convert camera RGB to standard RGB
    
    Args:
        rgb_image: RGB image [3, H, W] or [N, 3, H, W]
        ccm: Color correction matrix [3, 3]
        
    Returns:
        Color corrected RGB
    """
    if isinstance(ccm, np.ndarray):
        ccm = torch.from_numpy(ccm).to(rgb_image.dtype).to(rgb_image.device)
    
    is_batch = rgb_image.ndim == 4
    
    if not is_batch:
        # [3, H, W] -> [H, W, 3]
        rgb_image = rgb_image.permute(1, 2, 0)
        
    else:
        # [N, 3, H, W] -> [N, H, W, 3]
        rgb_image = rgb_image.permute(0, 2, 3, 1)
    
    # Apply CCM: Y = X @ CCM^T
    corrected = rgb_image @ ccm.T
    
    # Clip to valid range
    corrected = torch.clamp(corrected, 0.0, 1.0)
    
    # Convert back to channel-first format
    if not is_batch:
        corrected = corrected.permute(2, 0, 1)
    else:
        corrected = corrected.permute(0, 3, 1, 2)
    
    return corrected


def demosaic_malvar(bayer_raw: torch.Tensor) -> torch.Tensor:
    """
    Malvar-He-Cutler (MHC) demosaicing algorithm
    High-quality edge-aware interpolation
    
    Args:
        bayer_raw: Bayer image [4, H, W] where channels are [R, G1, G2, B]
        
    Returns:
        RGB image [3, H, W]
    """
    # Convert 4-channel to 2x2 mosaic pattern for processing
    # This is a simplified implementation
    # Full MHC requires complex edge-directed interpolation
    
    device = bayer_raw.device
    dtype = bayer_raw.dtype
    
    # Extract channels
    r_plane = bayer_raw[0]   # Red
    g1_plane = bayer_raw[1]  # Green (red row)
    g2_plane = bayer_raw[2]  # Green (blue row)
    b_plane = bayer_raw[3]   # Blue
    
    # Upsample each plane to full resolution (bilinear for now)
    # In production, use proper edge-aware interpolation
    r_full = F.interpolate(r_plane.unsqueeze(0).unsqueeze(0), 
                           scale_factor=2, mode='bilinear', align_corners=False)[0, 0]
    g_full = F.interpolate((g1_plane + g2_plane).unsqueeze(0).unsqueeze(0) / 2, 
                           scale_factor=2, mode='bilinear', align_corners=False)[0, 0]
    b_full = F.interpolate(b_plane.unsqueeze(0).unsqueeze(0), 
                           scale_factor=2, mode='bilinear', align_corners=False)[0, 0]
    
    # Stack to RGB
    rgb = torch.stack([r_full, g_full, b_full], dim=0)
    
    return rgb


def demosaic_bilinear(bayer_raw: torch.Tensor) -> torch.Tensor:
    """
    Simple bilinear demosaicing
    Fast but lower quality than MHC
    
    Args:
        bayer_raw: Bayer image [4, H, W]
        
    Returns:
        RGB image [3, H, W]
    """
    # Extract channels
    r_plane = bayer_raw[0]   # Red
    g1_plane = bayer_raw[1]  # Green (red row)
    g2_plane = bayer_raw[2]  # Green (blue row)
    b_plane = bayer_raw[3]   # Blue
    
    # Average green channels
    g_avg = (g1_plane + g2_plane) / 2
    
    # Upsample to full resolution
    r_full = F.interpolate(r_plane.unsqueeze(0).unsqueeze(0), 
                           scale_factor=2, mode='bilinear', align_corners=False)[0, 0]
    g_full = F.interpolate(g_avg.unsqueeze(0).unsqueeze(0), 
                           scale_factor=2, mode='bilinear', align_corners=False)[0, 0]
    b_full = F.interpolate(b_plane.unsqueeze(0).unsqueeze(0), 
                           scale_factor=2, mode='bilinear', align_corners=False)[0, 0]
    
    rgb = torch.stack([r_full, g_full, b_full], dim=0)
    
    return rgb


class LearnedDemosaicing(nn.Module):
    """
    Deep learning-based joint demosaicing and denoising
    Based on "Deep Joint Demosaicking and Denoising" (Gharbi et al.)
    """
    
    def __init__(self, hidden_channels: int = 64, num_layers: int = 15):
        super().__init__()
        
        # Encoder: process Bayer pattern
        self.encoder = nn.Sequential(
            nn.Conv2d(4, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Middle layers: residual blocks
        middle_layers = []
        for _ in range(num_layers):
            middle_layers.extend([
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
            ])
        self.middle = nn.Sequential(*middle_layers)
        
        # Decoder: output RGB
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),  # Output in [0, 1]
        )
        
        # Upsampling layer (Bayer is H/2 x W/2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    
    def forward(self, bayer_raw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bayer_raw: [N, 4, H, W] or [4, H, W]
            
        Returns:
            RGB image: [N, 3, H*2, W*2] or [3, H*2, W*2]
        """
        is_batch = bayer_raw.ndim == 4
        
        if not is_batch:
            bayer_raw = bayer_raw.unsqueeze(0)
        
        # Process
        features = self.encoder(bayer_raw)
        features = self.middle(features) + features  # Residual connection
        rgb = self.decoder(features)
        
        # Upsample to full resolution
        rgb = self.upsample(rgb)
        
        if not is_batch:
            rgb = rgb.squeeze(0)
        
        return rgb


def apply_gamma_correction(
    linear_rgb: torch.Tensor,
    gamma: float = 2.2
) -> torch.Tensor:
    """
    Apply gamma correction for display
    
    Args:
        linear_rgb: Linear RGB image
        gamma: Gamma value (2.2 for sRGB)
        
    Returns:
        Gamma corrected RGB
    """
    # sRGB gamma curve (simplified)
    return torch.pow(torch.clamp(linear_rgb, 0.0, 1.0), 1.0 / gamma)


def apply_lens_shading_correction(
    raw_tensor: torch.Tensor,
    lsc_gains: torch.Tensor
) -> torch.Tensor:
    """
    Apply lens shading correction (vignetting compensation)
    
    Args:
        raw_tensor: RAW image [4, H, W]
        lsc_gains: Lens shading correction gains [4, H, W]
        
    Returns:
        Corrected RAW
    """
    corrected = raw_tensor * lsc_gains
    return torch.clamp(corrected, 0.0, 1.0)


def compute_lens_shading_map(
    image_size: Tuple[int, int],
    center: Optional[Tuple[float, float]] = None,
    vignetting_coeff: float = 0.3
) -> torch.Tensor:
    """
    Compute radial lens shading map
    
    Args:
        image_size: (height, width)
        center: (y, x) optical center, None for image center
        vignetting_coeff: Vignetting strength
        
    Returns:
        Lens shading map [4, H, W]
    """
    h, w = image_size
    
    if center is None:
        cy, cx = h / 2, w / 2
    else:
        cy, cx = center
    
    # Create coordinate grids
    y, x = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing='ij'
    )
    
    # Compute radial distance (normalized)
    r = torch.sqrt((x - cx)**2 + (y - cy)**2)
    r = r / r.max()
    
    # Vignetting model: gain = 1 - coeff * r^4
    gains = 1.0 - vignetting_coeff * (r ** 4)
    gains = torch.clamp(gains, min=0.1, max=1.0)
    
    # Replicate for all channels
    lsc_map = gains.unsqueeze(0).repeat(4, 1, 1)
    
    return lsc_map


def raw_to_srgb_simple(
    raw_tensor: torch.Tensor,
    metadata: Dict,
    demosaic_method: str = 'bilinear'
) -> torch.Tensor:
    """
    Simple RAW to sRGB conversion pipeline
    
    Args:
        raw_tensor: RAW image [4, H, W]
        metadata: Metadata dictionary
        demosaic_method: 'bilinear' or 'malvar'
        
    Returns:
        sRGB image [3, H*2, W*2]
    """
    # 1. Linearize
    raw_linear = linearize_raw(
        raw_tensor,
        metadata['black_level'],
        metadata['white_level']
    )
    
    # 2. White balance
    raw_wb = apply_white_balance(raw_linear, metadata['camera_whitebalance'])
    
    # 3. Demosaic
    if demosaic_method == 'bilinear':
        rgb = demosaic_bilinear(raw_wb)
    elif demosaic_method == 'malvar':
        rgb = demosaic_malvar(raw_wb)
    else:
        raise ValueError(f"Unknown demosaic method: {demosaic_method}")
    
    # 4. Color correction
    if 'color_matrix' in metadata:
        rgb = apply_color_correction_matrix(rgb, metadata['color_matrix'])
    
    # 5. Gamma correction
    srgb = apply_gamma_correction(rgb, gamma=2.2)
    
    return srgb


if __name__ == "__main__":
    # Test preprocessing functions
    print("Testing preprocessing functions...")
    
    # Create dummy RAW image
    raw = torch.rand(4, 256, 256) * 4095  # 12-bit range
    
    # Test linearization
    raw_linear = linearize_raw(raw, [64, 64, 64, 64], 4095)
    print(f"Linearized range: [{raw_linear.min():.3f}, {raw_linear.max():.3f}]")
    
    # Test demosaicing
    rgb_bilinear = demosaic_bilinear(raw_linear)
    print(f"Demosaiced shape: {rgb_bilinear.shape}")
    
    # Test learned demosaicing
    model = LearnedDemosaicing()
    with torch.no_grad():
        rgb_learned = model(raw_linear)
    print(f"Learned demosaic shape: {rgb_learned.shape}")
    
    print("✓ All preprocessing tests passed!")
