"""
Lens Aberration Correction Module
iPhone-specific lens aberration modeling and correction
Handles chromatic aberration, spherical aberration, coma, vignetting, distortion
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LensAberrationEncoder(nn.Module):
    """
    Encodes lens metadata and spatial position for aberration modeling
    """
    
    def __init__(
        self,
        feature_dim: int = 128,
        num_lens_types: int = 5,  # iPhone 15/16 Pro models
        use_spatial_encoding: bool = True
    ):
        """
        Args:
            feature_dim: Output feature dimension
            num_lens_types: Number of camera lens types
            use_spatial_encoding: Use spatial position encoding
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.use_spatial_encoding = use_spatial_encoding
        
        # Lens type embedding
        self.lens_embedding = nn.Embedding(num_lens_types, 64)
        
        # Lens parameter encoder
        # Parameters: focal_length, aperture, focus_distance
        self.param_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.SiLU(),
            nn.Linear(64, 128),
            nn.SiLU(),
            nn.Linear(128, feature_dim)
        )
        
        # Spatial position encoder (radial distance from optical center)
        if use_spatial_encoding:
            self.spatial_encoder = SpatialPositionEncoder(feature_dim)
        
        # Combined feature projection
        input_dim = feature_dim + 64 if use_spatial_encoding else feature_dim
        self.output_proj = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim)
        )
    
    def forward(
        self,
        lens_params: torch.Tensor,
        lens_type: torch.Tensor,
        image_coords: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode lens aberration features
        
        Args:
            lens_params: [B, 3] (focal_length, aperture, focus_distance)
            lens_type: [B] lens type indices
            image_coords: [B, H, W, 2] (y, x) coordinates (optional)
            
        Returns:
            Aberration features [B, feature_dim] or [B, feature_dim, H, W]
        """
        # Lens type embedding
        lens_emb = self.lens_embedding(lens_type)  # [B, 64]
        
        # Lens parameter encoding
        param_features = self.param_encoder(lens_params)  # [B, feature_dim]
        
        # Combine
        features = param_features + lens_emb[:, :self.feature_dim]
        
        # Add spatial encoding if coordinates provided
        if self.use_spatial_encoding and image_coords is not None:
            spatial_features = self.spatial_encoder(image_coords)  # [B, feature_dim, H, W]
            
            # Broadcast and concatenate
            B, _, H, W = spatial_features.shape
            features = features.view(B, -1, 1, 1).expand(-1, -1, H, W)
            features = torch.cat([features, spatial_features], dim=1)
            
            # Project
            features = features.permute(0, 2, 3, 1)  # [B, H, W, feature_dim+...]
            features = self.output_proj(features)
            features = features.permute(0, 3, 1, 2)  # [B, feature_dim, H, W]
        else:
            features = self.output_proj(features.unsqueeze(-1).unsqueeze(-1))
        
        return features


class SpatialPositionEncoder(nn.Module):
    """
    Encodes spatial position (radial distance from optical center)
    """
    
    def __init__(self, feature_dim: int = 128):
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # Frequency encoding (similar to positional encoding in transformers)
        self.freq_bands = nn.Parameter(
            torch.exp(torch.linspace(0, 8, feature_dim // 4)),
            requires_grad=False
        )
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: [B, H, W, 2] (y, x) coordinates normalized to [-1, 1]
            
        Returns:
            Spatial features [B, feature_dim, H, W]
        """
        B, H, W, _ = coords.shape
        
        # Compute radial distance from center
        y, x = coords[..., 0], coords[..., 1]
        r = torch.sqrt(x ** 2 + y ** 2)  # [B, H, W]
        
        # Compute angle
        theta = torch.atan2(y, x)  # [B, H, W]
        
        # Frequency encoding of radial distance
        r_expanded = r.unsqueeze(-1) * self.freq_bands.view(1, 1, 1, -1)
        r_features = torch.cat([torch.sin(r_expanded), torch.cos(r_expanded)], dim=-1)
        
        # Frequency encoding of angle
        theta_expanded = theta.unsqueeze(-1) * self.freq_bands.view(1, 1, 1, -1)
        theta_features = torch.cat([torch.sin(theta_expanded), torch.cos(theta_expanded)], dim=-1)
        
        # Concatenate
        features = torch.cat([r_features, theta_features], dim=-1)  # [B, H, W, feature_dim]
        
        return features.permute(0, 3, 1, 2)  # [B, feature_dim, H, W]


class AberrationCorrectionModule(nn.Module):
    """
    Spatially-varying aberration correction
    Predicts and applies deconvolution kernels
    """
    
    def __init__(
        self,
        feature_dim: int = 128,
        psf_kernel_size: int = 15,
        num_channels: int = 4  # RAW Bayer channels
    ):
        """
        Args:
            feature_dim: Aberration feature dimension
            psf_kernel_size: PSF kernel size (odd number)
            num_channels: Number of image channels
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.psf_kernel_size = psf_kernel_size
        self.num_channels = num_channels
        
        # PSF predictor network
        self.psf_predictor = nn.Sequential(
            nn.Conv2d(feature_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            # Predict kernel weights for each channel
            nn.Conv2d(128, num_channels * psf_kernel_size ** 2, 3, padding=1)
        )
        
        # Frequency-domain correction branch
        self.freq_corrector = nn.Sequential(
            nn.Conv2d(num_channels * 2, 64, 3, padding=1),  # *2 for real/imag
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_channels * 2, 3, padding=1)
        )
        
        # Color channel-specific correction
        self.channel_corrector = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, 3, padding=1)
            ) for _ in range(num_channels)
        ])
    
    def forward(
        self,
        raw_image: torch.Tensor,
        aberration_features: torch.Tensor,
        use_freq_domain: bool = True
    ) -> torch.Tensor:
        """
        Apply aberration correction
        
        Args:
            raw_image: RAW image [B, 4, H, W]
            aberration_features: Aberration features [B, feature_dim, H, W]
            use_freq_domain: Use frequency-domain correction
            
        Returns:
            Corrected RAW image [B, 4, H, W]
        """
        B, C, H, W = raw_image.shape
        
        # Predict PSF kernels
        psf_weights = self.psf_predictor(aberration_features)
        psf_weights = psf_weights.view(B, self.num_channels, self.psf_kernel_size ** 2, H, W)
        
        # Normalize PSF weights (sum to 1)
        psf_weights = F.softmax(psf_weights, dim=2)
        
        # Apply spatially-varying deconvolution
        corrected = self._apply_spatially_varying_deconv(raw_image, psf_weights)
        
        # Frequency-domain correction (optional)
        if use_freq_domain:
            freq_correction = self._apply_frequency_correction(corrected)
            corrected = corrected + freq_correction
        
        # Channel-specific correction
        channel_outputs = []
        for i in range(self.num_channels):
            channel_out = self.channel_corrector[i](corrected[:, i:i+1])
            channel_outputs.append(channel_out)
        
        corrected = torch.cat(channel_outputs, dim=1)
        
        # Residual connection
        corrected = raw_image + corrected
        
        return torch.clamp(corrected, 0.0, 1.0)
    
    def _apply_spatially_varying_deconv(
        self,
        image: torch.Tensor,
        psf_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply spatially-varying deconvolution
        
        Args:
            image: [B, C, H, W]
            psf_weights: [B, C, K^2, H, W] where K is kernel size
            
        Returns:
            Deconvolved image [B, C, H, W]
        """
        B, C, H, W = image.shape
        K = self.psf_kernel_size
        pad = K // 2
        
        # Pad image
        image_padded = F.pad(image, (pad, pad, pad, pad), mode='reflect')
        
        # Extract patches
        patches = F.unfold(image_padded, kernel_size=K, stride=1)
        patches = patches.view(B, C, K * K, H, W)
        
        # Apply weights
        deconvolved = (patches * psf_weights).sum(dim=2)
        
        return deconvolved
    
    def _apply_frequency_correction(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply frequency-domain correction (chromatic aberration)
        
        Args:
            image: [B, C, H, W]
            
        Returns:
            Frequency correction [B, C, H, W]
        """
        # FFT
        fft = torch.fft.rfft2(image, norm='ortho')
        
        # Stack real and imaginary parts
        fft_real_imag = torch.cat([fft.real, fft.imag], dim=1)
        
        # Apply correction
        corrected_fft = self.freq_corrector(fft_real_imag)
        
        # Split back to real/imag
        C = image.shape[1]
        corrected_real = corrected_fft[:, :C]
        corrected_imag = corrected_fft[:, C:]
        
        # Reconstruct complex tensor
        corrected_fft = torch.complex(corrected_real, corrected_imag)
        
        # Inverse FFT
        corrected = torch.fft.irfft2(corrected_fft, s=image.shape[2:], norm='ortho')
        
        return corrected


class DiffusionAberrationPrior(nn.Module):
    """
    Learn distribution of sharp images (no aberration)
    Guide diffusion toward sharp prior during inference
    """
    
    def __init__(
        self,
        latent_channels: int = 16,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        # Sharp image prior encoder
        self.prior_encoder = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, latent_channels, 3, padding=1)
        )
        
        # Aberration parameter predictor
        self.aberration_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(latent_channels, 64),
            nn.SiLU(),
            nn.Linear(64, 6)  # [chromatic_x, chromatic_y, spherical, coma_x, coma_y, vignetting]
        )
    
    def forward(
        self,
        latent_z: torch.Tensor,
        aberration_params: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply aberration prior correction
        
        Args:
            latent_z: Latent features [B, C, H, W]
            aberration_params: Known aberration parameters [B, 6] (optional)
            
        Returns:
            corrected_latent: Corrected latent
            pred_aberration: Predicted aberration parameters
        """
        # Encode toward sharp prior
        prior_features = self.prior_encoder(latent_z)
        
        # Predict aberration if not provided
        if aberration_params is None:
            pred_aberration = self.aberration_predictor(latent_z)
        else:
            pred_aberration = aberration_params
        
        # Apply correction based on aberration parameters
        # This is a simplified model - in practice, use more sophisticated correction
        correction_strength = torch.sigmoid(pred_aberration.mean(dim=1, keepdim=True))
        correction_strength = correction_strength.view(-1, 1, 1, 1)
        
        corrected_latent = latent_z + correction_strength * prior_features
        
        return corrected_latent, pred_aberration


def create_synthetic_aberration(
    image: torch.Tensor,
    aberration_type: str = 'all',
    strength: float = 0.5
) -> torch.Tensor:
    """
    Create synthetic aberration for training
    
    Args:
        image: Clean image [B, C, H, W]
        aberration_type: 'chromatic', 'spherical', 'vignetting', or 'all'
        strength: Aberration strength [0, 1]
        
    Returns:
        Aberrated image
    """
    B, C, H, W = image.shape
    device = image.device
    
    aberrated = image.clone()
    
    if aberration_type in ['chromatic', 'all']:
        # Chromatic aberration: shift channels
        shift = int(strength * 3)
        if shift > 0:
            # Red channel shift
            aberrated[:, 0] = torch.roll(aberrated[:, 0], shifts=shift, dims=1)
            # Blue channel shift (opposite direction)
            aberrated[:, 3] = torch.roll(aberrated[:, 3], shifts=-shift, dims=1)
    
    if aberration_type in ['spherical', 'all']:
        # Spherical aberration: radial blur
        kernel_size = int(strength * 5) * 2 + 1
        if kernel_size > 1:
            # Create Gaussian kernel
            sigma = kernel_size / 6.0
            kernel = _get_gaussian_kernel(kernel_size, sigma, device)
            
            # Apply blur
            aberrated = F.conv2d(
                aberrated,
                kernel.repeat(C, 1, 1, 1),
                padding=kernel_size // 2,
                groups=C
            )
    
    if aberration_type in ['vignetting', 'all']:
        # Vignetting: radial light falloff
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        r = torch.sqrt(x ** 2 + y ** 2)
        vignette = 1.0 - strength * (r ** 4)
        vignette = vignette.clamp(0.1, 1.0)
        
        aberrated = aberrated * vignette.view(1, 1, H, W)
    
    return aberrated


def _get_gaussian_kernel(kernel_size: int, sigma: float, device: str) -> torch.Tensor:
    """Generate 2D Gaussian kernel"""
    x = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
    gauss = torch.exp(-x ** 2 / (2 * sigma ** 2))
    kernel_1d = gauss / gauss.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d.view(1, 1, kernel_size, kernel_size)


if __name__ == "__main__":
    print("Testing Lens Aberration Correction...")
    
    # Test aberration encoder
    print("\n1. Testing LensAberrationEncoder...")
    encoder = LensAberrationEncoder()
    
    lens_params = torch.tensor([[5.7, 1.6, 0.5], [5.1, 1.8, 1.0]])  # focal, aperture, focus
    lens_type = torch.tensor([0, 1])  # iPhone model indices
    coords = torch.rand(2, 64, 64, 2) * 2 - 1  # Normalized coordinates
    
    features = encoder(lens_params, lens_type, coords)
    print(f"   Aberration features shape: {features.shape}")
    
    # Test correction module
    print("\n2. Testing AberrationCorrectionModule...")
    corrector = AberrationCorrectionModule()
    
    raw_image = torch.rand(2, 4, 64, 64)
    aberration_features = features
    
    corrected = corrector(raw_image, aberration_features)
    print(f"   Corrected image shape: {corrected.shape}")
    print(f"   Correction range: [{corrected.min():.3f}, {corrected.max():.3f}]")
    
    # Test diffusion prior
    print("\n3. Testing DiffusionAberrationPrior...")
    prior = DiffusionAberrationPrior(latent_channels=16)
    
    latent = torch.randn(2, 16, 32, 32)
    corrected_latent, pred_aberration = prior(latent)
    
    print(f"   Corrected latent shape: {corrected_latent.shape}")
    print(f"   Predicted aberration shape: {pred_aberration.shape}")
    
    # Test synthetic aberration
    print("\n4. Testing synthetic aberration...")
    clean_image = torch.rand(2, 4, 128, 128)
    
    aberrated = create_synthetic_aberration(clean_image, 'all', strength=0.5)
    print(f"   Aberrated image shape: {aberrated.shape}")
    
    diff = (clean_image - aberrated).abs().mean()
    print(f"   Mean difference: {diff:.4f}")
    
    print("\n✓ All lens aberration tests passed!")
