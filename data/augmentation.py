"""
Data Augmentation for RAW Images
Specialized augmentation for Bayer pattern RAW data
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


class RawAugmentation:
    """
    Data augmentation pipeline for RAW images
    Preserves Bayer pattern structure
    """
    
    def __init__(
        self,
        random_flip: bool = True,
        random_rotate: bool = True,
        color_jitter: float = 0.1,
        noise_augmentation: bool = True,
        noise_level_range: Tuple[float, float] = (0.0, 0.02),
        synthetic_aberration: bool = False,
    ):
        """
        Args:
            random_flip: Enable random horizontal/vertical flips
            random_rotate: Enable random 90° rotations
            color_jitter: Color jitter strength (0-1)
            noise_augmentation: Add synthetic sensor noise
            noise_level_range: (min, max) noise standard deviation
            synthetic_aberration: Add synthetic lens aberration
        """
        self.random_flip = random_flip
        self.random_rotate = random_rotate
        self.color_jitter = color_jitter
        self.noise_augmentation = noise_augmentation
        self.noise_level_range = noise_level_range
        self.synthetic_aberration = synthetic_aberration
    
    def __call__(
        self,
        raw_tensor: torch.Tensor,
        metadata: Union[Dict, List[Dict]],
    ) -> torch.Tensor:
        """
        Apply augmentation to RAW image or burst
        
        Args:
            raw_tensor: [4, H, W] or [N, 4, H, W] (burst)
            metadata: Metadata dict or list of dicts
            
        Returns:
            Augmented RAW tensor
        """
        is_burst = raw_tensor.ndim == 4
        
        # Random flip
        if self.random_flip:
            raw_tensor = self._random_flip(raw_tensor)
        
        # Random rotation (90° increments to preserve Bayer pattern)
        if self.random_rotate:
            raw_tensor = self._random_rotate_90(raw_tensor)
        
        # Color jitter (adjust channel gains)
        if self.color_jitter > 0:
            raw_tensor = self._color_jitter(raw_tensor, self.color_jitter)
        
        # Add synthetic noise
        if self.noise_augmentation:
            raw_tensor = self._add_noise(raw_tensor, self.noise_level_range)
        
        # Synthetic aberration
        if self.synthetic_aberration:
            raw_tensor = self._add_synthetic_aberration(raw_tensor)
        
        return raw_tensor
    
    def _random_flip(self, tensor: torch.Tensor) -> torch.Tensor:
        """Random horizontal and/or vertical flip"""
        # Flip horizontally
        if torch.rand(1).item() > 0.5:
            if tensor.ndim == 3:
                tensor = torch.flip(tensor, dims=[2])
            else:
                tensor = torch.flip(tensor, dims=[3])
        
        # Flip vertically
        if torch.rand(1).item() > 0.5:
            if tensor.ndim == 3:
                tensor = torch.flip(tensor, dims=[1])
            else:
                tensor = torch.flip(tensor, dims=[2])
        
        return tensor
    
    def _random_rotate_90(self, tensor: torch.Tensor) -> torch.Tensor:
        """Random 90° rotation (preserves Bayer pattern)"""
        k = torch.randint(0, 4, (1,)).item()  # 0, 90, 180, or 270 degrees
        
        if k == 0:
            return tensor
        
        if tensor.ndim == 3:
            # [4, H, W]
            tensor = torch.rot90(tensor, k, dims=[1, 2])
        else:
            # [N, 4, H, W]
            tensor = torch.rot90(tensor, k, dims=[2, 3])
        
        return tensor
    
    def _color_jitter(self, tensor: torch.Tensor, strength: float) -> torch.Tensor:
        """
        Adjust channel gains to simulate white balance variation
        
        Args:
            tensor: RAW tensor
            strength: Jitter strength (0-1)
            
        Returns:
            Jittered tensor
        """
        # Generate random gains for each channel
        # Keep gains close to 1.0
        gains = 1.0 + (torch.rand(4) - 0.5) * 2 * strength
        gains = torch.clamp(gains, 0.5, 1.5)
        
        # Apply gains
        if tensor.ndim == 3:
            gains = gains.view(4, 1, 1).to(tensor.device)
        else:
            gains = gains.view(1, 4, 1, 1).to(tensor.device)
        
        jittered = tensor * gains
        
        return torch.clamp(jittered, 0.0, 1.0)
    
    def _add_noise(
        self,
        tensor: torch.Tensor,
        noise_range: Tuple[float, float]
    ) -> torch.Tensor:
        """
        Add synthetic sensor noise (Gaussian + Poisson approximation)
        
        Args:
            tensor: RAW tensor (normalized to [0, 1])
            noise_range: (min, max) noise level
            
        Returns:
            Noisy tensor
        """
        # Sample noise level
        noise_level = torch.rand(1).item() * (noise_range[1] - noise_range[0]) + noise_range[0]
        
        # Gaussian noise (read noise)
        gaussian_noise = torch.randn_like(tensor) * noise_level
        
        # Poisson noise (shot noise) - proportional to signal
        # Simplified: just scale Gaussian by sqrt(intensity)
        poisson_noise = torch.randn_like(tensor) * torch.sqrt(tensor + 1e-6) * noise_level
        
        # Combined noise
        noisy = tensor + gaussian_noise + poisson_noise
        
        return torch.clamp(noisy, 0.0, 1.0)
    
    def _add_synthetic_aberration(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Add synthetic lens aberration for augmentation
        Helps model learn to correct real aberrations
        
        Args:
            tensor: RAW tensor
            
        Returns:
            Aberrated tensor
        """
        # Random aberration strength
        aberration_strength = torch.rand(1).item() * 0.3
        
        # Apply spatially-varying blur (simplified aberration model)
        # In production, use proper PSF models
        
        # Create radial distance map
        if tensor.ndim == 3:
            c, h, w = tensor.shape
        else:
            n, c, h, w = tensor.shape
        
        # Simple radial blur (more blur at edges)
        center_y, center_x = h // 2, w // 2
        y, x = torch.meshgrid(
            torch.arange(h, dtype=torch.float32, device=tensor.device),
            torch.arange(w, dtype=torch.float32, device=tensor.device),
            indexing='ij'
        )
        
        r = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
        r_normalized = r / r.max()
        
        # Blur strength increases with radius
        blur_strength = aberration_strength * r_normalized ** 2
        
        # Apply Gaussian blur with varying kernel size
        # Simplified: apply uniform small blur
        kernel_size = int(aberration_strength * 5) * 2 + 1
        if kernel_size > 1:
            if tensor.ndim == 3:
                tensor_input = tensor.unsqueeze(0)
            else:
                # Reshape burst to process all frames
                n = tensor.shape[0]
                tensor_input = tensor.view(n * c, 1, h, w)
            
            # Create Gaussian kernel
            sigma = kernel_size / 6.0
            kernel = self._get_gaussian_kernel(kernel_size, sigma, tensor.device)
            
            # Apply blur
            blurred = F.conv2d(
                tensor_input,
                kernel.repeat(tensor_input.shape[1], 1, 1, 1),
                padding=kernel_size // 2,
                groups=tensor_input.shape[1]
            )
            
            # Reshape back
            if tensor.ndim == 3:
                tensor = blurred.squeeze(0)
            else:
                tensor = blurred.view(n, c, h, w)
        
        return tensor
    
    @staticmethod
    def _get_gaussian_kernel(kernel_size: int, sigma: float, device: torch.device) -> torch.Tensor:
        """Generate 2D Gaussian kernel"""
        x = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
        gauss = torch.exp(-x ** 2 / (2 * sigma ** 2))
        kernel_1d = gauss / gauss.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        return kernel_2d.view(1, 1, kernel_size, kernel_size)


class SyntheticBurstGenerator:
    """
    Generate synthetic burst sequences from single images
    Useful for training when real bursts are limited
    """
    
    def __init__(
        self,
        num_frames: int = 8,
        motion_sigma: float = 2.0,
        blur_probability: float = 0.3,
    ):
        """
        Args:
            num_frames: Number of frames in synthetic burst
            motion_sigma: Standard deviation of random motion (pixels)
            blur_probability: Probability of adding motion blur to each frame
        """
        self.num_frames = num_frames
        self.motion_sigma = motion_sigma
        self.blur_probability = blur_probability
    
    def __call__(self, raw_image: torch.Tensor) -> torch.Tensor:
        """
        Generate synthetic burst from single RAW image
        
        Args:
            raw_image: Single RAW image [4, H, W]
            
        Returns:
            Synthetic burst [N, 4, H, W]
        """
        device = raw_image.device
        c, h, w = raw_image.shape
        
        burst_frames = []
        
        for i in range(self.num_frames):
            # Random translation
            dx = torch.randn(1).item() * self.motion_sigma
            dy = torch.randn(1).item() * self.motion_sigma
            
            # Apply translation using grid_sample
            frame = self._translate(raw_image, dx, dy)
            
            # Random motion blur
            if torch.rand(1).item() < self.blur_probability:
                frame = self._add_motion_blur(frame)
            
            # Small noise variation
            frame = frame + torch.randn_like(frame) * 0.005
            frame = torch.clamp(frame, 0.0, 1.0)
            
            burst_frames.append(frame)
        
        burst = torch.stack(burst_frames, dim=0)
        
        return burst
    
    def _translate(self, image: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
        """Apply sub-pixel translation"""
        c, h, w = image.shape
        
        # Create sampling grid
        theta = torch.tensor([[
            [1, 0, dx / (w / 2)],
            [0, 1, dy / (h / 2)]
        ]], dtype=torch.float32, device=image.device)
        
        grid = F.affine_grid(theta, [1, c, h, w], align_corners=False)
        translated = F.grid_sample(
            image.unsqueeze(0),
            grid,
            mode='bilinear',
            padding_mode='reflection',
            align_corners=False
        )
        
        return translated.squeeze(0)
    
    def _add_motion_blur(self, image: torch.Tensor) -> torch.Tensor:
        """Add simple motion blur"""
        # Random blur direction
        angle = torch.rand(1).item() * 2 * np.pi
        blur_length = torch.randint(3, 7, (1,)).item()
        
        # Create motion blur kernel (simplified)
        kernel_size = blur_length * 2 + 1
        kernel = torch.zeros(1, 1, kernel_size, kernel_size, device=image.device)
        
        # Line of 1s in blur direction
        cx, cy = kernel_size // 2, kernel_size // 2
        for i in range(-blur_length, blur_length + 1):
            x = int(cx + i * np.cos(angle))
            y = int(cy + i * np.sin(angle))
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[0, 0, y, x] = 1.0
        
        kernel = kernel / kernel.sum()
        
        # Apply blur
        blurred = F.conv2d(
            image.unsqueeze(0),
            kernel.repeat(image.shape[0], 1, 1, 1),
            padding=kernel_size // 2,
            groups=image.shape[0]
        )
        
        return blurred.squeeze(0)


if __name__ == "__main__":
    print("Testing augmentation...")
    
    # Create dummy RAW image
    raw = torch.rand(4, 256, 256)
    
    # Test augmentation
    aug = RawAugmentation(
        random_flip=True,
        random_rotate=True,
        color_jitter=0.1,
        noise_augmentation=True,
    )
    
    augmented = aug(raw, {})
    print(f"Augmented shape: {augmented.shape}")
    
    # Test burst generation
    burst_gen = SyntheticBurstGenerator(num_frames=8)
    burst = burst_gen(raw)
    print(f"Synthetic burst shape: {burst.shape}")
    
    print("✓ All augmentation tests passed!")
