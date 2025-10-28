"""
RAW Image Loader for iPhone ProRAW (.DNG) files
Handles 12-bit and 14-bit RAW formats with metadata extraction
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import rawpy
import torch
from PIL import Image
from torch.utils.data import Dataset


def load_proraw_dng(
    filepath: Union[str, Path], 
    return_metadata: bool = True
) -> Union[Tuple[torch.Tensor, Dict], torch.Tensor]:
    """
    Load iPhone ProRAW .DNG file
    
    Args:
        filepath: Path to .DNG file
        return_metadata: Whether to return metadata dictionary
        
    Returns:
        raw_tensor: RAW image as torch.Tensor [4, H, W] (Bayer RGGB pattern)
        metadata_dict: Dictionary containing camera metadata (if return_metadata=True)
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"DNG file not found: {filepath}")
    
    try:
        with rawpy.imread(str(filepath)) as raw:
            # Get RAW image data
            raw_image = raw.raw_image.copy().astype(np.float32)
            
            # Get RAW pattern (usually RGGB for iPhone)
            raw_pattern = raw.raw_pattern
            raw_colors = raw.raw_colors
            
            # Extract metadata
            metadata = {
                'width': raw.sizes.width,
                'height': raw.sizes.height,
                'raw_width': raw.sizes.raw_width,
                'raw_height': raw.sizes.raw_height,
                'black_level': getattr(raw, 'black_level_per_channel', [0, 0, 0, 0]),
                'white_level': getattr(raw, 'white_level', 4095),  # 12-bit default
                'color_matrix': raw.color_matrix[:3, :3] if hasattr(raw, 'color_matrix') else np.eye(3),
                'camera_whitebalance': getattr(raw, 'camera_whitebalance', [1.0, 1.0, 1.0, 1.0]),
                'daylight_whitebalance': getattr(raw, 'daylight_whitebalance', [1.0, 1.0, 1.0, 1.0]),
                'raw_pattern': raw_pattern,
                'raw_colors': raw_colors,
            }
            
            # Try to get EXIF data
            try:
                import exifread
                with open(filepath, 'rb') as f:
                    tags = exifread.process_file(f)
                    
                metadata['iso'] = float(str(tags.get('EXIF ISOSpeedRatings', 100)))
                metadata['exposure_time'] = eval(str(tags.get('EXIF ExposureTime', '1/60')))
                metadata['f_number'] = float(str(tags.get('EXIF FNumber', 'f/1.6').replace('f/', '')))
                metadata['focal_length'] = float(str(tags.get('EXIF FocalLength', '5.7')).split('/')[0])
                
            except Exception as e:
                print(f"Warning: Could not extract EXIF data: {e}")
                metadata['iso'] = 100
                metadata['exposure_time'] = 1/60
                metadata['f_number'] = 1.6
                metadata['focal_length'] = 5.7
            
            # Convert Bayer pattern to 4-channel tensor [R, G1, G2, B]
            h, w = raw_image.shape
            bayer_tensor = demux_bayer(raw_image, raw_pattern)
            
            # Convert to torch tensor
            raw_tensor = torch.from_numpy(bayer_tensor).float()
            
            if return_metadata:
                return raw_tensor, metadata
            else:
                return raw_tensor
                
    except Exception as e:
        raise RuntimeError(f"Error loading DNG file {filepath}: {e}")


def demux_bayer(raw_image: np.ndarray, raw_pattern: np.ndarray) -> np.ndarray:
    """
    Convert Bayer mosaic to 4-channel image [R, G1, G2, B]
    
    Args:
        raw_image: Raw sensor data [H, W]
        raw_pattern: Bayer pattern (2x2 array indicating color at each position)
        
    Returns:
        bayer_channels: 4-channel array [4, H/2, W/2]
    """
    h, w = raw_image.shape
    
    # Ensure dimensions are even
    if h % 2 != 0:
        raw_image = raw_image[:-1, :]
        h -= 1
    if w % 2 != 0:
        raw_image = raw_image[:, :-1]
        w -= 1
    
    # Initialize 4-channel output
    bayer_channels = np.zeros((4, h // 2, w // 2), dtype=np.float32)
    
    # Extract channels based on Bayer pattern
    # Typical RGGB pattern:
    # R  G1
    # G2 B
    for i in range(2):
        for j in range(2):
            channel_idx = raw_pattern[i, j]
            bayer_channels[channel_idx] = raw_image[i::2, j::2]
    
    return bayer_channels


def parse_apple_metadata(dng_file: Union[str, Path]) -> Dict:
    """
    Parse Apple-specific metadata from ProRAW DNG
    
    Args:
        dng_file: Path to DNG file
        
    Returns:
        Dictionary containing:
        - deep_fusion_map: Deep Fusion processing map
        - semantic_mask: Semantic segmentation mask
        - gain_map: HDR gain map
        - lens_correction: Lens correction parameters
    """
    # Note: Apple ProRAW contains proprietary metadata in maker notes
    # This is a placeholder for actual implementation
    
    metadata = {
        'deep_fusion_map': None,
        'semantic_mask': None,
        'gain_map': None,
        'lens_correction': {
            'distortion_params': [0.0, 0.0, 0.0],
            'vignetting_params': [1.0, 0.0, 0.0],
        }
    }
    
    # TODO: Implement actual Apple metadata parsing
    # This requires reverse engineering Apple's maker note format
    
    return metadata


class BayerRawDataset(Dataset):
    """PyTorch Dataset for single RAW images"""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        crop_size: Optional[Tuple[int, int]] = None,
        normalize: bool = True,
        augmentation: Optional[object] = None,
    ):
        """
        Args:
            data_dir: Directory containing .DNG files
            crop_size: (height, width) for random crops, None for full image
            normalize: Whether to normalize RAW values to [0, 1]
            augmentation: Augmentation pipeline
        """
        self.data_dir = Path(data_dir)
        self.crop_size = crop_size
        self.normalize = normalize
        self.augmentation = augmentation
        
        # Find all DNG files
        self.dng_files = sorted(list(self.data_dir.glob('**/*.dng')) + 
                                list(self.data_dir.glob('**/*.DNG')))
        
        if len(self.dng_files) == 0:
            raise ValueError(f"No DNG files found in {data_dir}")
        
        print(f"Found {len(self.dng_files)} DNG files in {data_dir}")
    
    def __len__(self) -> int:
        return len(self.dng_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary containing:
            - 'raw': RAW image tensor [4, H, W]
            - 'metadata': Metadata dictionary
        """
        dng_path = self.dng_files[idx]
        
        # Load RAW image
        raw_tensor, metadata = load_proraw_dng(dng_path, return_metadata=True)
        
        # Normalize if requested
        if self.normalize:
            from .preprocessing import linearize_raw
            raw_tensor = linearize_raw(
                raw_tensor, 
                metadata['black_level'],
                metadata['white_level']
            )
        
        # Random crop if specified
        if self.crop_size is not None:
            raw_tensor = self._random_crop(raw_tensor, self.crop_size)
        
        # Apply augmentation
        if self.augmentation is not None:
            raw_tensor = self.augmentation(raw_tensor, metadata)
        
        return {
            'raw': raw_tensor,
            'metadata': metadata,
            'filename': dng_path.name,
        }
    
    def _random_crop(self, tensor: torch.Tensor, crop_size: Tuple[int, int]) -> torch.Tensor:
        """Random crop from tensor"""
        c, h, w = tensor.shape
        crop_h, crop_w = crop_size
        
        # Adjust crop size to Bayer dimensions (must be even)
        crop_h = crop_h // 2
        crop_w = crop_w // 2
        
        if h < crop_h or w < crop_w:
            # Pad if necessary
            tensor = torch.nn.functional.pad(
                tensor, 
                (0, max(0, crop_w - w), 0, max(0, crop_h - h)),
                mode='reflect'
            )
            h, w = tensor.shape[1], tensor.shape[2]
        
        # Random crop coordinates
        top = torch.randint(0, h - crop_h + 1, (1,)).item()
        left = torch.randint(0, w - crop_w + 1, (1,)).item()
        
        return tensor[:, top:top+crop_h, left:left+crop_w]


class BurstRawDataset(Dataset):
    """PyTorch Dataset for multi-frame RAW burst sequences"""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        num_frames: int = 8,
        crop_size: Optional[Tuple[int, int]] = None,
        normalize: bool = True,
        augmentation: Optional[object] = None,
    ):
        """
        Args:
            data_dir: Directory containing burst sequences (subdirs)
            num_frames: Number of frames per burst
            crop_size: (height, width) for random crops
            normalize: Whether to normalize RAW values
            augmentation: Augmentation pipeline
        """
        self.data_dir = Path(data_dir)
        self.num_frames = num_frames
        self.crop_size = crop_size
        self.normalize = normalize
        self.augmentation = augmentation
        
        # Find burst directories (each burst is in a separate folder)
        self.burst_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
        if len(self.burst_dirs) == 0:
            raise ValueError(f"No burst directories found in {data_dir}")
        
        print(f"Found {len(self.burst_dirs)} burst sequences in {data_dir}")
    
    def __len__(self) -> int:
        return len(self.burst_dirs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary containing:
            - 'burst': Burst frames [N, 4, H, W]
            - 'metadata_list': List of metadata dicts for each frame
            - 'reference_idx': Index of reference frame (usually 0)
        """
        burst_dir = self.burst_dirs[idx]
        
        # Get all DNG files in burst directory
        dng_files = sorted(list(burst_dir.glob('*.dng')) + 
                          list(burst_dir.glob('*.DNG')))
        
        # Take first num_frames
        dng_files = dng_files[:self.num_frames]
        
        if len(dng_files) < self.num_frames:
            print(f"Warning: Burst {burst_dir.name} has only {len(dng_files)} frames, padding...")
            # Pad by repeating last frame
            while len(dng_files) < self.num_frames:
                dng_files.append(dng_files[-1])
        
        # Load all frames
        raw_frames = []
        metadata_list = []
        
        for dng_file in dng_files:
            raw_tensor, metadata = load_proraw_dng(dng_file, return_metadata=True)
            
            # Normalize if requested
            if self.normalize:
                from .preprocessing import linearize_raw
                raw_tensor = linearize_raw(
                    raw_tensor,
                    metadata['black_level'],
                    metadata['white_level']
                )
            
            raw_frames.append(raw_tensor)
            metadata_list.append(metadata)
        
        # Stack frames
        burst_tensor = torch.stack(raw_frames, dim=0)  # [N, 4, H, W]
        
        # Apply same crop to all frames
        if self.crop_size is not None:
            burst_tensor = self._synchronized_crop(burst_tensor, self.crop_size)
        
        # Apply augmentation
        if self.augmentation is not None:
            burst_tensor = self.augmentation(burst_tensor, metadata_list)
        
        return {
            'burst': burst_tensor,
            'metadata_list': metadata_list,
            'reference_idx': 0,  # First frame is reference
            'burst_name': burst_dir.name,
        }
    
    def _synchronized_crop(self, burst: torch.Tensor, crop_size: Tuple[int, int]) -> torch.Tensor:
        """Apply same random crop to all frames in burst"""
        n, c, h, w = burst.shape
        crop_h, crop_w = crop_size[0] // 2, crop_size[1] // 2
        
        if h < crop_h or w < crop_w:
            # Pad if necessary
            burst = torch.nn.functional.pad(
                burst,
                (0, max(0, crop_w - w), 0, max(0, crop_h - h)),
                mode='reflect'
            )
            h, w = burst.shape[2], burst.shape[3]
        
        # Random crop coordinates (same for all frames)
        top = torch.randint(0, h - crop_h + 1, (1,)).item()
        left = torch.randint(0, w - crop_w + 1, (1,)).item()
        
        return burst[:, :, top:top+crop_h, left:left+crop_w]


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        dng_path = sys.argv[1]
        print(f"Loading {dng_path}...")
        
        raw_tensor, metadata = load_proraw_dng(dng_path)
        
        print(f"RAW tensor shape: {raw_tensor.shape}")
        print(f"RAW dtype: {raw_tensor.dtype}")
        print(f"RAW range: [{raw_tensor.min():.1f}, {raw_tensor.max():.1f}]")
        print(f"\nMetadata:")
        for key, value in metadata.items():
            if not isinstance(value, np.ndarray):
                print(f"  {key}: {value}")
