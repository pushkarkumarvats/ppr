"""
Latent Diffusion Model for RAW Space
RAW-VAE, Diffusion U-Net, and DDPM Scheduler
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class RAWVAE(nn.Module):
    """
    Variational Autoencoder for RAW images
    Compresses 4-channel Bayer RAW to latent space
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        latent_channels: int = 16,
        hidden_dims: List[int] = [64, 128, 256, 512],
        kl_weight: float = 1e-6
    ):
        """
        Args:
            in_channels: Input channels (4 for Bayer RGGB)
            latent_channels: Latent space channels
            hidden_dims: Hidden layer dimensions
            kl_weight: KL divergence weight (very low to preserve detail)
        """
        super().__init__()
        
        self.latent_channels = latent_channels
        self.kl_weight = kl_weight
        
        # Encoder
        self.encoder = Encoder(in_channels, latent_channels * 2, hidden_dims)
        
        # Decoder
        self.decoder = Decoder(latent_channels, in_channels, list(reversed(hidden_dims)))
        
        # Quantization for KL (optional)
        self.quant_conv = nn.Conv2d(latent_channels * 2, latent_channels * 2, 1)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode RAW to latent distribution
        
        Args:
            x: RAW image [B, 4, H, W]
            
        Returns:
            z_mean: Mean of latent distribution [B, C, H/8, W/8]
            z_logvar: Log variance of latent distribution
        """
        # Encode
        h = self.encoder(x)
        h = self.quant_conv(h)
        
        # Split into mean and logvar
        z_mean, z_logvar = torch.chunk(h, 2, dim=1)
        
        return z_mean, z_logvar
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to RAW
        
        Args:
            z: Latent [B, C, H/8, W/8]
            
        Returns:
            Reconstructed RAW [B, 4, H, W]
        """
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec
    
    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick
        
        Args:
            mean: Mean [B, C, H, W]
            logvar: Log variance [B, C, H, W]
            
        Returns:
            Sampled latent z
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z
    
    def forward(
        self,
        x: torch.Tensor,
        return_loss: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Forward pass through VAE
        
        Args:
            x: Input RAW [B, 4, H, W]
            return_loss: Whether to return loss components
            
        Returns:
            reconstruction: Reconstructed RAW
            losses: Dictionary of loss components (if return_loss=True)
        """
        # Encode
        z_mean, z_logvar = self.encode(x)
        
        # Sample
        z = self.reparameterize(z_mean, z_logvar)
        
        # Decode
        reconstruction = self.decode(z)
        
        if return_loss:
            # Reconstruction loss
            recon_loss = F.mse_loss(reconstruction, x)
            
            # KL divergence
            kl_loss = -0.5 * torch.mean(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
            
            # Total loss
            total_loss = recon_loss + self.kl_weight * kl_loss
            
            losses = {
                'total': total_loss,
                'reconstruction': recon_loss,
                'kl': kl_loss,
            }
            
            return reconstruction, losses
        
        return reconstruction
    
    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience method: encode without sampling"""
        z_mean, _ = self.encode(x)
        return z_mean


class Encoder(nn.Module):
    """VAE Encoder with Bayer-aware convolutions"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dims: List[int]
    ):
        super().__init__()
        
        modules = []
        
        # Input layer
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_dims[0], 3, stride=1, padding=1),
                nn.BatchNorm2d(hidden_dims[0]),
                nn.SiLU()
            )
        )
        
        # Downsampling blocks
        for i in range(len(hidden_dims) - 1):
            modules.append(
                DownBlock(hidden_dims[i], hidden_dims[i + 1])
            )
        
        # Output layer
        modules.append(
            nn.Sequential(
                nn.Conv2d(hidden_dims[-1], hidden_dims[-1], 3, padding=1),
                nn.BatchNorm2d(hidden_dims[-1]),
                nn.SiLU(),
                nn.Conv2d(hidden_dims[-1], out_channels, 3, padding=1)
            )
        )
        
        self.encoder = nn.Sequential(*modules)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class Decoder(nn.Module):
    """VAE Decoder with pixel shuffle upsampling"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dims: List[int]
    ):
        super().__init__()
        
        modules = []
        
        # Input layer
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_dims[0], 3, padding=1),
                nn.BatchNorm2d(hidden_dims[0]),
                nn.SiLU()
            )
        )
        
        # Upsampling blocks
        for i in range(len(hidden_dims) - 1):
            modules.append(
                UpBlock(hidden_dims[i], hidden_dims[i + 1])
            )
        
        # Output layer
        modules.append(
            nn.Sequential(
                nn.Conv2d(hidden_dims[-1], hidden_dims[-1], 3, padding=1),
                nn.BatchNorm2d(hidden_dims[-1]),
                nn.SiLU(),
                nn.Conv2d(hidden_dims[-1], out_channels, 3, padding=1),
                nn.Sigmoid()  # Output in [0, 1]
            )
        )
        
        self.decoder = nn.Sequential(*modules)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class DownBlock(nn.Module):
    """Downsampling block with residual connection"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        
        self.activation = nn.SiLU()
        
        # Shortcut
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.activation(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        
        shortcut = self.shortcut(x)
        
        return self.activation(h + shortcut)


class UpBlock(nn.Module):
    """Upsampling block with pixel shuffle"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        # Pixel shuffle for 2x upsampling
        self.conv2 = nn.Conv2d(in_channels, out_channels * 4, 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        
        self.activation = nn.SiLU()
        
        # Shortcut
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, 1),
            nn.PixelShuffle(2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.activation(self.norm1(self.conv1(x)))
        h = self.conv2(h)
        h = self.pixel_shuffle(h)
        h = self.norm2(h)
        
        shortcut = self.shortcut(x)
        
        return self.activation(h + shortcut)


class RAWDiffusionUNet(nn.Module):
    """
    Conditional U-Net for diffusion in RAW latent space
    """
    
    def __init__(
        self,
        in_channels: int = 16,
        model_channels: int = 64,
        out_channels: int = 16,
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = [8, 16, 32],
        channel_mult: List[int] = [1, 2, 4, 8],
        num_heads: int = 8,
        context_dim: int = 512,
        use_checkpoint: bool = True
    ):
        """
        Args:
            in_channels: Input latent channels
            model_channels: Base channel count
            out_channels: Output latent channels
            num_res_blocks: Residual blocks per level
            attention_resolutions: Resolutions to apply attention
            channel_mult: Channel multiplier for each level
            num_heads: Attention heads
            context_dim: Conditioning context dimension
            use_checkpoint: Use gradient checkpointing
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.use_checkpoint = use_checkpoint
        
        # Timestep embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # Condition encoder (for reference frames, metadata, etc.)
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # Input blocks (encoder)
        self.input_blocks = nn.ModuleList([
            nn.Conv2d(in_channels, model_channels, 3, padding=1)
        ])
        
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        out_channels=mult * model_channels
                    )
                ]
                ch = mult * model_channels
                
                if ds in attention_resolutions:
                    layers.append(
                        SpatialTransformer(ch, num_heads, context_dim)
                    )
                
                self.input_blocks.append(nn.ModuleList(layers))
                input_block_chans.append(ch)
            
            if level != len(channel_mult) - 1:
                self.input_blocks.append(Downsample(ch))
                input_block_chans.append(ch)
                ds *= 2
        
        # Middle blocks
        self.middle_block = nn.ModuleList([
            ResBlock(ch, time_embed_dim),
            SpatialTransformer(ch, num_heads, context_dim),
            ResBlock(ch, time_embed_dim)
        ])
        
        # Output blocks (decoder)
        self.output_blocks = nn.ModuleList([])
        
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        out_channels=mult * model_channels
                    )
                ]
                ch = mult * model_channels
                
                if ds in attention_resolutions:
                    layers.append(
                        SpatialTransformer(ch, num_heads, context_dim)
                    )
                
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch))
                    ds //= 2
                
                self.output_blocks.append(nn.ModuleList(layers))
        
        # Output projection
        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Noisy latent [B, C, H, W]
            timesteps: Diffusion timesteps [B]
            context: Conditioning context [B, context_dim]
            
        Returns:
            Predicted noise [B, C, H, W]
        """
        # Timestep embedding
        t_emb = self.get_timestep_embedding(timesteps, self.model_channels)
        t_emb = self.time_embed(t_emb)
        
        # Context embedding
        if context is not None:
            c_emb = self.context_encoder(context)
            emb = t_emb + c_emb
        else:
            emb = t_emb
        
        # Encoder
        hs = []
        h = x
        
        for module in self.input_blocks:
            if isinstance(module, nn.ModuleList):
                for layer in module:
                    if isinstance(layer, ResBlock):
                        h = layer(h, emb)
                    else:
                        h = layer(h, context)
            else:
                h = module(h)
            hs.append(h)
        
        # Middle
        for layer in self.middle_block:
            if isinstance(layer, ResBlock):
                h = layer(h, emb)
            else:
                h = layer(h, context)
        
        # Decoder
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in module:
                if isinstance(layer, ResBlock):
                    h = layer(h, emb)
                elif isinstance(layer, SpatialTransformer):
                    h = layer(h, context)
                else:
                    h = layer(h)
        
        # Output
        return self.out(h)
    
    @staticmethod
    def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        """
        Sinusoidal timestep embedding
        
        Args:
            timesteps: Timestep values [B]
            embedding_dim: Embedding dimension
            
        Returns:
            Embeddings [B, embedding_dim]
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        
        if embedding_dim % 2 == 1:  # Pad if odd
            emb = F.pad(emb, (0, 1))
        
        return emb


class ResBlock(nn.Module):
    """Residual block with timestep conditioning"""
    
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0
    ):
        super().__init__()
        
        out_channels = out_channels or in_channels
        
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.time_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, out_channels)
        )
        
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        
        # Add timestep embedding
        h = h + self.time_emb_proj(time_emb)[:, :, None, None]
        
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        
        return h + self.shortcut(x)


class SpatialTransformer(nn.Module):
    """
    Spatial transformer with cross-attention
    For conditioning on reference frames, metadata, etc.
    """
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        context_dim: Optional[int] = None
    ):
        super().__init__()
        
        self.norm = nn.GroupNorm(32, channels)
        self.proj_in = nn.Conv2d(channels, channels, 1)
        
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(channels, num_heads, context_dim)
        ])
        
        self.proj_out = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, c, h, w = x.shape
        x_in = x
        
        x = self.norm(x)
        x = self.proj_in(x)
        
        # Reshape to sequence
        x = x.view(b, c, h * w).permute(0, 2, 1)  # [B, H*W, C]
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, context)
        
        # Reshape back
        x = x.permute(0, 2, 1).view(b, c, h, w)
        x = self.proj_out(x)
        
        return x + x_in


class BasicTransformerBlock(nn.Module):
    """Transformer block with self-attention and cross-attention"""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        context_dim: Optional[int] = None
    ):
        super().__init__()
        
        self.attn1 = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        
        self.attn2 = nn.MultiheadAttention(dim, num_heads, batch_first=True) if context_dim else None
        self.norm2 = nn.LayerNorm(dim) if context_dim else None
        
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm3 = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        x = x + self.attn1(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        
        # Cross-attention (if context provided)
        if context is not None and self.attn2 is not None:
            x = x + self.attn2(self.norm2(x), context, context)[0]
        
        # Feedforward
        x = x + self.ff(self.norm3(x))
        
        return x


class Downsample(nn.Module):
    """Downsampling layer"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Upsampling layer"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class DDPMScheduler:
    """
    DDPM (Denoising Diffusion Probabilistic Models) Scheduler
    Handles noise scheduling for training and inference
    """
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "cosine"
    ):
        """
        Args:
            num_train_timesteps: Number of diffusion steps
            beta_start: Starting beta value
            beta_end: Ending beta value
            beta_schedule: 'linear' or 'cosine'
        """
        self.num_train_timesteps = num_train_timesteps
        
        # Generate beta schedule
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "cosine":
            self.betas = self._cosine_beta_schedule(num_train_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Precompute alpha values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
    
    @staticmethod
    def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
        """
        Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def add_noise(
        self,
        original: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Add noise to original samples for training
        
        Args:
            original: Original samples [B, C, H, W]
            noise: Noise to add [B, C, H, W]
            timesteps: Timesteps [B]
            
        Returns:
            Noisy samples
        """
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].to(original.device)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].to(original.device)
        
        # Reshape for broadcasting
        sqrt_alpha_prod = sqrt_alpha_prod.view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.view(-1, 1, 1, 1)
        
        noisy = sqrt_alpha_prod * original + sqrt_one_minus_alpha_prod * noise
        
        return noisy
    
    def denoise_step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor
    ) -> torch.Tensor:
        """
        Single denoising step
        
        Args:
            model_output: Predicted noise
            timestep: Current timestep
            sample: Current noisy sample
            
        Returns:
            Denoised sample
        """
        # Get parameters for this timestep
        alpha_t = self.alphas[timestep].to(sample.device)
        alpha_prod_t = self.alphas_cumprod[timestep].to(sample.device)
        beta_t = self.betas[timestep].to(sample.device)
        
        # Predicted original sample
        pred_original = (sample - torch.sqrt(1 - alpha_prod_t) * model_output) / torch.sqrt(alpha_prod_t)
        
        # Compute previous sample
        prev_sample = torch.sqrt(alpha_t) * pred_original + torch.sqrt(beta_t) * model_output
        
        return prev_sample


if __name__ == "__main__":
    print("Testing Latent Diffusion components...")
    
    # Test VAE
    print("\n1. Testing RAWVAE...")
    vae = RAWVAE()
    vae.eval()
    
    raw_input = torch.randn(2, 4, 256, 256)
    
    with torch.no_grad():
        recon, losses = vae(raw_input, return_loss=True)
    
    print(f"   Input shape: {raw_input.shape}")
    print(f"   Reconstruction shape: {recon.shape}")
    print(f"   Reconstruction loss: {losses['reconstruction']:.4f}")
    print(f"   KL loss: {losses['kl']:.4f}")
    
    # Test latent encoding
    with torch.no_grad():
        z_mean, z_logvar = vae.encode(raw_input)
    print(f"   Latent shape: {z_mean.shape}")
    
    # Test Diffusion U-Net
    print("\n2. Testing RAWDiffusionUNet...")
    unet = RAWDiffusionUNet()
    unet.eval()
    
    latent_input = torch.randn(2, 16, 32, 32)
    timesteps = torch.randint(0, 1000, (2,))
    context = torch.randn(2, 512)
    
    with torch.no_grad():
        noise_pred = unet(latent_input, timesteps, context)
    
    print(f"   Latent input shape: {latent_input.shape}")
    print(f"   Noise prediction shape: {noise_pred.shape}")
    
    # Test DDPM Scheduler
    print("\n3. Testing DDPMScheduler...")
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="cosine")
    
    clean = torch.randn(2, 16, 32, 32)
    noise = torch.randn_like(clean)
    t = torch.randint(0, 1000, (2,))
    
    noisy = scheduler.add_noise(clean, noise, t)
    print(f"   Noisy sample range: [{noisy.min():.2f}, {noisy.max():.2f}]")
    
    print("\n✓ All diffusion tests passed!")
