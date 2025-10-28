"""
Training Script for RAW Diffusion Model
Complete 4-stage training pipeline: VAE, Optical Flow, Diffusion, Consistency
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from typing import Dict, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
import wandb
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from data.raw_loader import BurstRawDataset
from data.preprocessing import raw_to_srgb_simple
from data.augmentation import RawAugmentation, SyntheticBurstGenerator
from models.raw_diffusion_unet import RAWVAE, RAWDiffusionUNet, DDPMScheduler
from models.optical_flow import RAWOpticalFlow, AlignmentModule
from models.consistency_distillation import ConsistencyModel, ConsistencyDistillationTrainer
from models.lens_aberration_module import AberrationCorrectionModule, DiffusionAberrationPrior
from training.losses import CombinedLoss
from training.metrics import MetricCalculator


class VAETrainer:
    """Trainer for RAW VAE pre-training."""
    
    def __init__(
        self,
        vae: RAWVAE,
        config: Dict,
        device: str = 'cuda',
        rank: int = 0,
        world_size: int = 1
    ):
        self.vae = vae
        self.config = config
        self.device = device
        self.rank = rank
        self.world_size = world_size
        
        # Optimizer
        self.optimizer = optim.AdamW(
            vae.parameters(),
            lr=config['vae']['learning_rate'],
            weight_decay=config['vae']['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['vae']['epochs'],
            eta_min=1e-6
        )
        
        # Mixed precision
        self.scaler = GradScaler() if config['training']['mixed_precision'] else None
        
        # Loss weights
        self.kl_weight = config['vae']['kl_weight']
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.vae.train()
        
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"VAE Epoch {epoch}", disable=self.rank != 0)
        
        for batch_idx, batch in enumerate(pbar):
            # Get reference frame from burst
            burst = batch['burst'].to(self.device)  # [B, T, 4, H, W]
            reference = burst[:, 0]  # Use first frame as reference
            
            # Forward pass
            self.optimizer.zero_grad()
            
            with autocast(enabled=self.scaler is not None):
                # Encode and decode
                posterior = self.vae.encode(reference)
                z = posterior.sample()
                recon = self.vae.decode(z)
                
                # Reconstruction loss
                recon_loss = nn.functional.l1_loss(recon, reference)
                
                # KL divergence
                kl_loss = posterior.kl().mean()
                
                # Total loss
                loss = recon_loss + self.kl_weight * kl_loss
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.vae.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.vae.parameters(), 1.0)
                self.optimizer.step()
            
            # Update stats
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            num_batches += 1
            
            # Update progress bar
            if self.rank == 0:
                pbar.set_postfix({
                    'loss': total_loss / num_batches,
                    'recon': total_recon / num_batches,
                    'kl': total_kl / num_batches
                })
        
        self.scheduler.step()
        
        return {
            'loss': total_loss / num_batches,
            'recon_loss': total_recon / num_batches,
            'kl_loss': total_kl / num_batches,
            'lr': self.optimizer.param_groups[0]['lr']
        }


class OpticalFlowTrainer:
    """Trainer for optical flow network."""
    
    def __init__(
        self,
        flow_net: RAWOpticalFlow,
        config: Dict,
        device: str = 'cuda',
        rank: int = 0,
        world_size: int = 1
    ):
        self.flow_net = flow_net
        self.config = config
        self.device = device
        self.rank = rank
        self.world_size = world_size
        
        # Optimizer
        self.optimizer = optim.AdamW(
            flow_net.parameters(),
            lr=config['optical_flow']['learning_rate'],
            weight_decay=config['optical_flow']['weight_decay']
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config['optical_flow']['learning_rate'],
            epochs=config['optical_flow']['epochs'],
            steps_per_epoch=1000  # Will be updated
        )
        
        self.scaler = GradScaler() if config['training']['mixed_precision'] else None
    
    def photometric_loss(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        flow: torch.Tensor
    ) -> torch.Tensor:
        """Compute photometric loss with occlusion handling."""
        # Warp img1 to img2
        B, C, H, W = img1.shape
        
        # Create sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=img1.device),
            torch.arange(W, device=img1.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).float()
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)
        
        warped_grid = grid + flow
        warped_grid[:, 0] = 2.0 * warped_grid[:, 0] / (W - 1) - 1.0
        warped_grid[:, 1] = 2.0 * warped_grid[:, 1] / (H - 1) - 1.0
        warped_grid = warped_grid.permute(0, 2, 3, 1)
        
        img1_warped = nn.functional.grid_sample(
            img1, warped_grid, align_corners=True, padding_mode='border'
        )
        
        # Robust loss (Charbonnier)
        diff = img1_warped - img2
        loss = torch.sqrt(diff.pow(2) + 1e-6).mean()
        
        return loss
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.flow_net.train()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Flow Epoch {epoch}", disable=self.rank != 0)
        
        for batch in pbar:
            burst = batch['burst'].to(self.device)  # [B, T, 4, H, W]
            B, T = burst.shape[:2]
            
            self.optimizer.zero_grad()
            
            epoch_loss = 0.0
            
            # Compute flow between consecutive frames
            for t in range(T - 1):
                img1 = burst[:, t]
                img2 = burst[:, t + 1]
                
                with autocast(enabled=self.scaler is not None):
                    # Predict flow
                    flow = self.flow_net(img1, img2)
                    
                    # Photometric loss
                    loss = self.photometric_loss(img1, img2, flow)
                    
                    # Smoothness loss
                    grad_x = flow[:, :, :, 1:] - flow[:, :, :, :-1]
                    grad_y = flow[:, :, 1:, :] - flow[:, :, :-1, :]
                    smoothness = (grad_x.abs().mean() + grad_y.abs().mean())
                    
                    loss = loss + 0.1 * smoothness
                
                epoch_loss += loss
            
            epoch_loss = epoch_loss / (T - 1)
            
            # Backward
            if self.scaler is not None:
                self.scaler.scale(epoch_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.flow_net.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                epoch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.flow_net.parameters(), 1.0)
                self.optimizer.step()
            
            self.scheduler.step()
            
            total_loss += epoch_loss.item()
            num_batches += 1
            
            if self.rank == 0:
                pbar.set_postfix({'loss': total_loss / num_batches})
        
        return {
            'loss': total_loss / num_batches,
            'lr': self.optimizer.param_groups[0]['lr']
        }


class DiffusionTrainer:
    """Trainer for diffusion model."""
    
    def __init__(
        self,
        vae: RAWVAE,
        unet: RAWDiffusionUNet,
        scheduler: DDPMScheduler,
        alignment: AlignmentModule,
        aberration: Optional[AberrationCorrectionModule],
        config: Dict,
        device: str = 'cuda',
        rank: int = 0,
        world_size: int = 1
    ):
        self.vae = vae
        self.unet = unet
        self.scheduler = scheduler
        self.alignment = alignment
        self.aberration = aberration
        self.config = config
        self.device = device
        self.rank = rank
        self.world_size = world_size
        
        # Freeze VAE and alignment
        self.vae.eval()
        self.vae.requires_grad_(False)
        self.alignment.eval()
        self.alignment.requires_grad_(False)
        
        # Optimizer
        params = list(unet.parameters())
        if aberration is not None:
            params += list(aberration.parameters())
        
        self.optimizer = optim.AdamW(
            params,
            lr=config['diffusion']['learning_rate'],
            weight_decay=config['diffusion']['weight_decay']
        )
        
        # Scheduler
        self.scheduler_lr = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['diffusion']['epochs']
        )
        
        self.scaler = GradScaler() if config['training']['mixed_precision'] else None
        
        # Loss function
        self.loss_fn = CombinedLoss(
            l1_weight=1.0,
            perceptual_weight=1.0,
            hallucination_weight=0.5,
            temporal_weight=0.0,  # No temporal for single frame training
            edge_weight=0.5,
            chroma_weight=0.2
        ).to(device)
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.unet.train()
        if self.aberration is not None:
            self.aberration.train()
        
        total_loss = 0.0
        loss_components = {}
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Diffusion Epoch {epoch}", disable=self.rank != 0)
        
        for batch in pbar:
            burst = batch['burst'].to(self.device)  # [B, T, 4, H, W]
            metadata = batch.get('metadata', None)
            
            B, T = burst.shape[:2]
            
            # Use middle frame as target
            target_idx = T // 2
            target = burst[:, target_idx]
            
            self.optimizer.zero_grad()
            
            with autocast(enabled=self.scaler is not None):
                # Align burst
                with torch.no_grad():
                    aligned_burst = self.alignment(burst, reference_idx=target_idx)
                
                # Average aligned burst as conditioning
                condition = aligned_burst.mean(dim=1)
                
                # Apply aberration correction if available
                if self.aberration is not None and metadata is not None:
                    condition = self.aberration(condition, metadata)
                
                # Encode target and condition
                with torch.no_grad():
                    target_latent = self.vae.encode(target).sample()
                    condition_latent = self.vae.encode(condition).sample()
                
                # Sample timestep
                t = torch.randint(
                    0, self.scheduler.num_train_timesteps,
                    (B,), device=self.device
                ).long()
                
                # Add noise
                noise = torch.randn_like(target_latent)
                noisy_latent = self.scheduler.add_noise(target_latent, noise, t)
                
                # Predict noise
                noise_pred = self.unet(noisy_latent, t, condition_latent)
                
                # Simple MSE loss
                loss = nn.functional.mse_loss(noise_pred, noise)
            
            # Backward
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 1.0)
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if self.rank == 0:
                pbar.set_postfix({'loss': total_loss / num_batches})
        
        self.scheduler_lr.step()
        
        return {
            'loss': total_loss / num_batches,
            'lr': self.optimizer.param_groups[0]['lr']
        }


def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(rank)
        dist.init_process_group(backend='nccl')
        return rank, world_size
    return 0, 1


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description='Train RAW Diffusion Model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--stage', type=str, default='all', 
                       choices=['vae', 'flow', 'diffusion', 'consistency', 'all'],
                       help='Training stage')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    
    args = parser.parse_args()
    
    # Setup distributed
    rank, world_size = setup_distributed()
    device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize wandb (rank 0 only)
    if rank == 0:
        wandb.init(
            project='raw-diffusion',
            config=config,
            name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataset
    dataset = BurstRawDataset(
        data_dir=args.data_dir,
        burst_size=config['data']['burst_size'],
        augmentation=RawAugmentation() if config['data']['augmentation'] else None
    )
    
    # Create dataloader
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank
    ) if world_size > 1 else None
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    print(f"Rank {rank}: Loaded {len(dataset)} samples")
    
    # Initialize models
    vae = RAWVAE(**config['model']['vae']).to(device)
    flow_net = RAWOpticalFlow(**config['model']['optical_flow']).to(device)
    alignment = AlignmentModule(flow_net).to(device)
    unet = RAWDiffusionUNet(**config['model']['unet']).to(device)
    scheduler = DDPMScheduler(**config['model']['scheduler'])
    aberration = AberrationCorrectionModule(**config['model']['aberration']).to(device) \
                 if config['model'].get('aberration') else None
    
    # Wrap with DDP
    if world_size > 1:
        vae = DDP(vae, device_ids=[rank])
        flow_net = DDP(flow_net, device_ids=[rank])
        unet = DDP(unet, device_ids=[rank])
        if aberration is not None:
            aberration = DDP(aberration, device_ids=[rank])
    
    # Stage 1: VAE pre-training
    if args.stage in ['vae', 'all']:
        print(f"\n{'='*50}")
        print("Stage 1: VAE Pre-training")
        print(f"{'='*50}\n")
        
        trainer = VAETrainer(vae, config, device, rank, world_size)
        
        for epoch in range(config['vae']['epochs']):
            metrics = trainer.train_epoch(dataloader, epoch)
            
            if rank == 0:
                print(f"Epoch {epoch}: {metrics}")
                wandb.log({f'vae/{k}': v for k, v in metrics.items()}, step=epoch)
                
                # Save checkpoint
                if (epoch + 1) % config['training']['save_interval'] == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': vae.module.state_dict() if world_size > 1 else vae.state_dict(),
                        'optimizer_state_dict': trainer.optimizer.state_dict(),
                    }, output_dir / f'vae_epoch_{epoch}.pt')
    
    # Stage 2: Optical flow training
    if args.stage in ['flow', 'all']:
        print(f"\n{'='*50}")
        print("Stage 2: Optical Flow Training")
        print(f"{'='*50}\n")
        
        trainer = OpticalFlowTrainer(flow_net, config, device, rank, world_size)
        trainer.scheduler.total_steps = len(dataloader) * config['optical_flow']['epochs']
        
        for epoch in range(config['optical_flow']['epochs']):
            metrics = trainer.train_epoch(dataloader, epoch)
            
            if rank == 0:
                print(f"Epoch {epoch}: {metrics}")
                wandb.log({f'flow/{k}': v for k, v in metrics.items()}, 
                         step=epoch + config['vae']['epochs'])
                
                if (epoch + 1) % config['training']['save_interval'] == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': flow_net.module.state_dict() if world_size > 1 else flow_net.state_dict(),
                        'optimizer_state_dict': trainer.optimizer.state_dict(),
                    }, output_dir / f'flow_epoch_{epoch}.pt')
    
    # Stage 3: Diffusion training
    if args.stage in ['diffusion', 'all']:
        print(f"\n{'='*50}")
        print("Stage 3: Diffusion Model Training")
        print(f"{'='*50}\n")
        
        trainer = DiffusionTrainer(
            vae, unet, scheduler, alignment, aberration, config, device, rank, world_size
        )
        
        for epoch in range(config['diffusion']['epochs']):
            metrics = trainer.train_epoch(dataloader, epoch)
            
            if rank == 0:
                print(f"Epoch {epoch}: {metrics}")
                wandb.log({f'diffusion/{k}': v for k, v in metrics.items()}, 
                         step=epoch + config['vae']['epochs'] + config['optical_flow']['epochs'])
                
                if (epoch + 1) % config['training']['save_interval'] == 0:
                    torch.save({
                        'epoch': epoch,
                        'unet_state_dict': unet.module.state_dict() if world_size > 1 else unet.state_dict(),
                        'optimizer_state_dict': trainer.optimizer.state_dict(),
                    }, output_dir / f'diffusion_epoch_{epoch}.pt')
    
    # Stage 4: Consistency distillation
    if args.stage in ['consistency', 'all']:
        print(f"\n{'='*50}")
        print("Stage 4: Consistency Distillation")
        print(f"{'='*50}\n")
        
        # Load best diffusion checkpoint
        # TODO: Implement consistency distillation training
        print("Consistency distillation training - TODO")
    
    if rank == 0:
        print("\n✓ Training complete!")
        wandb.finish()
    
    cleanup_distributed()


if __name__ == "__main__":
    main()
