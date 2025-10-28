"""Training Orchestrator - 4-Stage Training Pipeline"""

import os
import sys
import argparse
import yaml
import torch
import torch.distributed as dist
from pathlib import Path
from datetime import datetime
import json
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from training.train_diffusion import (
    VAETrainer, OpticalFlowTrainer, DiffusionTrainer,
    setup_distributed, cleanup_distributed
)
from models.raw_diffusion_unet import RAWVAE, RAWDiffusionUNet, DDPMScheduler
from models.optical_flow import RAWOpticalFlow, AlignmentModule
from models.lens_aberration_module import AberrationCorrectionModule
from data.raw_loader import BurstRawDataset
from data.augmentation import RawAugmentation


class TrainingOrchestrator:
    """
    Orchestrates complete training pipeline with automatic progression.
    """
    
    def __init__(
        self,
        config_path: str,
        data_dir: str,
        output_dir: str,
        resume_from: str = None,
        rank: int = 0,
        world_size: int = 1
    ):
        self.rank = rank
        self.world_size = world_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_dir = data_dir
        self.resume_from = resume_from
        
        # Training state
        self.state = {
            'current_stage': 'vae',
            'completed_stages': [],
            'best_metrics': {},
            'checkpoints': {}
        }
        
        # Load state if resuming
        if resume_from:
            self._load_state()
        
        logger.info(f"Initialized orchestrator (rank {rank}/{world_size})")
    
    def _load_state(self):
        """Load training state from checkpoint."""
        state_file = Path(self.resume_from) / 'training_state.json'
        if state_file.exists():
            with open(state_file, 'r') as f:
                self.state = json.load(f)
            logger.info(f"Resumed from stage: {self.state['current_stage']}")
    
    def _save_state(self):
        """Save current training state."""
        state_file = self.output_dir / 'training_state.json'
        with open(state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
        logger.info("Saved training state")
    
    def _create_dataloader(self, batch_size: int, num_workers: int = 4):
        """Create training dataloader."""
        dataset = BurstRawDataset(
            data_dir=self.data_dir,
            burst_size=self.config['data']['burst_size'],
            augmentation=RawAugmentation() if self.config['data']['augmentation'] else None
        )
        
        # Distributed sampler
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=self.world_size, rank=self.rank
        ) if self.world_size > 1 else None
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        return dataloader
    
    def train_stage_1_vae(self):
        """Stage 1: VAE Pre-training."""
        if 'vae' in self.state['completed_stages']:
            logger.info("Stage 1 (VAE) already completed, skipping...")
            return
        
        logger.info("="*70)
        logger.info("STAGE 1: VAE PRE-TRAINING")
        logger.info("="*70)
        
        device = f'cuda:{self.rank}' if torch.cuda.is_available() else 'cpu'
        
        # Initialize model
        vae = RAWVAE(**self.config['model']['vae']).to(device)
        
        if self.world_size > 1:
            vae = torch.nn.parallel.DistributedDataParallel(
                vae, device_ids=[self.rank]
            )
        
        # Create trainer
        trainer = VAETrainer(vae, self.config, device, self.rank, self.world_size)
        
        # Create dataloader
        dataloader = self._create_dataloader(
            batch_size=self.config['training']['batch_size']
        )
        
        # Training loop
        best_loss = float('inf')
        for epoch in range(self.config['vae']['epochs']):
            metrics = trainer.train_epoch(dataloader, epoch)
            
            if self.rank == 0:
                logger.info(f"Epoch {epoch}: {metrics}")
                
                # Save best checkpoint
                if metrics['loss'] < best_loss:
                    best_loss = metrics['loss']
                    checkpoint_path = self.output_dir / 'vae_best.pt'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': vae.module.state_dict() if self.world_size > 1 else vae.state_dict(),
                        'optimizer_state_dict': trainer.optimizer.state_dict(),
                        'metrics': metrics
                    }, checkpoint_path)
                    logger.info(f"Saved best checkpoint: {checkpoint_path}")
                
                # Save periodic checkpoint
                if (epoch + 1) % 5 == 0:
                    checkpoint_path = self.output_dir / f'vae_epoch_{epoch}.pt'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': vae.module.state_dict() if self.world_size > 1 else vae.state_dict(),
                        'optimizer_state_dict': trainer.optimizer.state_dict(),
                        'metrics': metrics
                    }, checkpoint_path)
        
        # Save final checkpoint
        if self.rank == 0:
            final_path = self.output_dir / 'vae_final.pt'
            torch.save({
                'epoch': self.config['vae']['epochs'] - 1,
                'model_state_dict': vae.module.state_dict() if self.world_size > 1 else vae.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'config': self.config['model']['vae']
            }, final_path)
            
            self.state['completed_stages'].append('vae')
            self.state['checkpoints']['vae'] = str(final_path)
            self.state['best_metrics']['vae'] = {'loss': best_loss}
            self._save_state()
            
            logger.info(f"✓ Stage 1 (VAE) complete! Best loss: {best_loss:.4f}")
    
    def train_stage_2_flow(self):
        """Stage 2: Optical Flow Training."""
        if 'flow' in self.state['completed_stages']:
            logger.info("Stage 2 (Flow) already completed, skipping...")
            return
        
        logger.info("\n" + "="*70)
        logger.info("STAGE 2: OPTICAL FLOW TRAINING")
        logger.info("="*70)
        
        device = f'cuda:{self.rank}' if torch.cuda.is_available() else 'cpu'
        
        # Initialize model
        flow_net = RAWOpticalFlow(**self.config['model']['optical_flow']).to(device)
        
        if self.world_size > 1:
            flow_net = torch.nn.parallel.DistributedDataParallel(
                flow_net, device_ids=[self.rank]
            )
        
        # Create trainer
        trainer = OpticalFlowTrainer(flow_net, self.config, device, self.rank, self.world_size)
        
        # Create dataloader
        dataloader = self._create_dataloader(
            batch_size=self.config['training']['batch_size'] // 2  # Flow uses more memory
        )
        
        trainer.scheduler.total_steps = len(dataloader) * self.config['optical_flow']['epochs']
        
        # Training loop
        best_loss = float('inf')
        for epoch in range(self.config['optical_flow']['epochs']):
            metrics = trainer.train_epoch(dataloader, epoch)
            
            if self.rank == 0:
                logger.info(f"Epoch {epoch}: {metrics}")
                
                if metrics['loss'] < best_loss:
                    best_loss = metrics['loss']
                    checkpoint_path = self.output_dir / 'flow_best.pt'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': flow_net.module.state_dict() if self.world_size > 1 else flow_net.state_dict(),
                        'optimizer_state_dict': trainer.optimizer.state_dict(),
                        'metrics': metrics
                    }, checkpoint_path)
        
        # Save final
        if self.rank == 0:
            final_path = self.output_dir / 'flow_final.pt'
            torch.save({
                'epoch': self.config['optical_flow']['epochs'] - 1,
                'model_state_dict': flow_net.module.state_dict() if self.world_size > 1 else flow_net.state_dict(),
                'config': self.config['model']['optical_flow']
            }, final_path)
            
            self.state['completed_stages'].append('flow')
            self.state['checkpoints']['flow'] = str(final_path)
            self.state['best_metrics']['flow'] = {'loss': best_loss}
            self._save_state()
            
            logger.info(f"✓ Stage 2 (Flow) complete! Best loss: {best_loss:.4f}")
    
    def train_stage_3_diffusion(self):
        """Stage 3: Diffusion Model Training."""
        if 'diffusion' in self.state['completed_stages']:
            logger.info("Stage 3 (Diffusion) already completed, skipping...")
            return
        
        logger.info("\n" + "="*70)
        logger.info("STAGE 3: DIFFUSION MODEL TRAINING")
        logger.info("="*70)
        
        device = f'cuda:{self.rank}' if torch.cuda.is_available() else 'cpu'
        
        # Load VAE and Flow
        vae = RAWVAE(**self.config['model']['vae']).to(device)
        vae.load_state_dict(torch.load(self.state['checkpoints']['vae'], map_location=device)['model_state_dict'])
        vae.eval()
        
        flow_net = RAWOpticalFlow(**self.config['model']['optical_flow']).to(device)
        flow_net.load_state_dict(torch.load(self.state['checkpoints']['flow'], map_location=device)['model_state_dict'])
        alignment = AlignmentModule(flow_net)
        alignment.eval()
        
        # Initialize diffusion
        unet = RAWDiffusionUNet(**self.config['model']['unet']).to(device)
        scheduler = DDPMScheduler(**self.config['model']['scheduler'])
        
        aberration = None
        if self.config['model'].get('aberration'):
            aberration = AberrationCorrectionModule(**self.config['model']['aberration']).to(device)
        
        if self.world_size > 1:
            unet = torch.nn.parallel.DistributedDataParallel(unet, device_ids=[self.rank])
            if aberration:
                aberration = torch.nn.parallel.DistributedDataParallel(aberration, device_ids=[self.rank])
        
        # Create trainer
        trainer = DiffusionTrainer(
            vae, unet, scheduler, alignment, aberration,
            self.config, device, self.rank, self.world_size
        )
        
        # Create dataloader
        dataloader = self._create_dataloader(
            batch_size=self.config['training']['batch_size'] // 2
        )
        
        # Training loop
        best_loss = float('inf')
        for epoch in range(self.config['diffusion']['epochs']):
            metrics = trainer.train_epoch(dataloader, epoch)
            
            if self.rank == 0:
                logger.info(f"Epoch {epoch}: {metrics}")
                
                if metrics['loss'] < best_loss:
                    best_loss = metrics['loss']
                    checkpoint_path = self.output_dir / 'diffusion_best.pt'
                    torch.save({
                        'epoch': epoch,
                        'unet_state_dict': unet.module.state_dict() if self.world_size > 1 else unet.state_dict(),
                        'metrics': metrics
                    }, checkpoint_path)
                
                if (epoch + 1) % 10 == 0:
                    checkpoint_path = self.output_dir / f'diffusion_epoch_{epoch}.pt'
                    torch.save({
                        'epoch': epoch,
                        'unet_state_dict': unet.module.state_dict() if self.world_size > 1 else unet.state_dict(),
                    }, checkpoint_path)
        
        # Save final
        if self.rank == 0:
            final_path = self.output_dir / 'diffusion_final.pt'
            torch.save({
                'epoch': self.config['diffusion']['epochs'] - 1,
                'unet_state_dict': unet.module.state_dict() if self.world_size > 1 else unet.state_dict(),
                'config': self.config['model']['unet']
            }, final_path)
            
            self.state['completed_stages'].append('diffusion')
            self.state['checkpoints']['diffusion'] = str(final_path)
            self.state['best_metrics']['diffusion'] = {'loss': best_loss}
            self._save_state()
            
            logger.info(f"✓ Stage 3 (Diffusion) complete! Best loss: {best_loss:.4f}")
    
    def train_stage_4_consistency(self):
        """Stage 4: Consistency Distillation."""
        logger.info("\n" + "="*70)
        logger.info("STAGE 4: CONSISTENCY DISTILLATION")
        logger.info("="*70)
        logger.info("⚠️  This stage requires the consistency trainer implementation")
        logger.info("    Placeholder for consistency distillation training")
        
        # TODO: Implement consistency distillation training
        # This would involve:
        # 1. Load trained diffusion model as teacher
        # 2. Initialize consistency model as student
        # 3. Progressive distillation (1000 -> 50 -> 10 -> 4 -> 2 steps)
        # 4. Train with consistency loss + distillation loss
        
        if self.rank == 0:
            logger.info("✓ Stage 4 (Consistency) placeholder complete")
    
    def run_complete_training(self):
        """Run all training stages in sequence."""
        logger.info("\n" + "="*70)
        logger.info("STARTING COMPLETE 4-STAGE TRAINING PIPELINE")
        logger.info("="*70)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"World size: {self.world_size}")
        
        start_time = datetime.now()
        
        try:
            # Stage 1: VAE
            self.train_stage_1_vae()
            
            # Stage 2: Optical Flow
            self.train_stage_2_flow()
            
            # Stage 3: Diffusion
            self.train_stage_3_diffusion()
            
            # Stage 4: Consistency
            self.train_stage_4_consistency()
            
            # Training complete
            if self.rank == 0:
                elapsed = datetime.now() - start_time
                logger.info("\n" + "="*70)
                logger.info("🎉 TRAINING COMPLETE! 🎉")
                logger.info("="*70)
                logger.info(f"Total time: {elapsed}")
                logger.info(f"Checkpoints saved to: {self.output_dir}")
                logger.info("\nCompleted stages:")
                for stage in self.state['completed_stages']:
                    metrics = self.state['best_metrics'].get(stage, {})
                    logger.info(f"  ✓ {stage}: {metrics}")
                
                # Save final summary
                summary = {
                    'training_time': str(elapsed),
                    'completed_stages': self.state['completed_stages'],
                    'best_metrics': self.state['best_metrics'],
                    'checkpoints': self.state['checkpoints'],
                    'config': self.config
                }
                
                summary_path = self.output_dir / 'training_summary.json'
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)
                
                logger.info(f"\nTraining summary saved to: {summary_path}")
                logger.info("\nNext steps:")
                logger.info("1. Run validation: python scripts/evaluate.py")
                logger.info("2. Export models: python scripts/final_integration.py")
                logger.info("3. Deploy to production")
        
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            raise


def main():
    parser = argparse.ArgumentParser(description='Complete Training Pipeline')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                       help='Path to training config')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing training data')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory for checkpoints')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Resume from checkpoint directory')
    
    args = parser.parse_args()
    
    # Setup distributed training
    rank, world_size = setup_distributed()
    
    try:
        # Create orchestrator
        orchestrator = TrainingOrchestrator(
            config_path=args.config,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            resume_from=args.resume_from,
            rank=rank,
            world_size=world_size
        )
        
        # Run training
        orchestrator.run_complete_training()
    
    finally:
        # Cleanup
        cleanup_distributed()


if __name__ == "__main__":
    main()
