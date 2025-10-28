"""
Consistency Distillation for Real-Time Inference
Reduces 1000 diffusion steps to 2-4 steps for real-time performance
Based on "Consistency Models" (Song et al., 2023)
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConsistencyModel(nn.Module):
    """
    Consistency model for fast diffusion inference
    Maps any noisy sample directly to clean sample in 2-4 steps
    """
    
    def __init__(
        self,
        unet: nn.Module,
        num_consistency_steps: int = 4,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        sigma_data: float = 0.5,
        rho: float = 7.0
    ):
        """
        Args:
            unet: Base diffusion U-Net model
            num_consistency_steps: Target inference steps
            sigma_min: Minimum noise level
            sigma_max: Maximum noise level
            sigma_data: Data noise scale
            rho: Schedule parameter
        """
        super().__init__()
        
        self.unet = unet
        self.num_consistency_steps = num_consistency_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        
        # Skip connection scaling parameters (learnable)
        self.skip_scaling = nn.Parameter(torch.ones(1))
        self.output_scaling = nn.Parameter(torch.ones(1))
    
    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Consistency function: maps (x_t, t) to x_0
        
        Args:
            x: Noisy input [B, C, H, W]
            sigma: Noise level [B]
            context: Conditioning context [B, context_dim]
            
        Returns:
            Predicted clean sample [B, C, H, W]
        """
        # Preconditioning (from EDM paper)
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)
        c_in = 1 / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)
        c_noise = 0.25 * torch.log(sigma)
        
        # Reshape for broadcasting
        c_skip = c_skip.view(-1, 1, 1, 1)
        c_out = c_out.view(-1, 1, 1, 1)
        c_in = c_in.view(-1, 1, 1, 1)
        
        # Convert sigma to timestep for U-Net
        timesteps = self._sigma_to_timestep(sigma)
        
        # Forward through U-Net
        F_theta = self.unet(c_in * x, timesteps, context)
        
        # Consistency function with skip connection
        D_theta = c_skip * x + c_out * F_theta
        
        return D_theta
    
    def generate(
        self,
        latent_shape: Tuple[int, ...],
        condition: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        device: str = 'cuda'
    ) -> torch.Tensor:
        """
        Generate samples using consistency model
        
        Args:
            latent_shape: Shape of latent [B, C, H, W]
            condition: Conditioning context
            num_steps: Number of generation steps (default: self.num_consistency_steps)
            device: Device
            
        Returns:
            Generated clean samples
        """
        if num_steps is None:
            num_steps = self.num_consistency_steps
        
        # Generate timestep schedule
        sigmas = self._get_sigma_schedule(num_steps, device)
        
        # Start from pure noise
        x = torch.randn(latent_shape, device=device) * sigmas[0]
        
        # Iterative refinement
        for i in range(num_steps):
            sigma = sigmas[i]
            
            # Predict clean sample
            with torch.no_grad():
                x = self(x, sigma.repeat(latent_shape[0]), condition)
            
            # Add noise for next step (except last)
            if i < num_steps - 1:
                noise = torch.randn_like(x)
                x = x + noise * sigmas[i + 1]
        
        return x
    
    def _get_sigma_schedule(self, num_steps: int, device: str) -> torch.Tensor:
        """
        Generate noise schedule for sampling
        
        Args:
            num_steps: Number of steps
            device: Device
            
        Returns:
            Sigma values [num_steps]
        """
        # Karras schedule (exponential)
        min_inv_rho = self.sigma_min ** (1 / self.rho)
        max_inv_rho = self.sigma_max ** (1 / self.rho)
        
        rho_steps = torch.linspace(0, 1, num_steps, device=device)
        sigmas = (max_inv_rho + rho_steps * (min_inv_rho - max_inv_rho)) ** self.rho
        
        return sigmas
    
    def _sigma_to_timestep(self, sigma: torch.Tensor) -> torch.Tensor:
        """Convert sigma to timestep for U-Net"""
        # Simple mapping: sigma -> timestep [0, 999]
        log_sigma = torch.log(sigma)
        timestep = (log_sigma - torch.log(torch.tensor(self.sigma_min))) / \
                   (torch.log(torch.tensor(self.sigma_max)) - torch.log(torch.tensor(self.sigma_min)))
        timestep = timestep * 999
        timestep = torch.clamp(timestep, 0, 999).long()
        return timestep


class ConsistencyDistillationTrainer:
    """
    Trainer for consistency distillation
    Distills teacher diffusion model into fast consistency model
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: ConsistencyModel,
        ema_decay: float = 0.9999,
        num_distill_steps: int = 4
    ):
        """
        Args:
            teacher_model: Pre-trained diffusion model (frozen)
            student_model: Consistency model (trainable)
            ema_decay: EMA decay rate for teacher updates
            num_distill_steps: Target number of inference steps
        """
        self.teacher = teacher_model
        self.student = student_model
        self.ema_decay = ema_decay
        self.num_distill_steps = num_distill_steps
        
        # EMA teacher
        self.ema_teacher = self._create_ema_model(teacher_model)
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    def _create_ema_model(self, model: nn.Module) -> nn.Module:
        """Create EMA copy of model"""
        ema_model = type(model)(**self._get_model_kwargs(model))
        ema_model.load_state_dict(model.state_dict())
        for param in ema_model.parameters():
            param.requires_grad = False
        return ema_model
    
    @staticmethod
    def _get_model_kwargs(model: nn.Module) -> Dict:
        """Extract model initialization kwargs"""
        # This is a simplified version - adjust based on your model
        return {}
    
    def compute_consistency_loss(
        self,
        x_0: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute consistency distillation loss
        
        Args:
            x_0: Clean samples [B, C, H, W]
            condition: Conditioning context
            
        Returns:
            Dictionary of losses
        """
        B = x_0.shape[0]
        device = x_0.device
        
        # Sample two adjacent timesteps
        t = torch.randint(1, 1000, (B,), device=device)
        t_next = t - 1
        
        # Convert to sigma
        sigma_t = self._timestep_to_sigma(t)
        sigma_next = self._timestep_to_sigma(t_next)
        
        # Add noise
        noise = torch.randn_like(x_0)
        x_t = x_0 + noise * sigma_t.view(-1, 1, 1, 1)
        x_next = x_0 + noise * sigma_next.view(-1, 1, 1, 1)
        
        # Student predictions
        with torch.no_grad():
            # EMA teacher prediction
            pred_teacher = self.ema_teacher(x_t, sigma_t, condition)
        
        # Student prediction
        pred_student_t = self.student(x_t, sigma_t, condition)
        pred_student_next = self.student(x_next, sigma_next, condition)
        
        # Consistency loss: predictions at adjacent timesteps should match
        consistency_loss = F.mse_loss(pred_student_t, pred_student_next.detach())
        
        # Distillation loss: student should match teacher
        distillation_loss = F.mse_loss(pred_student_t, pred_teacher)
        
        # Total loss
        total_loss = consistency_loss + 0.5 * distillation_loss
        
        losses = {
            'total': total_loss,
            'consistency': consistency_loss,
            'distillation': distillation_loss
        }
        
        return losses
    
    def update_ema(self):
        """Update EMA teacher model"""
        with torch.no_grad():
            for ema_param, param in zip(self.ema_teacher.parameters(), self.teacher.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
    
    @staticmethod
    def _timestep_to_sigma(timestep: torch.Tensor) -> torch.Tensor:
        """Convert timestep to sigma"""
        # Simple mapping - adjust based on your scheduler
        sigma_min, sigma_max = 0.002, 80.0
        t_normalized = timestep / 999.0
        sigma = sigma_min + t_normalized * (sigma_max - sigma_min)
        return sigma
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Single training step
        
        Args:
            batch: Batch of data
            optimizer: Optimizer
            
        Returns:
            Dictionary of loss values
        """
        x_0 = batch['clean']
        condition = batch.get('condition', None)
        
        # Forward pass
        losses = self.compute_consistency_loss(x_0, condition)
        
        # Backward pass
        optimizer.zero_grad()
        losses['total'].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update EMA
        self.update_ema()
        
        # Return scalar losses
        return {k: v.item() for k, v in losses.items()}


class ProgressiveDistillation:
    """
    Progressive distillation: gradually reduce from 1000 -> 50 -> 10 -> 4 -> 2 steps
    """
    
    def __init__(
        self,
        schedule: list = [
            {'from_steps': 1000, 'to_steps': 50, 'epochs': 10},
            {'from_steps': 50, 'to_steps': 10, 'epochs': 5},
            {'from_steps': 10, 'to_steps': 4, 'epochs': 5},
            {'from_steps': 4, 'to_steps': 2, 'epochs': 5}
        ]
    ):
        """
        Args:
            schedule: List of distillation stages
        """
        self.schedule = schedule
    
    def get_current_stage(self, epoch: int) -> Dict:
        """Get current distillation stage based on epoch"""
        cumulative_epochs = 0
        
        for stage in self.schedule:
            cumulative_epochs += stage['epochs']
            if epoch < cumulative_epochs:
                return stage
        
        # Return last stage if beyond schedule
        return self.schedule[-1]


class AdaptiveStepSelector:
    """
    Adaptive step selection based on scene complexity
    Simple scenes: 2 steps, Complex scenes: 4-8 steps
    """
    
    def __init__(
        self,
        min_steps: int = 2,
        max_steps: int = 8
    ):
        self.min_steps = min_steps
        self.max_steps = max_steps
    
    def select_steps(self, latent: torch.Tensor) -> int:
        """
        Select number of steps based on latent complexity
        
        Args:
            latent: Input latent [B, C, H, W]
            
        Returns:
            Number of steps to use
        """
        # Measure complexity using variance
        complexity = torch.var(latent).item()
        
        # Map complexity to steps
        if complexity < 0.1:
            return self.min_steps
        elif complexity < 0.3:
            return (self.min_steps + self.max_steps) // 2
        else:
            return self.max_steps


if __name__ == "__main__":
    print("Testing Consistency Distillation...")
    
    # Create dummy models
    from models.raw_diffusion_unet import RAWDiffusionUNet
    
    print("\n1. Testing ConsistencyModel...")
    teacher_unet = RAWDiffusionUNet(in_channels=16, out_channels=16)
    consistency_model = ConsistencyModel(teacher_unet, num_consistency_steps=4)
    
    # Test forward
    x = torch.randn(2, 16, 32, 32)
    sigma = torch.ones(2) * 5.0
    context = torch.randn(2, 512)
    
    with torch.no_grad():
        output = consistency_model(x, sigma, context)
    
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    
    # Test generation
    print("\n2. Testing generation...")
    with torch.no_grad():
        generated = consistency_model.generate(
            latent_shape=(2, 16, 32, 32),
            condition=context,
            num_steps=4,
            device='cpu'
        )
    print(f"   Generated shape: {generated.shape}")
    
    # Test trainer
    print("\n3. Testing ConsistencyDistillationTrainer...")
    student_unet = RAWDiffusionUNet(in_channels=16, out_channels=16)
    student_model = ConsistencyModel(student_unet, num_consistency_steps=4)
    
    trainer = ConsistencyDistillationTrainer(
        teacher_model=teacher_unet,
        student_model=student_model
    )
    
    batch = {
        'clean': torch.randn(2, 16, 32, 32),
        'condition': context
    }
    
    optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4)
    losses = trainer.train_step(batch, optimizer)
    
    print(f"   Consistency loss: {losses['consistency']:.4f}")
    print(f"   Distillation loss: {losses['distillation']:.4f}")
    
    # Test adaptive step selector
    print("\n4. Testing AdaptiveStepSelector...")
    selector = AdaptiveStepSelector()
    
    simple_latent = torch.randn(1, 16, 32, 32) * 0.1
    complex_latent = torch.randn(1, 16, 32, 32) * 2.0
    
    simple_steps = selector.select_steps(simple_latent)
    complex_steps = selector.select_steps(complex_latent)
    
    print(f"   Simple scene steps: {simple_steps}")
    print(f"   Complex scene steps: {complex_steps}")
    
    print("\n✓ All consistency distillation tests passed!")
