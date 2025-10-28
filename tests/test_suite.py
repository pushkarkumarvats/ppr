"""
Comprehensive Test Suite for RAW Image Enhancement

This module contains unit tests, integration tests, and quality tests
for all components of the RAW enhancement pipeline.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.raw_diffusion_unet import RAWVAE, RAWDiffusionUNet, DDPMScheduler
from models.optical_flow import RAWOpticalFlow, AlignmentModule
from models.consistency_distillation import ConsistencyModel
from models.lens_aberration_module import AberrationCorrectionModule
from training.losses import (
    RAWPerceptualLoss, HallucinationPenaltyLoss,
    TemporalConsistencyLoss, EdgePreservationLoss
)
from training.metrics import psnr, ssim, LPIPS
from data.raw_loader import demux_bayer
from data.preprocessing import demosaic_bilinear, apply_white_balance


class TestRAWVAE(unittest.TestCase):
    """Test RAW VAE model."""
    
    def setUp(self):
        self.vae = RAWVAE(
            in_channels=4,
            latent_channels=16,
            channels=64,
            num_res_blocks=2
        )
        self.batch_size = 2
        self.img_size = 128
    
    def test_forward_pass(self):
        """Test forward pass through VAE."""
        x = torch.randn(self.batch_size, 4, self.img_size, self.img_size)
        
        # Encode
        posterior = self.vae.encode(x)
        z = posterior.sample()
        
        # Check latent shape (should be 8x compressed)
        expected_h = self.img_size // 8
        expected_w = self.img_size // 8
        self.assertEqual(z.shape, (self.batch_size, 16, expected_h, expected_w))
        
        # Decode
        recon = self.vae.decode(z)
        self.assertEqual(recon.shape, x.shape)
    
    def test_kl_divergence(self):
        """Test KL divergence computation."""
        x = torch.randn(self.batch_size, 4, self.img_size, self.img_size)
        
        posterior = self.vae.encode(x)
        kl = posterior.kl()
        
        self.assertEqual(kl.shape, (self.batch_size,))
        self.assertTrue(torch.all(kl >= 0))  # KL should be non-negative
    
    def test_reconstruction_quality(self):
        """Test reconstruction preserves information."""
        x = torch.randn(self.batch_size, 4, self.img_size, self.img_size)
        
        # Full encode-decode
        posterior = self.vae.encode(x)
        z = posterior.sample()
        recon = self.vae.decode(z)
        
        # Reconstruction error should be reasonable
        mse = nn.functional.mse_loss(recon, x)
        self.assertLess(mse.item(), 0.1)


class TestOpticalFlow(unittest.TestCase):
    """Test optical flow model."""
    
    def setUp(self):
        self.flow_net = RAWOpticalFlow(
            in_channels=4,
            feature_dim=128,
            num_levels=4
        )
        self.batch_size = 2
        self.img_size = 256
    
    def test_flow_estimation(self):
        """Test flow estimation between two frames."""
        img1 = torch.randn(self.batch_size, 4, self.img_size, self.img_size)
        img2 = torch.randn(self.batch_size, 4, self.img_size, self.img_size)
        
        flow = self.flow_net(img1, img2)
        
        # Flow should be 2-channel (dx, dy)
        self.assertEqual(flow.shape, (self.batch_size, 2, self.img_size, self.img_size))
    
    def test_identity_flow(self):
        """Test that identical images produce near-zero flow."""
        img = torch.randn(self.batch_size, 4, self.img_size, self.img_size)
        
        flow = self.flow_net(img, img)
        
        # Flow should be close to zero
        self.assertLess(flow.abs().mean().item(), 5.0)
    
    def test_alignment_module(self):
        """Test burst alignment."""
        alignment = AlignmentModule(self.flow_net)
        
        burst = torch.randn(self.batch_size, 8, 4, self.img_size, self.img_size)
        aligned = alignment(burst, reference_idx=0)
        
        self.assertEqual(aligned.shape, burst.shape)


class TestDiffusionModel(unittest.TestCase):
    """Test diffusion models."""
    
    def setUp(self):
        self.unet = RAWDiffusionUNet(
            in_channels=16,
            model_channels=128,
            out_channels=16,
            num_res_blocks=2,
            attention_resolutions=[8, 16]
        )
        self.scheduler = DDPMScheduler(num_train_timesteps=1000)
        self.batch_size = 2
        self.latent_size = 32
    
    def test_noise_prediction(self):
        """Test UNet noise prediction."""
        x = torch.randn(self.batch_size, 16, self.latent_size, self.latent_size)
        t = torch.randint(0, 1000, (self.batch_size,))
        condition = torch.randn(self.batch_size, 16, self.latent_size, self.latent_size)
        
        noise_pred = self.unet(x, t, condition)
        
        self.assertEqual(noise_pred.shape, x.shape)
    
    def test_noise_schedule(self):
        """Test noise scheduler."""
        x0 = torch.randn(self.batch_size, 16, self.latent_size, self.latent_size)
        noise = torch.randn_like(x0)
        t = torch.randint(0, 1000, (self.batch_size,))
        
        # Add noise
        xt = self.scheduler.add_noise(x0, noise, t)
        
        self.assertEqual(xt.shape, x0.shape)
        
        # At t=0, xt should be close to x0
        t_zero = torch.zeros(self.batch_size, dtype=torch.long)
        xt_zero = self.scheduler.add_noise(x0, noise, t_zero)
        self.assertTrue(torch.allclose(xt_zero, x0, atol=0.1))


class TestConsistencyModel(unittest.TestCase):
    """Test consistency model."""
    
    def setUp(self):
        self.model = ConsistencyModel(
            in_channels=16,
            model_channels=128,
            num_res_blocks=2,
            attention_resolutions=[8, 16]
        )
        self.batch_size = 2
        self.latent_size = 32
    
    def test_generation(self):
        """Test multi-step generation."""
        condition = torch.randn(self.batch_size, 16, self.latent_size, self.latent_size)
        
        # Test with different step counts
        for num_steps in [2, 4]:
            output = self.model.generate(condition, num_steps=num_steps)
            self.assertEqual(output.shape, condition.shape)
    
    def test_consistency_property(self):
        """Test that model exhibits consistency property."""
        condition = torch.randn(self.batch_size, 16, self.latent_size, self.latent_size)
        
        # Generate with 2 and 4 steps
        output_2 = self.model.generate(condition, num_steps=2)
        output_4 = self.model.generate(condition, num_steps=4)
        
        # Outputs should be similar (consistency property)
        diff = (output_2 - output_4).abs().mean()
        self.assertLess(diff.item(), 0.5)


class TestLossFunctions(unittest.TestCase):
    """Test loss functions."""
    
    def setUp(self):
        self.batch_size = 2
        self.img_size = 256
    
    def test_perceptual_loss(self):
        """Test RAW perceptual loss."""
        loss_fn = RAWPerceptualLoss()
        
        pred = torch.randn(self.batch_size, 4, self.img_size, self.img_size)
        target = torch.randn(self.batch_size, 4, self.img_size, self.img_size)
        
        loss = loss_fn(pred, target)
        
        self.assertIsInstance(loss.item(), float)
        self.assertGreater(loss.item(), 0)
    
    def test_hallucination_loss(self):
        """Test hallucination penalty loss."""
        loss_fn = HallucinationPenaltyLoss()
        
        pred = torch.randn(self.batch_size, 4, self.img_size, self.img_size)
        burst = torch.randn(self.batch_size, 8, 4, self.img_size, self.img_size)
        
        loss = loss_fn(pred, burst)
        
        self.assertIsInstance(loss.item(), float)
        self.assertGreater(loss.item(), 0)
    
    def test_temporal_consistency_loss(self):
        """Test temporal consistency loss."""
        loss_fn = TemporalConsistencyLoss()
        
        pred_t0 = torch.randn(self.batch_size, 4, self.img_size, self.img_size)
        pred_t1 = torch.randn(self.batch_size, 4, self.img_size, self.img_size)
        flow = torch.randn(self.batch_size, 2, self.img_size, self.img_size)
        
        loss = loss_fn(pred_t0, pred_t1, flow)
        
        self.assertIsInstance(loss.item(), float)
        self.assertGreater(loss.item(), 0)
    
    def test_edge_preservation_loss(self):
        """Test edge preservation loss."""
        loss_fn = EdgePreservationLoss()
        
        pred = torch.randn(self.batch_size, 4, self.img_size, self.img_size)
        target = torch.randn(self.batch_size, 4, self.img_size, self.img_size)
        
        loss = loss_fn(pred, target)
        
        self.assertIsInstance(loss.item(), float)
        self.assertGreater(loss.item(), 0)


class TestMetrics(unittest.TestCase):
    """Test quality metrics."""
    
    def setUp(self):
        self.batch_size = 2
        self.img_size = 256
    
    def test_psnr(self):
        """Test PSNR metric."""
        pred = torch.randn(self.batch_size, 4, self.img_size, self.img_size)
        target = pred + 0.01 * torch.randn_like(pred)
        
        psnr_val = psnr(pred, target)
        
        self.assertGreater(psnr_val.item(), 0)
        self.assertLess(psnr_val.item(), 100)
    
    def test_ssim(self):
        """Test SSIM metric."""
        pred = torch.randn(self.batch_size, 4, self.img_size, self.img_size)
        target = pred + 0.01 * torch.randn_like(pred)
        
        ssim_val = ssim(pred, target)
        
        self.assertGreater(ssim_val.item(), 0)
        self.assertLessEqual(ssim_val.item(), 1)
    
    def test_lpips(self):
        """Test LPIPS metric."""
        lpips_metric = LPIPS()
        
        pred = torch.randn(self.batch_size, 4, self.img_size, self.img_size)
        target = torch.randn(self.batch_size, 4, self.img_size, self.img_size)
        
        lpips_val = lpips_metric(pred, target)
        
        self.assertIsInstance(lpips_val.item(), float)
        self.assertGreater(lpips_val.item(), 0)


class TestDataProcessing(unittest.TestCase):
    """Test data processing functions."""
    
    def test_bayer_demux(self):
        """Test Bayer pattern demuxing."""
        # Create synthetic Bayer pattern
        bayer = np.random.rand(512, 512).astype(np.float32)
        
        demuxed = demux_bayer(bayer, pattern='RGGB')
        
        self.assertEqual(demuxed.shape, (4, 256, 256))
    
    def test_demosaic(self):
        """Test demosaicing."""
        bayer_channels = np.random.rand(4, 256, 256).astype(np.float32)
        
        rgb = demosaic_bilinear(bayer_channels)
        
        self.assertEqual(rgb.shape, (3, 512, 512))
    
    def test_white_balance(self):
        """Test white balance application."""
        raw = np.random.rand(4, 256, 256).astype(np.float32)
        wb_gains = np.array([2.0, 1.0, 1.0, 1.5])
        
        balanced = apply_white_balance(raw, wb_gains)
        
        self.assertEqual(balanced.shape, raw.shape)


class IntegrationTests(unittest.TestCase):
    """Integration tests for complete pipeline."""
    
    def test_end_to_end_inference(self):
        """Test complete inference pipeline."""
        # Create small models for testing
        vae = RAWVAE(in_channels=4, latent_channels=16, channels=32, num_res_blocks=1)
        flow_net = RAWOpticalFlow(in_channels=4, feature_dim=64, num_levels=3)
        alignment = AlignmentModule(flow_net)
        consistency = ConsistencyModel(in_channels=16, model_channels=64, num_res_blocks=1)
        
        # Create dummy burst
        burst = torch.randn(1, 8, 4, 128, 128)
        
        # Stage 1: Alignment
        aligned = alignment(burst, reference_idx=0)
        self.assertEqual(aligned.shape, burst.shape)
        
        # Stage 2: Merge
        merged = aligned.mean(dim=1)
        self.assertEqual(merged.shape, (1, 4, 128, 128))
        
        # Stage 3: VAE encode
        latent = vae.encode(merged).sample()
        self.assertEqual(latent.shape[1], 16)
        
        # Stage 4: Consistency model
        enhanced_latent = consistency.generate(latent, num_steps=2)
        self.assertEqual(enhanced_latent.shape, latent.shape)
        
        # Stage 5: VAE decode
        enhanced = vae.decode(enhanced_latent)
        self.assertEqual(enhanced.shape, merged.shape)
    
    def test_memory_efficiency(self):
        """Test that pipeline doesn't leak memory."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        device = 'cuda'
        vae = RAWVAE(in_channels=4, latent_channels=16, channels=32, num_res_blocks=1).to(device)
        
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated()
        
        # Run multiple iterations
        for _ in range(10):
            x = torch.randn(1, 4, 128, 128, device=device)
            with torch.no_grad():
                z = vae.encode(x).sample()
                recon = vae.decode(z)
            del x, z, recon
        
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated()
        
        # Memory shouldn't grow significantly
        memory_growth = (final_memory - initial_memory) / (1024 ** 2)  # MB
        self.assertLess(memory_growth, 50)  # Less than 50MB growth


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestRAWVAE))
    suite.addTests(loader.loadTestsFromTestCase(TestOpticalFlow))
    suite.addTests(loader.loadTestsFromTestCase(TestDiffusionModel))
    suite.addTests(loader.loadTestsFromTestCase(TestConsistencyModel))
    suite.addTests(loader.loadTestsFromTestCase(TestLossFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestDataProcessing))
    suite.addTests(loader.loadTestsFromTestCase(IntegrationTests))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
