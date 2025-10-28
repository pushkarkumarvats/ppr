# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2025-10-28

### Added
- Complete RAW image enhancement pipeline with multi-frame fusion
- RAW VAE encoder/decoder with 8x compression
- RAFT-based optical flow alignment for burst sequences
- Diffusion U-Net with attention mechanisms
- Consistency distillation for 2-4 step inference
- Lens aberration correction module for iPhone ProRAW
- Comprehensive training pipeline (VAE → Flow → Diffusion → Consistency)
- Real-time inference pipeline with adaptive step selection
- FastAPI REST API for model serving
- Gradio web interface for interactive demos
- Docker and Docker Compose support
- Render.com deployment configuration
- CoreML export for iOS/ANE deployment
- Multi-format model export (ONNX, TorchScript, TFLite, OpenVINO)
- Comprehensive evaluation suite with benchmarking
- Visualization toolkit for training and results
- CI/CD pipeline with GitHub Actions
- Complete documentation (README, installation, deployment guides)
- Example scripts and Jupyter notebooks
- Unit tests and integration tests
- Security policy and contributing guidelines

### Core Models
- `RAWVAE`: 4-channel Bayer RAW VAE with 16-channel latent space
- `RAWDiffusionUNet`: Conditional diffusion U-Net with cross-attention
- `RAWOpticalFlow`: Modified RAFT for RAW Bayer patterns
- `ConsistencyModel`: Fast inference model with 2-4 steps
- `AberrationCorrectionModule`: Spatially-varying aberration correction

### Training Features
- Multi-stage orchestrated training with automatic progression
- Distributed training support (DDP)
- Mixed precision training (FP16)
- Comprehensive loss functions (perceptual, hallucination penalty, temporal, edge)
- Advanced metrics (PSNR, SSIM, LPIPS, NIQE)
- Weights & Biases integration
- Checkpoint management with best model selection

### Deployment Features
- Production-ready FastAPI server
- Health checks and monitoring endpoints
- Render.com one-click deployment
- Docker containerization with GPU support
- CoreML conversion for iOS deployment
- Model quantization and optimization

### Documentation
- Comprehensive README with architecture diagrams
- Installation guide with platform-specific instructions
- Training and deployment documentation
- API documentation
- Example usage scripts
- Contributing guidelines
- Security policy

## [0.1.0] - 2024-10-28

### Added
- Initial project structure
- Basic configuration system
- Documentation framework

[Unreleased]: https://github.com/yourusername/raw-fusion-diffusion/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/yourusername/raw-fusion-diffusion/releases/tag/v1.0.0
[0.1.0]: https://github.com/yourusername/raw-fusion-diffusion/releases/tag/v0.1.0
