# Quick Start Guide# 🚀 Quick Start Guide



## Installation## Real-Time Multi-Frame RAW Fusion - Quick Reference



```bashThis is your quick reference for getting started with the RAW enhancement system.

git clone https://github.com/yourusername/raw-diffusion.git

cd raw-diffusion---



conda create -n raw-diffusion python=3.10## 📦 Installation (5 minutes)

conda activate raw-diffusion

pip install -r requirements.txt```bash

```# Clone repository

git clone https://github.com/yourusername/raw-diffusion.git

## Trainingcd raw-diffusion



```bash# Create environment

python training/train_diffusion.py \conda create -n raw-diffusion python=3.10

    --config configs/training_config.yaml \conda activate raw-diffusion

    --stage all \

    --data_dir /path/to/proraw/data \# Install dependencies

    --output_dir ./outputspip install -r requirements.txt

``````



## Inference---



```bash## 🎯 Quick Commands

python inference/realtime_pipeline.py \

    --checkpoint ./outputs/consistency_final.pt \### Training (Full Pipeline)

    --input_burst ./data/sample/*.dng \```bash

    --output ./output/enhanced.dng \python training/train_diffusion.py \

    --num_steps 4    --config configs/training_config.yaml \

```    --stage all \

    --data_dir /path/to/proraw/data \

## Deployment    --output_dir ./outputs

```

### Docker

```bash### Training (Single Stage)

docker build -t raw-diffusion .```bash

docker run -p 8000:8000 --gpus all raw-diffusion# VAE only

```python training/train_diffusion.py --stage vae --data_dir /path/to/data



### API# Optical flow only

```bashpython training/train_diffusion.py --stage flow --data_dir /path/to/data

uvicorn api.serve:app --host 0.0.0.0 --port 8000

```# Diffusion only

python training/train_diffusion.py --stage diffusion --data_dir /path/to/data

### CoreML Export

```bash# Consistency only

python deployment/coreml_converter.py \python training/train_diffusion.py --stage consistency --data_dir /path/to/data

    --checkpoint ./outputs/consistency_final.pt \```

    --output_dir ./ios_models \

    --optimize_for_ane### Testing

``````bash

# Run all tests

## Testingpython tests/test_suite.py



```bash# Run specific test

pytest tests/ -v --cov=.python -m unittest tests.test_suite.TestRAWVAE

```

# With coverage

## Configurationpytest tests/ --cov=. --cov-report=html

```

Edit `configs/training_config.yaml` for training parameters.

Edit `configs/deployment_config.yaml` for deployment settings.### Benchmarking

```bash
python scripts/benchmark.py \
    --checkpoint_dir ./outputs \
    --config configs/model_config.yaml
```

### Deployment
```bash
python scripts/final_integration.py \
    --checkpoint_dir ./outputs \
    --config configs/deployment_config.yaml \
    --output_dir ./deployment
```

---

## 💻 Python API Quick Reference

### Inference Pipeline

```python
import torch
from inference.realtime_pipeline import RealTimePipeline
from models.raw_diffusion_unet import RAWVAE
from models.consistency_distillation import ConsistencyModel
from models.optical_flow import RAWOpticalFlow, AlignmentModule

# Load models
vae = RAWVAE(**config['vae'])
vae.load_state_dict(torch.load('outputs/vae_final.pt')['model_state_dict'])

flow_net = RAWOpticalFlow(**config['flow'])
flow_net.load_state_dict(torch.load('outputs/flow_final.pt')['model_state_dict'])
alignment = AlignmentModule(flow_net)

consistency = ConsistencyModel(**config['consistency'])
consistency.load_state_dict(torch.load('outputs/consistency_final.pt')['model_state_dict'])

# Create pipeline
pipeline = RealTimePipeline(
    vae=vae,
    consistency_model=consistency,
    alignment=alignment,
    device='cuda',
    num_inference_steps=2
)

# Process burst
burst = torch.randn(1, 8, 4, 1024, 1024)  # Your RAW burst
results = pipeline.forward(burst)
enhanced = results['enhanced']
```

### Training Custom Model

```python
from training.train_diffusion import DiffusionTrainer
from models.raw_diffusion_unet import RAWVAE, RAWDiffusionUNet

# Initialize models
vae = RAWVAE(**config['vae'])
unet = RAWDiffusionUNet(**config['unet'])

# Create trainer
trainer = DiffusionTrainer(
    vae=vae,
    unet=unet,
    config=config,
    device='cuda'
)

# Train
for epoch in range(num_epochs):
    metrics = trainer.train_epoch(dataloader, epoch)
    print(f"Epoch {epoch}: {metrics}")
```

### Data Loading

```python
from data.raw_loader import BurstRawDataset
from data.augmentation import RawAugmentation

# Create dataset
dataset = BurstRawDataset(
    data_dir='/path/to/proraw',
    burst_size=8,
    augmentation=RawAugmentation()
)

# Load burst
burst_data = dataset[0]
burst = burst_data['burst']  # [T, 4, H, W]
metadata = burst_data['metadata']
```

### Metrics Computation

```python
from training.metrics import MetricCalculator

calculator = MetricCalculator(device='cuda')

metrics = calculator.compute_all_metrics(
    pred=enhanced,
    target=ground_truth,
    burst=input_burst
)

print(f"PSNR: {metrics['psnr']:.2f}")
print(f"SSIM: {metrics['ssim']:.4f}")
print(f"LPIPS: {metrics['lpips']:.4f}")
```

---

## 📁 Key File Locations

### Models
- `models/raw_diffusion_unet.py` - VAE + Diffusion
- `models/optical_flow.py` - Optical flow
- `models/consistency_distillation.py` - Fast inference
- `models/lens_aberration_module.py` - Aberration correction

### Training
- `training/train_diffusion.py` - Main training script
- `training/losses.py` - Loss functions
- `training/metrics.py` - Quality metrics

### Inference
- `inference/realtime_pipeline.py` - Real-time pipeline
- `inference/optimization.py` - Optimizations

### Deployment
- `deployment/coreml_converter.py` - CoreML conversion
- `deployment/iOS_Integration_Guide.md` - iOS guide

### Testing
- `tests/test_suite.py` - All tests

### Configuration
- `configs/model_config.yaml` - Model config
- `configs/training_config.yaml` - Training config
- `configs/deployment_config.yaml` - Deployment config

---

## 🔧 Common Tasks

### Change Model Size

Edit `configs/model_config.yaml`:
```yaml
model:
  vae:
    channels: 64  # Change to 32 for smaller, 128 for larger
    num_res_blocks: 2  # Increase for more capacity
```

### Adjust Training

Edit `configs/training_config.yaml`:
```yaml
training:
  batch_size: 4  # Adjust based on GPU memory
  learning_rate: 1e-4  # Learning rate
  mixed_precision: true  # Enable FP16
```

### Change Inference Steps

```python
# In code
pipeline = RealTimePipeline(..., num_inference_steps=2)  # 2-4 steps

# With adaptive selection
pipeline = RealTimePipeline(..., use_adaptive_steps=True)
```

### Export to Mobile

```bash
python deployment/coreml_converter.py \
    --checkpoint outputs/consistency_final.pt \
    --output_dir ./coreml_models
```

---

## 🐛 Troubleshooting

### Out of Memory
```python
# Reduce batch size
config['training']['batch_size'] = 2

# Enable gradient checkpointing
config['training']['gradient_checkpointing'] = True

# Use smaller image size
config['data']['image_size'] = 512  # Instead of 1024
```

### Slow Training
```python
# Enable mixed precision
config['training']['mixed_precision'] = True

# Use multiple GPUs
# Run with: python -m torch.distributed.launch --nproc_per_node=4 training/train_diffusion.py

# Reduce number of workers
config['training']['num_workers'] = 4
```

### Poor Quality
```python
# Increase model capacity
config['model']['vae']['channels'] = 128

# Train longer
config['diffusion']['epochs'] = 100

# Adjust loss weights
config['training']['perceptual_weight'] = 2.0
```

---

## 📊 Expected Training Time

| Stage | Epochs | Time (A100) | GPU Memory |
|-------|--------|-------------|------------|
| VAE | 20 | ~8 hours | ~12GB |
| Flow | 15 | ~6 hours | ~10GB |
| Diffusion | 50 | ~48 hours | ~16GB |
| Consistency | 30 | ~12 hours | ~14GB |
| **Total** | **115** | **~3 days** | **~16GB peak** |

---

## 🎯 Performance Expectations

| Resolution | Burst Size | Latency | Memory |
|------------|-----------|---------|--------|
| 512x512 | 8 | 8-12ms | 0.4GB |
| 1024x1024 | 8 | 20-26ms | 1.2GB |
| 2048x2048 | 8 | 60-80ms | 4.5GB |

---

## 📱 iOS Quick Start

1. **Export Models**
```bash
python deployment/coreml_converter.py --checkpoint outputs/ --output coreml/
```

2. **Add to Xcode**
- Drag `.mlpackage` files to Xcode project
- Check "Copy items if needed"

3. **Use in Swift**
```swift
let pipeline = try RAWEnhancementPipeline()
let enhanced = try await pipeline.process(burst: rawImages)
```

See `deployment/iOS_Integration_Guide.md` for complete instructions.

---

## 🔗 Useful Links

- **Main README**: `README.md`
- **Complete Status**: `PROJECT_COMPLETE.md`
- **API Docs**: See docstrings in each module
- **Training Guide**: Section in README.md
- **iOS Integration**: `deployment/iOS_Integration_Guide.md`

---

## 💡 Tips

1. **Start Small**: Train on 512x512 images first
2. **Use Wandb**: Enable logging to track training
3. **Checkpoint Often**: Save every few epochs
4. **Test Early**: Run test suite before full training
5. **Profile First**: Use profiler to find bottlenecks
6. **Gradual Optimization**: Start with optimization_level=1

---

## 🆘 Getting Help

1. Check `README.md` for detailed documentation
2. Run `python tests/test_suite.py` to validate setup
3. Review `PROJECT_COMPLETE.md` for complete overview
4. Check docstrings in source code
5. Look at example scripts in `scripts/`

---

**Quick Start Complete!** 🎉

For detailed information, see the full documentation in `README.md`.
