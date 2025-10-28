# Installation Guide# 📦 Installation Guide



## Quick Install## Quick Install (Minimal)



```bashFor basic functionality (training and inference):

pip install -r requirements-full.txt

``````bash

pip install -r requirements-minimal.txt

## Platform-Specific```



### Linux**Includes:**

```bash- PyTorch & torchvision

sudo apt-get update- Diffusion models (diffusers)

sudo apt-get install -y python3-dev build-essential- Image processing (OpenCV, Pillow, RawPy)

pip install -r requirements-full.txt- FastAPI (for deployment)

```- Basic utilities



### Windows---

```powershell

python -m venv .venv## Full Install (All Features)

.\.venv\Scripts\Activate.ps1

pip install -r requirements-full.txtFor complete functionality including all optional features:

```

```bash

### macOSpip install -r requirements-full.txt

```bash```

brew install python@3.10

pip install -r requirements-full.txt**Includes everything above, plus:**

```- Web interface (Gradio)

- Visualization tools (Matplotlib, Seaborn, Plotly)

## GPU Support- Testing framework (pytest)

- Code quality tools (Black, Flake8, MyPy)

### NVIDIA CUDA- Security scanning (Bandit, Safety)

```bash- Jupyter notebooks

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121- Model export (ONNX, TensorFlow Lite, CoreML)

```- Advanced metrics (LPIPS, PIQ)

- Experiment tracking (WandB, TensorBoard)

### Verify GPU

```bash---

python -c "import torch; print(torch.cuda.is_available())"

```## Platform-Specific Installation



## Minimal Installation### Linux (Ubuntu/Debian)



For core functionality only:```bash

```bash# Install system dependencies

pip install -r requirements-minimal.txtsudo apt-get update

```sudo apt-get install -y \

    python3-dev \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0

# Install Python packages
pip install -r requirements-full.txt
```

### macOS

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install system dependencies
brew install python@3.10

# Install Python packages
pip install -r requirements-full.txt
```

### Windows

```powershell
# Requires Visual Studio C++ Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Install Python packages
pip install -r requirements-full.txt
```

---

## GPU Support

### CUDA (NVIDIA GPUs)

```bash
# Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Then install other requirements
pip install -r requirements-full.txt
```

### ROCm (AMD GPUs)

```bash
# Install PyTorch with ROCm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6

# Then install other requirements
pip install -r requirements-full.txt
```

### MPS (Apple Silicon)

```bash
# PyTorch supports MPS out of the box on macOS
pip install -r requirements-full.txt
```

---

## Optional Components

### Web Interface Only

```bash
pip install gradio>=4.0.0
```

### Testing Only

```bash
pip install pytest pytest-cov pytest-xdist
```

### Visualization Only

```bash
pip install matplotlib seaborn plotly
```

### Model Export Only

```bash
pip install onnx onnxruntime coremltools
```

### Code Quality Tools Only

```bash
pip install black flake8 isort mypy pylint
```

---

## Virtual Environment Setup

### Using venv (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install packages
pip install -r requirements-full.txt
```

### Using conda

```bash
# Create conda environment
conda create -n raw-diffusion python=3.10 -y

# Activate
conda activate raw-diffusion

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install other requirements
pip install -r requirements-full.txt
```

---

## Verification

After installation, verify everything works:

```python
import torch
import gradio
import cv2
import rawpy

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Gradio: {gradio.__version__}")
print("✓ All packages installed successfully!")
```

---

## Troubleshooting

### Issue: RawPy installation fails

**Solution:**
```bash
# Install dependencies first
sudo apt-get install libraw-dev  # Linux
brew install libraw  # macOS

pip install rawpy
```

### Issue: OpenCV import error

**Solution:**
```bash
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python-headless
```

### Issue: CUDA out of memory

**Solution:**
- Reduce batch size in config
- Use gradient accumulation
- Enable mixed precision training

### Issue: Import errors after installation

**Solution:**
```bash
# Upgrade pip
pip install --upgrade pip setuptools wheel

# Reinstall requirements
pip install -r requirements-full.txt --force-reinstall
```

---

## Development Setup

For contributing to the project:

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/raw-diffusion.git
cd raw-diffusion

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in editable mode
pip install -e .

# Install development dependencies
pip install -r requirements-full.txt

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

---

## Docker Installation

If you prefer Docker:

```bash
# Build image
docker build -t raw-diffusion:latest .

# Run container
docker run -p 8000:8000 --gpus all raw-diffusion:latest
```

---

## Minimal Install Sizes

| Configuration | Size | Use Case |
|---------------|------|----------|
| requirements-minimal.txt | ~3GB | Basic training/inference |
| requirements-full.txt | ~8GB | All features |
| With CUDA | +2GB | GPU support |
| With all optional | +10GB | Complete development |

---

## Next Steps

After installation:

1. Verify installation: `python -c "import torch; print(torch.__version__)"`
2. Run tests: `pytest tests/`
3. Try quick start: `python quick_train.py --demo`
4. Launch web UI: `python web/gradio_interface.py`

---

## Support

If you encounter issues:
1. Check [Troubleshooting](#troubleshooting) section
2. See GitHub Issues
3. Join Discord/Slack community

---

**Recommended**: Start with `requirements-full.txt` for the best experience!
