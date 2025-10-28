#!/usr/bin/env python
"""Project Setup Validation"""

import sys
from pathlib import Path


def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} - Need 3.8+")
        return False


def check_imports():
    """Check if critical imports work."""
    print("\nChecking critical imports...")
    
    imports = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'PIL': 'Pillow',
        'yaml': 'PyYAML',
    }
    
    results = {}
    for module, name in imports.items():
        try:
            __import__(module)
            print(f"✓ {name}")
            results[module] = True
        except ImportError:
            print(f"✗ {name} - not installed")
            results[module] = False
    
    return all(results.values())


def check_optional_imports():
    """Check optional dependencies."""
    print("\nChecking optional imports...")
    
    optional = {
        'fastapi': 'FastAPI (for API server)',
        'gradio': 'Gradio (for web UI)',
        'wandb': 'Weights & Biases (for logging)',
        'lpips': 'LPIPS (for metrics)',
    }
    
    for module, description in optional.items():
        try:
            __import__(module)
            print(f"✓ {description}")
        except ImportError:
            print(f"○ {description} - not installed (optional)")


def check_cuda():
    """Check CUDA availability."""
    print("\nChecking CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
            return True
        else:
            print("○ CUDA not available - will use CPU")
            return False
    except ImportError:
        print("○ PyTorch not installed - cannot check CUDA")
        return False


def check_project_structure():
    """Check if project structure is intact."""
    print("\nChecking project structure...")
    
    required_dirs = [
        'models',
        'training',
        'inference',
        'data',
        'configs',
        'api',
        'tests',
    ]
    
    all_exist = True
    for dirname in required_dirs:
        path = Path(dirname)
        if path.exists():
            print(f"✓ {dirname}/")
        else:
            print(f"✗ {dirname}/ - missing")
            all_exist = False
    
    return all_exist


def check_configs():
    """Check if config files exist."""
    print("\nChecking configuration files...")
    
    config_files = [
        'configs/training_config.yaml',
        'configs/model_config.yaml',
        'configs/deployment_config.yaml',
    ]
    
    all_exist = True
    for config_file in config_files:
        path = Path(config_file)
        if path.exists():
            print(f"✓ {config_file}")
        else:
            print(f"✗ {config_file} - missing")
            all_exist = False
    
    return all_exist


def run_smoke_test():
    """Run a quick smoke test."""
    print("\nRunning smoke test...")
    
    try:
        import torch
        import numpy as np
        
        # Test basic tensor operations
        x = torch.randn(2, 4, 32, 32)
        y = x * 2 + 1
        assert y.shape == x.shape
        print("✓ Tensor operations work")
        
        # Test model imports
        from models.raw_diffusion_unet import RAWVAE
        vae = RAWVAE(in_channels=4, latent_channels=16, channels=32, num_res_blocks=1)
        with torch.no_grad():
            z = vae.encode(x).sample()
            recon = vae.decode(z)
        assert recon.shape == x.shape
        print("✓ Model instantiation works")
        
        return True
        
    except Exception as e:
        print(f"✗ Smoke test failed: {e}")
        return False


def main():
    """Main validation function."""
    print("="*70)
    print("RAW Fusion Diffusion - Project Validation")
    print("="*70)
    
    checks = {
        'Python Version': check_python_version(),
        'Critical Imports': check_imports(),
        'Project Structure': check_project_structure(),
        'Config Files': check_configs(),
    }
    
    check_optional_imports()
    check_cuda()
    
    # Run smoke test if basics are OK
    if all(checks.values()):
        checks['Smoke Test'] = run_smoke_test()
    
    # Summary
    print("\n" + "="*70)
    print("Validation Summary")
    print("="*70)
    
    for check_name, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{check_name:20s} {status}")
    
    print("="*70)
    
    if all(checks.values()):
        print("\n🎉 All checks passed! Project is ready to use.")
        print("\nNext steps:")
        print("  1. Prepare your data: Place RAW images in ./data/train")
        print("  2. Start training: python quick_train.py")
        print("  3. Or run demo: python example.py --demo")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please install missing dependencies.")
        print("\nTo install:")
        print("  pip install -r requirements-full.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
