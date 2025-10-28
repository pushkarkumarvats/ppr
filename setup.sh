#!/bin/bash
# Linux/macOS setup script

set -e

echo "========================================"
echo "RAW Fusion Diffusion - Linux/macOS Setup"
echo "========================================"
echo ""

# Check Python version
echo "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "✓ $PYTHON_VERSION"
else
    echo "✗ Python not found. Please install Python 3.8+"
    exit 1
fi

# Check if running in virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo ""
    echo "⚠️  Not in a virtual environment. Creating one..."
    echo ""
    
    # Create virtual environment
    python3 -m venv .venv
    
    if [ -f ".venv/bin/activate" ]; then
        echo "Virtual environment created. Please activate it:"
        echo "  source .venv/bin/activate"
        echo ""
        echo "Then run this script again."
        exit 0
    else
        echo "✗ Failed to create virtual environment"
        exit 1
    fi
fi

# Upgrade pip
echo ""
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Ask user which requirements to install
echo ""
echo "Choose installation type:"
echo "  1) Minimal (core dependencies only)"
echo "  2) Full (all features including API, UI, etc.)"
echo "  3) Dev (full + development tools)"
echo ""
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo ""
        echo "Installing minimal requirements..."
        pip install -r requirements-minimal.txt
        ;;
    2)
        echo ""
        echo "Installing full requirements..."
        pip install -r requirements-full.txt
        ;;
    3)
        echo ""
        echo "Installing full requirements + dev tools..."
        pip install -r requirements-full.txt
        pip install -e ".[dev]"
        ;;
    *)
        echo "Invalid choice. Installing full requirements..."
        pip install -r requirements-full.txt
        ;;
esac

echo ""
echo "✓ Installation complete!"

# Create necessary directories
echo ""
echo "Creating project directories..."

directories=(
    "data/train"
    "data/val"
    "data/test"
    "outputs"
    "checkpoints"
    "logs"
)

for dir in "${directories[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "✓ Created $dir"
    fi
done

# Run validation
echo ""
echo "Running validation checks..."
echo ""
python validate_setup.py

# Print next steps
echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Prepare your data:"
echo "   - Place training RAW images in: ./data/train"
echo "   - Place validation RAW images in: ./data/val"
echo ""
echo "2. Start training:"
echo "   python quick_train.py"
echo ""
echo "3. Or run demo:"
echo "   python example.py --demo"
echo ""
echo "4. Start web UI:"
echo "   python web/gradio_interface.py"
echo ""
echo "5. Start API server:"
echo "   python quick_deploy.py"
echo ""
echo "For more information, see:"
echo "  README.md"
echo "  INSTALLATION.md"
echo "  QUICKSTART.md"
echo ""
