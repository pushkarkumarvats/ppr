#!/bin/bash

# Production Deployment Script
# This script automates the deployment process

set -e  # Exit on error

echo "=========================================="
echo "RAW Diffusion Deployment Script"
echo "=========================================="

# Configuration
DEPLOY_ENV=${DEPLOY_ENV:-production}
DEPLOY_PLATFORM=${DEPLOY_PLATFORM:-render}
MODEL_DIR=${MODEL_DIR:-./models}
DATA_DIR=${DATA_DIR:-./data}

echo ""
echo "Configuration:"
echo "  Environment: $DEPLOY_ENV"
echo "  Platform: $DEPLOY_PLATFORM"
echo "  Model Directory: $MODEL_DIR"
echo ""

# Function to check prerequisites
check_prerequisites() {
    echo "Checking prerequisites..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo "❌ Python 3 not found"
        exit 1
    fi
    echo "✓ Python 3 found"
    
    # Check Git
    if ! command -v git &> /dev/null; then
        echo "❌ Git not found"
        exit 1
    fi
    echo "✓ Git found"
    
    # Check Docker (optional)
    if command -v docker &> /dev/null; then
        echo "✓ Docker found"
    else
        echo "⚠️  Docker not found (optional)"
    fi
    
    echo ""
}

# Function to prepare models
prepare_models() {
    echo "Preparing models..."
    
    if [ ! -d "$MODEL_DIR" ]; then
        echo "Creating model directory..."
        mkdir -p "$MODEL_DIR"
    fi
    
    # Check if models exist
    REQUIRED_MODELS=("vae_final.pt" "flow_final.pt" "diffusion_final.pt")
    MISSING_MODELS=0
    
    for model in "${REQUIRED_MODELS[@]}"; do
        if [ ! -f "$MODEL_DIR/$model" ]; then
            echo "⚠️  Missing: $model"
            MISSING_MODELS=$((MISSING_MODELS + 1))
        else
            echo "✓ Found: $model"
        fi
    done
    
    if [ $MISSING_MODELS -gt 0 ]; then
        echo ""
        echo "⚠️  $MISSING_MODELS model(s) missing"
        echo "Please train models first or download pre-trained weights"
        echo ""
        read -p "Continue without all models? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    echo ""
}

# Function to test locally
test_local() {
    echo "Testing locally..."
    
    # Install dependencies
    echo "Installing dependencies..."
    pip install -r requirements.txt -q
    
    # Run health check
    echo "Starting API server (background)..."
    python api/serve.py &
    SERVER_PID=$!
    
    # Wait for server to start
    echo "Waiting for server to start..."
    sleep 10
    
    # Test health endpoint
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        echo "✓ Health check passed"
    else
        echo "❌ Health check failed"
        kill $SERVER_PID 2>/dev/null || true
        exit 1
    fi
    
    # Stop server
    kill $SERVER_PID 2>/dev/null || true
    echo ""
}

# Function to deploy to Render
deploy_render() {
    echo "Deploying to Render.com..."
    
    # Check if render.yaml exists
    if [ ! -f "render.yaml" ]; then
        echo "❌ render.yaml not found"
        exit 1
    fi
    
    echo "✓ render.yaml found"
    
    # Check if git repo
    if [ ! -d ".git" ]; then
        echo "Initializing git repository..."
        git init
        git add .
        git commit -m "Initial deployment"
    fi
    
    echo ""
    echo "Next steps for Render deployment:"
    echo "1. Push this repository to GitHub"
    echo "2. Go to https://dashboard.render.com"
    echo "3. Click 'New +' -> 'Blueprint'"
    echo "4. Connect your GitHub repository"
    echo "5. Render will auto-detect render.yaml"
    echo ""
    echo "Or use Render CLI:"
    echo "  render deploy"
    echo ""
}

# Function to build Docker image
build_docker() {
    echo "Building Docker image..."
    
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker not installed"
        exit 1
    fi
    
    IMAGE_NAME="raw-diffusion:latest"
    
    echo "Building $IMAGE_NAME..."
    docker build -t $IMAGE_NAME .
    
    echo "✓ Docker image built successfully"
    echo ""
    echo "To run locally:"
    echo "  docker run -p 8000:8000 --gpus all $IMAGE_NAME"
    echo ""
    echo "To push to registry:"
    echo "  docker tag $IMAGE_NAME your-registry/$IMAGE_NAME"
    echo "  docker push your-registry/$IMAGE_NAME"
    echo ""
}

# Function to create deployment package
create_package() {
    echo "Creating deployment package..."
    
    PACKAGE_NAME="raw-diffusion-deploy-$(date +%Y%m%d-%H%M%S).tar.gz"
    
    echo "Packaging files..."
    tar -czf "$PACKAGE_NAME" \
        --exclude='.git' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='data/*' \
        --exclude='outputs/*' \
        --exclude='.venv' \
        --exclude='venv' \
        .
    
    echo "✓ Created: $PACKAGE_NAME"
    echo ""
}

# Function to validate configuration
validate_config() {
    echo "Validating configuration..."
    
    # Check required files
    REQUIRED_FILES=("api/serve.py" "requirements.txt" "render.yaml")
    
    for file in "${REQUIRED_FILES[@]}"; do
        if [ ! -f "$file" ]; then
            echo "❌ Missing required file: $file"
            exit 1
        fi
        echo "✓ Found: $file"
    done
    
    echo ""
}

# Main deployment flow
main() {
    check_prerequisites
    validate_config
    prepare_models
    
    echo "Select deployment action:"
    echo "1) Test locally"
    echo "2) Deploy to Render"
    echo "3) Build Docker image"
    echo "4) Create deployment package"
    echo "5) Full deployment (test + build + deploy)"
    echo ""
    read -p "Enter choice [1-5]: " choice
    
    case $choice in
        1)
            test_local
            ;;
        2)
            deploy_render
            ;;
        3)
            build_docker
            ;;
        4)
            create_package
            ;;
        5)
            test_local
            build_docker
            deploy_render
            ;;
        *)
            echo "Invalid choice"
            exit 1
            ;;
    esac
    
    echo ""
    echo "=========================================="
    echo "✓ Deployment process complete!"
    echo "=========================================="
}

# Run main function
main
