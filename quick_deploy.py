#!/usr/bin/env python3
"""
Quick Deploy Script - Simplified Deployment Entry Point

Provides easy commands for testing and deploying the API.
"""

import sys
import argparse
import subprocess
from pathlib import Path

def check_models():
    """Check if required models exist."""
    model_dir = Path("./models")
    required = ["vae_final.pt", "flow_final.pt", "diffusion_final.pt"]
    
    if not model_dir.exists():
        return False, "Model directory not found"
    
    missing = []
    for model in required:
        if not (model_dir / model).exists():
            missing.append(model)
    
    if missing:
        return False, f"Missing models: {', '.join(missing)}"
    
    return True, "All models found"

def run_local():
    """Run API server locally."""
    print("="*70)
    print("Starting API Server Locally")
    print("="*70)
    
    # Check models
    ok, msg = check_models()
    if not ok:
        print(f"\n⚠️  {msg}")
        print("API will run in demo mode without model inference.")
        print("To enable full functionality, train models first:")
        print("  python quick_train.py --data ./data/train\n")
    else:
        print(f"✓ {msg}\n")
    
    print("Starting server on http://localhost:8000")
    print("Press Ctrl+C to stop\n")
    print("Available endpoints:")
    print("  - Health check: http://localhost:8000/health")
    print("  - API docs: http://localhost:8000/docs")
    print("  - Metrics: http://localhost:8000/metrics\n")
    
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "api.serve:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n\nServer stopped")

def test_api():
    """Test API endpoints."""
    import requests
    import time
    
    print("="*70)
    print("Testing API Endpoints")
    print("="*70)
    
    base_url = "http://localhost:8000"
    
    # Start server in background
    print("\nStarting server...")
    import subprocess
    proc = subprocess.Popen([
        sys.executable, "-m", "uvicorn",
        "api.serve:app",
        "--host", "0.0.0.0",
        "--port", "8000"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    time.sleep(5)  # Wait for server to start
    
    try:
        # Test health endpoint
        print("\n1. Testing /health endpoint...")
        resp = requests.get(f"{base_url}/health")
        if resp.status_code == 200:
            print(f"   ✓ Health check passed: {resp.json()}")
        else:
            print(f"   ✗ Health check failed: {resp.status_code}")
        
        # Test metrics endpoint
        print("\n2. Testing /metrics endpoint...")
        resp = requests.get(f"{base_url}/metrics")
        if resp.status_code == 200:
            print(f"   ✓ Metrics endpoint working")
        else:
            print(f"   ⚠️  Metrics returned: {resp.status_code}")
        
        print("\n✓ Basic tests complete!")
        print("\nFor full API documentation, visit:")
        print(f"  {base_url}/docs")
    
    except requests.exceptions.ConnectionError:
        print("\n✗ Could not connect to API server")
        print("Make sure the server is running: python quick_deploy.py --local")
    
    finally:
        # Stop server
        proc.terminate()
        print("\nTest server stopped")

def build_docker():
    """Build Docker image."""
    print("="*70)
    print("Building Docker Image")
    print("="*70)
    
    if not Path("Dockerfile").exists():
        print("\n✗ Dockerfile not found")
        return
    
    print("\nBuilding image: raw-diffusion:latest")
    print("This may take several minutes...\n")
    
    try:
        subprocess.run([
            "docker", "build",
            "-t", "raw-diffusion:latest",
            "."
        ], check=True)
        
        print("\n✓ Docker image built successfully!")
        print("\nTo run:")
        print("  docker run -p 8000:8000 --gpus all raw-diffusion:latest")
    
    except subprocess.CalledProcessError:
        print("\n✗ Docker build failed")
    except FileNotFoundError:
        print("\n✗ Docker not installed")
        print("Install from: https://docs.docker.com/get-docker/")

def deploy_render():
    """Deploy to Render.com."""
    print("="*70)
    print("Deploy to Render.com")
    print("="*70)
    
    print("\nDeployment steps:")
    print("\n1. Push your code to GitHub:")
    print("   git init")
    print("   git add .")
    print("   git commit -m 'Ready for deployment'")
    print("   git remote add origin https://github.com/YOUR_USERNAME/raw-diffusion.git")
    print("   git push -u origin main")
    
    print("\n2. Go to Render Dashboard:")
    print("   https://dashboard.render.com")
    
    print("\n3. Create new Blueprint:")
    print("   - Click 'New +' → 'Blueprint'")
    print("   - Connect your GitHub repository")
    print("   - Render will auto-detect render.yaml")
    print("   - Click 'Apply'")
    
    print("\n4. Configure model storage:")
    print("   - Add disk: model-storage (20GB)")
    print("   - Mount at: /opt/render/project/models")
    print("   - Upload your trained models")
    
    print("\n5. Your API will be available at:")
    print("   https://raw-diffusion-api.onrender.com")
    
    print("\nFor detailed instructions, see DEPLOYMENT.md")

def main():
    parser = argparse.ArgumentParser(
        description='Quick Deploy Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run API locally
  python quick_deploy.py --local
  
  # Test API endpoints
  python quick_deploy.py --test
  
  # Build Docker image
  python quick_deploy.py --docker
  
  # Deploy to Render.com (instructions)
  python quick_deploy.py --render
        """
    )
    
    parser.add_argument('--local', action='store_true',
                       help='Run API server locally')
    parser.add_argument('--test', action='store_true',
                       help='Test API endpoints')
    parser.add_argument('--docker', action='store_true',
                       help='Build Docker image')
    parser.add_argument('--render', action='store_true',
                       help='Show Render.com deployment instructions')
    
    args = parser.parse_args()
    
    if args.local:
        run_local()
    elif args.test:
        test_api()
    elif args.docker:
        build_docker()
    elif args.render:
        deploy_render()
    else:
        parser.print_help()
        print("\n" + "="*70)
        print("Quick Start:")
        print("="*70)
        print("\n1. Run locally:    python quick_deploy.py --local")
        print("2. Test API:       python quick_deploy.py --test")
        print("3. Build Docker:   python quick_deploy.py --docker")
        print("4. Deploy Render:  python quick_deploy.py --render")

if __name__ == "__main__":
    main()
