# Deployment Guide# Deployment Guide for Render.com



## Docker Deployment## Quick Deploy to Render



```bash### Prerequisites

docker build -t raw-diffusion:latest .- GitHub account with this repository pushed

docker run -p 8000:8000 --gpus all -e MODEL_PATH=/models raw-diffusion:latest- Render.com account (free tier available)

```- Trained model checkpoints



## Render.com Deployment### Step 1: Prepare Your Repository



1. Push code to GitHub1. **Push code to GitHub:**

2. Connect repository to Render```bash

3. Render uses `render.yaml` for configurationgit init

4. Upload model checkpoints to persistent diskgit add .

5. Set environment variable: `MODEL_PATH=/opt/render/project/models`git commit -m "Initial commit with RAW diffusion system"

git remote add origin https://github.com/YOUR_USERNAME/raw-diffusion.git

## API Endpointsgit push -u origin main

```

- `GET /health` - Health check

- `POST /enhance` - Enhance RAW images2. **Ensure these files are present:**

- `GET /metrics` - Performance metrics- `render.yaml` - Render configuration

- `POST /benchmark` - Run benchmark- `requirements.txt` - Python dependencies

- `api/serve.py` - FastAPI application

## Environment Variables- `Dockerfile` - Container configuration



- `MODEL_PATH` - Path to model checkpoints### Step 2: Deploy on Render

- `PORT` - API server port (default: 8000)

- `PYTHON_VERSION` - Python version (3.10+)#### Option A: Using render.yaml (Recommended)



## Production Checklist1. Go to [Render Dashboard](https://dashboard.render.com)

2. Click **"New +"** → **"Blueprint"**

- [ ] Models uploaded to storage3. Connect your GitHub repository

- [ ] Environment variables configured4. Render will auto-detect `render.yaml`

- [ ] Health checks passing5. Review services:

- [ ] HTTPS enabled   - **Web Service**: API endpoint

- [ ] Rate limiting configured   - **Worker**: Training job (optional)

- [ ] Monitoring configured6. Click **"Apply"**


#### Option B: Manual Web Service Setup

1. Click **"New +"** → **"Web Service"**
2. Connect repository
3. Configure:
   - **Name**: `raw-diffusion-api`
   - **Environment**: `Python 3`
   - **Region**: `Oregon (US West)`
   - **Branch**: `main`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn api.serve:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Select based on needs
     - Free: Good for testing
     - Starter ($7/mo): Basic production
     - Standard ($25/mo): More resources
     - Pro/Pro Plus: GPU support

4. **Environment Variables:**
   - `PYTHON_VERSION`: `3.10.12`
   - `MODEL_PATH`: `/opt/render/project/models`

5. Click **"Create Web Service"**

### Step 3: Upload Model Checkpoints

Since model files are large, you have several options:

#### Option A: Using Render Disks (Recommended)

1. In your service, go to **"Disks"** tab
2. Click **"Add Disk"**
   - **Name**: `model-storage`
   - **Mount Path**: `/opt/render/project/models`
   - **Size**: 20 GB
3. Upload models via SSH or API

#### Option B: Download from Cloud Storage

Add to your startup script:

```bash
# In api/serve.py or startup script
import os
from pathlib import Path

def download_models():
    model_path = Path(os.environ.get('MODEL_PATH', './models'))
    model_path.mkdir(exist_ok=True)
    
    # Download from S3, GCS, or similar
    # aws s3 sync s3://your-bucket/models/ ./models/
    pass
```

#### Option C: Git LFS (for smaller models)

```bash
git lfs track "*.pt"
git add .gitattributes
git add models/*.pt
git commit -m "Add model files"
git push
```

### Step 4: Configure Persistent Storage

For training or data persistence:

1. Go to service **Settings** → **Disks**
2. Add disks:
   - **Training Data**: `/opt/render/project/data` (100 GB)
   - **Model Storage**: `/opt/render/project/models` (20 GB)
   - **Outputs**: `/opt/render/project/outputs` (50 GB)

### Step 5: Verify Deployment

Once deployed, your API will be available at:
```
https://raw-diffusion-api.onrender.com
```

Test endpoints:
```bash
# Health check
curl https://raw-diffusion-api.onrender.com/health

# API docs
open https://raw-diffusion-api.onrender.com/docs
```

### Step 6: Monitor Your Service

1. **Logs**: View in Render dashboard under "Logs" tab
2. **Metrics**: Check "Metrics" tab for CPU/Memory usage
3. **Alerts**: Set up in "Settings" → "Alerts"

## Advanced Configuration

### GPU Support (Pro Plan Required)

Update `render.yaml`:
```yaml
services:
  - type: web
    name: raw-diffusion-api
    plan: pro-plus  # GPU instance
    dockerCommand: uvicorn api.serve:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: NVIDIA_VISIBLE_DEVICES
        value: all
```

### Auto-Scaling

For high traffic, configure auto-scaling:

```yaml
services:
  - type: web
    name: raw-diffusion-api
    autoDeploy: true
    scaling:
      minInstances: 2
      maxInstances: 10
      targetCPUPercent: 70
```

### Custom Domain

1. Go to service **Settings** → **Custom Domains**
2. Add your domain: `api.yourdomain.com`
3. Update DNS records as instructed

### Environment-Based Configuration

Create `.env.production`:
```bash
MODEL_PATH=/opt/render/project/models
LOG_LEVEL=INFO
MAX_BATCH_SIZE=8
ENABLE_PROFILING=false
```

## Alternative Deployment Platforms

### Deploy to Docker/Kubernetes

```bash
# Build image
docker build -t raw-diffusion:latest .

# Run locally
docker run -p 8000:8000 --gpus all raw-diffusion:latest

# Push to registry
docker tag raw-diffusion:latest your-registry/raw-diffusion:latest
docker push your-registry/raw-diffusion:latest
```

### Deploy to AWS ECS

1. Push Docker image to ECR
2. Create ECS task definition
3. Configure service with GPU support
4. Set up load balancer

### Deploy to Google Cloud Run

```bash
gcloud run deploy raw-diffusion-api \
  --image gcr.io/YOUR_PROJECT/raw-diffusion:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## Production Checklist

- [ ] Models uploaded to persistent storage
- [ ] Environment variables configured
- [ ] Health checks passing
- [ ] HTTPS enabled (automatic on Render)
- [ ] Monitoring/alerts configured
- [ ] Backup strategy for models
- [ ] Rate limiting configured
- [ ] API authentication added (if needed)
- [ ] Load testing completed
- [ ] Documentation updated with API URL

## Training on Render

For training jobs, use a Worker service:

```yaml
services:
  - type: worker
    name: raw-diffusion-trainer
    env: python
    plan: pro-plus  # GPU instance
    buildCommand: pip install -r requirements.txt
    startCommand: python scripts/train_orchestrator.py --data_dir /data --output_dir /outputs
```

**Note**: Training on Render is expensive. Consider:
- Using Render only for inference
- Training on AWS/GCP with GPU instances
- Using managed ML platforms (SageMaker, Vertex AI)

## Cost Optimization

1. **Use smaller models** for inference
2. **Implement caching** for common requests
3. **Use CPU instances** if latency allows
4. **Auto-sleep** during low traffic (Render Free)
5. **Batch requests** when possible

## Troubleshooting

### Service won't start
- Check build logs for dependency errors
- Verify `requirements.txt` is complete
- Ensure Python version compatibility

### Models not found
- Verify disk mount paths
- Check MODEL_PATH environment variable
- Ensure models were uploaded correctly

### Out of memory
- Reduce batch size
- Use smaller model variants
- Upgrade to larger plan
- Implement model quantization

### Slow inference
- Enable GPU (Pro+ plan)
- Use optimized models
- Implement caching
- Consider edge deployment

## Support

- Render Docs: https://render.com/docs
- GitHub Issues: [Your repo]/issues
- Email: your-support@email.com

## Next Steps

After deployment:
1. Test API thoroughly
2. Set up monitoring
3. Configure CI/CD pipeline
4. Add authentication if needed
5. Optimize for production traffic
6. Document API for users
