"""FastAPI REST API for RAW Image Enhancement"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import torch
import io
import base64
import numpy as np
from pathlib import Path
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAW Image Enhancement API",
    description="Real-time multi-frame RAW image enhancement using diffusion models",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
pipeline = None
device = "cuda" if torch.cuda.is_available() else "cpu"


class HealthResponse(BaseModel):
    status: str
    device: str
    models_loaded: bool


class EnhancementRequest(BaseModel):
    burst_images: List[str]  # Base64 encoded images
    num_steps: Optional[int] = 2
    use_adaptive_steps: Optional[bool] = True


class EnhancementResponse(BaseModel):
    enhanced_image: str  # Base64 encoded
    latency_ms: float
    num_steps_used: int


def load_models():
    """Load trained models on startup."""
    global pipeline
    
    try:
        logger.info("Loading models...")
        model_path = Path(os.environ.get('MODEL_PATH', './models'))
        
        # Import here to avoid issues if not installed
        from inference.realtime_pipeline import RealTimePipeline
        from models.raw_diffusion_unet import RAWVAE
        from models.consistency_distillation import ConsistencyModel
        from models.optical_flow import RAWOpticalFlow, AlignmentModule
        
        # Load checkpoints
        vae_checkpoint = model_path / 'vae_final.pt'
        flow_checkpoint = model_path / 'flow_final.pt'
        consistency_checkpoint = model_path / 'consistency_final.pt'
        
        if not all([vae_checkpoint.exists(), flow_checkpoint.exists(), consistency_checkpoint.exists()]):
            logger.warning("Model checkpoints not found. API will run in demo mode.")
            return
        
        # Load models
        vae = RAWVAE(in_channels=4, latent_channels=16, channels=64, num_res_blocks=2)
        vae.load_state_dict(torch.load(vae_checkpoint, map_location=device)['model_state_dict'])
        
        flow_net = RAWOpticalFlow(in_channels=4, feature_dim=128, num_levels=4)
        flow_net.load_state_dict(torch.load(flow_checkpoint, map_location=device)['model_state_dict'])
        alignment = AlignmentModule(flow_net)
        
        consistency = ConsistencyModel(in_channels=16, model_channels=128, num_res_blocks=2)
        consistency.load_state_dict(torch.load(consistency_checkpoint, map_location=device)['model_state_dict'])
        
        # Create pipeline
        pipeline = RealTimePipeline(
            vae=vae,
            consistency_model=consistency,
            alignment=alignment,
            device=device,
            num_inference_steps=2,
            use_adaptive_steps=True,
            enable_profiling=False
        )
        
        logger.info(f"✓ Models loaded successfully on {device}")
    
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        logger.info("API will run in demo mode")


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    logger.info("Starting RAW Enhancement API...")
    load_models()
    logger.info("API ready!")


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "RAW Image Enhancement API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "enhance": "/enhance",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        device=device,
        models_loaded=pipeline is not None
    )


@app.post("/enhance", response_model=EnhancementResponse)
async def enhance_images(request: EnhancementRequest):
    """
    Enhance a burst of RAW images.
    
    Args:
        request: EnhancementRequest with base64 encoded images
        
    Returns:
        EnhancementResponse with enhanced image and metrics
    """
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. API is in demo mode."
        )
    
    try:
        import time
        start_time = time.time()
        
        # Decode base64 images
        burst_images = []
        for img_b64 in request.burst_images:
            img_bytes = base64.b64decode(img_b64)
            img_array = np.frombuffer(img_bytes, dtype=np.float32)
            # Reshape to [4, H, W] - assumes RAW Bayer format
            # You'll need to adjust based on your actual data format
            burst_images.append(img_array)
        
        # Convert to tensor [B, T, 4, H, W]
        burst_tensor = torch.tensor(burst_images).unsqueeze(0).to(device)
        
        # Run inference
        results = pipeline.forward(
            burst_tensor,
            return_intermediate=False
        )
        
        enhanced = results['enhanced']
        num_steps = results['num_steps']
        
        # Convert back to base64
        enhanced_np = enhanced.squeeze(0).cpu().numpy()
        enhanced_bytes = enhanced_np.tobytes()
        enhanced_b64 = base64.b64encode(enhanced_bytes).decode('utf-8')
        
        latency_ms = (time.time() - start_time) * 1000
        
        return EnhancementResponse(
            enhanced_image=enhanced_b64,
            latency_ms=latency_ms,
            num_steps_used=num_steps
        )
    
    except Exception as e:
        logger.error(f"Enhancement failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/enhance_file")
async def enhance_file(files: List[UploadFile] = File(...)):
    """
    Enhance uploaded RAW image files.
    
    Args:
        files: List of uploaded RAW image files
        
    Returns:
        Enhanced image file
    """
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. API is in demo mode."
        )
    
    try:
        # Read uploaded files
        burst_images = []
        for file in files:
            contents = await file.read()
            # Parse RAW file (e.g., DNG)
            # This is a placeholder - you'll need proper RAW parsing
            img_array = np.frombuffer(contents, dtype=np.uint8)
            burst_images.append(img_array)
        
        # Process...
        # (Similar to enhance_images endpoint)
        
        return JSONResponse(content={"status": "success", "message": "File upload endpoint placeholder"})
    
    except Exception as e:
        logger.error(f"File enhancement failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """Get current performance metrics."""
    if pipeline is None:
        return {"status": "Models not loaded"}
    
    try:
        # Get model sizes
        sizes = pipeline.get_model_size()
        
        # Get memory info
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            memory_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        else:
            memory_allocated = 0
            memory_reserved = 0
        
        return {
            "device": device,
            "model_sizes_mb": sizes,
            "memory_allocated_mb": memory_allocated,
            "memory_reserved_mb": memory_reserved
        }
    
    except Exception as e:
        return {"error": str(e)}


@app.post("/benchmark")
async def run_benchmark():
    """Run performance benchmark."""
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded"
        )
    
    try:
        results = pipeline.benchmark(
            burst_sizes=[(512, 512)],
            burst_lengths=[8],
            num_iterations=10,
            warmup_iterations=2
        )
        
        return results
    
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    import os
    
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
