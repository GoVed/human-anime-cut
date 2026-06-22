import os
import io
import time
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Append the project root to sys.path so we can import our modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import Generator

app = FastAPI(title="Human-to-Anime Style Transfer Server")

# Allow CORS for development ease
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and device settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
IMAGE_SIZE = 256

@app.on_event("startup")
def load_generator():
    global model
    checkpoint_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "checkpoint_latest.pt"
    )
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        # Initialize an untrained generator as a fallback so server starts
        model = Generator().to(device)
        model.eval()
        return
        
    print(f"Loading checkpoint from: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = Generator().to(device)
        
        # Load weights from either CycleGAN (G_XtoY) or CUT (G) key
        if 'G' in checkpoint:
            model.load_state_dict(checkpoint['G'])
            print("Successfully loaded CUT generator weights.")
        elif 'G_XtoY' in checkpoint:
            model.load_state_dict(checkpoint['G_XtoY'])
            print("Successfully loaded CycleGAN generator weights.")
        else:
            # Direct state dict load fallback
            model.load_state_dict(checkpoint)
            print("Successfully loaded model directly as state dict.")
            
        # Convert to half-precision (FP16) to save VRAM and accelerate inference
        if device.type == 'cuda':
            model = model.half()
            print("Converted generator model to Half-Precision (FP16) for memory efficiency.")
            
        model.eval()
        
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        # Initialize fallback
        model = Generator().to(device)
        model.eval()

def preprocess_image(pil_img: Image.Image, size: int) -> torch.Tensor:
    """Preprocess PIL Image to normalized FP16 tensor."""
    # Convert to RGB if not already
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
        
    # Crop to center square to avoid squeezing aspect ratio
    width, height = pil_img.size
    if width != height:
        min_size = min(width, height)
        left = (width - min_size) // 2
        top = (height - min_size) // 2
        right = left + min_size
        bottom = top + min_size
        pil_img = pil_img.crop((left, top, right, bottom))
        
    # Resize keeping aspect ratio (optional, but standard square resize matching dataset)
    pil_img = pil_img.resize((size, size), Image.Resampling.LANCZOS)
    
    # Convert to numpy and normalize to [-1, 1]
    np_img = np.array(pil_img, dtype=np.float32) / 255.0
    np_img = (np_img - 0.5) / 0.5
    
    # Transpose to [C, H, W]
    np_img = np.transpose(np_img, (2, 0, 1))
    
    # Create tensor, add batch dimension and move to device
    tensor = torch.from_numpy(np_img).unsqueeze(0).to(device)
    
    # Cast to float16 if model is FP16
    if device.type == 'cuda':
        tensor = tensor.half()
        
    return tensor

def postprocess_tensor(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor output back to PIL Image."""
    # Remove batch dimension, move to CPU, convert to float32
    img = tensor.squeeze(0).detach().cpu().float().numpy()
    
    # Transpose back to [H, W, C]
    img = np.transpose(img, (1, 2, 0))
    
    # Denormalize from [-1, 1] to [0, 255]
    img = (img + 1.0) * 127.5
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    return Image.fromarray(img)

# In-memory IP request tracker for rate limiting
ip_cooldowns = {}

@app.post("/api/translate")
async def translate_image(request: Request, file: UploadFile = File(...)):
    """Receives user photo, runs styles transfer on GPU (FP16), and streams back styled PNG with rate-limiting."""
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model is loading or unavailable.")
        
    client_ip = request.client.host if request.client else "unknown"
    current_time = time.time()
    
    if client_ip in ip_cooldowns:
        elapsed = current_time - ip_cooldowns[client_ip]
        if elapsed < 3.0:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Please wait {3.0 - elapsed:.1f}s before uploading again."
            )
            
    try:
        # Read uploaded image bytes
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents))
        
        # Preprocess
        input_tensor = preprocess_image(pil_image, IMAGE_SIZE)
        
        # Inference
        with torch.no_grad():
            output_tensor = model(input_tensor)
            
        # Postprocess
        output_image = postprocess_tensor(output_tensor)
        
        # Upscale to 512x512 (twice the size of 256px) for crisp display and download
        output_image = output_image.resize((512, 512), Image.Resampling.LANCZOS)
        
        # Save output to bytes stream
        img_io = io.BytesIO()
        output_image.save(img_io, format="PNG")
        img_io.seek(0)
        
        # Explicit clean-up to prevent memory accumulation in PyTorch cache
        del input_tensor, output_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Update cooldown timestamp after successful inference
        ip_cooldowns[client_ip] = time.time()
            
        return StreamingResponse(img_io, media_type="image/png")
        
    except Exception as e:
        print(f"Error during translation: {e}")
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

@app.get("/api/health")
def health_check():
    """Verify backend and memory footprint statistics."""
    stats = {
        "status": "healthy" if model is not None else "initializing",
        "device": device.type,
        "precision": "FP16 (Half)" if device.type == 'cuda' else "FP32 (Full)"
    }
    
    if torch.cuda.is_available():
        stats["vram_allocated_mb"] = torch.cuda.memory_allocated(device) / (1024 * 1024)
        stats["vram_reserved_mb"] = torch.cuda.memory_reserved(device) / (1024 * 1024)
        
    return stats

# Serve the static files (UI) on root index
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
else:
    @app.get("/")
    def read_root():
        return {"message": "API running. Static files directory is missing."}
