#!/usr/bin/env python3
"""
FastAPI model inference server (port 8000).

Loads model once, serves batch inference and health-check endpoints.
"""

import os

# --- Model Loading (shared by both servers) ---
import gc
import numpy as np
import torch
from diffusers import QwenImageEditPlusPipeline

print("[Server] Starting model load...")
dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"

model_cache_dir = os.environ.get("TRANSFORMERS_CACHE", "/models")
model_cache_path = os.path.join(
    model_cache_dir, "models--ovedrive--Qwen-Image-Edit-2511-4bit"
)
snapshot_path = os.path.join(
    model_cache_path, "snapshots", "4104233c114f9b7b2e9c235d72ae4d216720aaac"
)

if os.path.exists(snapshot_path):
    print(f"[Server] Loading model from node-local cache: {snapshot_path}")
    pretrained_id = snapshot_path
    load_kwargs = {"torch_dtype": dtype}
else:
    print("[Server] Loading model by ID: ovedrive/Qwen-Image-Edit-2511-4bit")
    pretrained_id = "ovedrive/Qwen-Image-Edit-2511-4bit"
    load_kwargs = {"torch_dtype": dtype, "cache_dir": model_cache_dir}

pipe = QwenImageEditPlusPipeline.from_pretrained(pretrained_id, **load_kwargs).to(
    device
)
print("[Server] Model loaded successfully")


def run_fastapi():
    """Run FastAPI on port 8000."""
    import asyncio
    import base64
    import io
    import random
    import time
    from typing import List, Optional

    import uvicorn
    from fastapi import FastAPI, HTTPException
    from PIL import Image
    from pydantic import BaseModel, Field

    MAX_SEED = np.iinfo(np.int32).max
    DEFAULT_NEGATIVE_PROMPT = (
        "blurry, out of focus, low resolution, low detail, low sharpness, "
        "soft edges, motion blur, depth of field blur, hazy, unclear, artifact, noisy"
    )

    class ImageInput(BaseModel):
        data: str
        filename: Optional[str] = None

    class ImageOutput(BaseModel):
        data: str
        seed: int
        index: int

    class BatchInferenceRequest(BaseModel):
        images: List[ImageInput]
        prompt: str
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
        seed: int = Field(default=42, ge=0, le=MAX_SEED)
        randomize_seed: bool = False
        guidance_scale: float = Field(default=3.0, ge=1.0, le=10.0)
        num_inference_steps: int = Field(default=30, ge=1, le=50)
        height: int = Field(default=1024, ge=256, le=2048)
        width: int = Field(default=1024, ge=256, le=2048)
        num_images_per_prompt: int = Field(default=1, ge=1, le=4)
        style_reference_mode: bool = Field(default=True)

    class BatchInferenceResponse(BaseModel):
        success: bool
        images: List[ImageOutput] = []
        total_time_seconds: float
        error: Optional[str] = None

    class HealthResponse(BaseModel):
        status: str
        model_loaded: bool
        gpu_available: bool
        gpu_memory_used_gb: Optional[float] = None
        gpu_memory_total_gb: Optional[float] = None

    def decode_base64_image(data: str) -> Image.Image:
        if data.startswith("data:"):
            _, data = data.split(",", 1)
        return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")

    def encode_image_base64(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    app = FastAPI(
        title="Qwen Image Edit API",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    @app.get("/")
    async def root():
        """Root endpoint."""
        return {"status": "ok", "service": "qwen-model-api"}

    @app.get("/healthz")
    async def healthz():
        """Lightweight health check for ALB target group."""
        return {"status": "ok"}

    @app.get("/api/v1/health", response_model=HealthResponse)
    async def health_check():
        gpu_available = torch.cuda.is_available()
        gpu_mem_used = (
            torch.cuda.memory_allocated() / (1024**3) if gpu_available else None
        )
        gpu_mem_total = (
            torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_available
            else None
        )
        return HealthResponse(
            status="healthy" if pipe else "unhealthy",
            model_loaded=pipe is not None,
            gpu_available=gpu_available,
            gpu_memory_used_gb=round(gpu_mem_used, 2) if gpu_mem_used else None,
            gpu_memory_total_gb=round(gpu_mem_total, 2) if gpu_mem_total else None,
        )

    def _run_inference(
        pil_images: list,
        prompt: str,
        height: int,
        width: int,
        negative_prompt: str,
        num_inference_steps: int,
        generator: torch.Generator,
        guidance_scale: float,
        num_images_per_prompt: int,
    ) -> list:
        """Run the diffusion pipeline (blocking). Called via asyncio.to_thread()
        so the event loop stays free for health checks during inference."""
        try:
            result = pipe(
                image=pil_images,
                prompt=prompt,
                height=height,
                width=width,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                generator=generator,
                true_cfg_scale=guidance_scale,
                num_images_per_prompt=num_images_per_prompt,
            ).images
        except IndexError:
            # Workaround for off-by-one bug in the scheduler's sigma lookup
            # that triggers with certain num_inference_steps values.
            adjusted = num_inference_steps - 1
            print(
                f"[API] Scheduler IndexError at steps={num_inference_steps}, "
                f"retrying with steps={adjusted}"
            )
            result = pipe(
                image=pil_images,
                prompt=prompt,
                height=height,
                width=width,
                negative_prompt=negative_prompt,
                num_inference_steps=adjusted,
                generator=generator,
                true_cfg_scale=guidance_scale,
                num_images_per_prompt=num_images_per_prompt,
            ).images

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()

        return result

    @app.post("/api/v1/batch/infer", response_model=BatchInferenceResponse)
    async def batch_infer(request: BatchInferenceRequest):
        if pipe is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        start_time = time.time()
        output_images = []

        try:
            # Decode all input images first
            pil_images = []
            for idx, img_input in enumerate(request.images):
                pil_image = decode_base64_image(img_input.data)
                pil_images.append(pil_image)
                print(f"[API] Loaded image {idx + 1}/{len(request.images)}")

            seed = (
                random.randint(0, MAX_SEED) if request.randomize_seed else request.seed
            )
            generator = torch.Generator(device=device).manual_seed(seed)

            mode_str = (
                "style reference mode" if request.style_reference_mode else "batch mode"
            )
            print(
                f"[API] Processing {len(pil_images)} images ({mode_str}), seed={seed}"
            )

            # Run pipeline in a thread so health checks remain responsive
            result = await asyncio.to_thread(
                _run_inference,
                pil_images,
                request.prompt,
                request.height,
                request.width,
                request.negative_prompt,
                request.num_inference_steps,
                generator,
                request.guidance_scale,
                request.num_images_per_prompt,
            )

            print(f"[API] Received {len(result)} output images from pipeline")

            if request.style_reference_mode:
                # Return only the last image (target edit), previous images were style references
                if result:
                    final_image = result[-1]
                    output_images.append(
                        ImageOutput(
                            data=encode_image_base64(final_image),
                            seed=seed,
                            index=0,
                        )
                    )
                    print(
                        f"[API] Style reference mode: Returning final image (last of {len(result)})"
                    )
            else:
                # Return all edited images (batch mode)
                for idx, img in enumerate(result):
                    output_images.append(
                        ImageOutput(
                            data=encode_image_base64(img),
                            seed=seed,
                            index=idx,
                        )
                    )
                print(f"[API] Batch mode: Returning all {len(result)} images")

            return BatchInferenceResponse(
                success=True,
                images=output_images,
                total_time_seconds=round(time.time() - start_time, 2),
            )
        except Exception as e:
            return BatchInferenceResponse(
                success=False,
                images=[],
                total_time_seconds=round(time.time() - start_time, 2),
                error=str(e),
            )

    print("[Server] Starting FastAPI on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    run_fastapi()
