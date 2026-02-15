#!/usr/bin/env python3
"""
FastAPI model inference server (port 8000).

Loads model once, serves batch inference and health-check endpoints.
"""

import asyncio
import base64
import gc
import io
import json
import logging
import os
import queue
import random
import sys
import time
from typing import List, Optional

import numpy as np
import torch
import uvicorn
from diffusers import QwenImageEditPlusPipeline
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GIB = 1024**3
MAX_SEED = np.iinfo(np.int32).max
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, out of focus, low resolution, low detail, low sharpness, "
    "soft edges, motion blur, depth of field blur, hazy, unclear, artifact, noisy"
)
# HuggingFace snapshot commit for the 4-bit cached model
_4BIT_SNAPSHOT_HASH = "4104233c114f9b7b2e9c235d72ae4d216720aaac"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("qwen-server")

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"


def _load_8bit_pipeline(path: str) -> QwenImageEditPlusPipeline:
    """Load a pipeline with 8-bit quantized transformer via bitsandbytes.

    Loads the transformer from the full base model with int8 quantization,
    then builds the pipeline with the remaining components in bf16.
    Uses enable_model_cpu_offload() for device management.

    Expected VRAM: ~25 GB (8-bit transformer + bf16 text encoder + VAE).
    """
    from diffusers import BitsAndBytesConfig as DiffusersBnBConfig
    from diffusers.models import AutoModel

    logger.info("Loading transformer with 8-bit quantization...")
    quantization_config = DiffusersBnBConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
    )
    transformer = AutoModel.from_pretrained(
        path,
        subfolder="transformer",
        quantization_config=quantization_config,
        torch_dtype=dtype,
    )

    logger.info("Loading pipeline components (bf16)...")
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        path,
        transformer=transformer,
        torch_dtype=dtype,
    )

    logger.info("Enabling model CPU offload...")
    pipeline.enable_model_cpu_offload()

    if torch.cuda.is_available():
        vram_gb = torch.cuda.memory_allocated() / GIB
        logger.info("GPU memory after setup: %.1f GB", vram_gb)

    return pipeline


def _load_pipeline() -> QwenImageEditPlusPipeline:
    """Load the diffusion pipeline based on environment configuration."""
    model_path = os.environ.get("MODEL_PATH", "")
    model_cache_dir = os.environ.get("TRANSFORMERS_CACHE", "/models")
    load_in_8bit = os.environ.get("LOAD_IN_8BIT", "").lower() == "true"

    if model_path and os.path.isdir(model_path) and load_in_8bit:
        logger.info("Loading 8-bit model from: %s", model_path)
        return _load_8bit_pipeline(model_path)

    if model_path and os.path.isdir(model_path):
        logger.info("Loading model from MODEL_PATH: %s", model_path)
        return QwenImageEditPlusPipeline.from_pretrained(
            model_path, torch_dtype=dtype
        ).to(device)

    # Fall back to 4-bit HF cache layout
    model_cache_path = os.path.join(
        model_cache_dir, "models--ovedrive--Qwen-Image-Edit-2511-4bit"
    )
    snapshot_path = os.path.join(
        model_cache_path, "snapshots", _4BIT_SNAPSHOT_HASH
    )
    if os.path.exists(snapshot_path):
        logger.info("Loading model from node-local cache: %s", snapshot_path)
        pretrained_id = snapshot_path
        load_kwargs = {"torch_dtype": dtype}
    else:
        logger.info("Loading model by ID: ovedrive/Qwen-Image-Edit-2511-4bit")
        pretrained_id = "ovedrive/Qwen-Image-Edit-2511-4bit"
        load_kwargs = {"torch_dtype": dtype, "cache_dir": model_cache_dir}

    return QwenImageEditPlusPipeline.from_pretrained(
        pretrained_id, **load_kwargs
    ).to(device)


logger.info("Starting model load...")
try:
    pipe = _load_pipeline()
except Exception:
    logger.exception("Model loading failed")
    sys.exit(1)
logger.info("Model loaded successfully")

# ---------------------------------------------------------------------------
# GPU utilities
# ---------------------------------------------------------------------------


def _cleanup_gpu() -> None:
    """Synchronize GPU, release cached memory, and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()


def _log_gpu_memory(label: str) -> None:
    """Log GPU memory usage for diagnostics."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / GIB
        reserved = torch.cuda.memory_reserved() / GIB
        total = torch.cuda.get_device_properties(0).total_memory / GIB
        free = total - reserved
        logger.info(
            "[GPU %s] Allocated: %.2f GiB, Reserved: %.2f GiB, Free: %.2f GiB",
            label, allocated, reserved, free,
        )


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ImageInput(BaseModel):
    data: str
    filename: Optional[str] = None


class ImageOutput(BaseModel):
    data: str
    seed: int
    index: int


class BatchInferenceRequest(BaseModel):
    images: List[ImageInput]  # images[0] = source (already padded to 1024x1024)
    mask_image: Optional[ImageInput] = None  # white=editable, black=protected
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


# ---------------------------------------------------------------------------
# Image encoding helpers
# ---------------------------------------------------------------------------


def decode_base64_image(data: str) -> Image.Image:
    """Decode a base64-encoded (optionally data-URI-prefixed) string to PIL."""
    if data.startswith("data:"):
        _, data = data.split(",", 1)
    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")


def encode_image_base64(image: Image.Image) -> str:
    """Encode a PIL image as a base64 PNG string."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def _apply_mask_blend(
    src_image: Image.Image,
    generated_images: List[Image.Image],
    mask_image: Image.Image,
) -> List[Image.Image]:
    """Blend generated images with original using mask (post-processing).

    mask_image: white (255) = editable (use generated),
                black (0)   = protected (keep original).
    All blending is done in src_image resolution.
    """
    if mask_image is None:
        return generated_images

    src = src_image.convert("RGB")
    src_w, src_h = src.size

    mask = mask_image.convert("L").resize(
        (src_w, src_h), Image.Resampling.NEAREST,
    )
    mask_arr = np.array(mask, dtype=np.float32) / 255.0
    mask_arr = mask_arr[..., None]  # (H, W, 1)

    src_arr = np.array(src, dtype=np.float32)

    blended = []
    for gen_img in generated_images:
        gen = gen_img.convert("RGB").resize(
            (src_w, src_h), Image.Resampling.BILINEAR,
        )
        gen_arr = np.array(gen, dtype=np.float32)
        out_arr = src_arr * (1.0 - mask_arr) + gen_arr * mask_arr
        out_arr = np.clip(out_arr, 0, 255).astype(np.uint8)
        blended.append(Image.fromarray(out_arr, mode="RGB"))
    return blended


def _run_inference(
    src_image: Image.Image,
    mask_image: Optional[Image.Image],
    style_refs: List[Image.Image],
    prompt: str,
    height: int,
    width: int,
    negative_prompt: str,
    num_inference_steps: int,
    generator: torch.Generator,
    guidance_scale: float,
    num_images_per_prompt: int,
    callback=None,
) -> List[Image.Image]:
    """Run the diffusion pipeline (blocking).

    Called via asyncio.to_thread() so the event loop stays free for
    health checks during inference.
    """
    image_input = ([src_image] + style_refs) if style_refs else src_image

    pipe_kwargs = {
        "image": image_input,
        "prompt": prompt,
        "height": height,
        "width": width,
        "negative_prompt": negative_prompt,
        "num_inference_steps": num_inference_steps,
        "generator": generator,
        "true_cfg_scale": guidance_scale,
        "num_images_per_prompt": num_images_per_prompt,
    }

    if callback is not None:
        pipe_kwargs["callback_on_step_end"] = callback

    _log_gpu_memory("pre-inference")
    try:
        try:
            result = pipe(**pipe_kwargs).images
        except TypeError as e:
            if "callback_on_step_end" in str(e) and callback is not None:
                logger.warning("callback_on_step_end not supported, falling back")
                del pipe_kwargs["callback_on_step_end"]
                result = pipe(**pipe_kwargs).images
            else:
                raise
        except IndexError:
            adjusted = num_inference_steps - 1
            logger.warning(
                "Scheduler IndexError at steps=%d, retrying with steps=%d",
                num_inference_steps, adjusted,
            )
            pipe_kwargs["num_inference_steps"] = adjusted
            result = pipe(**pipe_kwargs).images

        if mask_image is not None:
            logger.info("Applying mask blend to %d output images", len(result))
            result = _apply_mask_blend(src_image, result, mask_image)

        return result
    finally:
        _cleanup_gpu()
        _log_gpu_memory("post-cleanup")


# ---------------------------------------------------------------------------
# Request preparation (shared by batch and stream endpoints)
# ---------------------------------------------------------------------------


def _prepare_request(
    request: BatchInferenceRequest,
) -> tuple:
    """Decode images, resolve seed, and build a generator from a request.

    Returns (src_image, style_refs, mask_pil, seed, generator).
    """
    if not request.images:
        raise HTTPException(
            status_code=400, detail="At least one image is required",
        )

    pil_images = [decode_base64_image(img.data) for img in request.images]
    src_image = pil_images[0]
    style_refs = pil_images[1:]

    mask_pil = None
    if request.mask_image is not None:
        mask_pil = decode_base64_image(request.mask_image.data)

    seed = random.randint(0, MAX_SEED) if request.randomize_seed else request.seed
    generator = torch.Generator(device=device).manual_seed(seed)

    mode_str = "style reference" if request.style_reference_mode else "batch"
    logger.info(
        "Prepared request: mode=%s, images=%d, style_refs=%d, "
        "has_mask=%s, seed=%d",
        mode_str, len(pil_images), len(style_refs),
        mask_pil is not None, seed,
    )

    return src_image, style_refs, mask_pil, seed, generator


def _encode_results(
    result: List[Image.Image],
    seed: int,
    style_reference_mode: bool,
) -> List[dict]:
    """Encode pipeline output images to serializable dicts."""
    output_images: List[dict] = []
    if style_reference_mode:
        if result:
            output_images.append({
                "data": encode_image_base64(result[-1]),
                "seed": seed,
                "index": 0,
            })
            logger.info(
                "Style reference mode: returning final image (last of %d)",
                len(result),
            )
    else:
        for idx, img in enumerate(result):
            output_images.append({
                "data": encode_image_base64(img),
                "seed": seed,
                "index": idx,
            })
        logger.info("Batch mode: returning all %d images", len(result))
    return output_images


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

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
    """Detailed health check with GPU memory stats."""
    gpu_available = torch.cuda.is_available()
    gpu_mem_used = (
        torch.cuda.memory_allocated() / GIB if gpu_available else None
    )
    gpu_mem_total = (
        torch.cuda.get_device_properties(0).total_memory / GIB
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


@app.post("/api/v1/batch/infer", response_model=BatchInferenceResponse)
async def batch_infer(request: BatchInferenceRequest):
    """Batch inference endpoint."""
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    try:
        src_image, style_refs, mask_pil, seed, generator = _prepare_request(
            request,
        )

        result = await asyncio.to_thread(
            _run_inference,
            src_image,
            mask_pil,
            style_refs,
            request.prompt,
            request.height,
            request.width,
            request.negative_prompt,
            request.num_inference_steps,
            generator,
            request.guidance_scale,
            request.num_images_per_prompt,
        )

        logger.info("Received %d output images from pipeline", len(result))

        encoded = _encode_results(result, seed, request.style_reference_mode)
        output_images = [
            ImageOutput(data=img["data"], seed=img["seed"], index=img["index"])
            for img in encoded
        ]

        return BatchInferenceResponse(
            success=True,
            images=output_images,
            total_time_seconds=round(time.time() - start_time, 2),
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Batch inference error")
        _cleanup_gpu()
        _log_gpu_memory("batch-error-cleanup")
        return BatchInferenceResponse(
            success=False,
            images=[],
            total_time_seconds=round(time.time() - start_time, 2),
            error="Inference failed. Check server logs for details.",
        )


def _sse_event(data: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data)}\n\n"


@app.post("/api/v1/stream/infer")
async def stream_infer(request: BatchInferenceRequest):
    """SSE streaming inference — sends first byte immediately to avoid
    CloudFront 60s origin read timeout."""
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    async def event_stream():
        yield _sse_event({
            "type": "started",
            "total_steps": request.num_inference_steps,
        })

        start_time = time.time()

        try:
            src_image, style_refs, mask_pil, seed, generator = (
                _prepare_request(request)
            )

            progress_q: queue.Queue[int] = queue.Queue()

            def step_callback(_pipe, step_index, _timestep, kwargs):
                progress_q.put(step_index + 1)
                return kwargs

            loop = asyncio.get_running_loop()
            future = loop.run_in_executor(
                None,
                _run_inference,
                src_image,
                mask_pil,
                style_refs,
                request.prompt,
                request.height,
                request.width,
                request.negative_prompt,
                request.num_inference_steps,
                generator,
                request.guidance_scale,
                request.num_images_per_prompt,
                step_callback,
            )

            while not future.done():
                await asyncio.sleep(1)
                got_progress = False
                while not progress_q.empty():
                    try:
                        step = progress_q.get_nowait()
                        got_progress = True
                        yield _sse_event({
                            "type": "progress",
                            "step": step,
                            "total_steps": request.num_inference_steps,
                        })
                    except queue.Empty:
                        break
                if not got_progress:
                    yield _sse_event({"type": "heartbeat"})

            result = future.result()
            logger.info(
                "Received %d output images from pipeline", len(result),
            )

            output_images = _encode_results(
                result, seed, request.style_reference_mode,
            )

            yield _sse_event({
                "type": "complete",
                "success": True,
                "images": output_images,
                "total_time_seconds": round(time.time() - start_time, 2),
            })

        except Exception:
            logger.exception("Stream inference error")
            _cleanup_gpu()
            _log_gpu_memory("stream-error-cleanup")
            yield _sse_event({
                "type": "error",
                "error": "Inference failed. Check server logs for details.",
            })

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Starting FastAPI on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
