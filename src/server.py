#!/usr/bin/env python3
"""
FastAPI model inference server (port 8000).

Loads model once, serves batch inference and health-check endpoints.
"""

import os
import gc
import numpy as np
import torch
from diffusers import QwenImageEditPlusPipeline

print("[Server] Starting model load...")
dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = os.environ.get("MODEL_PATH", "")
model_cache_dir = os.environ.get("TRANSFORMERS_CACHE", "/models")
load_in_8bit = os.environ.get("LOAD_IN_8BIT", "").lower() == "true"


def _load_8bit_pipeline(path):
    """Load a pipeline with 8-bit quantized transformer via bitsandbytes."""
    from diffusers import BitsAndBytesConfig as DiffusersBnBConfig
    from diffusers.models import AutoModel

    print("[Server] Loading transformer with 8-bit quantization...")
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

    print("[Server] Loading pipeline components (bf16)...")
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        path,
        transformer=transformer,
        torch_dtype=dtype,
    )

    print("[Server] Enabling model CPU offload...")
    pipeline.enable_model_cpu_offload()

    if torch.cuda.is_available():
        vram_gb = torch.cuda.memory_allocated() / (1024**3)
        print(f"[Server] GPU memory after setup: {vram_gb:.1f} GB")

    return pipeline


if model_path and os.path.isdir(model_path) and load_in_8bit:
    print(f"[Server] Loading 8-bit model from: {model_path}")
    import sys
    try:
        pipe = _load_8bit_pipeline(model_path)
    except Exception as e:
        import traceback
        print(f"[Server] 8-bit loading failed: {e}", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(1)
elif model_path and os.path.isdir(model_path):
    print(f"[Server] Loading model from MODEL_PATH: {model_path}")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        model_path, torch_dtype=dtype
    ).to(device)
else:
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
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        pretrained_id, **load_kwargs
    ).to(device)

print("[Server] Model loaded successfully")


def _cleanup_gpu():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()


def _log_gpu_memory(label: str):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        free = total - reserved
        print(
            f"[GPU {label}] Allocated: {allocated:.2f} GiB, "
            f"Reserved: {reserved:.2f} GiB, Free: {free:.2f} GiB"
        )


def run_fastapi():
    import asyncio
    import base64
    import io
    import random
    import time
    from typing import List, Optional

    import json
    import queue

    import uvicorn
    from fastapi import FastAPI, HTTPException
    from PIL import Image
    from pydantic import BaseModel, Field
    from starlette.responses import StreamingResponse

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
        # images[0] = source (already padded to 1024x1024)
        images: List[ImageInput]
        # optional mask: white=editable, black=protected
        mask_image: Optional[ImageInput] = None

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
        return {"status": "ok", "service": "qwen-model-api"}

    @app.get("/healthz")
    async def healthz():
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

    def _apply_mask_blend(src_image, generated_images, mask_image):
        """
        Blend generated images with original using mask (post-processing).

        mask_image: white (255) = editable (use generated),
                    black (0)   = protected (keep original).
        All blending is done in src_image resolution.
        """
        if mask_image is None:
            return generated_images

        src = src_image.convert("RGB")
        src_w, src_h = src.size

        mask = mask_image.convert("L").resize((src_w, src_h), Image.NEAREST)
        mask_arr = np.array(mask, dtype=np.float32) / 255.0
        mask_arr = mask_arr[..., None]  # (H, W, 1)

        src_arr = np.array(src, dtype=np.float32)

        blended = []
        for gen_img in generated_images:
            gen = gen_img.convert("RGB").resize((src_w, src_h), Image.BILINEAR)
            gen_arr = np.array(gen, dtype=np.float32)
            out_arr = src_arr * (1.0 - mask_arr) + gen_arr * mask_arr
            out_arr = np.clip(out_arr, 0, 255).astype(np.uint8)
            blended.append(Image.fromarray(out_arr, mode="RGB"))
        return blended

    def _run_inference_with_mask_and_refs(
        src_image,
        mask_image,
        style_refs,
        prompt,
        height,
        width,
        negative_prompt,
        num_inference_steps,
        generator,
        guidance_scale,
        num_images_per_prompt,
        callback=None,
    ):
        # Pass all images (source + style refs) as a list, matching the
        # original behavior where the pipeline receives the full image set.
        image_input = [src_image] + style_refs if style_refs else src_image

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
                    print("[API] callback_on_step_end not supported, falling back")
                    del pipe_kwargs["callback_on_step_end"]
                    result = pipe(**pipe_kwargs).images
                else:
                    raise
            except IndexError:
                adjusted = num_inference_steps - 1
                print(
                    f"[API] Scheduler IndexError at steps={num_inference_steps}, "
                    f"retrying with steps={adjusted}"
                )
                pipe_kwargs["num_inference_steps"] = adjusted
                result = pipe(**pipe_kwargs).images

            if mask_image is not None:
                print(f"[API] Applying mask blend to {len(result)} output images")
                result = _apply_mask_blend(src_image, result, mask_image)

            return result
        finally:
            _cleanup_gpu()
            _log_gpu_memory("post-cleanup")

    @app.post("/api/v1/batch/infer", response_model=BatchInferenceResponse)
    async def batch_infer(request: BatchInferenceRequest):
        if pipe is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        start_time = time.time()
        output_images = []

        try:
            if not request.images:
                raise HTTPException(status_code=400, detail="At least one image is required")

            pil_images = []
            for idx, img_input in enumerate(request.images):
                pil_image = decode_base64_image(img_input.data)
                pil_images.append(pil_image)
                print(f"[API] Loaded image {idx + 1}/{len(request.images)}")

            src_image = pil_images[0]
            style_refs = pil_images[1:]

            mask_pil = None
            if request.mask_image is not None:
                mask_pil = decode_base64_image(request.mask_image.data)
                print("[API] Loaded mask image")

            seed = (
                random.randint(0, MAX_SEED) if request.randomize_seed else request.seed
            )
            generator = torch.Generator(device=device).manual_seed(seed)

            mode_str = (
                "style reference mode" if request.style_reference_mode else "batch mode"
            )
            print(
                f"[API] Processing images (mode={mode_str}), seed={seed}, "
                f"style_refs={len(style_refs)}, has_mask={mask_pil is not None}"
            )

            result = await asyncio.to_thread(
                _run_inference_with_mask_and_refs,
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

            print(f"[API] Received {len(result)} output images from pipeline")

            if request.style_reference_mode:
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
            import traceback
            print(f"[API] Batch inference error: {e}")
            traceback.print_exc()
            _cleanup_gpu()
            _log_gpu_memory("batch-error-cleanup")
            return BatchInferenceResponse(
                success=False,
                images=[],
                total_time_seconds=round(time.time() - start_time, 2),
                error="Inference failed. Check server logs for details.",
            )

    def _sse_event(data: dict) -> str:
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
                if not request.images:
                    raise ValueError("At least one image is required")

                pil_images = []
                for idx, img_input in enumerate(request.images):
                    pil_images.append(decode_base64_image(img_input.data))
                    print(f"[Stream] Loaded image {idx + 1}/{len(request.images)}")

                src_image = pil_images[0]
                style_refs = pil_images[1:]

                mask_pil = None
                if request.mask_image is not None:
                    mask_pil = decode_base64_image(request.mask_image.data)
                    print("[Stream] Loaded mask image")

                seed = (
                    random.randint(0, MAX_SEED)
                    if request.randomize_seed
                    else request.seed
                )
                generator = torch.Generator(device=device).manual_seed(seed)

                mode_str = (
                    "style reference mode"
                    if request.style_reference_mode
                    else "batch mode"
                )
                print(
                    f"[Stream] Processing images (mode={mode_str}), seed={seed}, "
                    f"style_refs={len(style_refs)}, has_mask={mask_pil is not None}"
                )

                progress_q: queue.Queue[int] = queue.Queue()

                def step_callback(_pipe, step_index, _timestep, kwargs):
                    progress_q.put(step_index + 1)
                    return kwargs

                loop = asyncio.get_event_loop()
                future = loop.run_in_executor(
                    None,
                    _run_inference_with_mask_and_refs,
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
                print(
                    f"[Stream] Received {len(result)} output images from pipeline"
                )

                output_images = []
                if request.style_reference_mode:
                    if result:
                        output_images.append({
                            "data": encode_image_base64(result[-1]),
                            "seed": seed,
                            "index": 0,
                        })
                        print(
                            f"[Stream] Style reference mode: "
                            f"Returning final image (last of {len(result)})"
                        )
                else:
                    for idx, img in enumerate(result):
                        output_images.append({
                            "data": encode_image_base64(img),
                            "seed": seed,
                            "index": idx,
                        })
                    print(
                        f"[Stream] Batch mode: Returning all {len(result)} images"
                    )

                yield _sse_event({
                    "type": "complete",
                    "success": True,
                    "images": output_images,
                    "total_time_seconds": round(time.time() - start_time, 2),
                })

            except Exception as e:
                import traceback
                print(f"[Stream] Error: {e}")
                traceback.print_exc()
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

    print("[Server] Starting FastAPI on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    run_fastapi()
