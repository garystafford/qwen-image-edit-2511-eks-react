#!/usr/bin/env python3
"""
Batch process images using the FastAPI model endpoint.

Iterates through the samples_images/ folder and sends each image
to the /api/v1/batch/infer endpoint on the model service.

Usage:
    python scripts/batch_process_fastapi.py --url https://your-domain.example.com
    python scripts/batch_process_fastapi.py --url https://your-domain.example.com --prompt "Make it a watercolor painting"
    python scripts/batch_process_fastapi.py --url https://your-domain.example.com --output ./my_output
"""

import argparse
import base64
import sys
import time
from pathlib import Path

import requests

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".avif"}

DEFAULT_PROMPT = (
    "Restyle each image in a clean illustrative look with sharp line art, "
    "flat rich colors, soft cel shading, and minimal textured backgrounds. "
    "Extend or generate additional background as needed to fill a the new canvas, "
    "without cropping or changing the original subject, pose, or framing."
)

DEFAULT_NEGATIVE_PROMPT = (
    "blurry, out of focus, low resolution, low detail, low sharpness, "
    "double-image, soft edges, motion blur, depth of field blur, hazy, unclear, "
    "artifact, noisy, pixelated, compression artifacts, 'AI generated' text and surrounding box"
)

def image_to_base64(image_path: Path) -> str:
    """Read an image file and return its base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def save_base64_image(b64_data: str, output_path: Path) -> None:
    """Decode a base64 string and save it as a PNG image."""
    img_bytes = base64.b64decode(b64_data)
    with open(output_path, "wb") as f:
        f.write(img_bytes)


def check_health(base_url: str) -> bool:
    """Check the health endpoint and print GPU info."""
    try:
        resp = requests.get(f"{base_url}/api/v1/health", timeout=10)
        resp.raise_for_status()
        health = resp.json()
        print(f"  Status:     {health['status']}")
        print(f"  Model:      {'loaded' if health['model_loaded'] else 'NOT loaded'}")
        print(
            f"  GPU:        {'available' if health['gpu_available'] else 'NOT available'}"
        )
        if health.get("gpu_memory_used_gb") is not None:
            print(
                f"  GPU Memory: {health['gpu_memory_used_gb']:.1f} / {health['gpu_memory_total_gb']:.1f} GB"
            )
        return health["status"] == "healthy"
    except requests.exceptions.ConnectionError:
        print(f"  Cannot connect to {base_url}")
        print("  Make sure port-forwarding is active:")
        print("    kubectl port-forward -n qwen svc/qwen-model-service 8000:8000")
        return False
    except Exception as e:
        print(f"  Health check failed: {e}")
        return False


def process_image(
    base_url: str,
    image_path: Path,
    prompt: str,
    negative_prompt: str,
    seed: int,
    randomize_seed: bool,
    guidance_scale: float,
    steps: int,
    height: int,
    width: int,
    timeout: int,
    num_images_per_prompt: int = 1,
    max_retries: int = 3,
    retry_delay: float = 10.0,
) -> dict:
    """Send a single image to the FastAPI inference endpoint.

    Retries on 503 (Service Unavailable) and 504 (Gateway Timeout) errors,
    which can occur when the model pod is restarting, the ALB target is
    draining, or inference exceeds the ALB idle timeout.
    """
    b64_image = image_to_base64(image_path)

    payload = {
        "images": [{"data": b64_image, "filename": image_path.name}],
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "randomize_seed": randomize_seed,
        "guidance_scale": guidance_scale,
        "num_inference_steps": steps,
        "height": height,
        "width": width,
        "num_images_per_prompt": num_images_per_prompt,
    }

    for attempt in range(1, max_retries + 1):
        resp = requests.post(
            f"{base_url}/api/v1/batch/infer",
            json=payload,
            timeout=timeout,
        )
        if resp.status_code in (503, 504) and attempt < max_retries:
            print(f"{resp.status_code} (retry {attempt}/{max_retries}, waiting {retry_delay:.0f}s)...", end=" ", flush=True)
            time.sleep(retry_delay)
            continue
        resp.raise_for_status()
        return resp.json()

    # Should not reach here, but just in case
    resp.raise_for_status()
    return resp.json()


def format_time(seconds: float) -> str:
    """Format seconds into a readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    return f"{minutes:.1f}m"


def main():
    parser = argparse.ArgumentParser(
        description="Batch process sample images via FastAPI model endpoint"
    )
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="Model service URL (e.g. https://your-domain.example.com)",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Input directory (default: samples_images/ in project root)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory (default: batch_output/ in project root)",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default=DEFAULT_PROMPT,
        help="Editing prompt to apply to all images",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt describing what to avoid in the output",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--randomize-seed",
        action="store_true",
        default=False,
        help="Randomize seed for each image",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=3.0,
        help="Guidance scale (default: 3.0)",
    )
    parser.add_argument(
        "--steps", type=int, default=20, help="Number of inference steps (default: 20)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Output height in pixels (default: 1024)",
    )
    parser.add_argument(
        "--width", type=int, default=1024, help="Output width in pixels (default: 1024)"
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=1,
        help="Number of image variants per prompt (1-4, default: 1)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Request timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between images in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Max retries on 503 errors (default: 3)",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=10.0,
        help="Seconds to wait between retries (default: 10.0)",
    )

    args = parser.parse_args()

    # Strip trailing slash to avoid double-slash in URL paths
    args.url = args.url.rstrip("/")

    # Resolve directories relative to project root
    project_root = Path(__file__).resolve().parent.parent
    input_dir = Path(args.input) if args.input else project_root / "samples_images"
    output_dir = Path(args.output) if args.output else project_root / "batch_output"

    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find image files
    image_files = sorted(
        f for f in input_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_files:
        print(f"No image files found in '{input_dir}'")
        sys.exit(1)

    # Print configuration
    print("=" * 70)
    print("BATCH PROCESSING - FastAPI Model Endpoint")
    print("=" * 70)
    print(f"  Endpoint:   {args.url}")
    print(f"  Input:      {input_dir}")
    print(f"  Output:     {output_dir}")
    print(f"  Images:     {len(image_files)}")
    print(f"  Steps:      {args.steps}")
    print(f"  Guidance:   {args.guidance_scale}")
    print(f"  Size:       {args.width}x{args.height}")
    print(f"  Seed:       {'random' if args.randomize_seed else args.seed}")
    print(f"  Variants:   {args.num_images} per image")
    print(f"  Prompt:     {args.prompt[:80]}...")
    print(f"  Neg prompt: {args.negative_prompt[:80]}...")
    print("=" * 70)

    # Health check
    print("\nChecking model service health...")
    if not check_health(args.url):
        print("\nModel service is not healthy. Exiting.")
        sys.exit(1)
    print()

    # Process images
    success_count = 0
    fail_count = 0
    times = []
    batch_start = time.time()

    for i, image_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] {image_path.name}...", end=" ", flush=True)

        start = time.time()
        try:
            result = process_image(
                base_url=args.url,
                image_path=image_path,
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                seed=args.seed,
                randomize_seed=args.randomize_seed,
                guidance_scale=args.guidance_scale,
                steps=args.steps,
                height=args.height,
                width=args.width,
                timeout=args.timeout,
                num_images_per_prompt=args.num_images,
                max_retries=args.retries,
                retry_delay=args.retry_delay,
            )

            elapsed = time.time() - start

            if result["success"] and result["images"]:
                for idx, img_out in enumerate(result["images"]):
                    if len(result["images"]) > 1:
                        out_name = f"{image_path.stem}_edited_{idx + 1}.png"
                    else:
                        out_name = f"{image_path.stem}_edited.png"
                    out_path = output_dir / out_name
                    save_base64_image(img_out["data"], out_path)

                seeds = ", ".join(str(img["seed"]) for img in result["images"])
                n_variants = len(result["images"])
                variant_info = f", {n_variants} variants" if n_variants > 1 else ""
                print(
                    f"OK ({format_time(elapsed)}, seed={seeds}{variant_info})"
                )
                success_count += 1
                times.append(elapsed)
            else:
                error = result.get("error", "unknown error")
                print(f"FAILED ({format_time(elapsed)}) - {error}")
                fail_count += 1

        except requests.exceptions.Timeout:
            elapsed = time.time() - start
            print(f"TIMEOUT ({format_time(elapsed)})")
            fail_count += 1
        except Exception as e:
            elapsed = time.time() - start
            print(f"ERROR ({format_time(elapsed)}) - {e}")
            fail_count += 1

        if i < len(image_files) and args.delay > 0:
            time.sleep(args.delay)

    # Summary
    total_elapsed = time.time() - batch_start
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Processed:  {success_count}/{len(image_files)}")
    print(f"  Failed:     {fail_count}/{len(image_files)}")
    print(f"  Total time: {format_time(total_elapsed)}")
    if times:
        print(f"  Avg/image:  {format_time(sum(times) / len(times))}")
        print(f"  Fastest:    {format_time(min(times))}")
        print(f"  Slowest:    {format_time(max(times))}")
    print(f"  Output:     {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
