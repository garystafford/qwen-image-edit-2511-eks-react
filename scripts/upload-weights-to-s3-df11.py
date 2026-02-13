#!/usr/bin/env python3
"""
Build Qwen-Image-Edit-2511-DF11 model artifacts and upload to S3.
Base model: https://huggingface.co/Qwen/Qwen-Image-Edit-2511
DF11 weights: https://huggingface.co/m9e/Qwen-Image-Edit-2511-DF11

Downloads the full Qwen-Image-Edit-2511 diffusers pipeline, then overlays
the transformer weights with DF11 (DFloat11) safetensors from m9e. The
result is a complete model artifact set ready for EKS deployment.

Configuration is loaded from the project .env file.
Requires: huggingface_hub
"""
import os
import sys
import argparse
import glob
import shutil
import subprocess
from pathlib import Path

BASE_MODEL_ID = "Qwen/Qwen-Image-Edit-2511"
DF11_MODEL_ID = "m9e/Qwen-Image-Edit-2511-DF11"
S3_PREFIX_DEFAULT = "Qwen-Image-Edit-2511-DF11"


def load_env():
    """Load .env file from the project root."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    env_file = project_root / ".env"

    if not env_file.exists():
        print(f"Error: {env_file} not found.")
        print("Copy the example and fill in your values:")
        print("  cp .env.example .env")
        sys.exit(1)

    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if key and key not in os.environ:
                os.environ[key] = value


def validate_env():
    """Validate that required .env variables are set."""
    required = ["AWS_ACCOUNT_ID", "AWS_REGION"]
    missing = [var for var in required if not os.environ.get(var)]
    if missing:
        print("Error: Missing required variables in .env:")
        for var in missing:
            print(f"  - {var}")
        sys.exit(1)


def is_model_cached(local_dir):
    """Check if a model directory already has downloaded files."""
    if not os.path.isdir(local_dir):
        return False
    # Check for safetensors or model_index.json as evidence of a completed download
    has_safetensors = glob.glob(os.path.join(local_dir, "**", "*.safetensors"), recursive=True)
    has_index = os.path.exists(os.path.join(local_dir, "model_index.json"))
    return bool(has_safetensors) or has_index


def download_model(model_id, local_dir):
    """Download model from HuggingFace to a local directory."""
    from huggingface_hub import snapshot_download

    if is_model_cached(local_dir):
        print(f"  Already cached: {local_dir} (skipping download)")
        return local_dir

    print(f"Downloading {model_id} from HuggingFace...")
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
    )
    print(f"  Downloaded to: {local_dir}")
    return local_dir


def overlay_df11_weights(base_dir, df11_dir):
    """Overlay DF11 transformer weights onto the base model."""
    transformer_dir = os.path.join(base_dir, "transformer")

    if not os.path.isdir(transformer_dir):
        print(f"✗ Error: transformer/ directory not found in {base_dir}")
        sys.exit(1)

    # Remove existing transformer safetensors and index
    old_files = glob.glob(os.path.join(transformer_dir, "diffusion_pytorch_model*"))
    print(f"  Removing {len(old_files)} existing transformer weight files...")
    for f in old_files:
        os.remove(f)
        print(f"    Removed: {os.path.basename(f)}")

    # Copy DF11 safetensors, config, and index into transformer/
    df11_files = (
        glob.glob(os.path.join(df11_dir, "model-*.safetensors"))
        + glob.glob(os.path.join(df11_dir, "model.safetensors.index.json"))
        + glob.glob(os.path.join(df11_dir, "config.json"))
    )

    print(f"  Copying {len(df11_files)} DF11 files into transformer/...")
    for f in df11_files:
        dest = os.path.join(transformer_dir, os.path.basename(f))
        shutil.copy2(f, dest)
        size_gb = os.path.getsize(dest) / (1024**3)
        print(f"    Copied: {os.path.basename(f)} ({size_gb:.2f} GB)")

    print(f"  DF11 weights overlaid into {transformer_dir}")


def upload_to_s3(local_dir, s3_bucket, s3_prefix, aws_region):
    """Upload model directory to S3 using AWS CLI sync."""
    print(f"Uploading to s3://{s3_bucket}/{s3_prefix}/...")
    print(f"Local directory: {local_dir}")

    s3_uri = f"s3://{s3_bucket}/{s3_prefix}/"
    cmd = ["aws", "s3", "sync", local_dir, s3_uri, "--region", aws_region]

    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"✗ Upload failed with exit code {result.returncode}")
        sys.exit(1)

    print(f"\n✓ Upload complete!")
    print(f"✓ S3 location: {s3_uri}")


def main():
    """Main execution."""
    # Load configuration from .env
    load_env()
    validate_env()

    # Read config from environment (populated by .env)
    aws_account_id = os.environ["AWS_ACCOUNT_ID"]
    aws_region = os.environ["AWS_REGION"]

    # Auto-construct S3 bucket from account ID and region
    s3_bucket = f"{aws_account_id}-sagemaker-{aws_region}"

    # Parse command-line argument overrides
    parser = argparse.ArgumentParser(
        description="Build Qwen-Image-Edit-2511-DF11 model and upload to S3"
    )
    parser.add_argument(
        "--base-model", default=BASE_MODEL_ID, help="Base HuggingFace model ID"
    )
    parser.add_argument(
        "--df11-model", default=DF11_MODEL_ID, help="DF11 HuggingFace model ID"
    )
    parser.add_argument("--s3-bucket", default=s3_bucket, help="S3 bucket name")
    parser.add_argument(
        "--s3-prefix", default=S3_PREFIX_DEFAULT, help="S3 prefix/folder"
    )
    parser.add_argument(
        "--cache-dir", default="./model_cache", help="Local cache directory"
    )
    parser.add_argument("--region", default=aws_region, help="AWS region")
    args = parser.parse_args()

    base_dir = os.path.join(args.cache_dir, "Qwen-Image-Edit-2511")
    df11_dir = os.path.join(args.cache_dir, "Qwen-Image-Edit-2511-DF11")
    merged_dir = base_dir  # Overlay in-place

    print("=== Build Qwen-Image-Edit-2511-DF11 and Upload to S3 ===")
    print(f"Base Model:  {args.base_model}")
    print(f"DF11 Model:  {args.df11_model}")
    print(f"S3 Bucket:   {args.s3_bucket}")
    print(f"S3 Prefix:   {args.s3_prefix}")
    print(f"Region:      {args.region}")
    print(f"Cache Dir:   {args.cache_dir}")
    print()

    # Check for HuggingFace token
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login

        login(token=hf_token)
        print("✓ Logged into HuggingFace")
    else:
        print("Warning: HF_TOKEN not set. Downloads will be unauthenticated (slower).")
        print("  Set HF_TOKEN in .env or export HF_TOKEN=hf_... for faster downloads.")
        print()

    # Step 1: Download full base pipeline
    print("[1/4] Downloading base model pipeline...")
    download_model(args.base_model, base_dir)
    print()

    # Step 2: Download DF11 transformer weights
    print("[2/4] Downloading DF11 transformer weights...")
    download_model(args.df11_model, df11_dir)
    print()

    # Step 3: Overlay DF11 weights onto base model
    print("[3/4] Overlaying DF11 weights onto base transformer...")
    overlay_df11_weights(merged_dir, df11_dir)
    print()

    # Step 4: Upload to S3
    print("[4/4] Uploading merged model to S3...")
    upload_to_s3(merged_dir, args.s3_bucket, args.s3_prefix, args.region)

    print(f"\n✓ All done! Qwen-Image-Edit-2511-DF11 model is now in S3.")
    print(f"  S3 URI: s3://{args.s3_bucket}/{args.s3_prefix}/")


if __name__ == "__main__":
    main()
