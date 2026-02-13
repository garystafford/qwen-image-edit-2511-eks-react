#!/usr/bin/env python3
"""
Download Qwen2.5-VL Image Edit (4-bit quantized) model weights and upload to S3.
Model: https://huggingface.co/ovedrive/Qwen-Image-Edit-2511-4bit

Downloads the model from HuggingFace and syncs to S3 for EKS deployment.
Configuration is loaded from the project .env file.
Requires: huggingface_hub
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path

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
    required = ["AWS_ACCOUNT_ID", "AWS_REGION", "MODEL_ID", "S3_PREFIX"]
    missing = [var for var in required if not os.environ.get(var)]
    if missing:
        print("Error: Missing required variables in .env:")
        for var in missing:
            print(f"  - {var}")
        sys.exit(1)


def download_model(model_id, local_cache_dir):
    """Download model from HuggingFace to local cache."""
    from huggingface_hub import snapshot_download

    print(f"Downloading {model_id} from HuggingFace...")

    # Use cache_dir to get proper HuggingFace cache structure
    # This creates: cache_dir/models--{org}--{model}/snapshots/{hash}/
    cache_dir = os.path.join(local_cache_dir, "hub")

    snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        local_dir_use_symlinks=False,  # Copy files instead of symlinks for S3
    )

    # Find the model directory in cache
    model_cache_name = model_id.replace("/", "--")
    model_cache_dir = os.path.join(cache_dir, f"models--{model_cache_name}")

    print(f"Model downloaded to cache: {model_cache_dir}")
    return model_cache_dir


def upload_to_s3(local_dir, s3_bucket, s3_prefix, aws_region):
    """Upload model cache directory to S3 using AWS CLI sync."""
    print(f"Uploading model cache to s3://{s3_bucket}/{s3_prefix}/...")
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
    print(f"✓ Cache structure preserved for HuggingFace compatibility")


def main():
    """Main execution."""
    # Load configuration from .env
    load_env()
    validate_env()

    # Read config from environment (populated by .env)
    aws_account_id = os.environ["AWS_ACCOUNT_ID"]
    aws_region = os.environ["AWS_REGION"]
    model_id = os.environ["MODEL_ID"]
    s3_prefix = os.environ["S3_PREFIX"]

    # Auto-construct S3 bucket from account ID and region
    s3_bucket = f"{aws_account_id}-sagemaker-{aws_region}"

    # Parse command-line argument overrides
    parser = argparse.ArgumentParser(description="Upload Qwen model weights to S3")
    parser.add_argument("--model-id", default=model_id, help="HuggingFace model ID")
    parser.add_argument("--s3-bucket", default=s3_bucket, help="S3 bucket name")
    parser.add_argument("--s3-prefix", default=s3_prefix, help="S3 prefix/folder")
    parser.add_argument(
        "--cache-dir", default="./model_cache", help="Local cache directory"
    )
    parser.add_argument("--region", default=aws_region, help="AWS region")
    args = parser.parse_args()

    print("=== Upload Model Weights to S3 ===")
    print(f"Model ID:  {args.model_id}")
    print(f"S3 Bucket: {args.s3_bucket}")
    print(f"S3 Prefix: {args.s3_prefix}")
    print(f"Region:    {args.region}")
    print(f"Cache Dir: {args.cache_dir}")
    print()

    # Check for HuggingFace token if needed
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login

        login(token=hf_token)
        print("✓ Logged into HuggingFace")

    # Download model
    local_dir = download_model(args.model_id, args.cache_dir)

    # Upload to S3
    upload_to_s3(local_dir, args.s3_bucket, args.s3_prefix, args.region)

    print(f"\n✓ All done! Model weights are now in S3.")
    print(f"  S3 URI: s3://{args.s3_bucket}/{args.s3_prefix}/")


if __name__ == "__main__":
    main()
