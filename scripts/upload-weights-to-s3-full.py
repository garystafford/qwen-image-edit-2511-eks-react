#!/usr/bin/env python3
"""
Download full Qwen-Image-Edit-2511 base model weights and upload to S3.
Model: https://huggingface.co/Qwen/Qwen-Image-Edit-2511

Downloads the complete (unquantized) model from HuggingFace and syncs to S3
for EKS deployment with 8-bit bitsandbytes quantization at load time.
Configuration is loaded from the project .env file.
Requires: huggingface_hub
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path

MODEL_ID_DEFAULT = "Qwen/Qwen-Image-Edit-2511"
S3_PREFIX_DEFAULT = "qwen-image-edit-2511-full"


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


def download_model(model_id, local_dir):
    """Download model from HuggingFace to a local directory.

    Uses a .download_complete marker file to track successful downloads.
    If the marker is missing, snapshot_download runs and resumes any
    partial downloads automatically.
    """
    from huggingface_hub import snapshot_download

    marker = os.path.join(local_dir, ".download_complete")

    if os.path.exists(marker):
        print(f"  Already cached: {local_dir} (skipping download)")
        return local_dir

    print(f"Downloading {model_id} from HuggingFace...")
    if os.path.isdir(local_dir):
        print(f"  Resuming incomplete download in {local_dir}...")

    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
    )

    # Mark download as complete
    with open(marker, "w") as f:
        f.write(f"{model_id}\n")

    print(f"  Downloaded to: {local_dir}")
    return local_dir


def upload_to_s3(local_dir, s3_bucket, s3_prefix, aws_region):
    """Upload model directory to S3 using AWS CLI sync."""
    print(f"Uploading to s3://{s3_bucket}/{s3_prefix}/...")
    print(f"Local directory: {local_dir}")

    s3_uri = f"s3://{s3_bucket}/{s3_prefix}/"
    cmd = [
        "aws", "s3", "sync", local_dir, s3_uri,
        "--region", aws_region,
        "--exclude", ".cache/*",
        "--exclude", ".download_complete",
    ]

    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"Upload failed with exit code {result.returncode}")
        sys.exit(1)

    print(f"\nUpload complete!")
    print(f"S3 location: {s3_uri}")


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
        description="Download full Qwen-Image-Edit-2511 model and upload to S3"
    )
    parser.add_argument(
        "--model-id", default=MODEL_ID_DEFAULT, help="HuggingFace model ID"
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

    local_dir = os.path.join(args.cache_dir, "Qwen-Image-Edit-2511")

    print("=== Upload Full Qwen-Image-Edit-2511 Weights to S3 ===")
    print(f"Model ID:  {args.model_id}")
    print(f"S3 Bucket: {args.s3_bucket}")
    print(f"S3 Prefix: {args.s3_prefix}")
    print(f"Region:    {args.region}")
    print(f"Cache Dir: {args.cache_dir}")
    print()

    # Check for HuggingFace token
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login

        login(token=hf_token)
        print("Logged into HuggingFace")
    else:
        print("Warning: HF_TOKEN not set. Downloads will be unauthenticated (slower).")
        print("  Set HF_TOKEN in .env or export HF_TOKEN=hf_... for faster downloads.")
        print()

    # Step 1: Download full model
    print("[1/2] Downloading full model...")
    download_model(args.model_id, local_dir)
    print()

    # Step 2: Upload to S3
    print("[2/2] Uploading model to S3...")
    upload_to_s3(local_dir, args.s3_bucket, args.s3_prefix, args.region)

    print(f"\nAll done! Full model weights are now in S3.")
    print(f"  S3 URI: s3://{args.s3_bucket}/{args.s3_prefix}/")


if __name__ == "__main__":
    main()
