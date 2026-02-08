#!/bin/bash
# Create ECR repositories for UI and Model containers

usage() {
    echo "Usage: $(basename "$0")"
    echo ""
    echo "Create ECR repositories for the UI and Model containers."
    echo "Skips creation if repositories already exist."
    echo ""
    echo "Requires .env to be configured (see .env.example)."
    exit 0
}
[[ "${1:-}" =~ ^(-h|--help)$ ]] && usage

set -e
source "$(dirname "$0")/common.sh"

echo "Creating ECR repositories..."
echo ""

# Create UI repository
echo "[1/2] Creating ${ECR_REPO_UI} repository..."
if aws ecr describe-repositories --repository-names "${ECR_REPO_UI}" --region "${AWS_REGION}" >/dev/null 2>&1; then
    echo "  ✓ ${ECR_REPO_UI} repository already exists"
else
    aws ecr create-repository \
        --repository-name "${ECR_REPO_UI}" \
        --region "${AWS_REGION}" \
        --image-scanning-configuration scanOnPush=true \
        --encryption-configuration encryptionType=AES256 \
        --tags Key=Project,Value=qwen-image-edit Key=Component,Value=ui
    echo "  ✓ ${ECR_REPO_UI} repository created"
fi

# Create Model repository
echo "[2/2] Creating ${ECR_REPO_MODEL} repository..."
if aws ecr describe-repositories --repository-names "${ECR_REPO_MODEL}" --region "${AWS_REGION}" >/dev/null 2>&1; then
    echo "  ✓ ${ECR_REPO_MODEL} repository already exists"
else
    aws ecr create-repository \
        --repository-name "${ECR_REPO_MODEL}" \
        --region "${AWS_REGION}" \
        --image-scanning-configuration scanOnPush=true \
        --encryption-configuration encryptionType=AES256 \
        --tags Key=Project,Value=qwen-image-edit Key=Component,Value=model
    echo "  ✓ ${ECR_REPO_MODEL} repository created"
fi

echo ""
echo "ECR repositories ready!"
echo ""
echo "Repository URIs:"
echo "  UI:    ${ECR_REGISTRY}/${ECR_REPO_UI}"
echo "  Model: ${ECR_REGISTRY}/${ECR_REPO_MODEL}"
echo ""
echo "Login to ECR:"
echo "  aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_REGISTRY}"
