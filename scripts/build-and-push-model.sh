#!/bin/bash
# Build and push model container to ECR
# First build: ~10 minutes (with base layer caching)
# Subsequent builds: ~2-3 minutes (if only server.py changes)
# Push time: ~5 minutes (6GB image)

usage() {
    echo "Usage: $(basename "$0") [VERSION]"
    echo ""
    echo "Build and push the model container to ECR."
    echo ""
    echo "Arguments:"
    echo "  VERSION   Image tag (default: 1.0.0)"
    echo ""
    echo "Requires .env to be configured (see .env.example)."
    exit 0
}
[[ "${1:-}" =~ ^(-h|--help)$ ]] && usage

set -e
source "$(dirname "$0")/common.sh"

VERSION="${1:-1.0.0}"

echo "Building and pushing model container..."
echo "Version: ${VERSION}"
echo ""

# Step 1: Build model image (~2-10 minutes)
echo "[1/3] Building model image..."
docker build \
  --platform linux/amd64 \
  -f Dockerfile.model \
  -t "${ECR_REPO_MODEL}:${VERSION}" \
  -t "${ECR_REPO_MODEL}:latest" \
  -t "${ECR_REGISTRY}/${ECR_REPO_MODEL}:${VERSION}" \
  -t "${ECR_REGISTRY}/${ECR_REPO_MODEL}:latest" \
  .

# Step 2: Login to ECR
echo "[2/3] Logging into ECR..."
aws ecr get-login-password --region "${AWS_REGION}" | \
  docker login --username AWS --password-stdin "${ECR_REGISTRY}" > /dev/null 2>&1

# Step 3: Push to ECR (~5 minutes for 6GB image)
echo "[3/3] Pushing to ECR..."
docker push "${ECR_REGISTRY}/${ECR_REPO_MODEL}:${VERSION}"
docker push "${ECR_REGISTRY}/${ECR_REPO_MODEL}:latest"

echo ""
echo "Model container pushed successfully!"
echo ""
echo "Image: ${ECR_REGISTRY}/${ECR_REPO_MODEL}:${VERSION}"
echo ""
echo "To deploy:"
echo "  kubectl set image deployment/qwen-model model=${ECR_REGISTRY}/${ECR_REPO_MODEL}:${VERSION} -n ${K8S_NAMESPACE}"
echo "  kubectl rollout status deployment/qwen-model -n ${K8S_NAMESPACE}"
echo ""
echo "Or update config.yaml and use Kustomize:"
echo "  vim k8s/base/config.yaml  # Change MODEL_VERSION"
echo "  kubectl apply -k k8s/base/"
