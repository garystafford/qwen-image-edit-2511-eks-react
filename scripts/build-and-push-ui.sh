#!/bin/bash
# Build and push UI container to ECR
# Total time: ~1 minute

usage() {
    echo "Usage: $(basename "$0") [VERSION]"
    echo ""
    echo "Build and push the UI container to ECR."
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

echo "Building and pushing UI container..."
echo "Version: ${VERSION}"
echo ""

# Step 1: Build UI image (~30 seconds)
echo "[1/3] Building UI image..."
docker build \
  --platform linux/amd64 \
  -f Dockerfile.ui-react \
  -t "${ECR_REPO_UI}:${VERSION}" \
  -t "${ECR_REPO_UI}:latest" \
  -t "${ECR_REGISTRY}/${ECR_REPO_UI}:${VERSION}" \
  -t "${ECR_REGISTRY}/${ECR_REPO_UI}:latest" \
  .

# Step 2: Login to ECR
echo "[2/3] Logging into ECR..."
aws ecr get-login-password --region "${AWS_REGION}" | \
  docker login --username AWS --password-stdin "${ECR_REGISTRY}" > /dev/null 2>&1

# Step 3: Push to ECR (~20-30 seconds)
echo "[3/3] Pushing to ECR..."
docker push "${ECR_REGISTRY}/${ECR_REPO_UI}:${VERSION}"
docker push "${ECR_REGISTRY}/${ECR_REPO_UI}:latest"

echo ""
echo "UI container pushed successfully!"
echo ""
echo "Image: ${ECR_REGISTRY}/${ECR_REPO_UI}:${VERSION}"
echo ""
echo "To deploy:"
echo "  kubectl set image deployment/qwen-ui ui=${ECR_REGISTRY}/${ECR_REPO_UI}:${VERSION} -n ${K8S_NAMESPACE}"
echo "  kubectl rollout status deployment/qwen-ui -n ${K8S_NAMESPACE}"
