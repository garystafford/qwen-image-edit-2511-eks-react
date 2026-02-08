#!/bin/bash
# Build and push both UI and Model containers to ECR
# Total time: First build ~15 minutes, subsequent ~5-7 minutes

usage() {
    echo "Usage: $(basename "$0") [VERSION]"
    echo ""
    echo "Build and push both model and UI containers to ECR."
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

echo "Building and pushing all containers..."
echo "Version: ${VERSION}"
echo ""

# Build and push model container first (takes longer)
echo "==========================================================="
echo "  MODEL CONTAINER"
echo "==========================================================="
./scripts/build-and-push-model.sh "${VERSION}"

echo ""
echo "==========================================================="
echo "  UI CONTAINER"
echo "==========================================================="
./scripts/build-and-push-ui.sh "${VERSION}"

echo ""
echo "==========================================================="
echo "All containers pushed successfully!"
echo "==========================================================="
echo ""
echo "Images:"
echo "  UI:    ${ECR_REGISTRY}/${ECR_REPO_UI}:${VERSION}"
echo "  Model: ${ECR_REGISTRY}/${ECR_REPO_MODEL}:${VERSION}"
echo ""
echo "To deploy:"
echo "  1. Update config.yaml with version ${VERSION}:"
echo "     vim k8s/base/config.yaml"
echo "  2. Apply with Kustomize:"
echo "     kubectl apply -k k8s/base/"
echo "  3. Watch rollout:"
echo "     kubectl rollout status deployment/qwen-model -n ${K8S_NAMESPACE}"
echo "     kubectl rollout status deployment/qwen-ui -n ${K8S_NAMESPACE}"
