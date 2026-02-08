#!/bin/bash
# Verify all prerequisites are met before deployment

usage() {
    echo "Usage: $(basename "$0")"
    echo ""
    echo "Verify all prerequisites for EKS deployment."
    echo "Checks: AWS CLI, kubectl, cluster access, GPU nodes,"
    echo "ALB controller, ECR repositories, and Docker."
    echo ""
    echo "Requires .env to be configured (see .env.example)."
    exit 0
}
[[ "${1:-}" =~ ^(-h|--help)$ ]] && usage

set -e
source "$(dirname "$0")/common.sh"

echo "Checking prerequisites for Qwen Image Edit deployment..."
echo ""

EXIT_CODE=0

# Check AWS CLI
echo "[1/7] AWS CLI..."
if command -v aws >/dev/null 2>&1; then
    AWS_VERSION=$(aws --version | cut -d' ' -f1 | cut -d'/' -f2)
    echo "  ✓ AWS CLI installed (${AWS_VERSION})"
else
    echo "  ✗ AWS CLI not found"
    EXIT_CODE=1
fi

# Check kubectl
echo "[2/7] kubectl..."
if command -v kubectl >/dev/null 2>&1; then
    KUBECTL_VERSION=$(kubectl version --client -o json | grep -o '"gitVersion":"[^"]*"' | cut -d'"' -f4)
    echo "  ✓ kubectl installed (${KUBECTL_VERSION})"
else
    echo "  ✗ kubectl not found"
    EXIT_CODE=1
fi

# Check cluster access
echo "[3/7] EKS cluster access..."
if kubectl get nodes >/dev/null 2>&1; then
    NODE_COUNT=$(kubectl get nodes --no-headers | wc -l)
    echo "  ✓ Connected to cluster (${NODE_COUNT} nodes)"
else
    echo "  ✗ Cannot access cluster"
    EXIT_CODE=1
fi

# Check GPU nodes
echo "[4/7] GPU nodes..."
GPU_NODES=$(kubectl get nodes -l nvidia.com/gpu.present=true --no-headers 2>/dev/null | wc -l)
if [ "${GPU_NODES}" -gt 0 ]; then
    echo "  ✓ ${GPU_NODES} GPU node(s) available"
else
    echo "  ✗ No GPU nodes found"
    EXIT_CODE=1
fi

# Check AWS Load Balancer Controller
echo "[5/7] AWS Load Balancer Controller..."
if kubectl get deployment -n kube-system aws-load-balancer-controller >/dev/null 2>&1; then
    ALB_READY=$(kubectl get deployment -n kube-system aws-load-balancer-controller -o jsonpath='{.status.readyReplicas}')
    echo "  ✓ ALB Controller installed (${ALB_READY} replicas ready)"
else
    echo "  ✗ ALB Controller not found - run ./scripts/install-alb-controller.sh"
    EXIT_CODE=1
fi

# Check ECR repositories
echo "[6/7] ECR repositories..."
UI_REPO_EXISTS=$(aws ecr describe-repositories --repository-names "${ECR_REPO_UI}" --region "${AWS_REGION}" 2>/dev/null && echo "yes" || echo "no")
MODEL_REPO_EXISTS=$(aws ecr describe-repositories --repository-names "${ECR_REPO_MODEL}" --region "${AWS_REGION}" 2>/dev/null && echo "yes" || echo "no")

if [ "${UI_REPO_EXISTS}" = "yes" ] && [ "${MODEL_REPO_EXISTS}" = "yes" ]; then
    echo "  ✓ ECR repositories exist (${ECR_REPO_UI}, ${ECR_REPO_MODEL})"
else
    echo "  ✗ ECR repositories missing - run ./scripts/create-ecr-repos.sh"
    [ "${UI_REPO_EXISTS}" = "no" ] && echo "    - ${ECR_REPO_UI} not found"
    [ "${MODEL_REPO_EXISTS}" = "no" ] && echo "    - ${ECR_REPO_MODEL} not found"
    EXIT_CODE=1
fi

# Check Docker
echo "[7/7] Docker..."
if command -v docker >/dev/null 2>&1; then
    DOCKER_VERSION=$(docker --version | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+' | head -1)
    echo "  ✓ Docker installed (${DOCKER_VERSION})"
else
    echo "  ✗ Docker not found"
    EXIT_CODE=1
fi

echo ""
if [ "${EXIT_CODE}" -eq 0 ]; then
    echo "All prerequisites met!"
    echo ""
    echo "Next steps:"
    echo "  1. Build and push images:"
    echo "     ./scripts/build-and-push-all.sh v1"
    echo ""
    echo "  2. Deploy to EKS:"
    echo "     ./scripts/deploy.sh"
else
    echo "Some prerequisites are missing. Please fix the issues above."
fi

exit "${EXIT_CODE}"
