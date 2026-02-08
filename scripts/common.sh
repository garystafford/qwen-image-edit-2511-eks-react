#!/bin/bash
# Common configuration for all scripts.
# Sources .env from the project root and validates required variables.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Source .env if it exists
if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a
    source "${PROJECT_ROOT}/.env"
    set +a
else
    echo "Error: ${PROJECT_ROOT}/.env not found."
    echo "Copy the example and fill in your values:"
    echo "  cp .env.example .env"
    exit 1
fi

# Validate required variables
REQUIRED_VARS=(
    AWS_ACCOUNT_ID
    AWS_REGION
    EKS_CLUSTER_NAME
    EKS_NODEGROUP_NAME
    K8S_NAMESPACE
    ECR_REPO_UI
    ECR_REPO_MODEL
)

MISSING=()
for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        MISSING+=("$var")
    fi
done

if [ "${#MISSING[@]}" -gt 0 ]; then
    echo "Error: Missing required variables in .env:"
    for var in "${MISSING[@]}"; do
        echo "  - $var"
    done
    exit 1
fi

# Derived values
export ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
