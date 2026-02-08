#!/bin/bash
# Install AWS Load Balancer Controller on EKS cluster
# This enables ALB Ingress support for path-based routing

usage() {
    echo "Usage: $(basename "$0")"
    echo ""
    echo "Install the AWS Load Balancer Controller on the EKS cluster."
    echo "Creates IAM policy/role and installs via Helm."
    echo "Skips if the controller is already installed."
    echo ""
    echo "Environment:"
    echo "  ALB_CONTROLLER_VERSION   Controller version (default: v2.9.2)"
    echo ""
    echo "Requires .env to be configured (see .env.example)."
    exit 0
}
[[ "${1:-}" =~ ^(-h|--help)$ ]] && usage

set -e
source "$(dirname "$0")/common.sh"

ALB_VERSION="${ALB_CONTROLLER_VERSION:-v2.9.2}"

echo "Installing AWS Load Balancer Controller..."
echo "Cluster: ${EKS_CLUSTER_NAME}"
echo "Region: ${AWS_REGION}"
echo ""

# Check if controller already exists
if kubectl get deployment -n kube-system aws-load-balancer-controller >/dev/null 2>&1; then
    echo "AWS Load Balancer Controller already installed"
    kubectl get deployment -n kube-system aws-load-balancer-controller
    echo ""
    echo "To update, run:"
    echo "  helm upgrade aws-load-balancer-controller eks/aws-load-balancer-controller -n kube-system"
    exit 0
fi

echo "[1/5] Downloading IAM policy..."
curl -sS "https://raw.githubusercontent.com/kubernetes-sigs/aws-load-balancer-controller/${ALB_VERSION}/docs/install/iam_policy.json" -o /tmp/iam_policy.json

echo "[2/5] Creating IAM policy..."
POLICY_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:policy/AWSLoadBalancerControllerIAMPolicy"

if aws iam get-policy --policy-arn "${POLICY_ARN}" >/dev/null 2>&1; then
    echo "  IAM policy already exists"
else
    aws iam create-policy \
        --policy-name AWSLoadBalancerControllerIAMPolicy \
        --policy-document file:///tmp/iam_policy.json
    echo "  IAM policy created"
fi

echo "[3/5] Creating IAM role and service account..."
eksctl create iamserviceaccount \
    --cluster="${EKS_CLUSTER_NAME}" \
    --namespace=kube-system \
    --name=aws-load-balancer-controller \
    --role-name AmazonEKSLoadBalancerControllerRole \
    --attach-policy-arn="${POLICY_ARN}" \
    --region="${AWS_REGION}" \
    --approve \
    --override-existing-serviceaccounts || echo "  Service account already exists"

echo "[4/5] Adding EKS Helm repository..."
helm repo add eks https://aws.github.io/eks-charts
helm repo update

echo "[5/5] Installing AWS Load Balancer Controller..."
VPC_ID=$(aws eks describe-cluster --name "${EKS_CLUSTER_NAME}" --region "${AWS_REGION}" --query "cluster.resourcesVpcConfig.vpcId" --output text)
helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
    -n kube-system \
    --set clusterName="${EKS_CLUSTER_NAME}" \
    --set serviceAccount.create=false \
    --set serviceAccount.name=aws-load-balancer-controller \
    --set region="${AWS_REGION}" \
    --set vpcId="${VPC_ID}"

echo ""
echo "AWS Load Balancer Controller installation complete!"
echo ""
echo "Verify installation:"
echo "  kubectl get deployment -n kube-system aws-load-balancer-controller"
echo "  kubectl get pods -n kube-system -l app.kubernetes.io/name=aws-load-balancer-controller"
echo ""
echo "View logs:"
echo "  kubectl logs -n kube-system -l app.kubernetes.io/name=aws-load-balancer-controller --tail=50"
