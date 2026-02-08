#!/bin/bash
# Setup EKS prerequisites for Qwen Image Edit deployment
# This script sets up IAM roles, EFS, and other required resources

usage() {
    echo "Usage: $(basename "$0")"
    echo ""
    echo "Setup EKS prerequisites for Qwen Image Edit deployment."
    echo "Creates IAM roles (IRSA), EFS filesystem and mount targets,"
    echo "EFS StorageClass, and verifies S3 bucket access."
    echo ""
    echo "Requires .env to be configured (see .env.example)."
    exit 0
}
[[ "${1:-}" =~ ^(-h|--help)$ ]] && usage

set -e
source "$(dirname "$0")/common.sh"

SERVICE_ACCOUNT="${K8S_SERVICE_ACCOUNT:-qwen-s3-access}"

# Auto-construct S3 bucket from account ID and region
S3_BUCKET="${AWS_ACCOUNT_ID}-sagemaker-${AWS_REGION}"

echo "=== EKS Prerequisites Setup ==="
echo "Cluster: $EKS_CLUSTER_NAME"
echo "Region: $AWS_REGION"
echo "Account: $AWS_ACCOUNT_ID"
echo "Namespace: $K8S_NAMESPACE"
echo "S3 Bucket: $S3_BUCKET"
echo ""

# Step 1: Create IAM role for S3 access using IRSA
echo "Step 1: Creating IAM role for S3 access (IRSA)..."
eksctl create iamserviceaccount \
	--name "$SERVICE_ACCOUNT" \
	--namespace "$K8S_NAMESPACE" \
	--cluster "$EKS_CLUSTER_NAME" \
	--region "$AWS_REGION" \
	--attach-policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess \
	--approve \
	--override-existing-serviceaccounts

echo "IAM service account created"
echo ""

# Step 2: Setup EFS for shared model cache (optional but recommended for multi-pod)
echo "Step 2: Setting up EFS for shared model cache..."
echo "Installing AWS EFS CSI Driver..."

# Add EFS CSI driver IAM policy
eksctl create iamserviceaccount \
	--name efs-csi-controller-sa \
	--namespace kube-system \
	--cluster "$EKS_CLUSTER_NAME" \
	--region "$AWS_REGION" \
	--attach-policy-arn arn:aws:iam::aws:policy/service-role/AmazonEFSCSIDriverPolicy \
	--approve \
	--override-existing-serviceaccounts

# Install EFS CSI driver
kubectl apply -k "github.com/kubernetes-sigs/aws-efs-csi-driver/deploy/kubernetes/overlays/stable/?ref=master"

echo "EFS CSI driver installed"
echo ""

# Get VPC ID and create EFS filesystem
VPC_ID=$(aws eks describe-cluster --name "$EKS_CLUSTER_NAME" --region "$AWS_REGION" --query "cluster.resourcesVpcConfig.vpcId" --output text)
echo "VPC ID: $VPC_ID"

# Create EFS filesystem
echo "Creating EFS filesystem..."
EFS_ID=$(aws efs create-file-system \
	--creation-token "qwen-model-cache-$(date +%s)" \
	--performance-mode generalPurpose \
	--throughput-mode bursting \
	--encrypted \
	--tags Key=Name,Value=qwen-model-cache \
	--region "$AWS_REGION" \
	--query 'FileSystemId' \
	--output text)

echo "EFS filesystem created: $EFS_ID"
echo ""

# Wait for EFS to be available
echo "Waiting for EFS to be available..."
aws efs describe-file-systems --file-system-id "$EFS_ID" --region "$AWS_REGION" --query 'FileSystems[0].LifeCycleState' --output text
sleep 10

# Create mount targets in each subnet
echo "Creating EFS mount targets..."
SUBNET_IDS=$(aws eks describe-cluster --name "$EKS_CLUSTER_NAME" --region "$AWS_REGION" --query "cluster.resourcesVpcConfig.subnetIds" --output text)
SECURITY_GROUP_ID=$(aws eks describe-cluster --name "$EKS_CLUSTER_NAME" --region "$AWS_REGION" --query "cluster.resourcesVpcConfig.clusterSecurityGroupId" --output text)

for SUBNET_ID in $SUBNET_IDS; do
	echo "  Creating mount target in subnet: $SUBNET_ID"
	aws efs create-mount-target \
		--file-system-id "$EFS_ID" \
		--subnet-id "$SUBNET_ID" \
		--security-groups "$SECURITY_GROUP_ID" \
		--region "$AWS_REGION" || echo "  Mount target may already exist"
done

echo "EFS mount targets created"
echo ""

# Step 3: Create StorageClass for EFS
echo "Step 3: Creating EFS StorageClass..."
cat <<EOF | kubectl apply -f -
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: efs-sc
provisioner: efs.csi.aws.com
parameters:
  provisioningMode: efs-ap
  fileSystemId: $EFS_ID
  directoryPerms: "700"
EOF

echo "EFS StorageClass created"
echo ""

# Step 4: Verify S3 bucket access
echo "Step 4: Verifying S3 bucket access..."
if aws s3 ls "s3://$S3_BUCKET/" --region "$AWS_REGION" >/dev/null 2>&1; then
	echo "S3 bucket accessible: s3://$S3_BUCKET"
else
	echo "Warning: S3 bucket not accessible. Make sure to upload model weights."
fi
echo ""

echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Upload model weights to S3"
echo ""
echo "2. Update k8s/base/config.yaml with:"
echo "   - EFS filesystem ID: $EFS_ID"
echo "   - IAM role ARN from eksctl output above"
echo ""
echo "3. Deploy to EKS:"
echo "   ./scripts/deploy.sh"
