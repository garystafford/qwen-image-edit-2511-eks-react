#!/bin/bash

# Script to check GPU instance availability across AWS availability zones
# Usage: ./check-gpu-availability.sh [instance-type]

usage() {
    echo "Usage: $(basename "$0") [INSTANCE_TYPE]"
    echo ""
    echo "Check GPU instance availability across AWS availability zones."
    echo "Tests capacity via dry-run launches and reports ASG status."
    echo ""
    echo "Arguments:"
    echo "  INSTANCE_TYPE   EC2 instance type (default: g6e.xlarge)"
    echo ""
    echo "Requires .env to be configured (see .env.example)."
    exit 0
}
[[ "${1:-}" =~ ^(-h|--help)$ ]] && usage

source "$(dirname "$0")/common.sh"

INSTANCE_TYPE="${1:-g6e.xlarge}"
REGION="${AWS_REGION}"

echo "=================================================="
echo "GPU Instance Availability Checker"
echo "=================================================="
echo "Region: $REGION"
echo "Instance Type: $INSTANCE_TYPE"
echo "Cluster: $EKS_CLUSTER_NAME"
echo ""

# Get VPC and subnets from cluster
echo "1. Getting cluster network configuration..."
VPC_ID=$(aws eks describe-cluster --name "$EKS_CLUSTER_NAME" --region "$REGION" --query 'cluster.resourcesVpcConfig.vpcId' --output text 2>/dev/null)
if [ -z "$VPC_ID" ] || [ "$VPC_ID" = "None" ]; then
  echo "   Could not find cluster: $EKS_CLUSTER_NAME"
  exit 1
fi
echo "   VPC: $VPC_ID"

# Get all subnets in VPC with their AZs
echo ""
echo "2. Available subnets in VPC:"
aws ec2 describe-subnets \
  --filters "Name=vpc-id,Values=$VPC_ID" \
  --region "$REGION" \
  --query 'Subnets[*].[SubnetId, AvailabilityZone, CidrBlock, Tags[?Key==`Name`].Value | [0]]' \
  --output table

# Get ASG name for nodegroup
echo ""
echo "3. Checking Auto Scaling Group activities..."
ASG_NAME=$(aws autoscaling describe-auto-scaling-groups \
  --region "$REGION" \
  --query "AutoScalingGroups[?contains(AutoScalingGroupName, '$EKS_NODEGROUP_NAME')].AutoScalingGroupName" \
  --output text 2>/dev/null)

if [ -n "$ASG_NAME" ] && [ "$ASG_NAME" != "None" ]; then
  echo "   ASG: $ASG_NAME"
  echo ""
  echo "   Recent scaling activities (last 20):"

  # Get activities as JSON for better parsing
  aws autoscaling describe-scaling-activities \
    --auto-scaling-group-name "$ASG_NAME" \
    --region "$REGION" \
    --max-records 20 \
    --output json | jq -r '.Activities[] | "\(.StartTime)\t\(.StatusCode)\t\(.Details)"' | while IFS=$'\t' read -r time status details; do
      # Extract AZ from details JSON
      AZ=$(echo "$details" | jq -r '."Availability Zone" // "unknown"' 2>/dev/null || echo "unknown")

      # Format time (just date and time, no milliseconds)
      TIME_FORMATTED=$(echo "$time" | cut -d'.' -f1 | sed 's/T/ /')

      if [ "$status" = "Failed" ]; then
        echo "   FAILED $TIME_FORMATTED - AZ: $AZ"
      elif [ "$status" = "Successful" ]; then
        echo "   OK $TIME_FORMATTED - AZ: $AZ"
      else
        echo "   PENDING $TIME_FORMATTED - $status - AZ: $AZ"
      fi
    done
else
  echo "   ASG not found for nodegroup: $EKS_NODEGROUP_NAME"
fi

# Check availability zones in region
echo ""
echo "4. Checking all availability zones in $REGION..."
AZS=$(aws ec2 describe-availability-zones \
  --region "$REGION" \
  --query 'AvailabilityZones[?State==`available`].ZoneName' \
  --output text)

echo ""
echo "5. Attempting dry-run launches to test availability:"
echo ""

for AZ in $AZS; do
  echo -n "   Testing $AZ... "

  # Get a subnet in this AZ
  SUBNET=$(aws ec2 describe-subnets \
    --filters "Name=vpc-id,Values=$VPC_ID" "Name=availability-zone,Values=$AZ" \
    --region "$REGION" \
    --query 'Subnets[0].SubnetId' \
    --output text 2>/dev/null)

  if [ "$SUBNET" = "None" ] || [ -z "$SUBNET" ]; then
    echo "No subnet in VPC"
    continue
  fi

  # Get a security group for dry-run
  SG=$(aws ec2 describe-security-groups \
    --filters "Name=vpc-id,Values=$VPC_ID" \
    --region "$REGION" \
    --query 'SecurityGroups[0].GroupId' \
    --output text 2>/dev/null)

  # Get a minimal AMI for dry-run
  AMI=$(aws ec2 describe-images \
    --owners amazon \
    --filters "Name=name,Values=amzn2-ami-hvm-*-x86_64-gp2" \
    --region "$REGION" \
    --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
    --output text 2>/dev/null)

  if [ -z "$AMI" ] || [ "$AMI" = "None" ]; then
    echo "Cannot find test AMI"
    continue
  fi

  # Try dry-run launch
  RESULT=$(aws ec2 run-instances \
    --image-id "$AMI" \
    --instance-type "$INSTANCE_TYPE" \
    --subnet-id "$SUBNET" \
    --security-group-ids "$SG" \
    --dry-run \
    --region "$REGION" \
    2>&1 || true)

  if echo "$RESULT" | grep -q "InsufficientInstanceCapacity"; then
    echo "No capacity"
  elif echo "$RESULT" | grep -q "DryRunOperation"; then
    echo "Capacity available"
  elif echo "$RESULT" | grep -q "Unsupported"; then
    echo "Instance type not supported"
  elif echo "$RESULT" | grep -q "MissingParameter"; then
    echo "Missing parameter (cannot test)"
  elif echo "$RESULT" | grep -q "does not exist"; then
    echo "Resource not found"
  else
    # Show first error line only
    ERROR=$(echo "$RESULT" | grep "An error occurred" | head -1)
    echo "Unknown: ${ERROR:0:60}"
  fi
done

# Check instance type offerings
echo ""
echo "6. Instance type availability by AZ:"
aws ec2 describe-instance-type-offerings \
  --location-type availability-zone \
  --filters "Name=instance-type,Values=$INSTANCE_TYPE" \
  --region "$REGION" \
  --query 'InstanceTypeOfferings[*].[Location, InstanceType]' \
  --output table

echo ""
echo "=================================================="
echo "Summary"
echo "=================================================="
echo ""

if [ -n "$ASG_NAME" ] && [ "$ASG_NAME" != "None" ]; then
  echo "Current ASG Configuration:"
  ASG_INFO=$(aws autoscaling describe-auto-scaling-groups \
    --auto-scaling-group-name "$ASG_NAME" \
    --region "$REGION" \
    --output json)

  DESIRED=$(echo "$ASG_INFO" | jq -r '.AutoScalingGroups[0].DesiredCapacity')
  MIN=$(echo "$ASG_INFO" | jq -r '.AutoScalingGroups[0].MinSize')
  MAX=$(echo "$ASG_INFO" | jq -r '.AutoScalingGroups[0].MaxSize')
  SUBNETS=$(echo "$ASG_INFO" | jq -r '.AutoScalingGroups[0].VPCZoneIdentifier')

  echo "  Desired: $DESIRED, Min: $MIN, Max: $MAX"
  echo "  Subnets:"

  IFS=',' read -ra SUBNET_ARRAY <<< "$SUBNETS"
  for subnet in "${SUBNET_ARRAY[@]}"; do
    SUBNET_AZ=$(aws ec2 describe-subnets --subnet-ids "$subnet" --region "$REGION" --query 'Subnets[0].AvailabilityZone' --output text 2>/dev/null)
    echo "    - $subnet ($SUBNET_AZ)"
  done

  echo ""
  echo "Recent Capacity Issues:"
  aws autoscaling describe-scaling-activities \
    --auto-scaling-group-name "$ASG_NAME" \
    --region "$REGION" \
    --max-records 10 \
    --output json | jq -r '.Activities[] | select(.StatusCode == "Failed") | "\(.StartTime | split(".")[0] | gsub("T"; " "))\t\(.StatusMessage)"' | while IFS=$'\t' read -r time msg; do
      # Truncate message intelligently
      if [ "${#msg}" -gt 120 ]; then
        echo "  $time:"
        echo "    ${msg:0:120}..."
      else
        echo "  $time: $msg"
      fi
    done
fi

echo ""
echo "Recommendations:"
echo "  - Add subnets in AZs with available capacity (if any found)"
echo "  - Consider alternative instance types: g6.2xlarge, g5.xlarge, g5.2xlarge"
echo "  - Try Spot instances (different capacity pool)"
echo "  - Consider different regions: us-west-2, us-east-2"
echo "  - Set up capacity reservations for guaranteed availability"
echo ""
