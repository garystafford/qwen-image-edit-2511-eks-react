# Scripts

All scripts source `common.sh`, which reads `.env` from the project root. No AWS-specific values are hardcoded.

## Prerequisites

```bash
cp .env.example .env    # Fill in your AWS values
chmod +x scripts/*.sh   # Ensure scripts are executable
```

## Setup (One-Time)

| Script | Purpose |
| ------ | ------- |
| `verify-prerequisites.sh` | Check that aws, kubectl, docker, helm are installed |
| `create-ecr-repos.sh` | Create ECR repositories for UI and model images |
| `setup-eks-prerequisites.sh` | Setup IAM roles (IRSA), EFS, S3 access |
| `install-alb-controller.sh` | Install AWS ALB ingress controller on EKS |

## Build and Push

| Script | Purpose |
| ------ | ------- |
| `build-and-push-all.sh <version>` | Build + push both containers to ECR |
| `build-and-push-model.sh <version>` | Build + push model container (~6GB) |
| `build-and-push-ui.sh <version>` | Build + push UI container (~200MB) |

## Deploy

| Script | Purpose |
| ------ | ------- |
| `deploy.sh` | Apply Kustomize manifests (`kubectl apply -k k8s/base/`) |

## Diagnostics

| Script | Purpose |
| ------ | ------- |
| `check-gpu-availability.sh [instance-type]` | Check GPU capacity across availability zones |

## Testing

| Script | Purpose |
| ------ | ------- |
| `batch_process_fastapi.py` | Batch process `samples_images/` via FastAPI endpoint |

## Shared

| Script | Purpose |
| ------ | ------- |
| `common.sh` | Sources `.env`, validates required variables, exports `ECR_REGISTRY` |

## Typical Workflow

```bash
# 1. Verify tools
./scripts/verify-prerequisites.sh

# 2. Create ECR repos
./scripts/create-ecr-repos.sh

# 3. Build and push both containers
./scripts/build-and-push-all.sh v1

# 4. Deploy to EKS
./scripts/deploy.sh

# 5. Test
kubectl port-forward -n qwen svc/qwen-model-service 8000:8000
python scripts/batch_process_fastapi.py
```
