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
| `setup-cloudfront-auth.sh` | Setup CloudFront + WAF + Cognito authentication |

### CloudFront Authentication

`setup-cloudfront-auth.sh` creates the full CloudFront + WAF + Cognito stack:

1. Retrieves Cognito User Pool info and domain
2. Creates a new Cognito app client for ALB integration
3. Generates an origin verify secret and stores it in SSM Parameter Store
4. Creates a CloudFront distribution with the origin verify header
5. Updates Route 53 DNS to point to CloudFront
6. Creates a WAF WebACL and associates it with the ALB
7. Creates a CloudFront-restricted security group for the ALB (inbound limited to CloudFront IPs via AWS managed prefix list) and adds backend rules to the EKS cluster SG

Requires `.env` variables: `COGNITO_USER_POOL_ID`, `APP_DOMAIN`, `ALB_NAME`.

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
| `run-tests.sh` | End-to-end deployment tests (health, headers, security, inference) |
| `batch_process_fastapi.py` | Batch process `samples_images/` via FastAPI endpoint |

`run-tests.sh` validates the full deployed stack: pod health, service endpoints,
nginx and CloudFront security headers, asset caching, NetworkPolicies, PDB,
security contexts, and optionally GPU inference (both batch and SSE streaming).

```bash
# Infrastructure tests only (fast, no GPU needed)
./scripts/run-tests.sh

# Full suite including GPU inference
./scripts/run-tests.sh --include-inference
```

> **Note**: With CloudFront + Cognito auth enabled, use `kubectl port-forward`
> to connect directly to the model service. The public URL requires browser login.
>
> ```bash
> kubectl port-forward -n qwen svc/qwen-model-service 8000:8000 &
> sleep 3
> python scripts/batch_process_fastapi.py --url http://localhost:8000
> ```

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

# 5. Setup CloudFront authentication (optional)
./scripts/setup-cloudfront-auth.sh

# 6. Test (port-forward bypasses CloudFront/Cognito auth)
kubectl port-forward -n qwen svc/qwen-model-service 8000:8000 &
sleep 3
python scripts/batch_process_fastapi.py --url http://localhost:8000
```
