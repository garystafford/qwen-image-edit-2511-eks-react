# Load .env file if it exists
ifneq (,$(wildcard ./.env))
    include .env
    export
    ECR_REGISTRY = $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com
endif

.PHONY: help verify build-and-push-ui build-and-push-model build-and-push-all deploy port-forward status logs logs-ui logs-model clean batch lint lint-python lint-bash lint-markdown format

# Default target
help:
	@echo "Qwen Image Edit - EKS Deployment"
	@echo ""
	@echo "Prerequisites:"
	@echo "  cp .env.example .env   Copy and fill in your AWS values"
	@echo "  make verify            Verify prerequisites"
	@echo ""
	@echo "Build and Push:"
	@echo "  make build-and-push-ui      Build + push UI (requires VERSION=X.Y.Z)"
	@echo "  make build-and-push-model   Build + push model (requires VERSION=X.Y.Z)"
	@echo "  make build-and-push-all     Build + push both (requires VERSION=X.Y.Z)"
	@echo ""
	@echo "Deploy:"
	@echo "  make deploy          Deploy to EKS using Kustomize"
	@echo ""
	@echo "Status:"
	@echo "  make status          Show deployment status"
	@echo "  make logs            Tail application logs"
	@echo "  make logs-ui         Tail UI logs"
	@echo "  make logs-model      Tail model logs"
	@echo "  make port-forward    Start port-forward to model service"
	@echo ""
	@echo "Testing:"
	@echo "  make batch           Run batch test against FastAPI endpoint"
	@echo ""
	@echo "Linting:"
	@echo "  make lint            Run all linters (python, bash, markdown)"
	@echo "  make lint-python     Run black (check) + ruff on Python files"
	@echo "  make lint-bash       Run shellcheck on shell scripts"
	@echo "  make lint-markdown   Run markdownlint on Markdown files"
	@echo "  make format          Auto-format Python files (black + ruff fix)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean           Remove local build artifacts"
	@echo ""
	@echo "Examples:"
	@echo "  make build-and-push-all VERSION=1.0.0"
	@echo "  make deploy"
	@echo "  make lint"

verify:
	@echo "Verifying prerequisites..."
	./scripts/verify-prerequisites.sh

port-forward:
	@echo "Starting port-forward to model service..."
	kubectl port-forward -n $(K8S_NAMESPACE) svc/qwen-model-service 8000:8000

# Build and push (combined)
build-and-push-ui:
ifndef VERSION
	$(error VERSION is required. Usage: make build-and-push-ui VERSION=1.0.0)
endif
	@echo "Building and pushing UI container $(VERSION)..."
	./scripts/build-and-push-ui.sh $(VERSION)

build-and-push-model:
ifndef VERSION
	$(error VERSION is required. Usage: make build-and-push-model VERSION=1.0.0)
endif
	@echo "Building and pushing model container $(VERSION)..."
	./scripts/build-and-push-model.sh $(VERSION)

build-and-push-all:
ifndef VERSION
	$(error VERSION is required. Usage: make build-and-push-all VERSION=1.0.0)
endif
	@echo "Building and pushing all containers $(VERSION)..."
	./scripts/build-and-push-all.sh $(VERSION)

# Deploy
deploy:
	@echo "Deploying to EKS with Kustomize..."
	./scripts/deploy.sh

# Status and monitoring
status:
	@echo "=== Namespace Status ==="
	kubectl get all -n $(K8S_NAMESPACE)
	@echo ""
	@echo "=== Node Status ==="
	kubectl get nodes -l eks.amazonaws.com/nodegroup=$(EKS_NODEGROUP_NAME)
	@echo ""
	@echo "=== Ingress Status ==="
	kubectl get ingress -n $(K8S_NAMESPACE)

logs:
	@echo "Tailing all $(K8S_NAMESPACE) namespace logs..."
	kubectl logs -n $(K8S_NAMESPACE) -l app=qwen-image-edit --tail=100 -f

logs-ui:
	@echo "Tailing UI logs..."
	kubectl logs -n $(K8S_NAMESPACE) -l app=qwen-ui --tail=100 -f

logs-model:
	@echo "Tailing model logs..."
	kubectl logs -n $(K8S_NAMESPACE) -l app=qwen-model --tail=100 -f

# Testing
batch:
	@echo "Running batch test against FastAPI endpoint..."
	python scripts/batch_process_fastapi.py

# Linting
lint: lint-python lint-bash lint-markdown

lint-python:
	@echo "=== Python: black (check) ==="
	uvx black --check scripts/batch_process_fastapi.py
	@echo ""
	@echo "=== Python: ruff ==="
	uvx ruff check scripts/batch_process_fastapi.py

lint-bash:
	@echo "=== Bash: shellcheck ==="
	shellcheck -S warning scripts/*.sh

lint-markdown:
	@echo "=== Markdown: markdownlint ==="
	npx markdownlint-cli README.md

format:
	@echo "=== Formatting Python files ==="
	uvx black scripts/batch_process_fastapi.py
	uvx ruff check --fix scripts/batch_process_fastapi.py
	@echo "Done."

# Cleanup
clean:
	@echo "Cleaning local artifacts..."
	rm -rf .venv-local/
	rm -rf model_cache/
	rm -rf output_images/
	rm -rf batch_output/
	rm -rf interim_images/
	rm -rf src/__pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "Cleanup complete."
