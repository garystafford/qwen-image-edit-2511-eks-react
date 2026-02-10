#!/bin/bash
# End-to-end tests for the Qwen Image Edit deployment.
# Tests health endpoints, security headers, inference (batch + streaming),
# and NetworkPolicy enforcement.
#
# Usage:
#   ./scripts/run-tests.sh                    # All tests (skips inference)
#   ./scripts/run-tests.sh --include-inference # All tests including GPU inference
#   ./scripts/run-tests.sh --help

set -uo pipefail

usage() {
    echo "Usage: $(basename "$0") [OPTIONS]"
    echo ""
    echo "Run end-to-end tests against the deployed Qwen Image Edit stack."
    echo ""
    echo "Options:"
    echo "  --include-inference   Run inference tests (requires GPU pod, ~30s)"
    echo "  --image PATH          Image file for inference test (default: first in samples_images/)"
    echo "  --steps N             Inference steps (default: 8, lower = faster)"
    echo "  -h, --help            Show this help"
    echo ""
    echo "Requires: kubectl configured for the EKS cluster, .env configured."
    exit 0
}

# --- Parse args ---
INCLUDE_INFERENCE=false
INFERENCE_IMAGE=""
INFERENCE_STEPS=8

while [[ $# -gt 0 ]]; do
    case "$1" in
        --include-inference) INCLUDE_INFERENCE=true; shift ;;
        --image) INFERENCE_IMAGE="$2"; shift 2 ;;
        --steps) INFERENCE_STEPS="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# --- Setup ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/common.sh"

PASS=0
FAIL=0
SKIP=0

pass() { echo "  ✓ $1"; PASS=$((PASS + 1)); }
fail() { echo "  ✗ $1"; FAIL=$((FAIL + 1)); }
skip() { echo "  - $1 (skipped)"; SKIP=$((SKIP + 1)); }

cleanup() {
    # Kill any port-forwards we started
    [[ -n "${PF_MODEL_PID:-}" ]] && kill "$PF_MODEL_PID" 2>/dev/null && wait "$PF_MODEL_PID" 2>/dev/null || true
    [[ -n "${PF_UI_PID:-}" ]] && kill "$PF_UI_PID" 2>/dev/null && wait "$PF_UI_PID" 2>/dev/null || true
    rm -f /tmp/qwen-test-*.json /tmp/qwen-test-*.txt 2>/dev/null || true
}
trap cleanup EXIT

echo "======================================"
echo " Qwen Image Edit — End-to-End Tests"
echo "======================================"
echo ""
echo "Namespace:  ${K8S_NAMESPACE}"
echo "Cluster:    ${EKS_CLUSTER_NAME}"
echo "Inference:  ${INCLUDE_INFERENCE}"
echo ""

# =====================================================================
# 1. Pod Health
# =====================================================================
echo "--- Pod Health ---"

# Check all pods are Running
NOT_RUNNING=$(kubectl get pods -n "${K8S_NAMESPACE}" --no-headers 2>/dev/null \
    | grep -v "Running" | grep -v "Completed" || true)
if [[ -z "$NOT_RUNNING" ]]; then
    TOTAL_PODS=$(kubectl get pods -n "${K8S_NAMESPACE}" --no-headers | wc -l | tr -d ' ')
    pass "All ${TOTAL_PODS} pods running"
else
    fail "Some pods not running:"
    echo "$NOT_RUNNING"
fi

# Health check via kubectl exec on model pod
MODEL_POD=$(kubectl get pods -n "${K8S_NAMESPACE}" -l app=qwen-model -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
if [[ -n "$MODEL_POD" ]]; then
    HEALTH=$(kubectl exec -n "${K8S_NAMESPACE}" "$MODEL_POD" -- \
        curl -sf http://localhost:8000/healthz 2>/dev/null || true)
    if echo "$HEALTH" | grep -q '"status"'; then
        pass "Model pod healthz (kubectl exec)"
    else
        fail "Model pod healthz (kubectl exec)"
    fi
else
    fail "No model pod found"
fi

# Health check via kubectl exec on UI pod
UI_POD=$(kubectl get pods -n "${K8S_NAMESPACE}" -l app=qwen-ui -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
if [[ -n "$UI_POD" ]]; then
    HEALTH=$(kubectl exec -n "${K8S_NAMESPACE}" "$UI_POD" -- \
        curl -sf http://localhost:80/healthz 2>/dev/null || true)
    if [[ "$HEALTH" == "ok" ]]; then
        pass "UI pod healthz (kubectl exec)"
    else
        fail "UI pod healthz (kubectl exec)"
    fi
else
    fail "No UI pod found"
fi

echo ""

# =====================================================================
# 2. Port-Forward & Service Tests
# =====================================================================
echo "--- Service Endpoints (via port-forward) ---"

# Start port-forwards
kubectl port-forward "svc/qwen-model-service" 18000:8000 -n "${K8S_NAMESPACE}" &>/dev/null &
PF_MODEL_PID=$!

kubectl port-forward "svc/qwen-ui-service" 18080:80 -n "${K8S_NAMESPACE}" &>/dev/null &
PF_UI_PID=$!

sleep 4

# Model endpoints
for ENDPOINT in "/" "/healthz" "/api/v1/health"; do
    STATUS=$(curl -sf -o /dev/null -w '%{http_code}' "http://localhost:18000${ENDPOINT}" 2>/dev/null || echo "000")
    if [[ "$STATUS" == "200" ]]; then
        pass "Model ${ENDPOINT} → ${STATUS}"
    else
        fail "Model ${ENDPOINT} → ${STATUS}"
    fi
done

# Model /api/v1/health details
HEALTH_JSON=$(curl -sf http://localhost:18000/api/v1/health 2>/dev/null || echo "{}")
GPU=$(echo "$HEALTH_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('gpu_available','?'))" 2>/dev/null || echo "?")
MODEL=$(echo "$HEALTH_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('model_loaded','?'))" 2>/dev/null || echo "?")
if [[ "$GPU" == "True" && "$MODEL" == "True" ]]; then
    pass "Model reports GPU available + model loaded"
else
    fail "Model reports GPU=${GPU}, model_loaded=${MODEL}"
fi

# UI endpoints
STATUS=$(curl -sf -o /dev/null -w '%{http_code}' "http://localhost:18080/healthz" 2>/dev/null || echo "000")
if [[ "$STATUS" == "200" ]]; then
    pass "UI /healthz → ${STATUS}"
else
    fail "UI /healthz → ${STATUS}"
fi

STATUS=$(curl -sf -o /dev/null -w '%{http_code}' "http://localhost:18080/" 2>/dev/null || echo "000")
if [[ "$STATUS" == "200" ]]; then
    pass "UI / → ${STATUS} (serves index.html)"
else
    fail "UI / → ${STATUS}"
fi

echo ""

# =====================================================================
# 3. Nginx Security Headers
# =====================================================================
echo "--- Nginx Security Headers ---"

HEADERS=$(curl -sI "http://localhost:18080/" 2>/dev/null || true)

check_header() {
    local name="$1"
    local expected="$2"
    if echo "$HEADERS" | grep -qi "^${name}:.*${expected}"; then
        pass "nginx: ${name} contains '${expected}'"
    else
        fail "nginx: ${name} missing or wrong (expected '${expected}')"
    fi
}

check_header "X-Frame-Options" "DENY"
check_header "X-Content-Type-Options" "nosniff"
check_header "Referrer-Policy" "strict-origin-when-cross-origin"
check_header "Permissions-Policy" "camera=()"

echo ""

# =====================================================================
# 4. Nginx Asset Caching
# =====================================================================
echo "--- Nginx Asset Caching ---"

# Find an actual asset path from the built HTML
ASSET_PATH=$(curl -sf "http://localhost:18080/" 2>/dev/null \
    | grep -oE '/assets/[^"]+' | head -1 || true)

if [[ -n "$ASSET_PATH" ]]; then
    ASSET_HEADERS=$(curl -sI "http://localhost:18080${ASSET_PATH}" 2>/dev/null || true)
    if echo "$ASSET_HEADERS" | grep -qi "immutable"; then
        pass "Asset Cache-Control includes 'immutable'"
    else
        fail "Asset Cache-Control missing 'immutable'"
    fi
    if echo "$ASSET_HEADERS" | grep -qi "max-age=31536000"; then
        pass "Asset max-age=31536000 (1 year)"
    else
        fail "Asset max-age not 1 year"
    fi
else
    skip "No /assets/* path found in index.html"
fi

echo ""

# =====================================================================
# 5. CloudFront Security Headers
# =====================================================================
echo "--- CloudFront Security Headers ---"

if [[ -n "${APP_DOMAIN:-}" ]]; then
    CF_HEADERS=$(curl -sI "https://${APP_DOMAIN}/" --max-time 10 2>/dev/null || true)
    if [[ -n "$CF_HEADERS" ]]; then
        check_cf_header() {
            local name="$1"
            local expected="$2"
            if echo "$CF_HEADERS" | grep -qi "^${name}:.*${expected}"; then
                pass "CloudFront: ${name} contains '${expected}'"
            else
                fail "CloudFront: ${name} missing or wrong (expected '${expected}')"
            fi
        }

        check_cf_header "strict-transport-security" "max-age"
        check_cf_header "x-frame-options" "SAMEORIGIN"
        check_cf_header "x-content-type-options" "nosniff"
        check_cf_header "referrer-policy" "strict-origin-when-cross-origin"
    else
        skip "CloudFront not reachable at ${APP_DOMAIN}"
    fi
else
    skip "APP_DOMAIN not set in .env"
fi

echo ""

# =====================================================================
# 6. NetworkPolicy Enforcement
# =====================================================================
echo "--- NetworkPolicy ---"

NP_COUNT=$(kubectl get networkpolicy -n "${K8S_NAMESPACE}" --no-headers 2>/dev/null | wc -l | tr -d ' ')
if [[ "$NP_COUNT" -ge 3 ]]; then
    pass "${NP_COUNT} NetworkPolicies applied"
else
    fail "Expected >= 3 NetworkPolicies, found ${NP_COUNT}"
fi

# Check VPC CNI network policy enforcement
NP_MODE=$(kubectl get daemonset -n kube-system aws-node \
    -o jsonpath='{.spec.template.spec.containers[0].env}' 2>/dev/null \
    | python3 -c "
import sys, json
envs = json.load(sys.stdin)
for e in envs:
    if e.get('name') == 'NETWORK_POLICY_ENFORCING_MODE':
        print(e.get('value', ''))
        break
" 2>/dev/null || echo "")

if [[ "$NP_MODE" == "standard" ]]; then
    pass "VPC CNI NETWORK_POLICY_ENFORCING_MODE=standard"
else
    fail "VPC CNI NETWORK_POLICY_ENFORCING_MODE='${NP_MODE}' (expected 'standard')"
fi

# Check aws-eks-nodeagent is running (container within aws-node daemonset)
NA_PRESENT=$(kubectl get daemonset aws-node -n kube-system \
    -o jsonpath='{.spec.template.spec.containers[*].name}' 2>/dev/null || echo "")
if echo "$NA_PRESENT" | grep -q "aws-eks-nodeagent"; then
    AWS_NODE_READY=$(kubectl get daemonset aws-node -n kube-system \
        -o jsonpath='{.status.numberReady}' 2>/dev/null || echo "0")
    pass "aws-eks-nodeagent running in aws-node daemonset (${AWS_NODE_READY} ready)"
else
    fail "aws-eks-nodeagent not found in aws-node daemonset"
fi

echo ""

# =====================================================================
# 7. PodDisruptionBudget
# =====================================================================
echo "--- PodDisruptionBudget ---"

PDB=$(kubectl get pdb -n "${K8S_NAMESPACE}" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
if [[ -n "$PDB" ]]; then
    MIN_AVAIL=$(kubectl get pdb "$PDB" -n "${K8S_NAMESPACE}" -o jsonpath='{.spec.minAvailable}' 2>/dev/null || echo "?")
    pass "PDB '${PDB}' exists (minAvailable=${MIN_AVAIL})"
else
    fail "No PodDisruptionBudget found"
fi

echo ""

# =====================================================================
# 8. Security Context
# =====================================================================
echo "--- Security Context ---"

# Model pod: runAsNonRoot + allowPrivilegeEscalation=false
MODEL_RUN_AS=$(kubectl get deployment qwen-model -n "${K8S_NAMESPACE}" \
    -o jsonpath='{.spec.template.spec.securityContext.runAsNonRoot}' 2>/dev/null || echo "")
if [[ "$MODEL_RUN_AS" == "true" ]]; then
    pass "Model deployment: runAsNonRoot=true"
else
    fail "Model deployment: runAsNonRoot not set"
fi

MODEL_PRIV=$(kubectl get deployment qwen-model -n "${K8S_NAMESPACE}" \
    -o jsonpath='{.spec.template.spec.containers[0].securityContext.allowPrivilegeEscalation}' 2>/dev/null || echo "")
if [[ "$MODEL_PRIV" == "false" ]]; then
    pass "Model container: allowPrivilegeEscalation=false"
else
    fail "Model container: allowPrivilegeEscalation not false"
fi

# UI pod: allowPrivilegeEscalation=false (no runAsNonRoot — nginx needs root for entrypoint)
UI_PRIV=$(kubectl get deployment qwen-ui -n "${K8S_NAMESPACE}" \
    -o jsonpath='{.spec.template.spec.containers[0].securityContext.allowPrivilegeEscalation}' 2>/dev/null || echo "")
if [[ "$UI_PRIV" == "false" ]]; then
    pass "UI container: allowPrivilegeEscalation=false"
else
    fail "UI container: allowPrivilegeEscalation not false"
fi

echo ""

# =====================================================================
# 9. Inference Tests (optional)
# =====================================================================
echo "--- Inference ---"

if [[ "$INCLUDE_INFERENCE" != "true" ]]; then
    skip "Batch inference (use --include-inference)"
    skip "SSE streaming inference (use --include-inference)"
    echo ""
else
    # Find a test image
    if [[ -z "$INFERENCE_IMAGE" ]]; then
        INFERENCE_IMAGE=$(ls "${PROJECT_ROOT}"/samples_images/*.jpg 2>/dev/null | head -1 || true)
    fi

    if [[ -z "$INFERENCE_IMAGE" || ! -f "$INFERENCE_IMAGE" ]]; then
        fail "No test image found (provide --image PATH or add images to samples_images/)"
        echo ""
    else
        # Create test payload
        python3 - "$INFERENCE_IMAGE" "$INFERENCE_STEPS" <<'PYEOF'
import base64, json, sys

image_path = sys.argv[1]
steps = int(sys.argv[2])

with open(image_path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "images": [{"data": b64, "filename": "test.jpg"}],
    "prompt": "Make this image look like a watercolor painting",
    "negative_prompt": "",
    "seed": 42,
    "randomize_seed": False,
    "guidance_scale": 3.5,
    "num_inference_steps": steps,
    "height": 512,
    "width": 512,
    "num_images_per_prompt": 1,
    "style_reference_mode": False,
}

with open("/tmp/qwen-test-payload.json", "w") as f:
    json.dump(payload, f)
PYEOF

        # --- Batch inference ---
        echo "  Running batch inference (${INFERENCE_STEPS} steps)..."
        BATCH_HTTP=$(curl -s -o /tmp/qwen-test-batch.json -w '%{http_code}' \
            -X POST http://localhost:18000/api/v1/batch/infer \
            -H "Content-Type: application/json" \
            -d @/tmp/qwen-test-payload.json \
            --max-time 300 2>/dev/null || echo "000")

        if [[ "$BATCH_HTTP" == "200" ]]; then
            BATCH_OK=$(python3 -c "
import json
with open('/tmp/qwen-test-batch.json') as f:
    d = json.load(f)
print(d.get('success', False), len(d.get('images', [])), d.get('total_time_seconds', 0))
" 2>/dev/null || echo "False 0 0")
            BATCH_SUCCESS=$(echo "$BATCH_OK" | awk '{print $1}')
            BATCH_IMAGES=$(echo "$BATCH_OK" | awk '{print $2}')
            BATCH_TIME=$(echo "$BATCH_OK" | awk '{print $3}')
            if [[ "$BATCH_SUCCESS" == "True" && "$BATCH_IMAGES" -ge 1 ]]; then
                pass "Batch inference: ${BATCH_IMAGES} image(s) in ${BATCH_TIME}s"
            else
                fail "Batch inference: success=${BATCH_SUCCESS}, images=${BATCH_IMAGES}"
            fi
        else
            fail "Batch inference: HTTP ${BATCH_HTTP}"
        fi

        # --- SSE streaming inference ---
        echo "  Running SSE streaming inference (${INFERENCE_STEPS} steps)..."
        curl -s -X POST http://localhost:18000/api/v1/stream/infer \
            -H "Content-Type: application/json" \
            -d @/tmp/qwen-test-payload.json \
            --no-buffer --max-time 300 \
            > /tmp/qwen-test-stream.txt 2>/dev/null || true

        STREAM_RESULT=$(python3 <<'PYEOF'
import json

events = {"started": 0, "progress": 0, "heartbeat": 0, "complete": 0, "error": 0}
complete_data = None

with open("/tmp/qwen-test-stream.txt") as f:
    for line in f:
        line = line.strip()
        if line.startswith("data: "):
            try:
                evt = json.loads(line[6:])
                t = evt.get("type", "")
                events[t] = events.get(t, 0) + 1
                if t == "complete":
                    complete_data = evt
            except json.JSONDecodeError:
                pass

if complete_data and complete_data.get("success"):
    imgs = len(complete_data.get("images", []))
    secs = complete_data.get("total_time_seconds", 0)
    print(f"OK {events['started']} {events['progress']} {events['heartbeat']} {imgs} {secs}")
elif events["error"] > 0:
    print("ERROR 0 0 0 0 0")
else:
    print("INCOMPLETE 0 0 0 0 0")
PYEOF
)
        STREAM_STATUS=$(echo "$STREAM_RESULT" | awk '{print $1}')
        STREAM_STARTED=$(echo "$STREAM_RESULT" | awk '{print $2}')
        STREAM_PROGRESS=$(echo "$STREAM_RESULT" | awk '{print $3}')
        STREAM_HEARTBEAT=$(echo "$STREAM_RESULT" | awk '{print $4}')
        STREAM_IMAGES=$(echo "$STREAM_RESULT" | awk '{print $5}')
        STREAM_TIME=$(echo "$STREAM_RESULT" | awk '{print $6}')

        if [[ "$STREAM_STATUS" == "OK" ]]; then
            pass "SSE streaming: ${STREAM_IMAGES} image(s) in ${STREAM_TIME}s"
            pass "SSE events: ${STREAM_STARTED} started, ${STREAM_PROGRESS} progress, ${STREAM_HEARTBEAT} heartbeat"
        else
            fail "SSE streaming: status=${STREAM_STATUS}"
        fi

        echo ""
    fi
fi

# =====================================================================
# Summary
# =====================================================================
TOTAL=$((PASS + FAIL + SKIP))
echo "======================================"
echo " Results: ${PASS} passed, ${FAIL} failed, ${SKIP} skipped (${TOTAL} total)"
echo "======================================"

if [[ "$FAIL" -gt 0 ]]; then
    exit 1
fi
