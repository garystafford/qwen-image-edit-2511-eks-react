#!/bin/bash
# Setup CloudFront + WAF + Cognito authentication for the Qwen Image Edit app.
#
# This script replaces IP-based ALB access control with:
#   CloudFront → WAF (origin verify header) → ALB (Cognito auth) → EKS pods
#
# Prerequisites:
#   - .env configured with COGNITO_USER_POOL_ID, APP_DOMAIN, ALB_NAME
#   - Cognito User Pool already exists with a configured domain
#   - ACM certificate in us-east-1 (for CloudFront)
#   - ALB deployed via kubectl apply -k k8s/base/
#   - K8s manifests updated with Cognito auth annotations (ingress.yaml)
#
# Usage: ./scripts/setup-cloudfront-auth.sh [step]
#   No argument: runs all steps in sequence
#   1-6: runs only the specified step
#
# Steps:
#   1  Retrieve Cognito User Pool info
#   2  Create Cognito App Client for ALB
#   3  Generate origin verify secret (stored in SSM)
#   4  Create CloudFront distribution
#   5  Update Route 53 DNS
#   6  Create WAF WebACL and associate with ALB

usage() {
    echo "Usage: $(basename "$0") [step]"
    echo ""
    echo "Setup CloudFront + WAF + Cognito authentication."
    echo ""
    echo "Steps:"
    echo "  1  Retrieve Cognito User Pool info"
    echo "  2  Create Cognito App Client for ALB"
    echo "  3  Generate origin verify secret (stored in SSM)"
    echo "  4  Create CloudFront distribution"
    echo "  5  Update Route 53 DNS"
    echo "  6  Create WAF WebACL and associate with ALB"
    echo ""
    echo "Run without arguments to execute all steps."
    echo "Run with a step number to execute only that step."
    echo ""
    echo "Requires .env to be configured (see .env.example)."
    exit 0
}
[[ "${1:-}" =~ ^(-h|--help)$ ]] && usage

set -e
source "$(dirname "$0")/common.sh"

# Validate CloudFront-specific variables
for var in COGNITO_USER_POOL_ID APP_DOMAIN ALB_NAME; do
    if [ -z "${!var}" ]; then
        echo "Error: $var is not set in .env"
        exit 1
    fi
done

SSM_PARAM_NAME="/qwen/prod/origin-verify-secret"
COGNITO_CLIENT_NAME="qwen-alb-client"

# ---------------------------------------------------------------------------
# Step 1: Retrieve Cognito User Pool info
# ---------------------------------------------------------------------------
step_1() {
    echo "============================================================"
    echo "Step 1: Retrieving Cognito User Pool info"
    echo "============================================================"

    COGNITO_USER_POOL_ARN=$(aws cognito-idp describe-user-pool \
        --user-pool-id "${COGNITO_USER_POOL_ID}" \
        --region "${AWS_REGION}" \
        --query 'UserPool.Arn' \
        --output text)
    echo "User Pool ARN: ${COGNITO_USER_POOL_ARN}"

    COGNITO_DOMAIN=$(aws cognito-idp describe-user-pool \
        --user-pool-id "${COGNITO_USER_POOL_ID}" \
        --region "${AWS_REGION}" \
        --query 'UserPool.Domain' \
        --output text)

    if [ -z "${COGNITO_DOMAIN}" ] || [ "${COGNITO_DOMAIN}" = "None" ]; then
        echo "Error: No Cognito domain configured on user pool ${COGNITO_USER_POOL_ID}"
        echo "Create one with: aws cognito-idp create-user-pool-domain --user-pool-id ${COGNITO_USER_POOL_ID} --domain <prefix>"
        exit 1
    fi

    COGNITO_FULL_DOMAIN="${COGNITO_DOMAIN}.auth.${AWS_REGION}.amazoncognito.com"
    echo "Cognito domain: ${COGNITO_FULL_DOMAIN}"
    echo ""
}

# ---------------------------------------------------------------------------
# Step 2: Create Cognito App Client for ALB
# ---------------------------------------------------------------------------
step_2() {
    echo "============================================================"
    echo "Step 2: Creating Cognito App Client for ALB"
    echo "============================================================"

    # Check if client already exists
    EXISTING_CLIENT_ID=$(aws cognito-idp list-user-pool-clients \
        --user-pool-id "${COGNITO_USER_POOL_ID}" \
        --region "${AWS_REGION}" \
        --query "UserPoolClients[?ClientName=='${COGNITO_CLIENT_NAME}'].ClientId" \
        --output text 2>/dev/null || true)

    if [ -n "${EXISTING_CLIENT_ID}" ] && [ "${EXISTING_CLIENT_ID}" != "None" ]; then
        echo "App Client already exists: ${EXISTING_CLIENT_ID}"
        COGNITO_CLIENT_ID="${EXISTING_CLIENT_ID}"
    else
        COGNITO_CLIENT_ID=$(aws cognito-idp create-user-pool-client \
            --user-pool-id "${COGNITO_USER_POOL_ID}" \
            --client-name "${COGNITO_CLIENT_NAME}" \
            --generate-secret \
            --allowed-o-auth-flows code \
            --allowed-o-auth-flows-user-pool-client \
            --allowed-o-auth-scopes openid \
            --callback-urls "https://${APP_DOMAIN}/oauth2/idpresponse" \
            --supported-identity-providers COGNITO \
            --region "${AWS_REGION}" \
            --query 'UserPoolClient.ClientId' \
            --output text)
        echo "Created App Client: ${COGNITO_CLIENT_ID}"
    fi

    echo ""
    echo ">>> Update k8s/base/config.yaml COGNITO_IDP_CONFIG with:"
    echo "    userPoolClientID: ${COGNITO_CLIENT_ID}"
    echo "    userPoolDomain: ${COGNITO_FULL_DOMAIN:-<run step 1 first>}"
    echo ""
}

# ---------------------------------------------------------------------------
# Step 3: Generate origin verify secret
# ---------------------------------------------------------------------------
step_3() {
    echo "============================================================"
    echo "Step 3: Generating origin verify secret"
    echo "============================================================"

    # Check if secret already exists in SSM
    EXISTING_SECRET=$(aws ssm get-parameter \
        --name "${SSM_PARAM_NAME}" \
        --with-decryption \
        --query "Parameter.Value" \
        --output text \
        --region "${AWS_REGION}" 2>/dev/null || true)

    if [ -n "${EXISTING_SECRET}" ] && [ "${EXISTING_SECRET}" != "None" ]; then
        echo "Using existing secret from SSM: ${SSM_PARAM_NAME}"
        ORIGIN_VERIFY_SECRET="${EXISTING_SECRET}"
    else
        ORIGIN_VERIFY_SECRET=$(python3 -c "import uuid; print(str(uuid.uuid4()))")
        aws ssm put-parameter \
            --name "${SSM_PARAM_NAME}" \
            --value "${ORIGIN_VERIFY_SECRET}" \
            --type SecureString \
            --description "Origin verify header secret for CloudFront-to-ALB WAF validation" \
            --region "${AWS_REGION}"
        echo "Stored new secret in SSM: ${SSM_PARAM_NAME}"
    fi

    echo ""
}

# ---------------------------------------------------------------------------
# Step 4: Create CloudFront distribution
# ---------------------------------------------------------------------------
step_4() {
    echo "============================================================"
    echo "Step 4: Creating CloudFront distribution"
    echo "============================================================"

    # Load origin verify secret
    ORIGIN_VERIFY_SECRET=$(aws ssm get-parameter \
        --name "${SSM_PARAM_NAME}" \
        --with-decryption \
        --query "Parameter.Value" \
        --output text \
        --region "${AWS_REGION}")

    # Get ALB DNS name
    ALB_DNS=$(aws elbv2 describe-load-balancers \
        --names "${ALB_NAME}" \
        --region "${AWS_REGION}" \
        --query 'LoadBalancers[0].DNSName' \
        --output text)
    echo "ALB DNS: ${ALB_DNS}"

    # Get certificate ARN from config.yaml
    CERT_ARN=$(grep 'CERTIFICATE_ARN:' "${PROJECT_ROOT}/k8s/base/config.yaml" \
        | head -1 | sed 's/.*CERTIFICATE_ARN: *"//' | sed 's/".*//')
    echo "Certificate ARN: ${CERT_ARN}"

    # Check if distribution already exists for this domain
    EXISTING_DIST=$(aws cloudfront list-distributions \
        --query "DistributionList.Items[?contains(Aliases.Items, '${APP_DOMAIN}')].Id" \
        --output text 2>/dev/null || true)

    if [ -n "${EXISTING_DIST}" ] && [ "${EXISTING_DIST}" != "None" ]; then
        echo "CloudFront distribution already exists: ${EXISTING_DIST}"
        CF_DOMAIN=$(aws cloudfront get-distribution \
            --id "${EXISTING_DIST}" \
            --query 'Distribution.DomainName' \
            --output text)
        echo "CloudFront domain: ${CF_DOMAIN}"
        echo ""
        return
    fi

    echo "Creating CloudFront distribution..."
    CALLER_REF="qwen-$(date +%s)"

    RESULT=$(aws cloudfront create-distribution \
        --distribution-config '{
            "CallerReference": "'"${CALLER_REF}"'",
            "Comment": "qwen-image-edit CDN",
            "Enabled": true,
            "HttpVersion": "http2and3",
            "PriceClass": "PriceClass_100",
            "Aliases": {
                "Quantity": 1,
                "Items": ["'"${APP_DOMAIN}"'"]
            },
            "ViewerCertificate": {
                "ACMCertificateArn": "'"${CERT_ARN}"'",
                "SSLSupportMethod": "sni-only",
                "MinimumProtocolVersion": "TLSv1.2_2021"
            },
            "DefaultCacheBehavior": {
                "TargetOriginId": "alb-origin",
                "ViewerProtocolPolicy": "redirect-to-https",
                "AllowedMethods": {
                    "Quantity": 7,
                    "Items": ["GET","HEAD","OPTIONS","PUT","PATCH","POST","DELETE"],
                    "CachedMethods": {
                        "Quantity": 2,
                        "Items": ["GET","HEAD"]
                    }
                },
                "CachePolicyId": "4135ea2d-6df8-44a3-9df3-4b5a84be39ad",
                "OriginRequestPolicyId": "216adef6-5c7f-47e4-b989-5492eafa07d3",
                "Compress": true
            },
            "Origins": {
                "Quantity": 1,
                "Items": [{
                    "Id": "alb-origin",
                    "DomainName": "'"${ALB_DNS}"'",
                    "CustomOriginConfig": {
                        "HTTPPort": 80,
                        "HTTPSPort": 443,
                        "OriginProtocolPolicy": "https-only",
                        "OriginSslProtocols": {
                            "Quantity": 1,
                            "Items": ["TLSv1.2"]
                        },
                        "OriginReadTimeout": 60,
                        "OriginKeepaliveTimeout": 5
                    },
                    "CustomHeaders": {
                        "Quantity": 1,
                        "Items": [{
                            "HeaderName": "X-Origin-Verify",
                            "HeaderValue": "'"${ORIGIN_VERIFY_SECRET}"'"
                        }]
                    }
                }]
            }
        }')

    CF_DIST_ID=$(echo "${RESULT}" | python3 -c "import sys,json; print(json.load(sys.stdin)['Distribution']['Id'])")
    CF_DOMAIN=$(echo "${RESULT}" | python3 -c "import sys,json; print(json.load(sys.stdin)['Distribution']['DomainName'])")

    echo "Created distribution: ${CF_DIST_ID}"
    echo "CloudFront domain: ${CF_DOMAIN}"
    echo ""
    echo "Waiting for distribution to deploy (this may take several minutes)..."
    aws cloudfront wait distribution-deployed --id "${CF_DIST_ID}"
    echo "Distribution deployed."
    echo ""
}

# ---------------------------------------------------------------------------
# Step 5: Update Route 53 DNS
# ---------------------------------------------------------------------------
step_5() {
    echo "============================================================"
    echo "Step 5: Updating Route 53 DNS"
    echo "============================================================"

    # Extract base domain from APP_DOMAIN (e.g., example.com from app.example.com)
    BASE_DOMAIN=$(echo "${APP_DOMAIN}" | awk -F. '{print $(NF-1)"."$NF}')

    HOSTED_ZONE_ID=$(aws route53 list-hosted-zones-by-name \
        --dns-name "${BASE_DOMAIN}" \
        --query 'HostedZones[0].Id' \
        --output text | sed 's|/hostedzone/||')

    if [ -z "${HOSTED_ZONE_ID}" ] || [ "${HOSTED_ZONE_ID}" = "None" ]; then
        echo "Error: No hosted zone found for ${BASE_DOMAIN}"
        exit 1
    fi
    echo "Hosted Zone: ${HOSTED_ZONE_ID}"

    # Get CloudFront domain
    CF_DOMAIN=$(aws cloudfront list-distributions \
        --query "DistributionList.Items[?contains(Aliases.Items, '${APP_DOMAIN}')].DomainName | [0]" \
        --output text)

    if [ -z "${CF_DOMAIN}" ] || [ "${CF_DOMAIN}" = "None" ]; then
        echo "Error: No CloudFront distribution found for ${APP_DOMAIN}"
        echo "Run step 4 first."
        exit 1
    fi
    echo "CloudFront domain: ${CF_DOMAIN}"

    echo "Updating ${APP_DOMAIN} → ${CF_DOMAIN}..."
    aws route53 change-resource-record-sets \
        --hosted-zone-id "${HOSTED_ZONE_ID}" \
        --change-batch '{
            "Changes": [{
                "Action": "UPSERT",
                "ResourceRecordSet": {
                    "Name": "'"${APP_DOMAIN}"'",
                    "Type": "A",
                    "AliasTarget": {
                        "DNSName": "'"${CF_DOMAIN}"'",
                        "HostedZoneId": "Z2FDTNDATAQYW2",
                        "EvaluateTargetHealth": false
                    }
                }
            }]
        }'

    echo "DNS updated. Propagation may take a few minutes."
    echo ""
}

# ---------------------------------------------------------------------------
# Step 6: Create WAF WebACL and associate with ALB
# ---------------------------------------------------------------------------
step_6() {
    echo "============================================================"
    echo "Step 6: Creating WAF WebACL and associating with ALB"
    echo "============================================================"

    # Load origin verify secret
    ORIGIN_VERIFY_SECRET=$(aws ssm get-parameter \
        --name "${SSM_PARAM_NAME}" \
        --with-decryption \
        --query "Parameter.Value" \
        --output text \
        --region "${AWS_REGION}")

    # Get ALB ARN
    ALB_ARN=$(aws elbv2 describe-load-balancers \
        --names "${ALB_NAME}" \
        --region "${AWS_REGION}" \
        --query 'LoadBalancers[0].LoadBalancerArn' \
        --output text)
    echo "ALB ARN: ${ALB_ARN}"

    # Check if WAF already associated
    EXISTING_WAF=$(aws wafv2 get-web-acl-for-resource \
        --resource-arn "${ALB_ARN}" \
        --region "${AWS_REGION}" \
        --query 'WebACL.Name' \
        --output text 2>/dev/null || true)

    if [ -n "${EXISTING_WAF}" ] && [ "${EXISTING_WAF}" != "None" ]; then
        echo "WAF WebACL already associated with ALB: ${EXISTING_WAF}"
        echo ""
        return
    fi

    # Check if WebACL already exists
    EXISTING_ACL_ARN=$(aws wafv2 list-web-acls \
        --scope REGIONAL \
        --region "${AWS_REGION}" \
        --query "WebACLs[?Name=='qwen-origin-verify'].ARN | [0]" \
        --output text 2>/dev/null || true)

    if [ -n "${EXISTING_ACL_ARN}" ] && [ "${EXISTING_ACL_ARN}" != "None" ]; then
        echo "WebACL already exists: ${EXISTING_ACL_ARN}"
        WEBACL_ARN="${EXISTING_ACL_ARN}"
    else
        echo "Creating WAF WebACL..."
        # Base64 encode the secret for the WAF SearchString blob field
        SECRET_B64=$(echo -n "${ORIGIN_VERIFY_SECRET}" | base64)
        WEBACL_ARN=$(aws wafv2 create-web-acl \
            --name "qwen-origin-verify" \
            --scope REGIONAL \
            --region "${AWS_REGION}" \
            --default-action Block={} \
            --description "Blocks requests not originating from CloudFront for qwen" \
            --rules '[{
                "Name": "verify-origin-header",
                "Priority": 0,
                "Action": {"Allow": {}},
                "Statement": {
                    "ByteMatchStatement": {
                        "FieldToMatch": {"SingleHeader": {"Name": "x-origin-verify"}},
                        "PositionalConstraint": "EXACTLY",
                        "SearchString": "'"${SECRET_B64}"'",
                        "TextTransformations": [{"Priority": 0, "Type": "NONE"}]
                    }
                },
                "VisibilityConfig": {
                    "CloudWatchMetricsEnabled": true,
                    "MetricName": "qwen-origin-verify-rule",
                    "SampledRequestsEnabled": true
                }
            }]' \
            --visibility-config '{
                "CloudWatchMetricsEnabled": true,
                "MetricName": "qwen-waf",
                "SampledRequestsEnabled": true
            }' \
            --query 'Summary.ARN' \
            --output text)
        echo "Created WebACL: ${WEBACL_ARN}"
    fi

    echo "Associating WAF with ALB..."
    aws wafv2 associate-web-acl \
        --web-acl-arn "${WEBACL_ARN}" \
        --resource-arn "${ALB_ARN}" \
        --region "${AWS_REGION}"
    echo "WAF associated with ALB."
    echo ""
    echo "WARNING: Direct access to the ALB is now blocked."
    echo "Only requests through CloudFront (with the X-Origin-Verify header) will succeed."
    echo ""
}

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
verify() {
    echo "============================================================"
    echo "Verification"
    echo "============================================================"

    # Check CloudFront
    CF_STATUS=$(aws cloudfront list-distributions \
        --query "DistributionList.Items[?contains(Aliases.Items, '${APP_DOMAIN}')].Status | [0]" \
        --output text 2>/dev/null || echo "NOT FOUND")
    echo "CloudFront status: ${CF_STATUS}"

    # Check WAF association
    ALB_ARN=$(aws elbv2 describe-load-balancers \
        --names "${ALB_NAME}" \
        --region "${AWS_REGION}" \
        --query 'LoadBalancers[0].LoadBalancerArn' \
        --output text 2>/dev/null || echo "NOT FOUND")
    WAF_NAME=$(aws wafv2 get-web-acl-for-resource \
        --resource-arn "${ALB_ARN}" \
        --region "${AWS_REGION}" \
        --query 'WebACL.Name' \
        --output text 2>/dev/null || echo "NOT ASSOCIATED")
    echo "WAF on ALB: ${WAF_NAME}"

    # Check DNS
    echo ""
    echo "DNS resolution for ${APP_DOMAIN}:"
    dig +short "${APP_DOMAIN}" 2>/dev/null || nslookup "${APP_DOMAIN}" 2>/dev/null || echo "dig/nslookup not available"

    # Check K8s ingress
    echo ""
    echo "K8s ingress auth-type:"
    kubectl get ingress qwen-ingress -n "${K8S_NAMESPACE}" \
        -o jsonpath='{.metadata.annotations.alb\.ingress\.kubernetes\.io/auth-type}' 2>/dev/null || echo "NOT SET"
    echo ""

    echo ""
    echo "To test the full flow, open https://${APP_DOMAIN} in a browser."
    echo "You should be redirected to the Cognito login page."
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
STEP="${1:-all}"

case "${STEP}" in
    1) step_1 ;;
    2) step_1; step_2 ;;
    3) step_3 ;;
    4) step_4 ;;
    5) step_5 ;;
    6) step_6 ;;
    all)
        step_1
        step_2
        step_3
        step_4
        step_5
        step_6
        verify
        ;;
    verify) verify ;;
    *)
        echo "Unknown step: ${STEP}"
        usage
        ;;
esac
