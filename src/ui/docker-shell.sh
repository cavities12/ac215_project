#!/bin/bash
set -euo pipefail

# Configuration details.
IMAGE="ui"
SCRIPT_DIR="$(dirname "$0")"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../" && pwd)"
SECRETS_DIR="$PROJECT_ROOT/secrets"
SERVICE_ACCOUNT="data-sa.json"

# Build the image.
docker build -t "${IMAGE}:latest" .

# Run a container (remove previously running instances).
docker rm -f "$IMAGE" 2>/dev/null || true
docker run --rm -ti \
  --name "$IMAGE" \
  -v "$SECRETS_DIR":/secrets:ro \
  -e GOOGLE_APPLICATION_CREDENTIALS="/secrets/$SERVICE_ACCOUNT" \
  "$IMAGE"
