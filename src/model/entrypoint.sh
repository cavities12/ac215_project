#!/bin/bash
set -euo pipefail

echo "[model] Container is running!"

if [ -f "/app/.venv/bin/activate" ]; then
  echo "[model] Starting scripts..."
  source /app/.venv/bin/activate
  echo "[model] Finished scripts!"
  exit 0
else
  echo "[model] ERROR: virtual environment was not found!"
  exit 1
fi
