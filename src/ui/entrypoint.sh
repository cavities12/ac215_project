#!/bin/bash
set -euo pipefail

echo "[ui] Container is running!"

if [ -f "/app/.venv/bin/activate" ]; then
  echo "[ui] Starting scripts..."
  source /app/.venv/bin/activate
  echo "[ui] Finished scripts!"
  exit 0
else
  echo "[ui] ERROR: virtual environment was not found!"
  exit 1
fi
