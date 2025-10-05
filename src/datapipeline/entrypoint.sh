#!/bin/bash
set -euo pipefail

echo "[datapipeline] Container is running!"

if [ -f "/app/.venv/bin/activate" ]; then
  echo "[datapipeline] Starting scripts..."
  source /app/.venv/bin/activate
  python -u /app/preprocess.py
  echo "[datapipeline] Finished scripts!"
  exit 0
else
  echo "[datapipeline] ERROR: virtual environment was not found!"
  exit 1
fi
