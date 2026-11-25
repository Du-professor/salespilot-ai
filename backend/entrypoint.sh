#!/usr/bin/env sh
set -eu

DATA_DIR="/srv/salespilot/.salespilot"
mkdir -p "$DATA_DIR"

exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
