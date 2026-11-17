#!/usr/bin/env bash
set -euo pipefail
python -m pip install --no-cache-dir -r requirements.txt
python download_models.py
python -m pip check
