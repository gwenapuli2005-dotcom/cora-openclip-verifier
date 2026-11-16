#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip

python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  "torch==2.5.1+cpu" \
  "torchvision==0.20.1+cpu"

python -m pip install -r requirements.txt
python -m pip check

python -c 'import torch, torchvision, open_clip; print("Dependencies OK")'

python -c 'import open_clip; open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k", device="cpu")'
